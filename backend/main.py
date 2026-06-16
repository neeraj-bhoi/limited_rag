import os
import json
import time
import requests
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# We default to qwen2.5-coder:7b as requested for faster inference
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEBUG_PRINT_CHUNKS = os.getenv("DEBUG_PRINT_CHUNKS", "false").lower() == "true"
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "processed_data", "rag_kb_manifest.json")

# Sarvam AI API Configuration
USE_SARVAM_API = os.getenv("USE_SARVAM_API", "false").lower() == "true"
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-30b")

app = FastAPI(title="SewaSetu RAG API Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory manifest cache
MANIFEST_DATA = None
SERVICES_MAP = {}

@app.on_event("startup")
def load_manifest():
    global MANIFEST_DATA, SERVICES_MAP
    if not os.path.exists(DATA_MANIFEST_PATH):
        print(f"[API] Warning: Manifest file not found at: {DATA_MANIFEST_PATH}")
        return
        
    try:
        with open(DATA_MANIFEST_PATH, "r", encoding="utf-8") as f:
            MANIFEST_DATA = json.load(f)
            
        for s in MANIFEST_DATA.get("services", []):
            SERVICES_MAP[str(s.get("sno"))] = s
        print(f"[API] Cached {len(SERVICES_MAP)} services from manifest.")
    except Exception as e:
        print(f"[API] Error loading manifest catalog: {e}")

    # Pre-load embedder model on startup to avoid first-query latency
    try:
        from backend.embeddings.embedder import get_embedder_model
        print("[API] Pre-loading embedding model on startup...")
        get_embedder_model()
        print("[API] Embedding model pre-loaded successfully.")
    except Exception as e:
        print(f"[API] Error pre-loading embedding model: {e}")

# API Validation schemas
class Message(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    selected_sno: Optional[str] = None
    language: str = "en" # 'en' or 'hi'

class SearchRequest(BaseModel):
    query: str
    language: str = "en"

@app.get("/api/services")
def list_services():
    """
    Returns the list of all services in the catalog for browsing.
    """
    if not MANIFEST_DATA:
        # Try loading manifest on the fly if not loaded yet
        load_manifest()
        if not MANIFEST_DATA:
            raise HTTPException(status_code=500, detail="Manifest database not loaded on the backend.")
    return MANIFEST_DATA.get("services", [])

@app.get("/api/services/{sno}")
def get_service_details(sno: str, lang: str = "en"):
    """
    Retrieves the detailed JSON profile of a specific service.
    """
    if not SERVICES_MAP:
        load_manifest()
        
    if sno not in SERVICES_MAP:
        raise HTTPException(status_code=404, detail="Service not found.")
        
    service_meta = SERVICES_MAP[sno]
    path_key = "path_hi" if lang == "hi" else "path_en"
    rel_path = service_meta.get(path_key)
    
    if not rel_path:
        raise HTTPException(status_code=404, detail=f"Service details path not found for language: {lang}")
        
    full_path = os.path.join(WORKSPACE_DIR, rel_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Service details file missing on disk.")
        
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            details_json = json.load(f)
            
        if lang == "hi":
            # Also try to load the English version for fallback values
            en_rel_path = service_meta.get("path_en")
            if en_rel_path:
                en_full_path = os.path.join(WORKSPACE_DIR, en_rel_path)
                if os.path.exists(en_full_path):
                    with open(en_full_path, "r", encoding="utf-8") as en_f:
                        en_details = json.load(en_f)
                    
                    # Merge keys if they are empty or missing in Hindi
                    for key in ["sla", "time_limit", "contact_details"]:
                        if not details_json.get(key) and en_details.get(key):
                            val = en_details.get(key)
                            if key in ["sla", "time_limit"] and isinstance(val, str):
                                val_translated = val.replace("Days", "दिन").replace("Day", "दिन")
                                details_json[key] = val_translated
                            elif key == "contact_details" and val == "Sewa Setu Kendra":
                                details_json[key] = "सेवा सेतु केंद्र"
                            else:
                                details_json[key] = val
                                
                    # Merge fees
                    hi_fees = details_json.get("fees")
                    en_fees = en_details.get("fees")
                    if en_fees:
                        if not hi_fees:
                            details_json["fees"] = en_fees
                        else:
                            for fee_key in ["kiosk_fee", "online_fee", "where_to_apply", "raw_text"]:
                                if not hi_fees.get(fee_key) and en_fees.get(fee_key):
                                    val = en_fees.get(fee_key)
                                    if fee_key == "raw_text" and isinstance(val, str):
                                        val_translated = val.replace("Where to Apply?", "कहाँ आवेदन करें?").replace("Sewa Setu Kendra", "सेवा सेतु केंद्र").replace("Online", "ऑनलाइन")
                                        hi_fees[fee_key] = val_translated
                                    elif fee_key == "where_to_apply" and val == "Sewa Setu Kendra":
                                        hi_fees[fee_key] = "सेवा सेतु केंद्र"
                                    else:
                                        hi_fees[fee_key] = val
                                        
                    # Merge downloaded_pdfs
                    if not details_json.get("downloaded_pdfs") and en_details.get("downloaded_pdfs"):
                        details_json["downloaded_pdfs"] = en_details.get("downloaded_pdfs")
                        
        return details_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading details file: {str(e)}")

@app.post("/api/chat")
def chat_with_bot(request: ChatRequest):
    """
    Core RAG Chat endpoint querying the local Ollama LLM via HTTP.
    1. Retrieves relevant context from modular ChromaStore.
    2. Constructs a bilingual semantic prompt.
    3. Queries Ollama local server and returns response.
    """
    user_query = request.messages[-1].content
    language = request.language
    
    # 1. Query Chroma vector store (with timing)
    candidates = []
    metadata_doc = None
    retrieval_start = time.perf_counter()
    try:
        from backend.vector_store.chroma_store import query_vector_store, get_collection
        
        # If a service is selected, retrieve its metadata profile directly
        if request.selected_sno:
            try:
                collection = get_collection()
                doc_id = f"meta_{request.selected_sno}_{language}_0"
                res = collection.get(ids=[doc_id])
                if res and "documents" in res and len(res["documents"]) > 0 and res["documents"][0]:
                    metadata_doc = res["documents"][0]
            except Exception as e:
                print(f"[API] Error fetching direct metadata doc: {e}")
                
        # Retrieve top chunks
        candidates = query_vector_store(user_query, lang=language, limit=3, sno=request.selected_sno)
    except Exception as e:
        print(f"[API] Error querying vector store: {e}")
    finally:
        retrieval_elapsed = time.perf_counter() - retrieval_start
        print(f"\n[TIMING] ⏱  Chunk retrieval took: {retrieval_elapsed:.3f}s")
        
    # Budget-based context compile with source labeling
    context_parts = []
    current_length = 0
    budget = 10000
    
    if metadata_doc:
        header = "[Official Service Specification Profile]"
        context_parts.append(f"{header}\n{metadata_doc}")
        current_length += len(header) + len(metadata_doc)
        
    for res in candidates:
        # Avoid duplicate metadata documents in context
        if res["metadata"].get("type") == "metadata" and metadata_doc:
            continue
        doc_text = res["document"]
        doc_len = len(doc_text)
        
        source_label = "[User Manual & Guidelines]" if res["metadata"].get("type") == "manual" else "[Official Service Specification Profile]"
        chunk_text = f"{source_label}\n{doc_text}"
        chunk_len = len(chunk_text)
        
        if current_length + chunk_len > budget:
            break
            
        context_parts.append(chunk_text)
        current_length += chunk_len
        
    retrieved_context = "\n\n---\n\n".join(context_parts)

    # ── DEBUG: Print chunks being passed to the LLM ──────────────────────────
    if DEBUG_PRINT_CHUNKS:
        print("\n" + "=" * 70)
        print(f"[DEBUG] Query: {user_query}")
        print(f"[DEBUG] Language: {language} | Selected SNO: {request.selected_sno}")
        print(f"[DEBUG] Total chunks passed to LLM: {len(context_parts)}")
        for idx, part in enumerate(context_parts, 1):
            print(f"\n--- Chunk {idx} ---")
            print(part[:1000] + ("..." if len(part) > 1000 else ""))  # Truncate very long chunks for readability
        print("=" * 70 + "\n")
    # ─────────────────────────────────────────────────────────────────────────
    
    # Construct System prompt instructions
    if language == "hi":
        system_instruction = (
            "आप SewaSetu (सेवा सेतु) छत्तीसगढ़ पोर्टल के एक विशेषज्ञ सहायक हैं।\n"
            "आपका उद्देश्य नागरिकों को सरकारी सेवाओं के आवेदन, आवश्यक दस्तावेजों और शुल्कों को समझने में मदद करना है।\n\n"
            "उत्तर देने के लिए केवल और केवल प्रदान किए गए संदर्भ (Context) का उपयोग करें। ढांचागत आवश्यकताओं, समय सीमा (SLA), शुल्क और आवश्यक दस्तावेजों के लिए '[Official Service Specification Profile]' वाले हिस्से को ही अंतिम और प्राथमिक आधार मानें। विस्तृत विवरण, दिशा-निर्देशों या चरण-दर-चरण निर्देशों के लिए '[User Manual & Guidelines]' वाले हिस्सों का उपयोग करें।\n\n"
            "यदि संदर्भ में उपयोगकर्ता के प्रश्न का उत्तर पर्याप्त रूप से उपलब्ध नहीं है, तो आपको अनिवार्य रूप से केवल यही उत्तर देना होगा: 'जानकारी उपलब्ध नहीं है।' और कुछ भी नहीं जोड़ना है। अपनी ओर से कोई काल्पनिक बात या बाहरी ज्ञान का उपयोग न करें।\n\n"
            "आवश्यक दस्तावेजों के लिए महत्वपूर्ण निर्देश:\n"
            "- आपको आवश्यक दस्तावेजों की सूची केवल और केवल '[Official Service Specification Profile]' खंड से प्राप्त करनी होगी। आवश्यक दस्तावेजों की सूची के लिए '[User Manual & Guidelines]' को न देखें, क्योंकि यह अपूर्ण, कटी हुई या विरोधाभासी हो सकती है।\n"
            "- आपको संदर्भ में '## Required Documents' (या '## आवश्यक दस्तावेज़') के अंतर्गत '[Official Service Specification Profile]' में दी गई प्रत्येक दस्तावेज़ श्रेणी (जैसे Category [1], Category [2] आदि) को अनिवार्य रूप से पूरी तरह सूचीबद्ध करना होगा।\n"
            "- आपको अनिवार्य (हाँ) और वैकल्पिक (नहीं) दोनों श्रेणियों को शामिल करना होगा। किसी भी श्रेणी या उप-विकल्प को छोड़ना या छोटा नहीं करना है।\n"
            "- प्रत्येक श्रेणी के लिए, उसके सभी विकल्पों/सहायक दस्तावेजों को संदर्भ के अनुसार ही लिखें।\n"
            "- उत्तर का प्रारूप संदर्भ जैसा ही होना चाहिए। उदाहरण के लिए:\n"
            "  - Category [X]: Name (Mandatory/अनिवार्य: हाँ या नहीं)\n"
            "    * Option X.Y: Name\n\n"
        )
        if retrieved_context:
            system_instruction += f"संदर्भ दस्तावेज़:\n{retrieved_context}\n\n"
    else:
        system_instruction = (
            "You are an expert assistant for the SewaSetu Chhattisgarh portal.\n"
            "Your goal is to help citizens understand how to apply for services, check required documents, kiosk/online fees, and timelines.\n\n"
            "Answer the question using ONLY the relevant context below. Use the '[Official Service Specification Profile]' chunk as the absolute primary authority for structured requirements, SLAs, fees, and required documents. Use the '[User Manual & Guidelines]' chunks for details, descriptions, or step-by-step instructions. If the context does not contain the answer or if there is insufficient information to answer the question, you MUST respond with exactly: 'Information not available.' and nothing else. Do not make up a response, extrapolate, or use outside knowledge.\n\n"
            "CRITICAL REQUIREMENT FOR REQUIRED DOCUMENTS:\n"
            "- You MUST retrieve the list of required documents ONLY from the '[Official Service Specification Profile]' section. Do NOT use or look at '[User Manual & Guidelines]' for listing required documents, as it may be incomplete, truncated, or conflicting.\n"
            "- You MUST list EVERY single document category (Category [1], Category [2], Category [3], Category [4], etc.) present under the '## Required Documents' section in '[Official Service Specification Profile]'.\n"
            "- You MUST include both mandatory (Mandatory: Yes or Mandatory/अनिवार्य: Yes) and optional (Mandatory: No or Mandatory/अनिवार्य: No) document categories. Do not omit, filter, or truncate any categories.\n"
            "- For each category, you MUST list all of its options/supporting documents exactly as written in the context.\n"
            "- Format the output exactly like the context. For example:\n"
            "  - Category [X]: Name (Mandatory/अनिवार्य: Yes or No)\n"
            "    * Option X.Y: Name\n\n"
        )
        if retrieved_context:
            system_instruction += f"Relevant Context:\n{retrieved_context}\n\n"
            
    # Format message history
    history = request.messages[-7:-1] if len(request.messages) > 1 else []
    
    # Check if we should use Sarvam AI API or fall back to local Ollama
    if USE_SARVAM_API and SARVAM_API_KEY:
        generation_start = time.perf_counter()
        try:
            # Construct standard chat messages format for OpenAI-compatible endpoint
            sarvam_messages = [{"role": "system", "content": system_instruction}]
            for msg in history:
                sarvam_messages.append({"role": msg.role, "content": msg.content})
            sarvam_messages.append({"role": "user", "content": user_query})
            
            url = "https://api.sarvam.ai/v1/chat/completions"
            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": SARVAM_MODEL,
                "messages": sarvam_messages,
                "temperature": 0.0
            }
            
            print(f"[API] Querying Sarvam AI API (Model: {SARVAM_MODEL})...")
            res = requests.post(url, json=payload, headers=headers, timeout=120)
            
            if res.status_code == 200:
                generation_elapsed = time.perf_counter() - generation_start
                print(f"[TIMING] ⚡ Answer generation (Sarvam AI) took: {generation_elapsed:.3f}s")
                print(f"[TIMING] 📊 Total request time: {retrieval_elapsed + generation_elapsed:.3f}s\n")
                bot_reply = res.json()["choices"][0]["message"]["content"].strip()
                # Clean reasoning/thinking tags
                bot_reply = re.sub(r'<think>.*?</think>', '', bot_reply, flags=re.DOTALL)
                bot_reply = bot_reply.strip()
                return {"response": bot_reply}
            else:
                generation_elapsed = time.perf_counter() - generation_start
                print(f"[TIMING] ⚡ Sarvam AI generation failed after: {generation_elapsed:.3f}s")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Sarvam AI API returned code {res.status_code}: {res.text}"
                )
        except Exception as e:
            print(f"[API] Sarvam AI API execution exception: {e}")
            raise HTTPException(status_code=500, detail=f"Error communicating with Sarvam AI: {str(e)}")
    else:
        # Fallback: Query Ollama HTTP API (with timing)
        prompt_parts = [f"System: {system_instruction}"]
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {msg.content}")
        prompt_parts.append(f"User: {user_query}")
        prompt_parts.append("Assistant: ")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        generation_start = time.perf_counter()
        try:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            }
            res = requests.post(url, json=payload, timeout=150)
            if res.status_code == 200:
                generation_elapsed = time.perf_counter() - generation_start
                print(f"[TIMING] ⚡ Answer generation took: {generation_elapsed:.3f}s")
                print(f"[TIMING] 📊 Total request time: {retrieval_elapsed + generation_elapsed:.3f}s\n")
                bot_reply = res.json().get("response", "").strip()
                # Clean reasoning/thinking tags
                bot_reply = re.sub(r'<think>.*?</think>', '', bot_reply, flags=re.DOTALL)
                bot_reply = bot_reply.strip()
                return {"response": bot_reply}
            else:
                generation_elapsed = time.perf_counter() - generation_start
                print(f"[TIMING] ⚡ Answer generation failed after: {generation_elapsed:.3f}s")
                raise HTTPException(status_code=500, detail=f"Ollama server returned code {res.status_code}")
                
        except Exception as e:
            print(f"[API] Ollama HTTP execution exception: {e}")
            raise HTTPException(status_code=500, detail=f"Error communicating with local LLM: {str(e)}")

@app.post("/api/search")
def search_services(request: SearchRequest):
    """
    LLM-based Search endpoint to identify the closest matching service catalog item.
    Supports English, Hindi, and Hinglish.
    """
    query = request.query.strip()
    if not query:
        return {"sno": None, "service_id": None}
        
    # Load manifest services dynamically to build the catalog list dynamically
    services = []
    if MANIFEST_DATA and "services" in MANIFEST_DATA:
        services = MANIFEST_DATA["services"]
    else:
        # Fallback load manifest on the fly if not cached yet
        load_manifest()
        services = MANIFEST_DATA.get("services", []) if MANIFEST_DATA else []
        
    services_list = []
    for s in services:
        services_list.append(
            f"{s.get('sno')}. Serial Number {s.get('sno')} (Service ID: {s.get('service_id')}): {s.get('name_en')} | {s.get('name_hi')}"
        )
    services_catalog_desc = "\n".join(services_list)
    
    prompt = (
        "You are an expert service mapping assistant for the SewaSetu Chhattisgarh portal.\n"
        "Your task is to identify which specific service from the catalog is the closest match to the user query.\n"
        "The query could be in English, Hindi, or Hinglish (e.g. 'shadi certificate', 'aay praman', 'pani connection').\n\n"
        "Here is the catalog of services:\n"
        f"{services_catalog_desc}\n\n"
        f"User Query: '{query}'\n\n"
        "Instructions:\n"
        "- Match the query to a service ONLY if the query explicitly mentions or clearly target that specific service (e.g., marriage, income, domicile, water tap, CMEGP, or film subsidy).\n"
        "- Do NOT match generic queries like 'certificate', 'fees', 'documents', 'registration', 'apply', or 'how to apply' to any specific service if the query does not specify which service it is about.\n"
        "- If the query is generic, ambiguous, or does not clearly map to a specific service in the catalog, you MUST return {\"sno\": null, \"service_id\": null}.\n"
        "- Return ONLY a JSON object containing the mapped 'sno' and 'service_id' as strings. For example: {\"sno\": \"1\", \"service_id\": \"3\"}\n"
        "- Do not explain your choice. Do not output markdown, only raw JSON.\n\n"
        "Output JSON:"
    )
    
    if USE_SARVAM_API and SARVAM_API_KEY:
        try:
            url = "https://api.sarvam.ai/v1/chat/completions"
            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": SARVAM_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }
            print(f"[API] Querying Sarvam AI API for service mapping (Model: {SARVAM_MODEL})...")
            res = requests.post(url, json=payload, headers=headers, timeout=60)
            if res.status_code == 200:
                reply = res.json()["choices"][0]["message"]["content"].strip()
                # Extract JSON from reply using Regex
                json_match = re.search(r'\{.*?\}', reply, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group(0))
                    sno_val = json_data.get("sno")
                    sid_val = json_data.get("service_id")
                    if sno_val and str(sno_val).lower() != "null":
                        return {
                            "sno": str(sno_val),
                            "service_id": str(sid_val) if sid_val else None
                        }
                else:
                    print(f"[Search API] Failed to extract JSON from Sarvam reply: {reply}")
            else:
                print(f"[Search API] Sarvam AI API returned code {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[Search API] Mapping via Sarvam failed: {e}")
    else:
        # Fallback: Query local Ollama
        try:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0 # Force deterministic output
                }
            }
            res = requests.post(url, json=payload, timeout=60) # Higher timeout for first run loading
            if res.status_code == 200:
                reply = res.json().get("response", "").strip()
                
                # Extract JSON from reply using Regex
                json_match = re.search(r'\{.*?\}', reply, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group(0))
                    sno_val = json_data.get("sno")
                    sid_val = json_data.get("service_id")
                    if sno_val and str(sno_val).lower() != "null":
                        return {
                            "sno": str(sno_val),
                            "service_id": str(sid_val) if sid_val else None
                        }
                else:
                    print(f"[Search API] Failed to extract JSON from reply: {reply}")
        except Exception as e:
            print(f"[Search API] Mapping failed with error: {e}")
        
    return {"sno": None, "service_id": None}

@app.post("/api/ingest")
def trigger_ingestion(background_tasks: BackgroundTasks):
    def run_process():
        try:
            from data_pipeline.data_processor import run_ingestion
            run_ingestion()
        except Exception as e:
            print(f"[API] Ingestion Failed: {e}")
            
    background_tasks.add_task(run_process)
    return {"message": "Ingestion pipeline triggered in the background."}
