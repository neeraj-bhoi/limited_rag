import os
import json
import sys
from dotenv import load_dotenv

# Append parent directory to system path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.chunking.chunker import chunk_text, convert_json_to_markdown, normalize_quotes
from backend.embeddings.embedder import generate_embeddings
from backend.vector_store.chroma_store import get_chroma_client, get_collection

# Load env
load_dotenv()

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRAPED_DATA_DIR = os.path.join(WORKSPACE_DIR, "scraped_data")
PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "processed_data")

MD_OUT_DIR_EN = os.path.join(PROCESSED_DIR, "markdown_details", "en")
MD_OUT_DIR_HI = os.path.join(PROCESSED_DIR, "markdown_details", "hi")

SERVICE_IDS = [3, 6, 7, 24, 337, 354]
LANGUAGES = ["en", "hi"]

def setup_directories():
    os.makedirs(MD_OUT_DIR_EN, exist_ok=True)
    os.makedirs(MD_OUT_DIR_HI, exist_ok=True)
    print(f"[Processor] Directories set up in {PROCESSED_DIR}")

def run_ingestion():
    setup_directories()
    
    # 1. Compile manifest catalog from active profiles
    print("[Processor] Compiling manifest from profiles...")
    services_manifest = []
    
    for idx, sid in enumerate(SERVICE_IDS):
        sno = str(idx + 1)
        # Load English Profile
        en_profile_path = os.path.join(SCRAPED_DATA_DIR, "profiles", f"service_{sid}_en.json")
        hi_profile_path = os.path.join(SCRAPED_DATA_DIR, "profiles", f"service_{sid}_hi.json")
        
        name_en, dept_en, is_internal = "", "", False
        name_hi, dept_hi = "", ""
        
        if os.path.exists(en_profile_path):
            with open(en_profile_path, "r", encoding="utf-8") as f:
                en_data = json.load(f)
                name_en = en_data.get("name", "")
                dept_en = en_data.get("department", "")
                # Resolve internal
                is_internal = "instractionPageNew.do" in en_data.get("details_link", "")
                
        if os.path.exists(hi_profile_path):
            with open(hi_profile_path, "r", encoding="utf-8") as f:
                hi_data = json.load(f)
                name_hi = hi_data.get("name", "")
                dept_hi = hi_data.get("department", "")
                
        # Locate manuals if they exist
        manual_rel_en = None
        manual_rel_hi = None
        manual_filename_en = f"combined_manual_{sid}_en.txt"
        manual_filename_hi = f"combined_manual_{sid}_hi.txt"
        
        if os.path.exists(os.path.join(SCRAPED_DATA_DIR, "extracted_text", manual_filename_en)):
            manual_rel_en = f"scraped_data/extracted_text/{manual_filename_en}"
        if os.path.exists(os.path.join(SCRAPED_DATA_DIR, "extracted_text", manual_filename_hi)):
            manual_rel_hi = f"scraped_data/extracted_text/{manual_filename_hi}"
            
        services_manifest.append({
            "sno": sno,
            "service_id": str(sid),
            "name_en": name_en or f"Service {sid}",
            "name_hi": name_hi or f"सेवा {sid}",
            "dept_en": dept_en,
            "dept_hi": dept_hi,
            "is_internal": is_internal,
            "path_en": f"scraped_data/profiles/service_{sid}_en.json",
            "path_hi": f"scraped_data/profiles/service_{sid}_hi.json",
            "manual_path_en": manual_rel_en,
            "manual_path_hi": manual_rel_hi
        })
        
    manifest = {
        "services": services_manifest
    }
    
    manifest_path = os.path.join(PROCESSED_DIR, "rag_kb_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[Processor] Created manifest at: {manifest_path}")

    # Delete existing collection to avoid duplication
    client = get_chroma_client()
    try:
        client.delete_collection("sewasetu_knowledge_base")
        print("[Processor] Deleted old vector index collection.")
    except Exception:
        pass
        
    # Recreate collection
    collection = get_collection()
    
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    # 2. Ingest structured JSON profiles (converted to Markdown)
    for s in services_manifest:
        sno = s.get("sno")
        service_id = s.get("service_id")
        name_en = s.get("name_en")
        name_hi = s.get("name_hi")
        dept_en = s.get("dept_en")
        dept_hi = s.get("dept_hi")
        
        for lang in LANGUAGES:
            path_key = f"path_{lang}"
            rel_path = s.get(path_key)
            if not rel_path:
                continue
                
            full_path = os.path.join(WORKSPACE_DIR, rel_path)
            if not os.path.exists(full_path):
                continue
                
            try:
                with open(full_path, "r", encoding="utf-8") as file:
                    detail_data = json.load(file)
                    
                # Format JSON to clean markdown, excluding massive form fields to optimize prompt size
                md_content = convert_json_to_markdown(detail_data, lang=lang, exclude_form_fields=True)
                md_content = normalize_quotes(md_content)
                
                # Write to processed_data folder for debugging/manual verification
                md_out_path = os.path.join(PROCESSED_DIR, "markdown_details", lang, f"{sno}.md")
                with open(md_out_path, "w", encoding="utf-8") as out_file:
                    out_file.write(md_content)
                    
                chunks = [md_content]
                for idx, chunk in enumerate(chunks):
                    doc_id = f"meta_{sno}_{lang}_{idx}"
                    metadata = {
                        "sno": str(sno),
                        "service_id": str(service_id),
                        "language": lang,
                        "type": "metadata",
                        "name": name_hi if lang == "hi" else name_en,
                        "department": dept_hi if lang == "hi" else dept_en,
                        "chunk_index": idx
                    }
                    all_documents.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(doc_id)
                
            except Exception as e:
                print(f"[Processor] Error parsing {full_path}: {e}")
                
        # 3. Ingest text manual files
        for lang in LANGUAGES:
            manual_key = f"manual_path_{lang}"
            rel_path = s.get(manual_key)
            if not rel_path:
                continue
                
            full_path = os.path.join(WORKSPACE_DIR, rel_path)
            if not os.path.exists(full_path):
                continue
                
            try:
                with open(full_path, "r", encoding="utf-8") as file:
                    manual_text = file.read()
                
                manual_text = normalize_quotes(manual_text)
                    
                # Split manual into chunks
                chunks = chunk_text(manual_text)
                for idx, chunk in enumerate(chunks):
                    doc_id = f"manual_{sno}_{lang}_{idx}"
                    metadata = {
                        "sno": str(sno),
                        "service_id": str(service_id),
                        "language": lang,
                        "type": "manual",
                        "name": name_hi if lang == "hi" else name_en,
                        "department": dept_hi if lang == "hi" else dept_en,
                        "chunk_index": idx
                    }
                    all_documents.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(doc_id)
            except Exception as e:
                print(f"[Processor] Error chunking manual {full_path}: {e}")

    # 4. Generate embeddings in batches
    total_docs = len(all_documents)
    print(f"[Processor] Generating embeddings for {total_docs} text segments...")
    
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, total_docs, batch_size):
        batch_docs = all_documents[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1} / {int((total_docs - 1) / batch_size) + 1}...")
        batch_embeddings = generate_embeddings(batch_docs)
        all_embeddings.extend(batch_embeddings)
        
    # 5. Populate ChromaDB
    chroma_batch_size = 50
    for i in range(0, total_docs, chroma_batch_size):
        print(f"  Writing database records {i} to {min(i + chroma_batch_size, total_docs)}...")
        collection.add(
            ids=all_ids[i:i + chroma_batch_size],
            embeddings=all_embeddings[i:i + chroma_batch_size],
            metadatas=all_metadatas[i:i + chroma_batch_size],
            documents=all_documents[i:i + chroma_batch_size]
        )
        
    print(f"\n[Processor] Success! Processed database created at {os.path.join(PROCESSED_DIR, 'chroma_db')}")

if __name__ == "__main__":
    run_ingestion()
