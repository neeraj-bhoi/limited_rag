import os
import chromadb
import threading
from backend.embeddings.embedder import generate_query_embedding

_thread_local = threading.local()

def get_chroma_client():
    """
    Initializes and returns a thread-safe ChromaDB Persistent Client.
    """
    if not hasattr(_thread_local, "chroma_client"):
        # Local relative path in the workspace
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.getenv("CHROMA_DB_PATH")
        if not db_path:
            db_path = os.path.join(workspace_dir, "processed_data", "chroma_db")
        print(f"[ChromaStore] Connecting to ChromaDB at: {db_path}")
        _thread_local.chroma_client = chromadb.PersistentClient(path=db_path)
    return _thread_local.chroma_client

def get_collection():
    """
    Returns the target knowledge base Chroma collection.
    """
    client = get_chroma_client()
    return client.get_or_create_collection("sewasetu_knowledge_base")

def query_vector_store(query_text, lang="en", limit=5, sno=None):
    """
    Queries ChromaDB using dense query embeddings and payload metadata filters.
    
    Args:
        query_text (str): Semantic text query.
        lang (str): Language filter ('en' or 'hi').
        limit (int): Number of documents to retrieve.
        sno (str): Optional service serial number filter.
        
    Returns:
        list: Retrieved document contexts with metadata and scores.
    """
    collection = get_collection()
    query_emb = generate_query_embedding(query_text)
    
    # Base language filter combining language and sno using $and if sno is present
    if sno:
        where_filter = {
            "$and": [
                {"language": lang},
                {"sno": str(sno)}
            ]
        }
    else:
        where_filter = {"language": lang}
        
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=limit,
        where=where_filter
    )
    
    # Package results
    retrieved_docs = []
    if results and "documents" in results and len(results["documents"]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
        
        for i in range(len(docs)):
            retrieved_docs.append({
                "document": docs[i],
                "metadata": metas[i],
                "score": float(distances[i])
            })
            
    return retrieved_docs
