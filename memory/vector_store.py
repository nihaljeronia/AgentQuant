import chromadb
import os

CHROMA_DIR = "memory/chroma_db"

def get_chroma_collection(collection_name="hypotheses"):
    """Get persistent ChromaDB collection."""
    os.makedirs("memory", exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(collection_name)
    return collection

def store_hypothesis(hypothesis_id: str, hypothesis_text: str, metadata: dict):
    """Store a hypothesis document and metadata in ChromaDB."""
    collection = get_chroma_collection()
    collection.add(documents=[hypothesis_text], ids=[hypothesis_id], metadatas=[metadata])

def find_similar(query_text: str, n_results=3):
    """Query similar past hypotheses from ChromaDB vector store."""
    collection = get_chroma_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(query_texts=[query_text], n_results=min(n_results, count))
    return results
