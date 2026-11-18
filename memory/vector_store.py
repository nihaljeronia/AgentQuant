import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHROMA_DIR, ensure_dirs  # noqa: E402


def get_chroma_collection(collection_name="hypotheses"):
    """Get persistent ChromaDB collection."""
    import chromadb

    ensure_dirs()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(collection_name)


def store_hypothesis(hypothesis_id: str, hypothesis_text: str, metadata: dict):
    """
    Store a hypothesis document and metadata in ChromaDB.
    upsert() (not add()) so re-running never collides with an existing id.
    """
    collection = get_chroma_collection()
    collection.upsert(documents=[hypothesis_text], ids=[hypothesis_id], metadatas=[metadata])


def find_similar(query_text: str, n_results=3, ticker: str | None = None, exclude_run: str | None = None):
    """
    Query similar past hypotheses from the ChromaDB vector store.
    When ticker is given, only that ticker's hypotheses are searched; hypotheses from
    the current run (exclude_run) are skipped.
    """
    collection = get_chroma_collection()
    count = collection.count()
    if count == 0:
        return []
    where = None
    clauses = []
    if ticker:
        clauses.append({"ticker": ticker})
    if exclude_run:
        clauses.append({"run_id": {"$ne": exclude_run}})
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}
    try:
        return collection.query(query_texts=[query_text], n_results=min(n_results, count), where=where)
    except Exception:
        # Fewer matching documents than n_results or no matches for the filter
        return []
