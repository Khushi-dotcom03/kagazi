from pathlib import Path
from rank_bm25 import BM25Okapi
import re

import chromadb

DATABASE_PATH = Path(__file__).resolve().parents[1] / "chroma_db"
client = chromadb.PersistentClient(path=str(DATABASE_PATH))


def create_collection():
    return client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space": "cosine"})


def replace_document(
    collection,
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
):
    existing = collection.get(where={"document_id": document_id})

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],

        documents=[chunk["text"] for chunk in chunks],

        embeddings=embeddings,

        metadatas=[
            {
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "version": chunk["version"],
                "source_type": chunk["source_type"],
                "extraction_method": chunk["method"],
                "department": chunk.get("department"),
                "access_level": chunk.get("access_level"),
            }
            for chunk in chunks
        ],
    )


def search(collection, query_embedding: list[float], top_k: int = 4, allowed_departments: list[str] = None, allowed_access_levels: list[str] = None):
    where = {}
    if not allowed_departments:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    where_filter = {
        "department": {
            "$in": allowed_departments
        }
    }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())
