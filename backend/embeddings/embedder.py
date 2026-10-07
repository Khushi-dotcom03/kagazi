from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1) #Run get_model() only once, and remember the result so I don't have to load the model again
def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(texts: list[str]):
    return get_model().encode(texts, normalize_embeddings=True).tolist()