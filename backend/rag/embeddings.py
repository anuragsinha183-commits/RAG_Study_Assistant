"""
Embedding Module

WHAT: Converts text into numerical vectors (embeddings).
WHY:  Embeddings let us measure semantic similarity between a question
      and document chunks using math (cosine similarity). Two pieces of
      text about the same topic will have similar embedding vectors.
"""
from sentence_transformers import SentenceTransformer
from backend import config

# Singleton: load the model once, reuse everywhere
_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model (lazy singleton — loads only on first call)."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {config.EMBEDDING_MODEL}...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"[Embeddings] Model loaded successfully.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each is a list of 384 floats
        for all-MiniLM-L6-v2).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Returns one embedding vector."""
    model = get_embedding_model()
    embedding = model.encode(query)
    return embedding.tolist()
