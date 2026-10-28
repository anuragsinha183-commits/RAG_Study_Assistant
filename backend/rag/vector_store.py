"""
Vector Store Module (ChromaDB)

WHAT: Stores and searches document embeddings using ChromaDB.
WHY:  ChromaDB persists vectors to disk so documents survive restarts.
      It handles similarity search efficiently using cosine distance.
"""
import chromadb
from backend import config
from backend.rag.chunker import TextChunk
from backend.rag.embeddings import embed_texts, embed_query


# Singleton ChromaDB client and collection
_client = None
_collection = None

COLLECTION_NAME = "documents"


def get_client() -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_DIR)
        )
    return _client


def get_collection():
    """Get or create the document collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    return _collection


def add_chunks(chunks: list[TextChunk]) -> int:
    """
    Embed and store chunks in ChromaDB.

    Args:
        chunks: List of TextChunk objects to store.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_collection()

    # Prepare data for ChromaDB
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "char_count": c.char_count,
        }
        for c in chunks
    ]

    # Generate embeddings for all chunks
    print(f"[VectorStore] Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # Add to ChromaDB in batches to avoid memory issues
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"[VectorStore] Stored {len(chunks)} chunks.")
    return len(chunks)


def search(
    query: str,
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[dict]:
    """
    Search for chunks relevant to a query.

    Args:
        query: The user's question.
        top_k: Number of results to return.
        document_id: If set, only search within this document.

    Returns:
        List of result dicts with keys: chunk_id, text, metadata, distance, score.
        Sorted by relevance (highest score first).
    """
    top_k = top_k or config.TOP_K
    collection = get_collection()

    # Build optional filter for document-scoped search
    where_filter = None
    if document_id:
        where_filter = {"document_id": document_id}

    # Embed the query
    query_embedding = embed_query(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Format results into clean dicts
    formatted = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1 - (distance / 2)

            formatted.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "score": round(score, 4),
            })

    return formatted


def delete_document(document_id: str) -> int:
    """
    Delete all chunks belonging to a document.

    Returns:
        Number of chunks deleted.
    """
    collection = get_collection()

    # Get IDs of chunks for this document
    results = collection.get(
        where={"document_id": document_id},
        include=[],
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"[VectorStore] Deleted {len(results['ids'])} chunks for document {document_id}")

    return len(results["ids"])


def get_document_chunk_count(document_id: str) -> int:
    """Get the number of chunks stored for a document."""
    collection = get_collection()
    results = collection.get(
        where={"document_id": document_id},
        include=[],
    )
    return len(results["ids"])
