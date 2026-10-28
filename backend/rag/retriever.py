"""
RAG Retriever — Main Orchestrator

WHAT: Combines retrieval + filtering + generation into one pipeline.
WHY:  This ties together all RAG components and enforces hallucination
      control by rejecting low-confidence retrievals before they reach the LLM.
"""
from backend import config
from backend.rag.vector_store import search
from backend.rag.prompts import build_rag_prompt
from backend.rag.generator import generate_answer


async def ask(
    question: str,
    document_id: str | None = None,
    answer_mode: str = "simple",
    chat_history: list[dict] | None = None,
    provider: str | None = None,
) -> dict:
    """
    The main RAG pipeline:
    1. Search vector store for relevant chunks
    2. Filter by similarity threshold (hallucination guard)
    3. If no good chunks → refuse to answer
    4. Build prompt with strict grounding instructions
    5. Generate grounded answer via Groq / Local Offline Reader / Ollama

    Args:
        question: The student's question.
        document_id: Scope to a specific document (None = all docs).
        answer_mode: One of: simple, exam, short, detailed, points.
        chat_history: Previous messages for conversational context.
        provider: auto | offline | groq | ollama

    Returns:
        Dict with: answer, sources, retrieval_scores, was_grounded,
                   chunks_retrieved, chunks_used.
    """
    # Step 1: Retrieve relevant chunks from vector store
    results = search(
        query=question,
        top_k=config.TOP_K,
        document_id=document_id,
    )

    # Step 2: Filter by similarity threshold — HALLUCINATION GUARD
    good_results = [
        r for r in results
        if r["score"] >= config.SIMILARITY_THRESHOLD
    ]

    # Step 3: Refuse if no relevant evidence found
    if not good_results:
        scope = "the selected document" if document_id else "any uploaded document"
        return {
            "answer": (
                f"I could not find enough information about this in {scope}. "
                f"Please make sure the relevant material has been uploaded."
            ),
            "sources": [],
            "retrieval_scores": [r["score"] for r in results],
            "was_grounded": False,
            "chunks_retrieved": len(results),
            "chunks_used": 0,
        }

    # Step 4: Build prompt
    messages = build_rag_prompt(
        question=question,
        context_chunks=good_results,
        answer_mode=answer_mode,
        chat_history=chat_history,
    )

    # Step 5: Generate answer via Groq / Offline Reader / Ollama
    answer = await generate_answer(
        messages=messages,
        provider=provider,
        context_chunks=good_results,
        question=question,
        answer_mode=answer_mode,
    )

    # Step 6: Extract deduplicated source citations
    sources = []
    seen = set()
    for r in good_results:
        key = (r["metadata"]["document_name"], r["metadata"]["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "document_name": r["metadata"]["document_name"],
                "document_id": r["metadata"]["document_id"],
                "page_number": r["metadata"]["page_number"],
                "relevance_score": r["score"],
            })

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_scores": [r["score"] for r in good_results],
        "was_grounded": True,
        "chunks_retrieved": len(results),
        "chunks_used": len(good_results),
    }

