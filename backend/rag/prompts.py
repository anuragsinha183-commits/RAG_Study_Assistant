"""
Prompt Templates Module

WHAT: System prompts and prompt construction for the LLM.
WHY:  A carefully crafted prompt is the PRIMARY defense against hallucination.
      The LLM must understand it can ONLY use the provided context.
"""

# ── The strict system prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a document-grounded educational assistant.

STRICT RULES:
1. Answer the user's question ONLY using the provided CONTEXT below.
2. The CONTEXT comes from documents uploaded by the user.
3. Do NOT use your pretrained knowledge to introduce facts that are not supported by the CONTEXT.
4. If the answer cannot be found in the CONTEXT, say: "I could not find this information in the uploaded document(s)."
5. Do NOT fabricate facts, citations, page numbers, formulas, examples, or quotations.
6. If the CONTEXT only partially answers the question, clearly state what can and cannot be determined from the uploaded material.
7. Use simple, educational language while preserving the meaning of the source material.
8. Always cite the source document name and page number at the end of your answer.

ANSWER FORMAT:
- Provide the answer first.
- Then list sources as: "Source: [document name] — Page [number]"
"""

# ── Answer mode instructions ──────────────────────────────────────────────
ANSWER_MODES = {
    "simple": (
        "Explain the answer in simple, beginner-friendly language "
        "that a student can easily understand."
    ),
    "exam": (
        "Write an exam-ready answer that is structured, concise, "
        "and covers key points suitable for writing in an exam."
    ),
    "short": (
        "Give a concise, brief answer in 2-3 sentences maximum."
    ),
    "detailed": (
        "Explain the topic in thorough detail using the provided context. "
        "Cover all relevant aspects mentioned in the material."
    ),
    "points": (
        "Present the answer as clear bullet points highlighting "
        "the most important information."
    ),
}


def build_rag_prompt(
    question: str,
    context_chunks: list[dict],
    answer_mode: str = "simple",
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the complete message list for the LLM.

    Args:
        question: The student's question.
        context_chunks: Retrieved chunks with metadata.
        answer_mode: One of: simple, exam, short, detailed, points.
        chat_history: Previous messages for conversational context.

    Returns:
        List of message dicts [{"role": ..., "content": ...}].
    """
    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Source {i}: {meta['document_name']} — Page {meta['page_number']}]\n"
            f"{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    # Get mode instruction
    mode_instruction = ANSWER_MODES.get(answer_mode, ANSWER_MODES["simple"])

    # Build user message
    user_message = f"""CONTEXT FROM UPLOADED DOCUMENTS:

{context_str}

---

QUESTION: {question}

INSTRUCTION: {mode_instruction}"""

    # Build messages list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history if present (for follow-up questions)
    # Only include the last 6 messages (3 exchanges) to keep context manageable
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    return messages
