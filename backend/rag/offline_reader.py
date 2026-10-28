"""
Offline Extractive Reader Module

WHAT: Synthesizes high-quality, grounded study answers directly from
      retrieved document passages without needing Ollama or internet.
WHY:  Enables 100% offline document QA on low-storage PCs (e.g. 0 GB free on C:),
      using the local embeddings and ChromaDB vector store already on disk.
"""
import re
from typing import List, Dict


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "explain", "describe", "define", "give"
}


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artifacts."""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into complete, cleaned sentences."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in raw_sentences:
        s = _clean_text(s)
        if len(s) < 20:
            continue
        if re.search(r'copyright|all rights reserved|page \d+', s, re.IGNORECASE):
            continue
        cleaned.append(s)
    return cleaned


def _score_sentence(
    sentence: str,
    query_tokens: set[str],
    chunk_score: float,
) -> float:
    """Score a sentence based on query keyword matches and informative cues."""
    s_lower = sentence.lower()
    words = set(re.findall(r'\b[a-z0-9]+\b', s_lower))

    overlap = len(words & query_tokens)
    if not overlap:
        return 0.0

    score = overlap * 2.0 + (chunk_score * 1.5)

    if re.search(r'\b(is defined as|refers to|means|consists of|includes|such as|because|due to|therefore|results in|classified into)\b', s_lower):
        score += 2.5

    if 40 <= len(sentence) <= 220:
        score += 1.0

    return score


def synthesize_offline_answer(
    question: str,
    context_chunks: List[Dict],
    answer_mode: str = "simple",
) -> str:
    """
    Generate a formatted study answer extracted from retrieved chunks.
    """
    if not context_chunks:
        return "I could not find enough information in the uploaded document(s) to answer this question."

    q_words = set(re.findall(r'\b[a-z0-9]+\b', question.lower()))
    query_tokens = {w for w in q_words if w not in STOP_WORDS and len(w) > 2}
    if not query_tokens:
        query_tokens = q_words

    candidates = []
    seen_texts = set()

    for chunk in context_chunks:
        c_score = chunk.get("score", 0.5)
        doc_name = chunk.get("metadata", {}).get("document_name", "Document")
        page_num = chunk.get("metadata", {}).get("page_number", 1)
        raw_text = chunk.get("text", "")

        sentences = _split_into_sentences(raw_text)
        for s in sentences:
            s_norm = s.lower().strip()
            if s_norm in seen_texts:
                continue
            seen_texts.add(s_norm)

            rel_score = _score_sentence(s, query_tokens, c_score)
            if rel_score > 0.5:
                candidates.append({
                    "sentence": s,
                    "score": rel_score,
                    "doc_name": doc_name,
                    "page_num": page_num,
                })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        top_text = _clean_text(context_chunks[0]["text"][:500])
        return (
            "> 📴 **Offline Reader**: Synthesized directly from your local document.\n\n"
            f"{top_text}..."
        )

    banner = "> 📴 **Offline Mode**: Running 100% locally from your document (no internet / no Ollama required).\n\n"

    if answer_mode == "short":
        selected = [c["sentence"] for c in candidates[:2]]
        body = " ".join(selected)
        return banner + body

    elif answer_mode == "points":
        selected = candidates[:6]
        points = [f"- **{c['sentence']}** (Page {c['page_num']})" for c in selected]
        return banner + "### Key Points from Document:\n" + "\n".join(points)

    elif answer_mode == "exam":
        lead = candidates[0]["sentence"]
        other_points = candidates[1:5]
        sections = [
            banner + "### Exam Summary",
            f"**Definition / Overview:**\n{lead}\n",
            "**Key Supporting Points:**",
        ]
        for p in other_points:
            sections.append(f"- {p['sentence']}")
        return "\n".join(sections)

    elif answer_mode == "detailed":
        top_sentences = [c["sentence"] for c in candidates[:7]]
        para1 = " ".join(top_sentences[:3])
        para2 = " ".join(top_sentences[3:6]) if len(top_sentences) > 3 else ""
        body = f"{para1}\n\n{para2}" if para2 else para1
        return banner + body

    else:
        top_sentences = [c["sentence"] for c in candidates[:4]]
        body = " ".join(top_sentences)
        return banner + body
