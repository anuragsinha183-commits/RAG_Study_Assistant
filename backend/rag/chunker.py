"""
Smart Text Chunking Module

WHAT: Splits document text into meaningful, overlapping chunks with metadata.
WHY:  Embedding models need small inputs (~512 tokens). Chunks must be
      meaningful (not cut mid-sentence) and carry metadata for citations.
"""
import re
import uuid
from dataclasses import dataclass
from backend.rag.loader import PageContent
from backend import config


@dataclass
class TextChunk:
    """A chunk of text with its metadata for storage and retrieval."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    char_count: int
    chunk_index: int  # Position within the document


def chunk_pages(
    pages: list[PageContent],
    document_id: str,
    document_name: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split page texts into overlapping chunks, preserving page info.

    Strategy:
    1. Process each page separately to preserve page numbers.
    2. Within a page, split on paragraph boundaries first.
    3. If a paragraph is still too large, split on sentences.
    4. Merge small consecutive paragraphs up to chunk_size.
    5. Add overlap from the end of the previous chunk.

    Args:
        pages: List of PageContent from PDF extraction.
        document_id: Unique ID for the document.
        document_name: Human-readable document name.
        chunk_size: Max characters per chunk (default from config).
        chunk_overlap: Overlap characters between chunks (default from config).

    Returns:
        List of TextChunk objects ready for embedding.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    all_chunks = []
    chunk_index = 0

    for page in pages:
        if not page.has_text:
            continue  # Skip empty/scanned pages (OCR handled in Phase 2)

        # Split page text into paragraphs
        paragraphs = _split_into_paragraphs(page.text)

        # Merge small paragraphs, split large ones
        segments = _create_segments(paragraphs, chunk_size)

        # Create chunks with overlap
        for i, segment in enumerate(segments):
            # Add overlap from previous segment
            if i > 0 and chunk_overlap > 0:
                prev_text = segments[i - 1]
                overlap_text = prev_text[-chunk_overlap:]
                # Don't break mid-word — find first space in overlap
                space_idx = overlap_text.find(' ')
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1:]
                segment = overlap_text + " " + segment

            chunk = TextChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                document_name=document_name,
                page_number=page.page_number,
                text=segment.strip(),
                char_count=len(segment.strip()),
                chunk_index=chunk_index,
            )
            all_chunks.append(chunk)
            chunk_index += 1

    return all_chunks


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on double newlines (paragraph boundaries)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _create_segments(paragraphs: list[str], max_size: int) -> list[str]:
    """
    Merge small paragraphs together (up to max_size).
    Split large paragraphs into sentence-level segments.
    """
    segments = []
    current = ""

    for para in paragraphs:
        if len(para) > max_size:
            # Save current buffer first
            if current:
                segments.append(current.strip())
                current = ""
            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_buf = ""
            for sent in sentences:
                if len(sent_buf) + len(sent) + 1 <= max_size:
                    sent_buf = (sent_buf + " " + sent).strip()
                else:
                    if sent_buf:
                        segments.append(sent_buf)
                    sent_buf = sent
            if sent_buf:
                segments.append(sent_buf)
        elif len(current) + len(para) + 2 <= max_size:
            # Merge small paragraph into current buffer
            current = (current + "\n\n" + para).strip()
        else:
            # Current buffer is full, start new one
            if current:
                segments.append(current.strip())
            current = para

    if current:
        segments.append(current.strip())

    return segments
