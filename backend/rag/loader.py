"""
PDF Text Extraction Module

WHAT: Extracts text from PDF files, page by page.
WHY:  Raw text is the starting point of the RAG pipeline.
      Page numbers are preserved for source citations.
"""
import pymupdf  # PyMuPDF (replaces deprecated fitz)
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents extracted text from one PDF page."""
    page_number: int       # 1-indexed page number
    text: str              # Extracted text content
    has_text: bool         # True if meaningful text was found
    char_count: int        # Number of characters extracted


def extract_text_from_pdf(file_path: str | Path) -> list[PageContent]:
    """
    Extract text from all pages of a PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of PageContent objects, one per page.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the file isn't a valid PDF.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    pages = []

    try:
        doc = pymupdf.open(str(file_path))

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # Extract as plain text

            # Clean: normalize whitespace but preserve paragraph breaks
            cleaned = _clean_extracted_text(text)

            pages.append(PageContent(
                page_number=page_num + 1,  # 1-indexed
                text=cleaned,
                has_text=len(cleaned.strip()) > 10,  # More than 10 chars = has content
                char_count=len(cleaned),
            ))

        doc.close()

    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")

    return pages


def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted text:
    - Remove excessive blank lines
    - Normalize whitespace within lines
    - Strip leading/trailing whitespace per line
    """
    # Replace multiple spaces/tabs (not newlines) with single space
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Replace 3+ consecutive newlines with 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip each line individually
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def get_pdf_metadata(file_path: str | Path) -> dict:
    """Extract PDF metadata (title, author, page count, file size)."""
    file_path = Path(file_path)
    doc = pymupdf.open(str(file_path))

    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "page_count": len(doc),
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
    }

    doc.close()
    return metadata
