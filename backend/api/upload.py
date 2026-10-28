"""
Upload Endpoint

WHAT: Handles PDF upload, processing, and indexing.
WHY:  This is how documents enter the RAG system.
"""
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend import config
from backend.rag.loader import extract_text_from_pdf, get_pdf_metadata
from backend.rag.chunker import chunk_pages
from backend.rag.vector_store import add_chunks
from backend.services.document_store import save_document_record

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    Steps:
    1. Validate file type and size
    2. Save to disk
    3. Extract text
    4. Chunk the text
    5. Generate embeddings and store in ChromaDB
    6. Save document metadata record
    """
    # ── Validation ────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Read file contents
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            400,
            f"File too large ({size_mb:.1f}MB). Maximum: {config.MAX_UPLOAD_SIZE_MB}MB"
        )

    # ── Save to disk ──────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    file_path = config.UPLOAD_DIR / f"{doc_id}.pdf"

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # ── Extract text ──────────────────────────────────────
        pages = extract_text_from_pdf(file_path)
        pdf_meta = get_pdf_metadata(file_path)

        # Check if any text was extracted
        text_pages = [p for p in pages if p.has_text]
        if not text_pages:
            raise HTTPException(
                422,
                "No text could be extracted from this PDF. "
                "It may be a scanned/image-based document "
                "(OCR support coming in Phase 2)."
            )

        # ── Chunk the text ────────────────────────────────────
        chunks = chunk_pages(pages, doc_id, file.filename)

        # ── Store embeddings ──────────────────────────────────
        num_chunks = add_chunks(chunks)

        # ── Save document record ──────────────────────────────
        doc_record = {
            "document_id": doc_id,
            "document_name": file.filename,
            "page_count": pdf_meta["page_count"],
            "text_pages": len(text_pages),
            "chunk_count": num_chunks,
            "file_size_mb": round(size_mb, 2),
            "title": pdf_meta.get("title", ""),
        }
        save_document_record(doc_record)

        return {
            "status": "success",
            "message": f"Document processed successfully. {num_chunks} chunks created.",
            "document": doc_record,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to process document: {str(e)}")
