"""
Documents Endpoint

WHAT: Document management — list, view, and delete uploaded documents.
WHY:  Users need to see what they've uploaded, select documents for
      scoped queries, and remove documents they no longer need.
"""
from fastapi import APIRouter, HTTPException

from backend.rag.vector_store import delete_document
from backend.services.document_store import (
    get_all_documents,
    get_document,
    remove_document_record,
)
from backend import config

router = APIRouter()


@router.get("/documents")
async def list_documents():
    """Get all uploaded documents with their metadata."""
    docs = get_all_documents()
    return {
        "documents": docs,
        "total": len(docs),
    }


@router.get("/documents/{document_id}")
async def get_document_info(document_id: str):
    """Get metadata for a specific document."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return {"document": doc}


@router.delete("/documents/{document_id}")
async def delete_doc(document_id: str):
    """
    Delete a document completely:
    1. Remove all chunks from vector store
    2. Delete the PDF file from disk
    3. Remove the document record
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")

    # Delete from vector store
    deleted_chunks = delete_document(document_id)

    # Delete PDF file from disk
    pdf_path = config.UPLOAD_DIR / f"{document_id}.pdf"
    pdf_path.unlink(missing_ok=True)

    # Delete metadata record
    remove_document_record(document_id)

    return {
        "status": "deleted",
        "document_id": document_id,
        "document_name": doc.get("document_name", ""),
        "chunks_deleted": deleted_chunks,
    }
