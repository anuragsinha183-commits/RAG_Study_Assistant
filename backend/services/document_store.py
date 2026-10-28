"""
Document Store — Simple JSON-based document record storage.

WHAT: Tracks which documents are uploaded, their names, page counts, etc.
WHY:  We need metadata about documents (names, sizes, chunk counts)
      separate from the vector embeddings. A JSON file is simple and
      sufficient for an MVP — no database server needed.
"""
import json
from pathlib import Path
from backend import config

STORE_FILE = config.PROJECT_ROOT / "data" / "documents.json"


def _load_store() -> dict:
    """Load the document store from disk."""
    if STORE_FILE.exists():
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_store(store: dict):
    """Save the document store to disk."""
    STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def save_document_record(record: dict):
    """Save or update a document record."""
    store = _load_store()
    store[record["document_id"]] = record
    _save_store(store)


def get_document(document_id: str) -> dict | None:
    """Get a document record by ID. Returns None if not found."""
    store = _load_store()
    return store.get(document_id)


def get_all_documents() -> list[dict]:
    """Get all document records as a list."""
    store = _load_store()
    return list(store.values())


def remove_document_record(document_id: str):
    """Remove a document record by ID."""
    store = _load_store()
    store.pop(document_id, None)
    _save_store(store)
