"""
FastAPI Application — Main Entry Point

WHAT: The application server that ties together all API routes.
WHY:  This is the single entry point. Run with:
      uvicorn backend.main:app --reload --port 8000
"""
import os

# Prevent OpenBLAS thread memory allocation issue on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.api import upload, chat, documents

app = FastAPI(
    title="AI Study Assistant",
    description="Local & Online Document Question Answering System",
    version="1.0.0",
)

# ── CORS Middleware ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register API Routes ──────────────────────────────────────────
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])


# ── Health Check ─────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
async def health_check():
    """Check system health: API status and LLM availability."""
    from backend.rag.generator import check_llm_health
    llm_status = await check_llm_health()
    return {
        "status": "ok",
        "llm": llm_status,
        "ollama": llm_status,  # For backward-compatibility
    }


# ── Serve Frontend ───────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )

