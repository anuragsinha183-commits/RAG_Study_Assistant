"""
Chat Endpoint

WHAT: Handles questions and returns grounded answers.
WHY:  The core feature — student asks a question, gets an answer from their docs.
"""
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import config
from backend.rag.retriever import ask
from backend.rag.prompts import ANSWER_MODES

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    question: str
    document_id: str | None = None  # None = search all documents
    answer_mode: str = "simple"     # simple | exam | short | detailed | points
    chat_history: list[dict] | None = None
    provider: str | None = None     # auto | offline | groq | ollama


class SourceInfo(BaseModel):
    """Source citation information."""
    document_name: str
    document_id: str
    page_number: int
    relevance_score: float


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""
    answer: str
    sources: list[dict]
    was_grounded: bool
    chunks_retrieved: int
    chunks_used: int


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Ask a question about uploaded documents.
    """
    # Validate question
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # Validate answer mode
    valid_modes = list(ANSWER_MODES.keys())
    if request.answer_mode not in valid_modes:
        raise HTTPException(
            400,
            f"Invalid answer mode '{request.answer_mode}'. "
            f"Valid modes: {', '.join(valid_modes)}"
        )

    try:
        result = await ask(
            question=request.question,
            document_id=request.document_id,
            answer_mode=request.answer_mode,
            chat_history=request.chat_history,
            provider=request.provider,
        )

        return ChatResponse(**result)

    except ValueError as e:
        raise HTTPException(400, str(e))
    except httpx.ConnectError:
        active_p = (request.provider or config.LLM_PROVIDER).lower()
        if active_p == "ollama":
            msg = (
                f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
                "Please ensure Ollama is installed and running (`ollama serve`)."
            )
        else:
            msg = (
                f"Cannot connect to {active_p.upper()} API. "
                "Please check your internet connection or switch to Offline Mode."
            )
        raise HTTPException(503, msg)
    except httpx.HTTPStatusError as e:
        active_p = (request.provider or config.LLM_PROVIDER).upper()
        error_msg = f"{active_p} API error ({e.response.status_code})"
        try:
            err_json = e.response.json()
            if "error" in err_json:
                if isinstance(err_json["error"], dict) and "message" in err_json["error"]:
                    error_msg = f"{active_p}: {err_json['error']['message']}"
                else:
                    error_msg = f"{active_p}: {err_json['error']}"
        except Exception:
            pass
        raise HTTPException(502, error_msg)
    except httpx.TimeoutException:
        active_p = (request.provider or config.LLM_PROVIDER).upper()
        raise HTTPException(
            504,
            f"Request to {active_p} timed out while generating an answer.",
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate answer: {str(e)}")

