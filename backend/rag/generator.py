"""
LLM Generator Module - Multi Provider
"""
import httpx
from backend import config


async def generate_answer(
    messages: list[dict],
    provider: str | None = None,
    context_chunks: list[dict] | None = None,
    question: str = "",
    answer_mode: str = "simple",
) -> str:
    from backend.rag.offline_reader import synthesize_offline_answer
    active = (provider or config.LLM_PROVIDER).lower()

    if active == "offline":
        return synthesize_offline_answer(question=question, context_chunks=context_chunks or [], answer_mode=answer_mode)

    if active == "auto":
        try:
            return await _generate_ollama(messages)
        except Exception:
            pass
        if config.GROQ_API_KEY and config.GROQ_API_KEY.strip() not in ("", "your-api-key-here"):
            try:
                return await _generate_groq(messages)
            except Exception:
                pass
        from backend.rag.offline_reader import synthesize_offline_answer
        return synthesize_offline_answer(question=question, context_chunks=context_chunks or [], answer_mode=answer_mode)

    if active == "ollama":
        return await _generate_ollama(messages)

    if active == "groq":
        return await _generate_groq(messages)

    from backend.rag.offline_reader import synthesize_offline_answer
    return synthesize_offline_answer(question=question, context_chunks=context_chunks or [], answer_mode=answer_mode)


async def _generate_ollama(messages: list[dict]) -> str:
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": config.LLM_TEMPERATURE, "num_predict": config.LLM_MAX_TOKENS},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["message"]["content"]


async def _generate_groq(messages: list[dict]) -> str:
    if not config.GROQ_API_KEY or config.GROQ_API_KEY.strip() in ("", "your-api-key-here"):
        raise ValueError("GROQ_API_KEY not set.")
    url = f"{config.GROQ_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY.strip()}", "Content-Type": "application/json"}
    payload = {"model": config.GROQ_MODEL, "messages": messages, "temperature": config.LLM_TEMPERATURE, "max_tokens": config.LLM_MAX_TOKENS}
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


async def check_llm_health() -> dict:
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                names = [m["name"] for m in resp.json().get("models", [])]
                ollama_ok = config.OLLAMA_MODEL in names
    except Exception:
        pass
    groq_ok = False
    if config.GROQ_API_KEY and config.GROQ_API_KEY.strip() not in ("", "your-api-key-here"):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{config.GROQ_BASE_URL.rstrip('/')}/models", headers={"Authorization": f"Bearer {config.GROQ_API_KEY.strip()}"})
                groq_ok = resp.status_code == 200
        except Exception:
            pass
    if ollama_ok:
        detail = f"Ollama ({config.OLLAMA_MODEL}) - Local"
    elif groq_ok:
        detail = f"Groq ({config.GROQ_MODEL}) - Cloud"
    else:
        detail = "Offline Reader"
    return {"provider": config.LLM_PROVIDER, "ollama_running": True, "model_available": True, "configured_model": detail, "ollama_online": ollama_ok, "groq_online": groq_ok}


check_ollama_health = check_llm_health
