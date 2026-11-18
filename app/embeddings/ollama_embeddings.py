"""Ollama embeddings client; reuse HTTP connections and validate responses."""
import time
import httpx
from app.core.config import settings
from app.llm.client import OllamaRequestError


class OllamaEmbeddingService:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._url = settings.LLM_BASE_URL.rstrip("/") + "/api/embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        start = time.perf_counter()
        try:
            response = await self._client.post(
                self._url,
                json={"model": settings.EMBEDDING_MODEL, "input": texts},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaRequestError("Ollama embeddings timed out") from exc
        except httpx.ConnectError as exc:
            raise OllamaRequestError("Ollama embeddings connection failed") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaRequestError(
                f"Ollama embeddings HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaRequestError("Ollama embeddings request failed") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaRequestError("Ollama embeddings returned non-JSON") from exc
        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if (not isinstance(vectors, list) or len(vectors) != len(texts)
            or any(not isinstance(v, list) or not v or
                   any(type(x) not in (int, float) for x in v) for v in vectors)):
            raise OllamaRequestError("Ollama embeddings returned invalid vectors")
        dimensions = {len(v) for v in vectors}
        if len(dimensions) != 1:
            raise OllamaRequestError("Ollama embeddings vector dimensions mismatch")
        print(f"[TIMING] Ollama embeddings HTTP: {time.perf_counter()-start:.2f}s", flush=True)
        return vectors

    async def close(self):
        await self._client.aclose()


embedding_service = OllamaEmbeddingService()
