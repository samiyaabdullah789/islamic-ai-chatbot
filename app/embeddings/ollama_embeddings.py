import httpx

from app.core.config import settings


class OllamaEmbeddingService:

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.LLM_BASE_URL}/api/embed",
                json={
                    "model": settings.EMBEDDING_MODEL,
                    "input": texts,
                },
            )

            response.raise_for_status()
            return response.json()["embeddings"]


embedding_service = OllamaEmbeddingService()