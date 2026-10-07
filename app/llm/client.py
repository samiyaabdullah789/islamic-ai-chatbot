import httpx

from app.core.config import settings


class OllamaClient:

    def __init__(self):
        self.base_url = settings.LLM_BASE_URL
        self.model = settings.LLM_MODEL

    async def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
            )

            response.raise_for_status()

            data = response.json()

            return data["message"]["content"]


ollama_client = OllamaClient()