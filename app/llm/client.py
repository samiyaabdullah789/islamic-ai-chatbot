
import time
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

        start = time.perf_counter()

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

            print(
                f"[TIMING] Ollama HTTP request: "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )

            # Ollama durations are reported in nanoseconds
            for key in (
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            ):
                duration = data.get(key)

                if isinstance(duration, (int, float)):
                    print(
                        f"[TIMING] Ollama {key}: "
                        f"{duration / 1_000_000_000:.2f}s",
                        flush=True,
                    )

            return data["message"]["content"]


ollama_client = OllamaClient()
