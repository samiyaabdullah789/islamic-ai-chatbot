
import time

import httpx

from app.core.config import settings


class OllamaClient:

    TASK_OPTIONS = {
        "rewrite": {
            "num_predict": 120,
            "temperature": 0.0,
        },
        "answer": {
            "num_predict": 650,
            "temperature": 0.0,
        },
        "translation": {
            "num_predict": 100,
            "temperature": 0.0,
        },
    }

    def __init__(self):
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL

        self.num_ctx = settings.OLLAMA_NUM_CTX

        if self.num_ctx <= 0:
            raise ValueError(
                "OLLAMA_NUM_CTX must be positive"
            )

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                180.0,
                connect=10.0,
            )
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        task: str = "answer",
    ) -> str:

        if task not in self.TASK_OPTIONS:
            raise ValueError(
                f"Unsupported Ollama task: {task}"
            )

        start = time.perf_counter()

        options = self.TASK_OPTIONS[task].copy()
        options["num_ctx"] = self.num_ctx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": options,
        }

        if json_mode:
            payload["format"] = "json"

        print(
            f"[OLLAMA] Model: {self.model}",
            flush=True,
        )

        print(
            f"[OLLAMA] Task: {task}",
            flush=True,
        )

        print(
            f"[OLLAMA] Requested context: {self.num_ctx}",
            flush=True,
        )

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        print(
            "[TIMING] Ollama HTTP request: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

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

        prompt_tokens = data.get("prompt_eval_count")

        if isinstance(prompt_tokens, int):
            print(
                "[OLLAMA] Prompt tokens:",
                prompt_tokens,
                flush=True,
            )

            if prompt_tokens >= self.num_ctx:
                print(
                    "[WARNING] Prompt token count reached "
                    "the requested context limit. "
                    "Input truncation is possible.",
                    flush=True,
                )

        generated_tokens = data.get("eval_count")

        if isinstance(generated_tokens, int):
            print(
                "[TIMING] Generated tokens:",
                generated_tokens,
                flush=True,
            )

        message = data.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Invalid Ollama response message"
            )

        content = message.get("content", "")

        if not isinstance(content, str):
            raise ValueError(
                "Invalid Ollama response content"
            )

        return content.strip()

    async def close(self):
        await self.client.aclose()


ollama_client = OllamaClient()
