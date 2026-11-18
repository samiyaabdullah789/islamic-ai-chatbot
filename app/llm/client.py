import re
import time

import httpx

from app.core.config import settings


class OllamaRequestError(RuntimeError):
    """An Ollama transport, HTTP, or response failure, not missing RAG evidence."""


class OllamaClient:
    TASK_OPTIONS = {
        "rewrite": {"num_predict": 120, "temperature": 0.0},
        "answer": {"num_predict": 1200, "temperature": 0.0},
        "verify": {"num_predict": 500, "temperature": 0.0},
        "translation": {"num_predict": 100, "temperature": 0.0},
    }

    def __init__(self):
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL
        self.num_ctx = settings.OLLAMA_NUM_CTX
        if self.num_ctx <= 0:
            raise ValueError("OLLAMA_NUM_CTX must be positive")
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=10.0)
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        task: str = "answer",
    ) -> str:
        if task not in self.TASK_OPTIONS:
            raise ValueError(f"Unsupported Ollama task: {task}")
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

        print(f"[OLLAMA] Model: {self.model}", flush=True)
        print(f"[OLLAMA] Task: {task}", flush=True)
        print(f"[OLLAMA] Requested context: {self.num_ctx}", flush=True)

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            print(f"[OLLAMA ERROR] Timeout during {task}: {error}", flush=True)
            raise OllamaRequestError(f"Ollama timed out during {task}") from error
        except httpx.ConnectError as error:
            print(f"[OLLAMA ERROR] Connection failed during {task}: {error}", flush=True)
            raise OllamaRequestError(f"Ollama connection failed during {task}") from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            print(f"[OLLAMA ERROR] HTTP {status} during {task}", flush=True)
            raise OllamaRequestError(
                f"Ollama HTTP {status} during {task}"
            ) from error
        except httpx.RequestError as error:
            print(f"[OLLAMA ERROR] Request failed during {task}: {error}", flush=True)
            raise OllamaRequestError(f"Ollama request failed during {task}") from error

        try:
            data = response.json()
        except ValueError as error:
            raise OllamaRequestError("Invalid JSON response from Ollama HTTP API") from error
        if not isinstance(data, dict):
            raise OllamaRequestError("Invalid Ollama response envelope")

        print(
            f"[TIMING] Ollama HTTP request: {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        for key in ("load_duration", "prompt_eval_duration", "eval_duration"):
            duration = data.get(key)
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                print(f"[TIMING] Ollama {key}: {duration / 1_000_000_000:.2f}s", flush=True)

        prompt_tokens = data.get("prompt_eval_count")
        if isinstance(prompt_tokens, int):
            print("[OLLAMA] Prompt tokens:", prompt_tokens, flush=True)
            if prompt_tokens >= self.num_ctx:
                print(
                    "[WARNING] Prompt token count reached the requested "
                    "context limit. Input truncation is possible.", flush=True,
                )

        generated_tokens = data.get("eval_count")
        if isinstance(generated_tokens, int):
            print("[TIMING] Generated tokens:", generated_tokens, flush=True)

        done_reason = data.get("done_reason")
        if done_reason:
            print(f"[OLLAMA] Done reason: {done_reason}", flush=True)
        if done_reason == "length":
            print(
                f"[WARNING] Generation for task '{task}' reached the token limit. "
                "Output may be truncated.", flush=True,
            )

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaRequestError("Invalid Ollama response message")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise OllamaRequestError("Invalid Ollama response content")
        content = re.sub(
            r"<think>.*?</think>", "", content, flags=re.S | re.I
        ).strip()
        if not content and done_reason == "length":
            raise OllamaRequestError("Ollama returned empty truncated output")
        return content

    async def close(self):
        await self.client.aclose()


ollama_client = OllamaClient()
