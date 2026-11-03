
import asyncio
import json
import time
from pathlib import Path

from app.llm.client import ollama_client


ROOT = Path(__file__).resolve().parents[1]

JSON_PATH = ROOT / "data" / "test_batch_1.json"
RULES_PATH = ROOT / "app" / "prompts" / "islamic_rules.txt"


async def main():
    start = time.perf_counter()

    data = json.loads(
        JSON_PATH.read_text(encoding="utf-8")
    )

    rules = RULES_PATH.read_text(encoding="utf-8")

    records = data.get("records", [])

    if not records:
        raise ValueError("No records found in test JSON")

    print(f"[TEST] Loaded records: {len(records)}")
    print(f"[TEST] JSON size: {JSON_PATH.stat().st_size} bytes")

    print("\nAvailable questions:\n")

    for index, record in enumerate(records, start=1):
        print(f"{index}. {record.get('question', '')[:200]}")

    question = input("\nEnter your question: ").strip()

    if not question:
        print("No question entered")
        return

    messages = [
        {
            "role": "system",
            "content": rules,
        },
        {
            "role": "user",
            "content": (
                "Use ONLY this JSON data:\n\n"
                + json.dumps(data, ensure_ascii=False)
                + "\n\nUser question:\n"
                + question
            ),
        },
    ]

    try:
        answer = await ollama_client.generate(
            messages=messages,
            task="answer",
        )

        print("\n========== QWEN ANSWER ==========\n")
        print(answer)

        print(
            "\n[TIMING] Total test: "
            f"{time.perf_counter() - start:.2f}s"
        )

    finally:
        await ollama_client.close()


if __name__ == "__main__":
    asyncio.run(main())
