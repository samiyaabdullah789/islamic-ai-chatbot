
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from app.embeddings.ollama_embeddings import embedding_service
from app.vector_store.chroma_client import chroma_client


BATCH_SIZE = 32


def normalize_text(value):
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return ", ".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )

    return ""


def load_records(file_path: Path) -> list[dict]:
    with file_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(
        data.get("records"), list
    ):
        return data["records"]

    raise ValueError(
        "JSON must contain a list or a 'records' list."
    )


def build_search_text(record: dict) -> str:
    question = normalize_text(record.get("question"))
    title = normalize_text(record.get("title"))
    category = normalize_text(
        record.get(
            "category",
            record.get("categories", ""),
        )
    )
    tags = normalize_text(record.get("tags"))
    madhhab = normalize_text(record.get("madhhab"))

    parts = []

    if title:
        parts.append(f"Title: {title}")

    parts.append(f"Question: {question}")

    if category:
        parts.append(f"Category: {category}")

    if tags:
        parts.append(f"Tags: {tags}")

    if madhhab:
        parts.append(f"Madhhab: {madhhab}")

    return "\n".join(parts)


async def index_file(
    file_path: Path,
    source_id: int,
):
    start = time.perf_counter()

    records = load_records(file_path)

    collection = chroma_client.qa_collection

    print(f"[QA] File: {file_path}")
    print(f"[QA] Source ID: {source_id}")
    print(f"[QA] Total records: {len(records)}")

    valid_records = []
    skipped = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            skipped += 1
            continue

        question = normalize_text(
            record.get("question")
        )
        answer = record.get("answer")

        if not question or not isinstance(answer, str):
            skipped += 1
            continue

        answer = answer.strip()

        if not answer:
            skipped += 1
            continue

        category = normalize_text(
            record.get(
                "category",
                record.get("categories", ""),
            )
        )

        metadata = {
            "source_id": source_id,
            "source_name": file_path.stem,
            "record_index": index,
            "question": question,
            "answer": answer,
            "madhhab": normalize_text(
                record.get("madhhab")
            ),
            "category": category,
            "tags": normalize_text(
                record.get("tags")
            ),
        }

        record_id = normalize_text(
            record.get("id")
        )

        if record_id:
            metadata["original_record_id"] = record_id

        valid_records.append({
            "id": f"qa-source-{source_id}-record-{index}",
            "search_text": build_search_text(record),
            "metadata": metadata,
        })

    print(f"[QA] Valid records: {len(valid_records)}")
    print(f"[QA] Skipped records: {skipped}")

    existing = collection.get(
        where={"source_id": source_id},
        include=[],
    )

    existing_ids = set(existing["ids"])

    pending = [
        item
        for item in valid_records
        if item["id"] not in existing_ids
    ]

    print(f"[QA] Already indexed: {len(valid_records) - len(pending)}")
    print(f"[QA] Pending: {len(pending)}")

    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset:offset + BATCH_SIZE]

        texts = [
            item["search_text"]
            for item in batch
        ]

        embed_start = time.perf_counter()

        embeddings = await embedding_service.embed(texts)

        print(
            "[TIMING] QA embedding batch: "
            f"{time.perf_counter() - embed_start:.2f}s",
            flush=True,
        )

        if len(embeddings) != len(batch):
            raise ValueError(
                "Embedding count does not match batch size"
            )

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                item["metadata"]
                for item in batch
            ],
        )

        print(
            f"[QA] Indexed {min(offset + BATCH_SIZE, len(pending))}"
            f"/{len(pending)} pending records",
            flush=True,
        )

    print(
        "[TIMING] QA indexing total: "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    print(
        f"[QA] Collection count: {collection.count()}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--source-id",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(
            f"File does not exist: {args.file}"
        )

    asyncio.run(
        index_file(
            file_path=args.file.resolve(),
            source_id=args.source_id,
        )
    )


if __name__ == "__main__":
    main()
