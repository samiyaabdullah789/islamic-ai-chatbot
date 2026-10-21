
import json
import re
from pathlib import Path


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def split_answer(
    answer: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk overlap")

    # Normalize excessive whitespace while
    # preserving paragraph breaks.
    answer = re.sub(r"[ \t]+", " ", answer).strip()

    if not answer:
        return []

    if len(answer) <= chunk_size:
        return [answer]

    chunks = []
    start = 0

    while start < len(answer):

        end = min(start + chunk_size, len(answer))

        if end < len(answer):
            # Prefer splitting at a paragraph or sentence.
            segment = answer[start:end]

            split_at = max(
                segment.rfind("\n\n"),
                segment.rfind(". "),
                segment.rfind("? "),
                segment.rfind("! "),
            )

            if split_at >= chunk_size // 2:
                end = start + split_at + 1

        chunk = answer[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(answer):
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def process_generic_json(file_path: str) -> list[dict]:

    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict) and isinstance(
        data.get("records"), list
    ):
        records = data["records"]

    else:
        raise ValueError(
            "JSON must contain a list or "
            "an object with a 'records' list."
        )

    documents = []
    skipped = 0

    for record_index, record in enumerate(records):

        if not isinstance(record, dict):
            skipped += 1
            continue

        question = record.get("question", "")
        answer = record.get("answer", "")

        if not isinstance(question, str):
            skipped += 1
            continue

        if not isinstance(answer, str):
            skipped += 1
            continue

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            skipped += 1
            continue

        madhhab = record.get("madhhab", "")

        if not isinstance(madhhab, str):
            madhhab = ""

        category = record.get(
            "category",
            record.get("categories", []),
        )

        if isinstance(category, list):
            category = ", ".join(
                str(item).strip()
                for item in category
                if str(item).strip()
            )

        elif not isinstance(category, str):
            category = ""

        tags = record.get("tags", [])

        if isinstance(tags, list):
            tags = ", ".join(
                str(tag).strip()
                for tag in tags
                if str(tag).strip()
            )

        elif not isinstance(tags, str):
            tags = ""

        metadata = {
            "madhhab": madhhab.strip(),
            "category": category.strip(),
            "tags": tags.strip(),
        }

        answer_chunks = split_answer(answer)

        for chunk_number, answer_chunk in enumerate(
            answer_chunks
        ):

            text = (
                f"Question: {question}\n"
                f"Answer: {answer_chunk}"
            )

            documents.append({
                "text": text,
                "metadata": metadata.copy(),
                "record_index": (
                    f"{record_index}-{chunk_number}"
                ),
            })

    print(f"[JSON] Total records: {len(records)}")
    print(f"[JSON] Valid Q&A records: {len(records) - skipped}")
    print(f"[JSON] Skipped records: {skipped}")
    print(f"[JSON] Total chunks: {len(documents)}")

    return documents
