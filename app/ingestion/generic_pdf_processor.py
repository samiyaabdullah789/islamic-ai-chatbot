
from pathlib import Path

from pypdf import PdfReader


def process_generic_pdf(
    file_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[dict]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be between 0 and chunk_size"
        )

    pdf_path = Path(file_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        text = " ".join(text.split())

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                        "chunk_index": len(chunks),
                    }
                )

            if end == len(text):
                break

            start = end - chunk_overlap

    return chunks
