
from app.ingestion.pdf_processor import process_pdf
from app.vector_store.chroma_store import chroma_store


SOURCE_ID = 1


def verify_bukhari() -> None:
    print("\nReading Sahih al-Bukhari PDF...")

    hadiths = process_pdf()

    usable_hadiths = [
        hadith
        for hadith in hadiths
        if hadith["text"].strip()
    ]

    expected_ids = {
        (
            f"source-{SOURCE_ID}"
            f"-v{hadith['volume']}"
            f"-b{hadith['book']}"
            f"-h{hadith['hadith_number']}"
        )
        for hadith in usable_hadiths
    }

    print("\nReading existing ChromaDB records...")

    stored_ids = set(
        chroma_store.get_source_document_ids(
            source_id=SOURCE_ID,
        )
    )

    missing_ids = expected_ids - stored_ids
    unexpected_ids = stored_ids - expected_ids

    print("\n" + "=" * 60)
    print("SAHIH AL-BUKHARI VERIFICATION REPORT")
    print("=" * 60)

    print(f"Extracted Hadiths: {len(hadiths)}")
    print(f"Usable Hadiths: {len(usable_hadiths)}")
    print(f"Unique Expected IDs: {len(expected_ids)}")
    print(f"Stored ChromaDB IDs: {len(stored_ids)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Unexpected IDs: {len(unexpected_ids)}")

    if missing_ids:
        print("\nSample missing IDs:")
        for document_id in sorted(missing_ids)[:10]:
            print(f"  {document_id}")

    if unexpected_ids:
        print("\nSample unexpected IDs:")
        for document_id in sorted(unexpected_ids)[:10]:
            print(f"  {document_id}")

    if (
        expected_ids
        and not missing_ids
        and not unexpected_ids
    ):
        print(
            "\nRESULT: All expected document IDs "
            "are present in ChromaDB."
        )
        print(
            "Note: This verifies ID coverage, "
            "not embedding or text accuracy."
        )
    else:
        print(
            "\nRESULT: Document ID mismatch detected "
            "or no usable Hadiths found."
        )

    print("=" * 60)


if __name__ == "__main__":
    verify_bukhari()
