import re
from pathlib import Path

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parents[2]

PDF_PATH = (
    BASE_DIR
    / "data"
    / "sources"
    / "pdfs"
    / "sahih_bukhari.pdf"
)


HADITH_START_PATTERN = re.compile(
    r"Volume\s+\d+,\s*Book\s+\d+,\s*Number\s+\d+:"
)


def clean_page_text(text: str) -> str:
    """
    Remove repeated PDF headers / footers while preserving Hadith text.
    """

    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        # Examples:
        # Volume 9 - 1700 / 1700
        # Volume 1 - 6 / 1700
        if re.fullmatch(
            r"Volume\s+\d+\s*-\s*\d+\s*/\s*\d+",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue

        # Example:
        # - 2 / 1700
        if re.fullmatch(
            r"-?\s*\d+\s*/\s*\d+",
            stripped,
        ):
            continue

        # Repeated navigation/header lines
        if stripped.upper().startswith("SAHIH BUKHARI"):
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def extract_hadiths(text: str) -> list[dict]:
    pattern = re.compile(
        r"Volume\s+(\d+),\s*Book\s+(\d+),\s*Number\s+(\d+):\s*"
        r"(.*?)"
        r"(?=Volume\s+\d+,\s*Book\s+\d+,\s*Number\s+\d+:|$)",
        re.DOTALL,
    )

    matches = pattern.findall(text)

    hadiths = []

    for volume, book, number, content in matches:
        content = content.strip()

        narrator = None
        hadith_text = content

        narrator_match = re.match(
            r"Narrated\s+([^:]+):\s*(.*)",
            content,
            re.DOTALL,
        )

        if narrator_match:
            narrator = narrator_match.group(1).strip()
            hadith_text = narrator_match.group(2).strip()

        hadiths.append(
            {
                "volume": int(volume),
                "book": int(book),
                "hadith_number": int(number),
                "narrator": narrator,
                "text": hadith_text,
            }
        )

    return hadiths


def split_complete_text(text: str) -> tuple[str, str]:
    """
    Keep the final Hadith in carry-over because it may continue
    into the next batch.

    Returns:
        complete_text -> safe to parse now
        carry_over    -> send into next batch
    """

    starts = list(HADITH_START_PATTERN.finditer(text))

    if not starts:
        return "", text

    last_start = starts[-1].start()

    complete_text = text[:last_start]
    carry_over = text[last_start:]

    return complete_text, carry_over


def process_pdf(batch_size: int = 25) -> list[dict]:
    reader = PdfReader(PDF_PATH)

    total_pages = len(reader.pages)
    all_hadiths = []

    # Last incomplete Hadith from previous batch
    carry_over = ""

    print(f"Total PDF pages: {total_pages}")
    print(
        f"Processing PDF in batches of "
        f"{batch_size} pages...\n"
    )

    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)

        print(
            f"Processing pages "
            f"{start + 1}-{end}..."
        )

        pages_text = []

        for page_number in range(start, end):
            try:
                text = reader.pages[page_number].extract_text()

                if text:
                    cleaned_text = clean_page_text(text)
                    pages_text.append(cleaned_text)

            except Exception as error:
                print(
                    f"Could not extract page "
                    f"{page_number + 1}: {error}"
                )

        batch_text = "\n".join(pages_text)

        # Add incomplete Hadith from previous batch
        combined_text = carry_over + "\n" + batch_text

        complete_text, carry_over = split_complete_text(
            combined_text
        )

        batch_hadiths = extract_hadiths(
            complete_text
        )

        all_hadiths.extend(batch_hadiths)

        print(
            f"Found {len(batch_hadiths)} "
            f"complete hadiths in this batch."
        )

    # Process final Hadith remaining in carry-over
    if carry_over.strip():
        final_hadiths = extract_hadiths(carry_over)

        all_hadiths.extend(final_hadiths)

        print(
            f"Found {len(final_hadiths)} "
            f"final hadith(s)."
        )

    return all_hadiths


def validate_hadiths(hadiths: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"Total Hadiths: {len(hadiths)}")

    # Find Hadith records with empty text
    empty_text = [
        hadith
        for hadith in hadiths
        if not hadith["text"].strip()
    ]

    print(
        f"Hadiths with empty text: "
        f"{len(empty_text)}"
    )

    # IMPORTANT:
    # Print the actual empty records so we can investigate them
    if empty_text:
        print("\nEmpty Hadith records:")

        for hadith in empty_text:
            print(hadith)

    # Check duplicate Volume + Book + Hadith references
    references = [
        (
            hadith["volume"],
            hadith["book"],
            hadith["hadith_number"],
        )
        for hadith in hadiths
    ]

    duplicate_count = (
        len(references) - len(set(references))
    )

    print(
        f"Duplicate references: "
        f"{duplicate_count}"
    )


if __name__ == "__main__":
    hadiths = process_pdf()

    validate_hadiths(hadiths)

    # Show first, middle and last records
    # for manual verification
    if hadiths:
        sample_indexes = [
            0,
            len(hadiths) // 2,
            len(hadiths) - 1,
        ]

        labels = [
            "FIRST HADITH",
            "MIDDLE HADITH",
            "LAST HADITH",
        ]

        for label, index in zip(
            labels,
            sample_indexes,
        ):
            print("\n" + "=" * 70)
            print(label)
            print("=" * 70)
            print(hadiths[index])