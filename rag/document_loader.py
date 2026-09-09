from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(directory: str = "data/knowledge") -> list[dict]:
    """
    Load knowledge documents from the specified directory.

    Each returned document contains:
    - source
    - text
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {directory_path}"
        )

    documents = []

    for file_path in sorted(directory_path.rglob("*")):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if not text:
            continue

        documents.append(
            {
                "source": str(file_path),
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            f"No knowledge documents found in {directory_path}"
        )

    return documents