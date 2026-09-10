"""
Resume file validation and safe storage.

Security model:
  - Files are stored in data/uploads/<user_id>/<uuid>.<ext>
  - Storage path is NEVER returned to the client
  - Original filenames are stored for display only (never used as paths)
  - MIME type is detected from file bytes, not from the Content-Type header
  - Extension whitelist: .pdf, .docx only
  - Maximum file size: 5 MB
  - Safe filename: UUID-based, no user input in path components
  - Path traversal: impossible because paths are constructed from UUIDs
  - Uploaded content is never executed

Text extraction:
  - PDF: pypdf (PdfReader) — text only, no macro execution
  - DOCX: python-docx — paragraph text only, no macro execution
  - Extraction errors are caught; status is set to 'failed' rather than crashing
  - Extracted text is treated as untrusted — never used as LLM system instructions
"""

import logging
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# MIME type signatures detected from file bytes (magic bytes)
# These are checked in addition to the extension whitelist.
_PDF_MAGIC = b"%PDF"
_DOCX_MAGIC = b"PK\x03\x04"  # ZIP-based (OOXML)

_UPLOADS_ROOT = Path("data/uploads")


# ── Validation ────────────────────────────────────────────────────────

class ResumeValidationError(ValueError):
    """Raised when an uploaded file fails validation."""
    pass


def _detect_mime(data: bytes) -> Optional[str]:
    """Detect MIME type from magic bytes. Returns None if unrecognised."""
    if data[:4] == _PDF_MAGIC:
        return "application/pdf"
    if data[:4] == _DOCX_MAGIC:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return None


def _sanitize_extension(filename: str) -> str:
    """
    Extract and validate the file extension.
    Returns the lowercase extension (e.g. '.pdf') or raises.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ResumeValidationError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return suffix


def validate_resume_upload(
    filename: str,
    file_data: bytes,
) -> Tuple[str, str]:
    """
    Validate an uploaded resume file.

    Args:
        filename:  Original filename as reported by the client (untrusted).
        file_data: Raw file bytes.

    Returns:
        (detected_mime, extension) on success.

    Raises:
        ResumeValidationError: on any validation failure.
    """
    # 1. Size check
    if len(file_data) == 0:
        raise ResumeValidationError("Uploaded file is empty.")
    if len(file_data) > MAX_FILE_SIZE_BYTES:
        raise ResumeValidationError(
            f"File is too large ({len(file_data) // 1024} KB). "
            f"Maximum allowed: {MAX_FILE_SIZE_BYTES // 1024} KB."
        )

    # 2. Extension check (whitelist)
    extension = _sanitize_extension(filename)

    # 3. Magic-byte MIME check — must match the claimed extension
    detected_mime = _detect_mime(file_data)
    if detected_mime is None:
        raise ResumeValidationError(
            "File content does not match a supported format (PDF or DOCX)."
        )

    # PDF extension must have PDF magic bytes
    if extension == ".pdf" and not file_data.startswith(_PDF_MAGIC):
        raise ResumeValidationError(
            "File extension is .pdf but content does not appear to be a PDF."
        )

    # DOCX extension must have ZIP magic bytes
    if extension == ".docx" and not file_data.startswith(_DOCX_MAGIC):
        raise ResumeValidationError(
            "File extension is .docx but content does not appear to be a DOCX."
        )

    return detected_mime, extension


# ── Safe storage ──────────────────────────────────────────────────────

def build_safe_path(user_id: int, extension: str) -> Tuple[Path, str]:
    """
    Build a safe, UUID-based storage path for a resume file.

    Path structure:
        data/uploads/<user_id>/<uuid><extension>

    Returns (absolute Path, safe_filename string).
    """
    user_dir = _UPLOADS_ROOT / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4()}{extension}"
    full_path = user_dir / safe_name

    return full_path, safe_name


def store_resume_file(
    user_id: int,
    file_data: bytes,
    extension: str,
) -> Tuple[Path, str]:
    """
    Write validated resume bytes to isolated, non-public storage.

    Returns (storage_path, safe_filename).
    Never exposes the path to the caller's HTTP response.
    """
    full_path, safe_name = build_safe_path(user_id, extension)

    full_path.write_bytes(file_data)
    logger.info(
        "Resume stored: user_id=%d safe_filename=%s size=%d",
        user_id, safe_name, len(file_data),
    )

    return full_path, safe_name


# ── Text extraction ───────────────────────────────────────────────────

def extract_text_from_pdf(file_data: bytes) -> str:
    """
    Extract plain text from PDF bytes using pypdf.

    The extracted text is treated as untrusted candidate input — never
    used as LLM system instructions without further sanitisation.
    """
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_data))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages).strip()

    except Exception as exc:
        logger.warning("PDF extraction failed: %s", type(exc).__name__)
        return ""


def extract_text_from_docx(file_data: bytes) -> str:
    """
    Extract plain text from DOCX bytes using python-docx.

    Only paragraph text is extracted — no macros, no embedded objects.
    """
    try:
        import io
        import docx

        doc = docx.Document(io.BytesIO(file_data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    except Exception as exc:
        logger.warning("DOCX extraction failed: %s", type(exc).__name__)
        return ""


def extract_resume_text(file_data: bytes, mime_type: str) -> Tuple[str, str]:
    """
    Extract text from a validated resume file.

    Args:
        file_data: Raw file bytes (already validated).
        mime_type: Detected MIME type.

    Returns:
        (extracted_text, status)
        status is 'success' or 'failed'.
    """
    if mime_type == "application/pdf":
        text = extract_text_from_pdf(file_data)
    else:
        text = extract_text_from_docx(file_data)

    if text:
        return text, "success"
    else:
        return "", "failed"


# ── Injection guard for extracted resume text ─────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"you\s+are\s+now|forget\s+everything|"
    r"system\s*:|<\s*/?system\s*>|"
    r"assistant\s*:|<\s*/?assistant\s*>)",
    re.IGNORECASE,
)

_MAX_CONTEXT_CHARS = 2000


def sanitize_resume_text_for_context(text: str) -> str:
    """
    Sanitise extracted resume text before passing it as candidate_context
    to interview prompts.

    Actions:
    - Truncate to MAX_CONTEXT_CHARS
    - Strip prompt injection patterns
    - Normalise whitespace

    The result is still treated as untrusted user input — it is passed
    only as candidate_context, never as a system instruction.
    """
    if not text:
        return ""

    # Truncate first to limit injection surface area
    truncated = text[:_MAX_CONTEXT_CHARS]

    # Strip injection patterns
    cleaned = _INJECTION_PATTERNS.sub("[REDACTED]", truncated)

    # Normalise Unicode and whitespace
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
