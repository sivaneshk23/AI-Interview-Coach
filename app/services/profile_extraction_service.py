"""
Resume Profile Extraction Service.

Responsibility: derive structured candidate profile fields from raw
resume text using IBM Granite via the existing LLM service.

Architecture:
    extract_profile_from_text(text) → ExtractedProfile
        ↑ called by the /resume/{id}/extract-profile endpoint
        ↑ result shown to candidate for review/editing before confirm

Security design:
    - Resume text is UNTRUSTED input.
    - It is passed only as USER-ROLE content — never as SYSTEM instructions.
    - Injection patterns are stripped before sending to LLM.
    - The LLM output is parsed as structured data, not executed.
    - If LLM extraction fails, a safe empty profile is returned — no crash.
    - Extracted fields are offered to the candidate for review; they are
      NOT automatically trusted until the candidate confirms.

Offline / test mode:
    When IBM_API_KEY is missing or the LLM call fails, the service
    returns an empty ExtractionResult with status="unavailable" so that
    tests can run completely offline without IBM access.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Max resume text to send to LLM (tokens are expensive) ────────────
_MAX_TEXT_FOR_EXTRACTION = 3000  # characters

# ── Injection guard (same patterns as file_handler.py) ────────────────
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"you\s+are\s+now|forget\s+everything|"
    r"system\s*:|<\s*/?system\s*>|"
    r"assistant\s*:|<\s*/?assistant\s*>)",
    re.IGNORECASE,
)


def _sanitize_for_llm(text: str) -> str:
    """Strip injection patterns and normalise before sending to LLM."""
    truncated = text[:_MAX_TEXT_FOR_EXTRACTION]
    cleaned   = _INJECTION_PATTERNS.sub("[REDACTED]", truncated)
    cleaned   = unicodedata.normalize("NFKC", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# ── Extracted Profile dataclass ───────────────────────────────────────

@dataclass
class ExtractedProfile:
    """
    Structured profile extracted from resume text.

    All fields are Optional — extraction may be partial.
    The candidate must review and confirm before any field is
    used as authoritative context.
    """
    full_name:       Optional[str] = None
    email:           Optional[str] = None
    phone:           Optional[str] = None
    college:         Optional[str] = None
    degree:          Optional[str] = None
    branch:          Optional[str] = None
    graduation_year: Optional[int] = None
    skills:          Optional[str] = None
    projects:        Optional[str] = None
    experience:      Optional[str] = None
    certifications:  Optional[str] = None
    achievements:    Optional[str] = None
    # Suggested role from resume — user must confirm/edit this
    suggested_role:  Optional[str] = None


@dataclass
class ExtractionResult:
    """
    Result returned by extract_profile_from_text().
    """
    status:  str                 # "success" | "partial" | "failed" | "unavailable"
    profile: ExtractedProfile    = field(default_factory=ExtractedProfile)
    message: str                 = ""


# ── Extraction prompt ─────────────────────────────────────────────────

_EXTRACTION_PROMPT_TEMPLATE = """
Extract structured candidate profile information from the resume text below.

Return ONLY a valid JSON object with the following keys (omit any key you cannot find):
  full_name       (string)
  email           (string)
  phone           (string)
  college         (string)
  degree          (string, e.g. "B.Tech", "B.E.", "M.Sc")
  branch          (string, e.g. "Computer Science", "Electronics")
  graduation_year (integer, e.g. 2024)
  skills          (string, comma-separated list of technical skills)
  projects        (string, brief description of key projects)
  experience      (string, work experience summary)
  certifications  (string, certifications if present)
  achievements    (string, notable achievements if present)
  suggested_role  (string, inferred target job role from resume content)

Rules:
- Return ONLY the JSON object. No explanation, no markdown, no extra text.
- If a field is not present, omit the key.
- Do not invent information not present in the resume.
- skills must be a single comma-separated string.
- graduation_year must be a 4-digit integer.

RESUME TEXT:
{resume_text}

JSON:
""".strip()


# ── Main extraction function ──────────────────────────────────────────

def extract_profile_from_text(resume_text: str) -> ExtractionResult:
    """
    Extract a structured candidate profile from raw resume text.

    Uses IBM Granite via IBMWatsonxService. Falls back gracefully if
    the LLM is unavailable (offline tests, missing credentials).

    Args:
        resume_text: Plain text extracted from a PDF or DOCX resume.

    Returns:
        ExtractionResult with status and extracted profile fields.
        Status values:
            "success"     — extraction successful, fields populated
            "partial"     — extraction ran but response was incomplete JSON
            "failed"      — LLM returned unusable output
            "unavailable" — LLM service not configured or unreachable
    """
    if not resume_text or not resume_text.strip():
        return ExtractionResult(
            status="failed",
            message="Empty resume text provided.",
        )

    safe_text = _sanitize_for_llm(resume_text)
    prompt    = _EXTRACTION_PROMPT_TEMPLATE.format(resume_text=safe_text)

    # Try LLM extraction
    raw_response = _call_llm(prompt)
    if raw_response is None:
        return ExtractionResult(
            status="unavailable",
            message="LLM service is not available. Profile extraction requires IBM watsonx.ai.",
        )

    # Parse JSON response
    return _parse_extraction_response(raw_response)


def _call_llm(prompt: str) -> Optional[str]:
    """
    Call IBM Granite for profile extraction.

    Returns the raw text response, or None if the service is unavailable.
    Never propagates credentials in exceptions.
    """
    try:
        from app.services.llm_service import IBMWatsonxService
        service  = IBMWatsonxService()
        response = service.generate(
            prompt         = prompt,
            max_new_tokens = 500,   # profile JSON is compact
            temperature    = 0.1,   # low temperature for structured extraction
        )
        return response
    except RuntimeError as exc:
        # IBMWatsonxService raises RuntimeError when the service is unavailable.
        logger.warning("LLM unavailable for profile extraction: %s", exc)
        return None
    except Exception as exc:
        logger.warning(
            "Unexpected error during profile extraction LLM call: %s",
            type(exc).__name__,
        )
        return None


def _parse_extraction_response(raw: str) -> ExtractionResult:
    """
    Parse the LLM's JSON response into an ExtractedProfile.

    Handles:
    - Clean JSON objects
    - JSON embedded in markdown code fences
    - Partial JSON (best-effort)
    """
    # Strip markdown code fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("Profile extraction: no JSON object found in LLM response.")
        return ExtractionResult(
            status="failed",
            message="Could not parse profile from resume. Please enter your profile manually.",
        )

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Profile extraction: JSON parse error.")
        return ExtractionResult(
            status="partial",
            message="Partial profile extracted. Please review and correct the fields.",
        )

    # Build profile — sanitize each field
    profile = ExtractedProfile(
        full_name       = _safe_str(data.get("full_name")),
        email           = _safe_email(data.get("email")),
        phone           = _safe_phone(data.get("phone")),
        college         = _safe_str(data.get("college")),
        degree          = _safe_str(data.get("degree")),
        branch          = _safe_str(data.get("branch")),
        graduation_year = _safe_year(data.get("graduation_year")),
        skills          = _safe_str(data.get("skills")),
        projects        = _safe_str(data.get("projects")),
        experience      = _safe_str(data.get("experience")),
        certifications  = _safe_str(data.get("certifications")),
        achievements    = _safe_str(data.get("achievements")),
        suggested_role  = _safe_str(data.get("suggested_role")),
    )

    # Determine status
    populated = sum(
        1 for v in [
            profile.full_name, profile.skills, profile.experience,
            profile.college, profile.degree,
        ] if v
    )
    status = "success" if populated >= 2 else "partial"

    return ExtractionResult(
        status  = status,
        profile = profile,
        message = (
            "Profile extracted successfully. Please review and confirm the fields."
            if status == "success"
            else "Partial profile extracted. Some fields may be missing — please fill them in."
        ),
    )


# ── Field sanitizers ──────────────────────────────────────────────────

def _safe_str(value) -> Optional[str]:
    """Return a clean string or None if empty/invalid."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _safe_email(value) -> Optional[str]:
    """Basic email validation — returns None if not a plausible email."""
    s = _safe_str(value)
    if s and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s):
        return s.lower()
    return None


def _safe_phone(value) -> Optional[str]:
    """Strip non-phone characters and return."""
    s = _safe_str(value)
    if s:
        cleaned = re.sub(r"[^\d\s+\-(). ext]", "", s)
        return cleaned.strip() or None
    return None


def _safe_year(value) -> Optional[int]:
    """Return a graduation year (1980–2040) or None."""
    try:
        year = int(str(value).strip())
        if 1980 <= year <= 2040:
            return year
    except (ValueError, TypeError):
        pass
    return None
