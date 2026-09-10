"""
Resume upload routes.

POST /resume/upload                      — upload PDF or DOCX, extract text, persist
GET  /resume/                            — list candidate's resume records
GET  /resume/{resume_id}/extracted       — get extracted text for review
POST /resume/{resume_id}/extract-profile — extract structured profile from resume text
DELETE /resume/{resume_id}               — soft-delete a resume

The storage path is NEVER returned in API responses.
Original filenames are returned for display only.

Flow:
    1. POST /resume/upload          → validate, store, extract text
    2. GET  /resume/{id}/extracted  → candidate reviews extracted text
    3. POST /resume/{id}/extract-profile → extract structured profile from resume
    4. PUT  /candidate/profile      → candidate reviews/edits extracted fields
    5. POST /candidate/profile/confirm  → confirms profile
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import require_auth
from app.db.base import get_db
from app.db.models import CandidateProfile, Resume, User
from app.resume.file_handler import (
    ResumeValidationError,
    extract_resume_text,
    sanitize_resume_text_for_context,
    store_resume_file,
    validate_resume_upload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resume", tags=["resume"])

# ── Schemas ───────────────────────────────────────────────────────────

class ResumeUploadResponse(BaseModel):
    resume_id: int
    original_filename: Optional[str]
    mime_type: Optional[str]
    file_size_bytes: Optional[int]
    extraction_status: str
    message: str


class ResumeSummary(BaseModel):
    resume_id: int
    original_filename: Optional[str]
    mime_type: Optional[str]
    file_size_bytes: Optional[int]
    extraction_status: str


class ResumeExtractedResponse(BaseModel):
    resume_id: int
    extraction_status: str
    extracted_text: Optional[str]
    # Sanitised context snippet suitable for passing to the interview engine
    context_preview: Optional[str]


class ExtractedProfileResponse(BaseModel):
    """
    Response from POST /resume/{id}/extract-profile.
    Contains suggested field values for the candidate to review.
    The candidate must PUT /candidate/profile and then
    POST /candidate/profile/confirm to make these authoritative.
    """
    resume_id:       int
    status:          str   # "success" | "partial" | "failed" | "unavailable"
    message:         str
    # Suggested fields — all optional; candidate reviews/edits before confirming
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
    suggested_role:  Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=ResumeUploadResponse, status_code=201)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF or DOCX resume.

    - Validates MIME type via magic bytes (not the Content-Type header)
    - Validates extension whitelist (.pdf, .docx)
    - Enforces 5 MB size limit
    - Stores file at a UUID-based, non-public path
    - Extracts text immediately
    - Optionally pre-populates candidate profile interview_context
    - Never returns the storage path
    """
    # Read file bytes (bounded by the max-size check below)
    file_data = await file.read()
    original_filename = file.filename or "unknown"

    try:
        detected_mime, extension = validate_resume_upload(
            filename=original_filename,
            file_data=file_data,
        )
    except ResumeValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Store file safely
    try:
        storage_path, safe_filename = store_resume_file(
            user_id=current_user.id,
            file_data=file_data,
            extension=extension,
        )
    except Exception:
        logger.exception("Failed to store resume for user_id=%d", current_user.id)
        raise HTTPException(
            status_code=500,
            detail="Failed to store resume. Please try again.",
        )

    # Extract text
    extracted_text, extraction_status = extract_resume_text(file_data, detected_mime)

    # Build DB record (storage_path stored server-side, never returned)
    resume = Resume(
        user_id=current_user.id,
        safe_filename=safe_filename,
        original_filename=original_filename,
        mime_type=detected_mime,
        file_size_bytes=len(file_data),
        storage_path=str(storage_path),
        extracted_text=extracted_text if extracted_text else None,
        extraction_status=extraction_status,
    )
    db.add(resume)

    # Pre-populate interview_context on candidate profile if not already set
    if extracted_text:
        context_snippet = sanitize_resume_text_for_context(extracted_text)
        profile = (
            db.query(CandidateProfile)
            .filter(CandidateProfile.user_id == current_user.id)
            .first()
        )
        if profile is None:
            profile = CandidateProfile(
                user_id=current_user.id,
                interview_context=context_snippet,
            )
            db.add(profile)
        elif not profile.interview_context:
            profile.interview_context = context_snippet
            profile.is_confirmed = False

    db.commit()
    db.refresh(resume)

    logger.info(
        "Resume uploaded: user_id=%d resume_id=%d status=%s",
        current_user.id, resume.id, extraction_status,
    )

    return ResumeUploadResponse(
        resume_id=resume.id,
        original_filename=original_filename,
        mime_type=detected_mime,
        file_size_bytes=len(file_data),
        extraction_status=extraction_status,
        message=(
            "Resume uploaded and text extracted successfully."
            if extraction_status == "success"
            else "Resume uploaded but text extraction failed. You can enter your profile manually."
        ),
    )


@router.get("/", response_model=List[ResumeSummary])
def list_resumes(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """List all active resumes for the authenticated candidate."""
    resumes = (
        db.query(Resume)
        .filter(Resume.user_id == current_user.id, Resume.is_active == True)
        .order_by(Resume.created_at.desc())
        .all()
    )
    return [
        ResumeSummary(
            resume_id=r.id,
            original_filename=r.original_filename,
            mime_type=r.mime_type,
            file_size_bytes=r.file_size_bytes,
            extraction_status=r.extraction_status,
        )
        for r in resumes
    ]


@router.get("/{resume_id}/extracted", response_model=ResumeExtractedResponse)
def get_extracted_text(
    resume_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Return the extracted text from a resume for candidate review.

    Only returns text — never the storage path or raw file.
    The candidate uses this to review what was extracted before confirming.
    """
    resume = db.get(Resume, resume_id)

    if resume is None or resume.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Resume not found.")

    if not resume.is_active:
        raise HTTPException(status_code=404, detail="Resume not found.")

    context_preview = None
    if resume.extracted_text:
        context_preview = sanitize_resume_text_for_context(resume.extracted_text)

    return ResumeExtractedResponse(
        resume_id=resume.id,
        extraction_status=resume.extraction_status,
        extracted_text=resume.extracted_text,
        context_preview=context_preview,
    )


@router.post("/{resume_id}/extract-profile", response_model=ExtractedProfileResponse)
def extract_profile_from_resume(
    resume_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Extract structured profile fields from the resume's text using IBM Granite.

    The extracted fields are SUGGESTIONS for the candidate to review.
    They are NOT automatically applied to the candidate profile.

    Workflow after calling this endpoint:
        1. Review the extracted fields.
        2. PUT /candidate/profile with corrected values.
        3. POST /candidate/profile/confirm to finalise.

    If the LLM service is unavailable (offline / missing credentials),
    status="unavailable" is returned — no error.
    """
    resume = db.get(Resume, resume_id)
    if resume is None or resume.user_id != current_user.id or not resume.is_active:
        raise HTTPException(status_code=404, detail="Resume not found.")

    if resume.extraction_status != "success" or not resume.extracted_text:
        raise HTTPException(
            status_code=422,
            detail="Resume text has not been successfully extracted yet.",
        )

    try:
        from app.services.profile_extraction_service import extract_profile_from_text
        result = extract_profile_from_text(resume.extracted_text)
    except Exception:
        logger.exception("Profile extraction failed for resume_id=%d", resume_id)
        raise HTTPException(
            status_code=500,
            detail="Profile extraction failed. Please enter your profile manually.",
        )

    p = result.profile
    return ExtractedProfileResponse(
        resume_id       = resume_id,
        status          = result.status,
        message         = result.message,
        full_name       = p.full_name,
        email           = p.email,
        phone           = p.phone,
        college         = p.college,
        degree          = p.degree,
        branch          = p.branch,
        graduation_year = p.graduation_year,
        skills          = p.skills,
        projects        = p.projects,
        experience      = p.experience,
        certifications  = p.certifications,
        achievements    = p.achievements,
        suggested_role  = p.suggested_role,
    )


@router.delete("/{resume_id}", status_code=204)
def delete_resume(
    resume_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Soft-delete a resume (sets is_active=False, records deleted_at).

    The physical file is NOT deleted immediately — this allows recovery
    and audit. A background job could purge old files later.
    Authorization: only the owning user can delete their own resume.
    """
    resume = db.get(Resume, resume_id)
    if resume is None or resume.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Resume not found.")

    if not resume.is_active:
        raise HTTPException(status_code=404, detail="Resume not found.")

    resume.is_active   = False
    resume.deleted_at  = datetime.now(tz=timezone.utc)
    db.commit()

    logger.info("Resume soft-deleted: user_id=%d resume_id=%d", current_user.id, resume_id)
    # 204 No Content — no response body
