"""
Candidate profile routes.

GET    /candidate/profile      — retrieve authenticated candidate's profile
PUT    /candidate/profile      — create or update profile
POST   /candidate/profile/confirm — mark profile as confirmed/ready for interview

The job_role field is always a free-form string.
No enum, no hardcoded role list, no role-specific conditional code.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.auth.dependencies import require_auth
from app.db.base import get_db
from app.db.models import CandidateProfile, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/candidate", tags=["candidate"])


# ── Schemas ───────────────────────────────────────────────────────────

class ProfileUpdateRequest(BaseModel):
    """
    Request body for creating or updating a candidate profile.

    All fields are optional so partial updates are supported
    (e.g. after resume extraction, only extracted fields are populated).
    """

    full_name: Optional[str] = Field(None, max_length=200)
    phone: Optional[str] = Field(None, max_length=30)
    college: Optional[str] = Field(None, max_length=300)
    degree: Optional[str] = Field(None, max_length=200)
    branch: Optional[str] = Field(None, max_length=200)
    graduation_year: Optional[int] = Field(None, ge=1980, le=2040)

    # Free-form — no validation against a role list.
    job_role: Optional[str] = Field(
        None,
        max_length=200,
        description="Target job role — any legitimate professional role.",
    )

    skills: Optional[str] = Field(None, max_length=3000)
    projects: Optional[str] = Field(None, max_length=5000)
    experience: Optional[str] = Field(None, max_length=5000)
    interview_context: Optional[str] = Field(
        None,
        max_length=2000,
        description=(
            "Passed as candidate_context to InterviewEngine. "
            "Summary of background and skills."
        ),
    )

    @field_validator("phone")
    @classmethod
    def phone_digits_only(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Allow digits, spaces, +, -, (, )
        import re
        cleaned = re.sub(r"[^\d\s+\-()]", "", v)
        return cleaned.strip() or None


class ProfileResponse(BaseModel):
    user_id: int
    full_name: Optional[str]
    phone: Optional[str]
    college: Optional[str]
    degree: Optional[str]
    branch: Optional[str]
    graduation_year: Optional[int]
    job_role: Optional[str]
    skills: Optional[str]
    projects: Optional[str]
    experience: Optional[str]
    interview_context: Optional[str]
    is_confirmed: bool

    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/profile", response_model=ProfileResponse)
def get_profile(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Return the authenticated candidate's profile."""
    profile = (
        db.query(CandidateProfile)
        .filter(CandidateProfile.user_id == current_user.id)
        .first()
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found.")

    return ProfileResponse(
        user_id=current_user.id,
        full_name=profile.full_name,
        phone=profile.phone,
        college=profile.college,
        degree=profile.degree,
        branch=profile.branch,
        graduation_year=profile.graduation_year,
        job_role=profile.job_role,
        skills=profile.skills,
        projects=profile.projects,
        experience=profile.experience,
        interview_context=profile.interview_context,
        is_confirmed=profile.is_confirmed,
    )


@router.put("/profile", response_model=ProfileResponse)
def upsert_profile(
    request: ProfileUpdateRequest,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Create or update the candidate's professional profile.

    Safe for repeated calls — idempotent upsert.
    Updating any field resets is_confirmed to False so the candidate
    must re-confirm before starting an interview.
    """
    profile = (
        db.query(CandidateProfile)
        .filter(CandidateProfile.user_id == current_user.id)
        .first()
    )

    if profile is None:
        profile = CandidateProfile(user_id=current_user.id)
        db.add(profile)

    # Apply updates — only provided (non-None) fields overwrite
    update_data = request.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(profile, field, value)

    # Any edit resets confirmation
    profile.is_confirmed = False

    db.commit()
    db.refresh(profile)

    logger.info("Profile updated for user_id=%d", current_user.id)

    return ProfileResponse(
        user_id=current_user.id,
        full_name=profile.full_name,
        phone=profile.phone,
        college=profile.college,
        degree=profile.degree,
        branch=profile.branch,
        graduation_year=profile.graduation_year,
        job_role=profile.job_role,
        skills=profile.skills,
        projects=profile.projects,
        experience=profile.experience,
        interview_context=profile.interview_context,
        is_confirmed=profile.is_confirmed,
    )


@router.post("/profile/confirm", response_model=ProfileResponse)
def confirm_profile(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Mark the candidate's profile as confirmed.

    Called after the candidate reviews (and optionally edits) their
    extracted profile. Sets is_confirmed = True.
    """
    profile = (
        db.query(CandidateProfile)
        .filter(CandidateProfile.user_id == current_user.id)
        .first()
    )
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Profile not found. Create a profile first.",
        )

    profile.is_confirmed = True
    db.commit()
    db.refresh(profile)

    logger.info("Profile confirmed for user_id=%d", current_user.id)

    return ProfileResponse(
        user_id=current_user.id,
        full_name=profile.full_name,
        phone=profile.phone,
        college=profile.college,
        degree=profile.degree,
        branch=profile.branch,
        graduation_year=profile.graduation_year,
        job_role=profile.job_role,
        skills=profile.skills,
        projects=profile.projects,
        experience=profile.experience,
        interview_context=profile.interview_context,
        is_confirmed=profile.is_confirmed,
    )
