"""
SQLAlchemy ORM models for the production foundation.

Entity map:
    User                 — account credentials + auth
    CandidateProfile     — professional profile (1:1 with User)
    Resume               — uploaded resume files (many:1 with User)
    InterviewSessionLink — links existing UUID-based sessions to a User

Design principles:
  - Primary keys: Integer (surrogate) for new entities; UUID text for sessions
    (backward-compatible with existing session_store.py)
  - Timestamps: created_at / updated_at on every table
  - Soft-deletion: is_active flag on User; deleted_at on Resume
  - Future entities (InterviewRound, InterviewQuestion, etc.) can reference
    InterviewSessionLink.session_id (the existing UUID) or the new integer PKs
  - PostgreSQL-compatible: no SQLite-specific types used except where abstracted
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


# ── Helpers ───────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── User ──────────────────────────────────────────────────────────────

class User(Base):
    """
    Authentication credential record.

    One User → one CandidateProfile (created lazily after registration).
    One User → many Resumes.
    One User → many InterviewSessionLinks.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        unique=True,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    profile: Mapped[Optional["CandidateProfile"]] = relationship(
        "CandidateProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    resumes: Mapped[List["Resume"]] = relationship(
        "Resume",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="Resume.created_at.desc()",
    )
    session_links: Mapped[List["InterviewSessionLink"]] = relationship(
        "InterviewSessionLink",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


# ── CandidateProfile ──────────────────────────────────────────────────

class CandidateProfile(Base):
    """
    Professional candidate profile.

    The job_role field is a free-form string — never an enum.
    Candidates can enter any legitimate job role.

    Skills, projects, and experience are stored as plain text so they
    are simple to display, edit, and pass to the interview context
    without structured parsing complexity.
    """

    __tablename__ = "candidate_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Personal info
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    # Academic info
    college: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    degree: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    branch: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    graduation_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Professional info — all free-form text
    job_role: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
        comment="Free-form target job role. Never an enum.",
    )
    skills: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    projects: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    experience: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Interview context (used as candidate_context in interview prompts)
    interview_context: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Passed as candidate_context to InterviewEngine.",
    )

    # Profile completion flag
    is_confirmed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="True once candidate has reviewed and confirmed the profile.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="profile")

    def __repr__(self) -> str:
        return f"<CandidateProfile user_id={self.user_id} role={self.job_role!r}>"


# ── Resume ────────────────────────────────────────────────────────────

class Resume(Base):
    """
    Uploaded resume record.

    The actual file is stored on the filesystem at a safe, non-public
    path. This record stores the metadata and extracted text only.

    The storage_path is the server-side path (never returned to clients).
    Clients receive only the resume ID.
    """

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Safe filename (UUID-based, not the original upload name)
    safe_filename: Mapped[str] = mapped_column(String(200), nullable=False)

    # Original upload name for display only — never used as a path
    original_filename: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # MIME type as detected, not as claimed by the client
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Server-side storage path — never exposed to clients
    storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Extracted text from the resume (plain text, not trusted as instructions)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Extraction status
    extraction_status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False,
        comment="pending | success | failed",
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="resumes")

    def __repr__(self) -> str:
        return f"<Resume id={self.id} user_id={self.user_id} status={self.extraction_status!r}>"


# ── InterviewRoundRecord ──────────────────────────────────────────────

class InterviewRoundRecord(Base):
    """
    Persists per-round results for a multi-round interview session.

    Each row corresponds to one completed InterviewRound.
    Links to InterviewSessionLink via session_id.

    This table is an audit/reporting table — the authoritative round state
    lives in the session JSON (via session_store.py).  This table enables
    SQL queries over round performance without deserialising JSON.
    """

    __tablename__ = "interview_round_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Link to the session (UUID from session_store.py)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Round identity
    round_id:   Mapped[str] = mapped_column(String(100), nullable=False)
    round_type: Mapped[str] = mapped_column(String(40),  nullable=False)
    round_order:Mapped[int] = mapped_column(Integer,     nullable=False)
    title:      Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Round lifecycle
    state:      Mapped[str] = mapped_column(String(20), nullable=False, default="not_started")

    # Aggregate evaluation (populated on round completion)
    score:           Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    turn_count:      Mapped[int]             = mapped_column(Integer, default=0, nullable=False)
    correct_count:   Mapped[Optional[int]]   = mapped_column(Integer, nullable=True)
    total_questions: Mapped[Optional[int]]   = mapped_column(Integer, nullable=True)
    feedback:        Mapped[Optional[str]]   = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewRoundRecord session={self.session_id!r} "
            f"type={self.round_type!r} state={self.state!r}>"
        )


# ── InterviewSessionLink ──────────────────────────────────────────────

class InterviewSessionLink(Base):
    """
    Links the existing UUID-based InterviewSession records to a User.

    The existing session_store.py uses its own SQLite table with TEXT
    primary keys (UUIDs). This table adds ownership and timestamps
    without touching the existing sessions table.

    Future entities (InterviewRound, etc.) can reference this table's
    session_id (the UUID) to join with the legacy sessions table.
    """

    __tablename__ = "interview_session_links"
    __table_args__ = (
        UniqueConstraint("user_id", "session_id", name="uq_user_session"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # UUID from the existing InterviewSession / sessions table
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Snapshot of the interview parameters at creation time
    role: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    experience_level: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    interview_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Aggregate score (populated when interview ends)
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="session_links")

    def __repr__(self) -> str:
        return (
            f"<InterviewSessionLink user_id={self.user_id} "
            f"session_id={self.session_id!r}>"
        )
