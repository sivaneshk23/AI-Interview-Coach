"""
Authentication routes: registration, login.

POST /auth/register   — create a new account
POST /auth/login      — exchange credentials for a JWT access token
GET  /auth/me         — return the current authenticated user

No plaintext passwords are returned in any response.
No passwords are written to logs.
"""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.dependencies import require_auth
from app.auth.passwords import hash_password, verify_password
from app.auth.tokens import create_access_token
from app.db.base import get_db
from app.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Minimum password requirements: at least 8 chars, mixed content.
_MIN_PASSWORD_LEN = 8


# ── Request / Response schemas ────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Candidate email address")
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < _MIN_PASSWORD_LEN:
            raise ValueError(
                f"Password must be at least {_MIN_PASSWORD_LEN} characters."
            )
        return v

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.lower().strip()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.lower().strip()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str


class UserResponse(BaseModel):
    user_id: int
    email: str
    is_active: bool
    has_profile: bool

    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new candidate account.

    Returns a JWT access token immediately so the candidate can proceed
    to upload a resume or build their profile without a separate login step.

    Raises 409 if the email is already registered.
    """
    # Check for duplicate email
    existing = db.query(User).filter(User.email == request.email).first()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="An account with this email address already exists.",
        )

    hashed = hash_password(request.password)
    user = User(email=request.email, hashed_password=hashed)

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with this email address already exists.",
        )

    logger.info("New user registered: id=%d", user.id)

    token = create_access_token(user_id=user.id, email=user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
    )


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate and return a JWT access token.

    Uses constant-time comparison via bcrypt.
    Never reveals whether the email or password was wrong separately.
    """
    user = db.query(User).filter(User.email == request.email).first()

    # Always run verify_password even if user is None to prevent timing attacks.
    # Use a pre-computed bcrypt hash of a throwaway string so passlib doesn't
    # reject a malformed dummy hash. The result is always False when user is None.
    _DUMMY_HASH = "$2b$12$TG.XPlALf98YrqKcOVbNb.mKNEbV2DBHB3onjAb8NzUVmMZCIdeh6"
    stored_hash = user.hashed_password if user else _DUMMY_HASH

    password_ok = verify_password(request.password, stored_hash)

    if not user or not password_ok or not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("User logged in: id=%d", user.id)

    token = create_access_token(user_id=user.id, email=user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(require_auth)):
    """Return the currently authenticated user's account info."""
    return UserResponse(
        user_id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        has_profile=current_user.profile is not None,
    )
