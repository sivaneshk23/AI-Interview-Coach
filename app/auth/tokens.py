"""
JWT token utilities.

Uses PyJWT (already installed as 'jwt') with HS256 signing.

Secret key is read from the JWT_SECRET_KEY environment variable.
If not set, a random secret is generated at startup (sessions won't
survive server restarts in that mode — acceptable for development,
not for production). Operators MUST set JWT_SECRET_KEY in production.

Token structure:
    sub   — user ID (integer, as string)
    email — user email
    exp   — expiry timestamp
    iat   — issued-at timestamp
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "480")  # 8 hours default
)

def _get_secret_key() -> str:
    """
    Return the JWT signing secret.

    Reads JWT_SECRET_KEY from the environment.
    Falls back to a per-process random secret if not set (dev mode only).
    Logs a warning so operators know to set it.
    """
    key = os.getenv("JWT_SECRET_KEY", "").strip()
    if key:
        return key

    # Development fallback: random secret (not persisted across restarts)
    global _dev_secret
    if not _dev_secret:
        _dev_secret = secrets.token_hex(32)
        logger.warning(
            "JWT_SECRET_KEY is not set. Using a random per-process secret. "
            "Sessions will not survive server restarts. "
            "Set JWT_SECRET_KEY in production."
        )
    return _dev_secret


_dev_secret: str = ""


# ── Token creation ────────────────────────────────────────────────────

def create_access_token(user_id: int, email: str) -> str:
    """Create a signed JWT access token for the given user."""
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": expire,
    }

    return jwt.encode(payload, _get_secret_key(), algorithm=_ALGORITHM)


# ── Token verification ────────────────────────────────────────────────

class TokenData:
    """Parsed token payload."""

    def __init__(self, user_id: int, email: str):
        self.user_id = user_id
        self.email = email


def verify_access_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode an access token.

    Returns TokenData on success, None on any failure.
    Never raises — callers check for None.
    """
    try:
        payload = jwt.decode(
            token,
            _get_secret_key(),
            algorithms=[_ALGORITHM],
        )
        user_id_str: str = payload.get("sub", "")
        email: str = payload.get("email", "")

        if not user_id_str or not email:
            return None

        return TokenData(user_id=int(user_id_str), email=email)

    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError):
        return None
