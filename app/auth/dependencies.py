"""
FastAPI authentication dependencies.

Usage:

    @router.get("/protected")
    def handler(current_user: User = Depends(require_auth)):
        ...

    @router.get("/optional-auth")
    def handler(current_user: Optional[User] = Depends(optional_auth)):
        ...

Tokens are read from the Authorization: Bearer <token> header.
Never log tokens or passwords.
"""

import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.auth.tokens import verify_access_token
from app.db.base import get_db
from app.db.models import User

logger = logging.getLogger(__name__)


def _extract_bearer(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """Extract the Bearer token from the Authorization header."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def get_current_user(
    token: Optional[str] = Depends(_extract_bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Decode the JWT and return the active User, or None if not authenticated.
    Does not raise — use require_auth for endpoints that must be protected.
    """
    if not token:
        return None

    token_data = verify_access_token(token)
    if not token_data:
        return None

    user = db.get(User, token_data.user_id)
    if user is None or not user.is_active:
        return None

    return user


def require_auth(
    current_user: Optional[User] = Depends(get_current_user),
) -> User:
    """
    Require an authenticated user.
    Raises HTTP 401 if not authenticated or if the account is inactive.
    """
    if current_user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
