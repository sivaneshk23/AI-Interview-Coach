"""
Password hashing utilities using bcrypt via passlib.

bcrypt is the industry standard for password hashing:
- adaptive work factor
- built-in salt generation
- resistant to GPU cracking

Never call these functions with plaintext passwords in log statements.
"""

from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plaintext password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return _pwd_context.verify(plain, hashed)
