"""
Database engine and session factory.

Uses SQLAlchemy 2.0 with SQLite for local development.
Switch to PostgreSQL by setting DATABASE_URL in the environment:

    DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname

The rest of the application is database-agnostic — it works with either
backend without any business-logic changes.
"""

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ── Database URL ──────────────────────────────────────────────────────
# Default: SQLite at data/app.db (development/local).
# Override: set DATABASE_URL env var for PostgreSQL in production.

_DEFAULT_SQLITE = "sqlite:///data/app.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", _DEFAULT_SQLITE)

# Ensure the data directory exists for SQLite.
if DATABASE_URL.startswith("sqlite:///"):
    _db_file = Path(DATABASE_URL.replace("sqlite:///", ""))
    _db_file.parent.mkdir(parents=True, exist_ok=True)

# ── Engine ────────────────────────────────────────────────────────────
# connect_args only applies to SQLite; ignored for PostgreSQL.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,  # set True only when debugging SQL
)

# Enable WAL mode for SQLite (better concurrent read performance).
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ── Session factory ───────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── Declarative base ──────────────────────────────────────────────────
class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Using DeclarativeBase (SQLAlchemy 2.0 style) means all mapped classes
    share the same metadata registry — important for alembic migrations.
    """
    pass


# ── FastAPI dependency ────────────────────────────────────────────────
def get_db():
    """
    Yield a SQLAlchemy session.

    Use as a FastAPI dependency:

        @router.get("/something")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Init DB ───────────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create all tables that do not yet exist.

    Safe to call on every startup — CREATE TABLE IF NOT EXISTS semantics.
    For production migrations, use Alembic instead of this function.
    """
    # Import all models so their metadata is registered before create_all.
    import app.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised (backend: %s)", DATABASE_URL.split(":")[0])
