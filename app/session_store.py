"""
SQLite-backed session store.

Persists InterviewSession objects as JSON rows so sessions survive
across API requests (and optionally across server restarts).

The store is a thin wrapper around a SQLite table:

    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        data       TEXT NOT NULL,          -- JSON-serialised session
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

Agent objects (_interviewer, _evaluator) are NOT serialised — they are
ephemeral and are re-attached whenever a session is loaded from the DB.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from app.models import InterviewSession, InterviewTurn

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/sessions.db")
_lock = Lock()


# ── Bootstrap ────────────────────────────────────────────────────────

def _ensure_db(path: Path = _DB_PATH) -> None:
    """Create the database file and sessions table if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def _db(path: Path = _DB_PATH):
    """Thread-safe SQLite connection context manager."""
    with _lock:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


# ── Serialisation ────────────────────────────────────────────────────

def _session_to_dict(session: InterviewSession) -> dict:
    """
    Convert an InterviewSession to a plain dict (JSON-safe).

    Backward-compatible: the 'plan' key is only present when the session
    has a multi-round plan attached (via session._plan).  Old sessions
    without a plan load and deserialise correctly.
    """
    d = {
        "session_id": session.session_id,
        "candidate_name": session.candidate_name,
        "role": session.role,
        "experience_level": session.experience_level,
        "interview_type": session.interview_type,
        "turns": [
            {
                "question": t.question,
                "answer": t.answer,
                "evaluation": t.evaluation,
            }
            for t in session.turns
        ],
    }
    # Persist multi-round plan if present
    if hasattr(session, "_plan") and session._plan is not None:
        d["plan"] = session._plan.to_dict()
    return d


def _dict_to_session(data: dict) -> InterviewSession:
    """
    Reconstruct an InterviewSession from a plain dict.

    Old sessions without 'plan' continue to load correctly.
    Multi-round sessions have _plan re-attached.
    """
    session = InterviewSession(
        session_id=data["session_id"],
        candidate_name=data["candidate_name"],
        role=data["role"],
        experience_level=data["experience_level"],
        interview_type=data["interview_type"],
        turns=[
            InterviewTurn(
                question=t["question"],
                answer=t["answer"],
                evaluation=t.get("evaluation", {}),
            )
            for t in data.get("turns", [])
        ],
    )
    # Re-attach multi-round plan if present in serialised data
    if "plan" in data and data["plan"]:
        try:
            from interview.plan import InterviewPlan
            session._plan = InterviewPlan.from_dict(data["plan"])
        except Exception as exc:
            logger.warning(
                "Could not deserialise InterviewPlan: %s. "
                "Session will operate without a plan.",
                exc,
            )
    return session


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ── Public API ───────────────────────────────────────────────────────

class SessionStore:
    """Thread-safe SQLite-backed session store."""

    def __init__(self, db_path: Path = _DB_PATH):
        self._path = db_path
        _ensure_db(self._path)

    def save(self, session: InterviewSession) -> None:
        """Insert or replace a session."""
        now = _now()
        data_json = json.dumps(_session_to_dict(session))

        with _db(self._path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, data, created_at, updated_at)
                VALUES (:sid, :data, :now, :now)
                ON CONFLICT(session_id) DO UPDATE SET
                    data       = excluded.data,
                    updated_at = excluded.updated_at
                """,
                {"sid": session.session_id, "data": data_json, "now": now},
            )

    def load(self, session_id: str) -> Optional[InterviewSession]:
        """Load a session by ID, or return None if not found."""
        with _db(self._path) as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        try:
            data = json.loads(row["data"])
            return _dict_to_session(data)
        except Exception as exc:
            logger.error(
                "Failed to deserialise session %s: %s",
                session_id,
                exc,
            )
            return None

    def delete(self, session_id: str) -> None:
        """Remove a session."""
        with _db(self._path) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )

    def exists(self, session_id: str) -> bool:
        """Return True if the session ID is present."""
        with _db(self._path) as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None


# ── Process-level singleton ─────────────────────────────────────────

_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
