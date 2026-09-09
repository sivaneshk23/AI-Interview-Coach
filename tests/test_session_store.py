"""
Tests: SQLite session store.

No IBM API calls — purely local I/O.
"""

import tempfile
from pathlib import Path

import pytest

from app.models import InterviewSession, InterviewTurn
from app.session_store import SessionStore


@pytest.fixture()
def store(tmp_path):
    return SessionStore(db_path=tmp_path / "test_sessions.db")


def _make_session(session_id: str = "abc-123") -> InterviewSession:
    session = InterviewSession(
        session_id=session_id,
        candidate_name="Alice",
        role="Data Engineer",
        experience_level="Mid-level",
        interview_type="technical",
    )
    return session


class TestSessionStoreSaveLoad:

    def test_save_and_load_empty_session(self, store):
        session = _make_session()
        store.save(session)

        loaded = store.load(session.session_id)
        assert loaded is not None
        assert loaded.session_id == "abc-123"
        assert loaded.candidate_name == "Alice"
        assert loaded.role == "Data Engineer"
        assert loaded.experience_level == "Mid-level"
        assert loaded.interview_type == "technical"
        assert loaded.turns == []

    def test_save_and_load_session_with_turns(self, store):
        session = _make_session("xyz-789")
        session.add_turn(
            question="What is Apache Spark?",
            answer="A distributed computing framework.",
            evaluation={"overall_score": 8, "evaluation": "Good."},
        )
        store.save(session)

        loaded = store.load("xyz-789")
        assert loaded is not None
        assert len(loaded.turns) == 1
        turn = loaded.turns[0]
        assert turn.question == "What is Apache Spark?"
        assert turn.answer == "A distributed computing framework."
        assert turn.evaluation["overall_score"] == 8

    def test_load_missing_session_returns_none(self, store):
        result = store.load("does-not-exist")
        assert result is None

    def test_exists_true(self, store):
        session = _make_session("ex-1")
        store.save(session)
        assert store.exists("ex-1") is True

    def test_exists_false(self, store):
        assert store.exists("no-such-id") is False

    def test_delete_removes_session(self, store):
        session = _make_session("del-1")
        store.save(session)
        assert store.exists("del-1") is True
        store.delete("del-1")
        assert store.exists("del-1") is False
        assert store.load("del-1") is None

    def test_save_overwrites_existing_session(self, store):
        session = _make_session("upd-1")
        store.save(session)

        # Add a turn and save again
        session.add_turn(
            question="Q1",
            answer="A1",
            evaluation={"overall_score": 5},
        )
        store.save(session)

        loaded = store.load("upd-1")
        assert len(loaded.turns) == 1

    def test_multiple_sessions_are_independent(self, store):
        s1 = _make_session("s1")
        s2 = _make_session("s2")
        s2.candidate_name = "Bob"

        store.save(s1)
        store.save(s2)

        l1 = store.load("s1")
        l2 = store.load("s2")
        assert l1.candidate_name == "Alice"
        assert l2.candidate_name == "Bob"
