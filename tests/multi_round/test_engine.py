"""
Tests for interview/engine.py — MultiRoundEngine

All IBM Granite / watsonx.ai calls are mocked.
All RAG calls are mocked.

Tests cover:
  - create_session: plan attached, role preserved, session_id assigned
  - start_next_round: first question generated, round starts
  - submit_answer: turn recorded, round auto-completes at max_turns
  - complete_round_manually: manual completion path
  - build_final_report: aggregation and structure
  - _find_round: not-found error
  - role-agnostic behaviour (arbitrary role strings accepted)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from interview.engine import MultiRoundEngine
from interview.plan import InterviewPlan, InterviewRound, RoundEvaluation, RoundTurn
from interview.planner import InterviewPlanner
from interview.round_types import RoundState, RoundType


# ── Helpers ────────────────────────────────────────────────────────────

def _make_minimal_planner(round_type: RoundType = RoundType.TECHNICAL, max_turns: int = 2):
    """
    Return an InterviewPlanner that produces a single-round plan.
    This avoids touching real blueprints for unit tests.
    """
    single_round = [
        {
            "round_type": round_type,
            "order": 1,
            "title": "Test Round",
            "purpose": "Testing",
            "difficulty": "medium",
            "max_turns": max_turns,
            "is_required": True,
        }
    ]
    # "general" key required as fallback in InterviewPlanner.create_plan
    blueprints = {"test_bp": single_round, "general": single_round}
    keywords = {"test_bp": [r".*"]}  # match any role
    return InterviewPlanner(blueprints=blueprints, role_keywords=keywords)


def _stub_turn(question: str = "Test question?") -> RoundTurn:
    return RoundTurn(question=question, answer="", score=None)


def _stub_evaluation() -> dict:
    return {
        "score": 7.5,
        "strengths": ["Good answer"],
        "weaknesses": ["Could improve"],
        "improvement_suggestions": ["Practice more"],
        "feedback": "Decent",
    }


def _make_engine(round_type=RoundType.TECHNICAL, max_turns=2):
    """Return a MultiRoundEngine with mocked agents and RAG."""
    planner = _make_minimal_planner(round_type=round_type, max_turns=max_turns)
    engine  = MultiRoundEngine.__new__(MultiRoundEngine)
    engine._planner = planner
    engine._rag = None  # RAG disabled for unit tests
    return engine


# ── Mock factories ─────────────────────────────────────────────────────

def _mock_interviewer():
    m = MagicMock()
    m.generate_question.return_value = "What is a Python list?"
    return m


def _mock_evaluator():
    m = MagicMock()
    m.evaluate.return_value = {
        "overall_score": 7.5,
        "technical_score": 7,
        "relevance_score": 8,
        "clarity_score": 7,
        "communication_score": 8,
        "completeness_score": 7,
        "strengths": ["Good communication"],
        "weaknesses": ["Could add detail"],
        "improvement_suggestions": ["Practice more"],
        "evaluation": "Decent answer.",
        "feedback": "Decent.",
    }
    return m


# ── create_session ─────────────────────────────────────────────────────

class TestCreateSession:
    def setup_method(self):
        self.engine = _make_engine()

    def test_session_id_assigned(self):
        session = self.engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        assert session.session_id and len(session.session_id) > 0

    def test_role_preserved_verbatim(self):
        role = "Quantum Computing Researcher"
        session = self.engine.create_session(
            candidate_name="Bob",
            role=role,
            experience_level="Senior",
        )
        assert session.role == role

    def test_plan_attached(self):
        session = self.engine.create_session(
            candidate_name="Carol",
            role="Data Scientist",
            experience_level="Junior",
        )
        assert hasattr(session, "_plan")
        assert isinstance(session._plan, InterviewPlan)

    def test_plan_has_rounds(self):
        session = self.engine.create_session(
            candidate_name="Dave",
            role="ML Engineer",
            experience_level="Mid",
        )
        assert session._plan.total_rounds >= 1

    def test_arbitrary_role_does_not_raise(self):
        for role in ["CEO", "Barista", "Nurse", "Teacher", "Chef", "Astronaut"]:
            session = self.engine.create_session(
                candidate_name="Test",
                role=role,
                experience_level="Mid",
            )
            assert session.role == role

    def test_candidate_name_stored(self):
        session = self.engine.create_session(
            candidate_name="Eve",
            role="DevOps Engineer",
            experience_level="Senior",
        )
        assert session.candidate_name == "Eve"

    def test_experience_level_stored(self):
        session = self.engine.create_session(
            candidate_name="Frank",
            role="HR Manager",
            experience_level="Lead",
        )
        assert session.experience_level == "Lead"


# ── start_next_round ──────────────────────────────────────────────────

class TestStartNextRound:
    def _session(self):
        engine = _make_engine(max_turns=2)
        return engine, engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )

    def test_round_transitions_to_in_progress(self):
        engine, session = self._session()
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, turn = engine.start_next_round(session)
        assert round_.state == RoundState.IN_PROGRESS

    def test_returns_turn_with_question(self):
        engine, session = self._session()
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, turn = engine.start_next_round(session)
        assert isinstance(turn, RoundTurn)
        assert len(turn.question) > 0

    def test_raises_if_all_rounds_complete(self):
        engine, session = self._session()
        plan = session._plan
        plan.rounds[0].start()
        plan.rounds[0].complete()
        with pytest.raises(ValueError, match="finished"):
            engine.start_next_round(session)


# ── submit_answer ─────────────────────────────────────────────────────

class TestSubmitAnswer:
    def _started_session(self, max_turns=2):
        engine = _make_engine(max_turns=max_turns)
        session = engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, _ = engine.start_next_round(session)
        return engine, session, round_

    def test_empty_answer_raises(self):
        engine, session, round_ = self._started_session()
        with pytest.raises(ValueError, match="empty"):
            engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="",
            )

    def test_turn_recorded(self):
        engine, session, round_ = self._started_session(max_turns=3)
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            turn, complete = engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="What is OOP?",
                answer="Object-oriented programming.",
            )
        assert len(round_.turns) == 1
        assert turn.question == "What is OOP?"

    def test_round_auto_completes_at_max_turns(self):
        engine, session, round_ = self._started_session(max_turns=1)
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            _, complete = engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="A.",
            )
        assert complete is True
        assert round_.state == RoundState.COMPLETED

    def test_round_not_complete_below_max_turns(self):
        engine, session, round_ = self._started_session(max_turns=3)
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            _, complete = engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="A.",
            )
        assert complete is False
        assert round_.state == RoundState.IN_PROGRESS

    def test_wrong_round_id_raises(self):
        engine, session, round_ = self._started_session()
        with pytest.raises(ValueError, match="not found"):
            engine.submit_answer(
                session=session,
                round_id="nonexistent-round-id",
                question="Q?",
                answer="A.",
            )

    def test_evaluation_score_stored_on_turn(self):
        engine, session, round_ = self._started_session(max_turns=3)
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            turn, _ = engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="A.",
            )
        assert turn.score is not None
        assert 0 <= turn.score <= 10


# ── complete_round_manually ───────────────────────────────────────────

class TestCompleteRoundManually:
    def test_manual_completion(self):
        engine = _make_engine(max_turns=5)
        session = engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, _ = engine.start_next_round(session)
            # Add one turn before manual completion
            engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="A.",
            )
            engine.complete_round_manually(session=session, round_id=round_.round_id)
        assert round_.state == RoundState.COMPLETED

    def test_manual_completion_without_turns(self):
        """Completing with 0 turns should not crash."""
        engine = _make_engine(max_turns=5)
        session = engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, _ = engine.start_next_round(session)
            engine.complete_round_manually(session=session, round_id=round_.round_id)
        assert round_.state == RoundState.COMPLETED


# ── build_final_report ────────────────────────────────────────────────

class TestBuildFinalReport:
    def test_report_keys_present(self):
        engine = _make_engine(max_turns=1)
        session = engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, _ = engine.start_next_round(session)
            engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="A.",
            )
        report = engine.build_final_report(session)
        expected_keys = [
            "session_id", "candidate_name", "role",
            "overall_score", "round_reports", "round_scores",
            "strengths", "weaknesses", "improvement_suggestions",
            "total_rounds", "completed_rounds",
        ]
        for key in expected_keys:
            assert key in report, f"Key '{key}' missing from final report"

    def test_report_role_matches_session(self):
        engine = _make_engine(max_turns=1)
        session = engine.create_session(
            candidate_name="Alice",
            role="Cybersecurity Analyst",
            experience_level="Senior",
        )
        with patch(
            "interview.engine._ensure_plan_agent",
            return_value=(_mock_interviewer(), _mock_evaluator()),
        ):
            round_, _ = engine.start_next_round(session)
            engine.submit_answer(
                session=session,
                round_id=round_.round_id,
                question="Q?",
                answer="A.",
            )
        report = engine.build_final_report(session)
        assert report["role"] == "Cybersecurity Analyst"

    def test_report_without_completed_rounds(self):
        engine = _make_engine(max_turns=5)
        session = engine.create_session(
            candidate_name="Alice",
            role="Product Manager",
            experience_level="Mid",
        )
        report = engine.build_final_report(session)
        assert report["overall_score"] == 0.0
        assert report["round_reports"] == []


# ── _find_round ───────────────────────────────────────────────────────

class TestFindRound:
    def test_finds_existing_round(self):
        engine = _make_engine()
        session = engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        plan  = session._plan
        round_ = plan.rounds[0]
        found = engine._find_round(plan, round_.round_id)
        assert found is round_

    def test_raises_on_missing_round_id(self):
        engine = _make_engine()
        session = engine.create_session(
            candidate_name="Alice",
            role="Software Engineer",
            experience_level="Mid",
        )
        plan = session._plan
        with pytest.raises(ValueError, match="not found"):
            engine._find_round(plan, "does-not-exist")


# ── No plan guard ─────────────────────────────────────────────────────

class TestNoPlanGuard:
    def test_get_plan_raises_without_plan(self):
        from app.models import InterviewSession
        engine = _make_engine()
        session = InterviewSession(
            session_id="s1",
            candidate_name="X",
            role="HR",
            experience_level="Mid",
            interview_type="technical",
        )
        with pytest.raises(RuntimeError, match="InterviewPlan"):
            engine._get_plan(session)
