"""
Tests: API/service layer → InterviewEngine integration.

These tests use mocks so no live IBM API call is needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.interview_engine import InterviewEngine
from app.models import InterviewSession


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_session(engine: InterviewEngine) -> InterviewSession:
    return engine.create_session(
        candidate_name="Test Candidate",
        role="Cloud Engineer",
        experience_level="Mid-level",
        interview_type="technical",
    )


# ------------------------------------------------------------------ #
# InterviewEngine unit tests (mocked LLM + mocked RAG)
# ------------------------------------------------------------------ #

class TestInterviewEngineIntegration:

    def setup_method(self):
        """Patch IBMWatsonxService.generate and RAGEngine for every test."""
        self._llm_patcher = patch(
            "app.services.llm_service.IBMWatsonxService.generate",
            return_value="What is Kubernetes?",
        )
        self._rag_patcher = patch(
            "app.interview_engine._get_rag_engine",
            return_value=None,  # disable RAG for isolation
        )
        self._llm_patcher.start()
        self._rag_patcher.start()

    def teardown_method(self):
        self._llm_patcher.stop()
        self._rag_patcher.stop()

    def test_create_session_returns_session_with_agents(self):
        engine = InterviewEngine()
        session = _make_session(engine)

        assert isinstance(session, InterviewSession)
        assert session.role == "Cloud Engineer"
        assert session.experience_level == "Mid-level"
        assert session.interview_type == "technical"
        assert session.question_count == 0
        assert hasattr(session, "_interviewer")
        assert hasattr(session, "_evaluator")

    def test_get_next_question_returns_string(self):
        engine = InterviewEngine()
        session = _make_session(engine)
        question = engine.get_next_question(session)

        assert isinstance(question, str)
        assert len(question.strip()) > 0

    def test_submit_answer_stores_turn(self):
        engine = InterviewEngine()
        session = _make_session(engine)

        # patch evaluator to return a fixed dict
        mock_eval = {
            "overall_score": 7,
            "technical_score": 7,
            "relevance_score": 7,
            "clarity_score": 7,
            "communication_score": 7,
            "completeness_score": 7,
            "strengths": ["Good explanation"],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": "Well answered.",
        }
        session._evaluator.evaluate = MagicMock(return_value=mock_eval)

        evaluation = engine.submit_answer(
            session=session,
            question="What is Kubernetes?",
            answer="Kubernetes is a container orchestration platform.",
        )

        assert isinstance(evaluation, dict)
        assert session.question_count == 1
        assert session.questions() == ["What is Kubernetes?"]
        assert session.answers() == [
            "Kubernetes is a container orchestration platform."
        ]

    def test_submit_empty_answer_raises_value_error(self):
        engine = InterviewEngine()
        session = _make_session(engine)

        with pytest.raises(ValueError, match="empty"):
            engine.submit_answer(
                session=session,
                question="What is Kubernetes?",
                answer="   ",
            )

    def test_get_next_question_before_create_session_raises(self):
        """Calling get_next_question on a plain InterviewSession raises."""
        engine = InterviewEngine()
        bare_session = InterviewSession(
            session_id="x",
            candidate_name="x",
            role="x",
            experience_level="x",
            interview_type="x",
        )
        with pytest.raises(RuntimeError):
            engine.get_next_question(bare_session)

    def test_interview_service_uses_engine(self):
        """InterviewService.generate_question delegates to InterviewEngine."""
        from app.services.interview_service import InterviewService

        svc = InterviewService()
        result = svc.generate_question(
            role="DevOps Engineer",
            experience_level="Senior",
            interview_type="technical",
        )

        assert "question" in result
        assert "category" in result
        assert "difficulty" in result
        assert isinstance(result["question"], str)

    def test_evaluation_service_uses_engine(self):
        """EvaluationService.evaluate delegates to InterviewEngine."""
        from app.services.evaluation_service import EvaluationService

        # Patch the evaluator agent's evaluate method to return a known dict
        fixed_eval = {
            "overall_score": 8,
            "technical_score": 8,
            "relevance_score": 8,
            "clarity_score": 8,
            "communication_score": 8,
            "completeness_score": 8,
            "strengths": ["Clear"],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": "Good.",
        }
        with patch(
            "agents.evaluator_agent.EvaluatorAgent.evaluate",
            return_value=fixed_eval,
        ):
            svc = EvaluationService()
            result = svc.evaluate(
                question="What is CI/CD?",
                answer="CI/CD is a practice for automating software delivery.",
                role="DevOps Engineer",
                experience_level="Senior",
            )

        assert isinstance(result, dict)
        assert "overall_score" in result
