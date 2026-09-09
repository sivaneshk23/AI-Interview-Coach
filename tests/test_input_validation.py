"""
Tests: Input validation — empty answers, field length limits, etc.

These tests do not require a live IBM API connection.
"""

import pytest
from pydantic import ValidationError

from app.schemas import AnswerRequest, InterviewRequest


class TestAnswerRequestValidation:

    def test_valid_answer_request_passes(self):
        req = AnswerRequest(
            question="What is Python?",
            answer="Python is a high-level programming language.",
            role="Software Engineer",
            experience_level="Junior",
        )
        assert req.answer == "Python is a high-level programming language."

    def test_empty_answer_fails_validation(self):
        with pytest.raises(ValidationError):
            AnswerRequest(
                question="What is Python?",
                answer="",
                role="Software Engineer",
                experience_level="Junior",
            )

    def test_answer_exceeding_max_length_fails(self):
        with pytest.raises(ValidationError):
            AnswerRequest(
                question="What is Python?",
                answer="x" * 5001,
                role="Software Engineer",
                experience_level="Junior",
            )

    def test_question_too_short_fails(self):
        with pytest.raises(ValidationError):
            AnswerRequest(
                question="Hi",  # less than min_length=5
                answer="Some answer here",
                role="Software Engineer",
                experience_level="Junior",
            )

    def test_role_too_short_fails(self):
        with pytest.raises(ValidationError):
            AnswerRequest(
                question="What is Python?",
                answer="Python is a language.",
                role="X",  # less than min_length=2
                experience_level="Junior",
            )


class TestInterviewRequestValidation:

    def test_valid_interview_request_passes(self):
        req = InterviewRequest(
            role="Data Scientist",
            experience_level="Senior",
            interview_type="technical",
        )
        assert req.role == "Data Scientist"

    def test_role_too_short_fails(self):
        with pytest.raises(ValidationError):
            InterviewRequest(
                role="X",  # too short
                experience_level="Senior",
                interview_type="technical",
            )

    def test_role_exceeding_max_length_fails(self):
        with pytest.raises(ValidationError):
            InterviewRequest(
                role="A" * 201,
                experience_level="Senior",
                interview_type="technical",
            )

    def test_candidate_context_optional(self):
        req = InterviewRequest(
            role="Cybersecurity Analyst",
            experience_level="Entry-level",
            interview_type="technical",
        )
        assert req.candidate_context is None

    def test_candidate_context_accepted_when_provided(self):
        req = InterviewRequest(
            role="Cybersecurity Analyst",
            experience_level="Entry-level",
            interview_type="technical",
            candidate_context="3 years in network security.",
        )
        assert req.candidate_context == "3 years in network security."


class TestInterviewEngineEmptyAnswerValidation:
    """Engine-level empty-answer guard (independent of Pydantic)."""

    def test_submit_empty_answer_raises_value_error(self):
        from unittest.mock import patch
        from app.interview_engine import InterviewEngine

        with patch("app.interview_engine._get_rag_engine", return_value=None):
            engine = InterviewEngine()

        session = engine.create_session(
            candidate_name="Candidate",
            role="HR Specialist",
            experience_level="Mid-level",
            interview_type="HR",
        )

        with pytest.raises(ValueError, match="empty"):
            engine.submit_answer(
                session=session,
                question="Tell me about yourself.",
                answer="",
            )

    def test_submit_whitespace_only_answer_raises_value_error(self):
        from unittest.mock import patch
        from app.interview_engine import InterviewEngine

        with patch("app.interview_engine._get_rag_engine", return_value=None):
            engine = InterviewEngine()

        session = engine.create_session(
            candidate_name="Candidate",
            role="HR Specialist",
            experience_level="Mid-level",
            interview_type="HR",
        )

        with pytest.raises(ValueError, match="empty"):
            engine.submit_answer(
                session=session,
                question="Tell me about yourself.",
                answer="    \n\t  ",
            )
