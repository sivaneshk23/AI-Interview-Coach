"""
Tests for HR round executor — resume-aware, adaptive, no repetition.

These are offline tests — no IBM network calls are made.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from interview.executors import HRRoundExecutor, build_executor
from interview.plan import InterviewRound, RoundTurn
from interview.round_types import RoundState, RoundType


# ── Fixtures ───────────────────────────────────────────────────────────

def _mock_interviewer(return_value: str = "Tell me about yourself."):
    m = MagicMock()
    m.generate_question.return_value = return_value
    return m


def _mock_evaluator(score: float = 7.0):
    m = MagicMock()
    m.evaluate.return_value = {
        "overall_score": score,
        "technical_score": 5,
        "relevance_score": 7,
        "clarity_score": 7,
        "communication_score": 7,
        "completeness_score": 6,
        "strengths": ["clear answer"],
        "weaknesses": [],
        "improvement_suggestions": [],
        "evaluation": "Good answer.",
    }
    return m


def _make_hr_round(max_turns: int = 6) -> InterviewRound:
    r = InterviewRound(
        round_id   = str(uuid.uuid4()),
        round_type = RoundType.HR,
        max_turns  = max_turns,
        difficulty = "easy",
    )
    r.state = RoundState.IN_PROGRESS
    return r


# ── Basic structure ────────────────────────────────────────────────────

def test_hr_executor_round_type():
    assert HRRoundExecutor.round_type == RoundType.HR


def test_build_executor_hr_requires_agents():
    """build_executor(HR) raises if agents are missing."""
    with pytest.raises(ValueError, match="requires both"):
        build_executor(RoundType.HR)


def test_build_executor_hr_with_agents():
    executor = build_executor(
        RoundType.HR,
        interviewer_agent=_mock_interviewer(),
        evaluator_agent=_mock_evaluator(),
    )
    assert isinstance(executor, HRRoundExecutor)


# ── generate_question ─────────────────────────────────────────────────

def test_hr_generate_question_returns_round_turn():
    interviewer = _mock_interviewer("Tell me about yourself.")
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    turn = executor.generate_question(
        round_            = round_,
        role              = "Product Manager",
        candidate_context = "Priya, MBA, 2 years product experience.",
    )

    assert isinstance(turn, RoundTurn)
    assert turn.question == "Tell me about yourself."
    assert turn.time_started_at is not None


def test_hr_generate_question_passes_full_history():
    """InterviewerAgent.generate_question must be called with all_previous_questions."""
    interviewer = _mock_interviewer("What is your greatest strength?")
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    # Add two prior turns
    round_.turns.append(RoundTurn(question="Tell me about yourself.", answer="I am Priya...", score=7.0, evaluation={"overall_score": 7}))
    round_.turns.append(RoundTurn(question="Why do you want this role?", answer="I love product...", score=8.0, evaluation={"overall_score": 8}))

    executor.generate_question(round_=round_, role="Product Manager", candidate_context="Resume context here.")

    call_kwargs = interviewer.generate_question.call_args[1]
    assert "all_previous_questions" in call_kwargs
    prev_qs = call_kwargs["all_previous_questions"]
    assert "Tell me about yourself." in prev_qs
    assert "Why do you want this role?" in prev_qs


def test_hr_generate_question_includes_candidate_context():
    """candidate_context (from resume) must be passed to the interviewer agent."""
    interviewer = _mock_interviewer("Describe a project you led.")
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    context = "Built a Python school management system with MySQL"
    executor.generate_question(round_=round_, role="SDE", candidate_context=context)

    call_kwargs = interviewer.generate_question.call_args[1]
    assert call_kwargs.get("candidate_context") == context


def test_hr_generate_question_passes_evaluations_for_adaptive():
    """Previous evaluations must be passed for adaptive difficulty."""
    interviewer = _mock_interviewer("Follow-up question.")
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    eval_ = {"overall_score": 9, "strengths": ["great"], "weaknesses": [], "improvement_suggestions": [], "evaluation": "Excellent."}
    round_.turns.append(RoundTurn(question="Q1?", answer="A1", score=9.0, evaluation=eval_))

    executor.generate_question(round_=round_, role="PM")

    call_kwargs = interviewer.generate_question.call_args[1]
    assert "all_previous_evaluations" in call_kwargs
    assert len(call_kwargs["all_previous_evaluations"]) == 1


# ── evaluate_answer ───────────────────────────────────────────────────

def test_hr_evaluate_answer_sets_score():
    interviewer = _mock_interviewer()
    evaluator   = _mock_evaluator(score=8.0)
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()
    turn        = RoundTurn(question="Tell me about yourself.", answer="I am Priya...")

    evaluated = executor.evaluate_answer(round_=round_, turn=turn, role="PM")

    assert evaluated.score == pytest.approx(8.0)
    assert isinstance(evaluated.evaluation, dict)


def test_hr_evaluate_answer_passes_voice_fields():
    """Voice fields on the turn must be forwarded to the evaluator."""
    interviewer = _mock_interviewer()
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    turn = RoundTurn(
        question           = "Tell me about yourself.",
        answer             = "transcript text",
        voice_input_mode   = "voice",
        transcript         = "My spoken answer.",
        voice_duration_sec = 30.0,
    )

    executor.evaluate_answer(round_=round_, turn=turn, role="PM")

    call_kwargs = evaluator.evaluate.call_args[1]
    assert call_kwargs.get("voice_input_mode") == "voice"
    assert call_kwargs.get("voice_duration_sec") == pytest.approx(30.0)
    assert call_kwargs.get("transcript") == "My spoken answer."


# ── build_round_evaluation ─────────────────────────────────────────────

def test_hr_build_round_evaluation_averages_scores():
    """build_round_evaluation should return the average of per-turn scores."""
    interviewer = _mock_interviewer()
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    round_.turns.extend([
        RoundTurn(question="Q1", answer="A1", score=8.0, evaluation={"strengths": ["S1"], "weaknesses": [], "improvement_suggestions": [], "evaluation": "E1"}),
        RoundTurn(question="Q2", answer="A2", score=6.0, evaluation={"strengths": [], "weaknesses": ["W1"], "improvement_suggestions": ["I1"], "evaluation": "E2"}),
    ])

    eval_ = executor.build_round_evaluation(round_, "PM")
    assert eval_.score == pytest.approx(7.0, abs=0.1)
    assert "S1" in eval_.strengths
    assert "W1" in eval_.weaknesses
    assert "I1" in eval_.improvement_suggestions


def test_hr_build_round_evaluation_no_turns():
    """build_round_evaluation returns score=0.0 when no turns exist."""
    executor = HRRoundExecutor(_mock_interviewer(), _mock_evaluator())
    round_   = _make_hr_round()
    eval_    = executor.build_round_evaluation(round_, "PM")
    assert eval_.score == 0.0


# ── Resume-aware context test ─────────────────────────────────────────

def test_hr_resume_context_passed_through():
    """
    HR executor must pass resume/profile context to the interviewer agent.

    This simulates the 'resume-aware interview' feature where the confirmed
    candidate profile is used to personalise HR questions.
    """
    interviewer = _mock_interviewer("Tell me about your Python project.")
    evaluator   = _mock_evaluator()
    executor    = HRRoundExecutor(interviewer, evaluator)
    round_      = _make_hr_round()

    resume_context = (
        "Candidate: Priya Sharma\n"
        "Skills: Python, Django, REST APIs\n"
        "Projects: Built a school management system with Python and MySQL\n"
        "College: VIT Chennai, B.Tech CSE, 2025"
    )

    executor.generate_question(
        round_            = round_,
        role              = "Software Engineer",
        candidate_context = resume_context,
    )

    call_kwargs = interviewer.generate_question.call_args[1]
    assert call_kwargs["candidate_context"] == resume_context
