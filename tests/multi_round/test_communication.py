"""
Tests for the communication round executor and communication evaluation dimensions.

These are offline tests — no IBM network calls are made.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from interview.executors import CommunicationRoundExecutor, build_executor
from interview.plan import InterviewRound, RoundTurn
from interview.round_types import RoundState, RoundType
from agents.evaluator_agent import EvaluatorAgent, _count_filler_words


# ── Fixtures ───────────────────────────────────────────────────────────

_STANDARD_COMM_EVAL = {
    "overall_score": 7,
    "technical_score": 5,
    "relevance_score": 7,
    "clarity_score": 8,
    "communication_score": 8,
    "completeness_score": 7,
    "language_quality_score": 8,
    "response_structure_score": 7,
    "filler_word_score": 9,
    "strengths": ["clear delivery", "good structure"],
    "weaknesses": ["slightly rushed"],
    "improvement_suggestions": ["slow down slightly"],
    "evaluation": "Good communication overall.",
}


def _mock_interviewer(question: str = "Describe a project you are proud of."):
    m = MagicMock()
    m.generate_question.return_value = question
    return m


def _mock_evaluator(result: dict = None):
    m = MagicMock()
    m.evaluate.return_value = result or _STANDARD_COMM_EVAL
    return m


def _make_comm_round(max_turns: int = 5) -> InterviewRound:
    r = InterviewRound(
        round_id   = str(uuid.uuid4()),
        round_type = RoundType.COMMUNICATION,
        max_turns  = max_turns,
        difficulty = "easy",
    )
    r.state = RoundState.IN_PROGRESS
    return r


# ── Basic structure ────────────────────────────────────────────────────

def test_communication_executor_round_type():
    assert CommunicationRoundExecutor.round_type == RoundType.COMMUNICATION


def test_build_executor_communication_requires_agents():
    with pytest.raises(ValueError, match="requires both"):
        build_executor(RoundType.COMMUNICATION)


def test_build_executor_communication_with_agents():
    executor = build_executor(
        RoundType.COMMUNICATION,
        interviewer_agent=_mock_interviewer(),
        evaluator_agent=_mock_evaluator(),
    )
    assert isinstance(executor, CommunicationRoundExecutor)


# ── generate_question ─────────────────────────────────────────────────

def test_comm_generate_question_returns_turn():
    executor = CommunicationRoundExecutor(_mock_interviewer("Tell me about a presentation."), _mock_evaluator())
    round_   = _make_comm_round()

    turn = executor.generate_question(round_=round_, role="Marketing Manager")

    assert isinstance(turn, RoundTurn)
    assert turn.question == "Tell me about a presentation."
    assert turn.time_started_at is not None


def test_comm_generate_question_passes_full_history():
    """CommunicationRoundExecutor must pass all_previous_questions to the agent."""
    interviewer = _mock_interviewer("Explain a concept to a non-expert.")
    executor    = CommunicationRoundExecutor(interviewer, _mock_evaluator())
    round_      = _make_comm_round()

    round_.turns.extend([
        RoundTurn(question="Q1?", answer="A1", score=7.0, evaluation={"overall_score": 7}),
        RoundTurn(question="Q2?", answer="A2", score=8.0, evaluation={"overall_score": 8}),
    ])

    executor.generate_question(round_=round_, role="Any")

    call_kwargs = interviewer.generate_question.call_args[1]
    assert "all_previous_questions" in call_kwargs
    assert "Q1?" in call_kwargs["all_previous_questions"]
    assert "Q2?" in call_kwargs["all_previous_questions"]


# ── evaluate_answer ───────────────────────────────────────────────────

def test_comm_evaluate_answer_score_from_communication_score():
    """CommunicationRoundExecutor uses communication_score as the primary score."""
    comm_eval = dict(_STANDARD_COMM_EVAL)
    comm_eval["communication_score"] = 9
    comm_eval["overall_score"]       = 6  # overall is lower — primary should be communication

    executor = CommunicationRoundExecutor(_mock_interviewer(), _mock_evaluator(comm_eval))
    round_   = _make_comm_round()
    turn     = RoundTurn(question="Q?", answer="Thoughtful answer.")

    evaluated = executor.evaluate_answer(round_=round_, turn=turn, role="Any")

    assert evaluated.score == pytest.approx(9.0)


def test_comm_evaluate_answer_uses_communication_score_when_zero_overall():
    """When overall_score is low but communication_score is higher, use communication_score."""
    eval_ = {
        "overall_score": 3,
        "technical_score": 2,
        "relevance_score": 3,
        "clarity_score": 7,
        "communication_score": 8,  # higher — this should be used as primary
        "completeness_score": 5,
        "strengths": [], "weaknesses": [],
        "improvement_suggestions": [], "evaluation": "",
    }
    executor = CommunicationRoundExecutor(_mock_interviewer(), _mock_evaluator(eval_))
    round_   = _make_comm_round()
    turn     = RoundTurn(question="Q?", answer="An answer.")

    evaluated = executor.evaluate_answer(round_=round_, turn=turn, role="Any")

    # communication_score=8 > overall_score=3 — the comm round should use comm_score
    assert evaluated.score == pytest.approx(8.0)


def test_comm_evaluate_passes_voice_fields():
    """Voice fields on the turn must be forwarded to the evaluator."""
    evaluator   = _mock_evaluator()
    executor    = CommunicationRoundExecutor(_mock_interviewer(), evaluator)
    round_      = _make_comm_round()
    turn = RoundTurn(
        question           = "Tell me about a project.",
        answer             = "Well, um, I worked on a Python app...",
        voice_input_mode   = "voice",
        transcript         = "I worked on a Python app for data processing.",
        voice_duration_sec = 22.0,
    )

    executor.evaluate_answer(round_=round_, turn=turn, role="Data Analyst")

    call_kwargs = evaluator.evaluate.call_args[1]
    assert call_kwargs.get("voice_input_mode")    == "voice"
    assert call_kwargs.get("voice_duration_sec")  == pytest.approx(22.0)
    assert call_kwargs.get("transcript")          == turn.transcript


# ── build_round_evaluation ─────────────────────────────────────────────

def test_comm_build_round_evaluation_no_turns():
    executor = CommunicationRoundExecutor(_mock_interviewer(), _mock_evaluator())
    round_   = _make_comm_round()
    eval_    = executor.build_round_evaluation(round_, "Any")
    assert eval_.score == 0.0
    assert "No turns evaluated" in eval_.feedback


def test_comm_build_round_evaluation_averages_comm_scores():
    """build_round_evaluation averages per-turn scores (which are communication_score based)."""
    executor = CommunicationRoundExecutor(_mock_interviewer(), _mock_evaluator())
    round_   = _make_comm_round()

    round_.turns.extend([
        RoundTurn(question="Q1", answer="A1", score=8.0, evaluation={"strengths": ["clear"], "weaknesses": [], "improvement_suggestions": [], "evaluation": "E1"}),
        RoundTurn(question="Q2", answer="A2", score=6.0, evaluation={"strengths": [], "weaknesses": ["vague"], "improvement_suggestions": ["expand"], "evaluation": "E2"}),
    ])

    eval_ = executor.build_round_evaluation(round_, "Any")
    assert eval_.score == pytest.approx(7.0, abs=0.1)
    assert "clear" in eval_.strengths
    assert "vague" in eval_.weaknesses
    assert "expand" in eval_.improvement_suggestions


# ── EvaluatorAgent communication-specific dimensions ──────────────────

def test_evaluator_communication_mode_adds_extra_dimensions():
    """communication interview_type should trigger communication-specific evaluation."""
    evaluator = EvaluatorAgent(role="PR Manager", interview_type="communication")

    comm_json = (
        '{"overall_score": 8, "technical_score": 5, "relevance_score": 8, '
        '"clarity_score": 9, "communication_score": 9, "completeness_score": 8, '
        '"language_quality_score": 8, "response_structure_score": 8, '
        '"filler_word_score": 9, "strengths": ["structured"], "weaknesses": [], '
        '"improvement_suggestions": [], "evaluation": "Excellent."}'
    )
    with patch("agents.evaluator_agent.generate_text", return_value=comm_json):
        result = evaluator.evaluate(question="Tell me.", answer="Sure, I will start with context, then provide examples.")

    assert "language_quality_score"   in result
    assert "response_structure_score" in result
    assert "filler_word_score"        in result
    # Scores must be clamped to [0, 10]
    assert 0 <= result["language_quality_score"]   <= 10
    assert 0 <= result["response_structure_score"] <= 10
    assert 0 <= result["filler_word_score"]        <= 10


def test_evaluator_voice_mode_triggers_communication_evaluation():
    """voice_input_mode='voice' should trigger communication-specific evaluation."""
    evaluator = EvaluatorAgent(role="Any", interview_type="technical")

    comm_json = (
        '{"overall_score": 7, "technical_score": 7, "relevance_score": 7, '
        '"clarity_score": 7, "communication_score": 7, "completeness_score": 7, '
        '"language_quality_score": 7, "response_structure_score": 7, '
        '"filler_word_score": 8, "strengths": [], "weaknesses": [], '
        '"improvement_suggestions": [], "evaluation": "OK."}'
    )
    with patch("agents.evaluator_agent.generate_text", return_value=comm_json):
        result = evaluator.evaluate(
            question         = "What is polymorphism?",
            answer           = "It means um like, multiple forms.",
            voice_input_mode = "voice",
        )

    # Even though interview_type is "technical", voice mode triggers comm dimensions
    assert "filler_word_score" in result


def test_evaluator_filler_count_stored_in_result():
    """_filler_word_count must be stored in the result as _filler_word_count."""
    evaluator = EvaluatorAgent(role="Any", interview_type="communication")

    comm_json = (
        '{"overall_score": 6, "technical_score": 5, "relevance_score": 6, '
        '"clarity_score": 6, "communication_score": 6, "completeness_score": 6, '
        '"language_quality_score": 6, "response_structure_score": 6, '
        '"filler_word_score": 5, "strengths": [], "weaknesses": [], '
        '"improvement_suggestions": [], "evaluation": "Average."}'
    )
    answer_text = "Um, so like, I would, you know, approach it basically by, uh, iterating."
    with patch("agents.evaluator_agent.generate_text", return_value=comm_json):
        result = evaluator.evaluate(
            question = "How do you approach debugging?",
            answer   = answer_text,
        )

    assert "_filler_word_count" in result
    assert result["_filler_word_count"] >= 3


def test_evaluator_standard_mode_no_extra_dimensions():
    """Standard (non-communication) evaluation should NOT require comm-specific fields."""
    evaluator = EvaluatorAgent(role="SDE", interview_type="technical")

    standard_json = (
        '{"overall_score": 8, "technical_score": 9, "relevance_score": 8, '
        '"clarity_score": 7, "communication_score": 7, "completeness_score": 8, '
        '"strengths": ["correct"], "weaknesses": [], '
        '"improvement_suggestions": [], "evaluation": "Accurate."}'
    )
    with patch("agents.evaluator_agent.generate_text", return_value=standard_json):
        result = evaluator.evaluate(
            question = "What is O(n log n)?",
            answer   = "It's the time complexity of efficient sorting algorithms.",
        )

    assert result["technical_score"] == 9
    # language_quality_score is NOT required in standard mode
    # (it may be absent or 0 — either is acceptable)


def test_evaluator_safe_fallback_on_invalid_json():
    """EvaluatorAgent must not raise on completely invalid LLM output."""
    evaluator = EvaluatorAgent(role="Any", interview_type="technical")

    with patch("agents.evaluator_agent.generate_text", return_value="This is not JSON at all."):
        result = evaluator.evaluate(question="Q?", answer="A.")

    assert isinstance(result, dict)
    assert result["overall_score"] == 0
    assert isinstance(result["strengths"], list)
    assert isinstance(result["weaknesses"], list)
