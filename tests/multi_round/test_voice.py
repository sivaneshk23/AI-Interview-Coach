"""
Tests for voice interview infrastructure.

These are offline tests — no IBM network calls are made.
Voice input arrives as a transcript string (Web Speech API output).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from interview.plan import InterviewRound, RoundTurn
from interview.round_types import RoundState, RoundType
from agents.evaluator_agent import EvaluatorAgent, _count_filler_words


# ── RoundTurn voice fields ─────────────────────────────────────────────

def test_round_turn_voice_defaults():
    """RoundTurn should have voice fields with safe defaults."""
    turn = RoundTurn(question="Q", answer="A")
    assert turn.voice_input_mode == "text"
    assert turn.transcript is None
    assert turn.voice_duration_sec is None


def test_round_turn_voice_fields_set():
    """Voice fields can be set on a RoundTurn."""
    turn = RoundTurn(
        question           = "Tell me about yourself.",
        answer             = "I am a software engineer.",
        voice_input_mode   = "voice",
        transcript         = "I am a software engineer with 3 years of experience.",
        voice_duration_sec = 12.5,
    )
    assert turn.voice_input_mode == "voice"
    assert turn.transcript is not None
    assert turn.voice_duration_sec == pytest.approx(12.5)


def test_round_turn_voice_fields_serialise():
    """Voice fields must survive to_dict / from_dict round-trip."""
    turn = RoundTurn(
        question           = "Q?",
        answer             = "A",
        voice_input_mode   = "voice",
        transcript         = "spoken answer text",
        voice_duration_sec = 45.2,
    )
    d = turn.to_dict()
    assert d["voice_input_mode"]   == "voice"
    assert d["transcript"]         == "spoken answer text"
    assert d["voice_duration_sec"] == pytest.approx(45.2)

    restored = RoundTurn.from_dict(d)
    assert restored.voice_input_mode   == "voice"
    assert restored.transcript         == "spoken answer text"
    assert restored.voice_duration_sec == pytest.approx(45.2)


def test_round_turn_time_started_at_serialises():
    """time_started_at is serialised and deserialised correctly."""
    turn = RoundTurn(
        question        = "Q?",
        time_started_at = "2025-01-01T10:00:00+00:00",
    )
    d = turn.to_dict()
    assert d["time_started_at"] == "2025-01-01T10:00:00+00:00"

    restored = RoundTurn.from_dict(d)
    assert restored.time_started_at == "2025-01-01T10:00:00+00:00"


# ── Filler word counter ────────────────────────────────────────────────

def test_filler_word_counter_zero_for_clean_text():
    """Clean professional text should have zero or very few filler words."""
    text = "My experience includes Python development and machine learning projects."
    count = _count_filler_words(text)
    assert count == 0


def test_filler_word_counter_detects_fillers():
    """Should detect common filler words: um, uh, like, you know."""
    text = "Um, so like, I was working on this project, you know, and uh, it was really challenging."
    count = _count_filler_words(text)
    assert count >= 4, f"Expected at least 4 fillers, got {count}"


def test_filler_word_counter_case_insensitive():
    """Filler word detection must be case-insensitive."""
    count_lower = _count_filler_words("um uh like")
    count_upper = _count_filler_words("UM UH LIKE")
    assert count_lower == count_upper


# ── EvaluatorAgent voice path ─────────────────────────────────────────

def test_evaluator_uses_transcript_when_voice_mode():
    """When voice_input_mode='voice', evaluator should use transcript as the effective answer."""
    evaluator = EvaluatorAgent(role="Software Engineer", interview_type="technical")

    expected_transcript = "My spoken answer about Python."

    with patch("agents.evaluator_agent.generate_text") as mock_llm:
        mock_llm.return_value = '{"overall_score": 7, "technical_score": 7, "relevance_score": 7, "clarity_score": 7, "communication_score": 7, "completeness_score": 7, "language_quality_score": 7, "response_structure_score": 7, "filler_word_score": 8, "strengths": [], "weaknesses": [], "improvement_suggestions": [], "evaluation": "Good."}'

        result = evaluator.evaluate(
            question           = "Explain Python.",
            answer             = "A",          # typed answer — should be ignored
            voice_input_mode   = "voice",
            transcript         = expected_transcript,
            voice_duration_sec = 15.0,
        )

    # The prompt should have been built with transcript, not "A" alone
    assert mock_llm.called, "LLM should have been called"
    call_prompt = mock_llm.call_args[0][0]  # first positional arg is the prompt
    assert expected_transcript in call_prompt, "Transcript should appear in the LLM prompt"


def test_evaluator_communication_mode_returns_extra_dimensions():
    """Communication mode evaluation must include extra voice quality dimensions."""
    evaluator = EvaluatorAgent(role="PR Manager", interview_type="communication")

    with patch("utils.llm.generate_text") as mock_llm:
        mock_llm.return_value = '{"overall_score": 7, "technical_score": 5, "relevance_score": 7, "clarity_score": 8, "communication_score": 8, "completeness_score": 7, "language_quality_score": 8, "response_structure_score": 7, "filler_word_score": 9, "strengths": ["clear"], "weaknesses": [], "improvement_suggestions": [], "evaluation": "Good."}'

        result = evaluator.evaluate(
            question = "Describe a time you presented a complex idea.",
            answer   = "I once presented our Q2 roadmap to senior leadership.",
        )

    assert "language_quality_score" in result
    assert "response_structure_score" in result
    assert "filler_word_score" in result


def test_evaluator_voice_mode_includes_duration_in_prompt():
    """When voice_duration_sec is provided, the prompt should mention the duration."""
    evaluator = EvaluatorAgent(role="Any", interview_type="communication")

    with patch("agents.evaluator_agent.generate_text") as mock_llm:
        mock_llm.return_value = '{"overall_score": 6, "technical_score": 5, "relevance_score": 6, "clarity_score": 6, "communication_score": 6, "completeness_score": 6, "language_quality_score": 6, "response_structure_score": 6, "filler_word_score": 7, "strengths": [], "weaknesses": [], "improvement_suggestions": [], "evaluation": "OK."}'

        evaluator.evaluate(
            question           = "Tell me about yourself.",
            answer             = "I am a developer.",
            voice_input_mode   = "voice",
            voice_duration_sec = 95.5,
        )

    assert mock_llm.called
    call_prompt = mock_llm.call_args[0][0]
    assert "95.5" in call_prompt or "duration" in call_prompt.lower()


def test_evaluator_voice_fallback_on_llm_failure():
    """EvaluatorAgent must return safe defaults when IBM Granite fails."""
    evaluator = EvaluatorAgent(role="Any", interview_type="communication")

    with patch("utils.llm.generate_text", side_effect=Exception("IBM down")):
        result = evaluator.evaluate(
            question         = "How do you handle conflict?",
            answer           = "I try to listen first.",
            voice_input_mode = "voice",
        )

    # Must return a dict with required keys — not crash
    assert isinstance(result, dict)
    assert "overall_score" in result
    assert "communication_score" in result


# ── InterviewRound timing fields ──────────────────────────────────────

def test_round_started_at_set_on_start():
    """round.start() should set round_started_at."""
    from interview.plan import InterviewRound
    round_ = InterviewRound(
        round_type = RoundType.TECHNICAL,
        max_turns  = 5,
    )
    assert round_.round_started_at is None
    round_.start()
    assert round_.round_started_at is not None


def test_round_elapsed_seconds_none_before_start():
    """elapsed_seconds() returns None for a round that hasn't started."""
    from interview.plan import InterviewRound
    round_ = InterviewRound(round_type=RoundType.TECHNICAL)
    assert round_.elapsed_seconds() is None


def test_round_elapsed_seconds_positive_after_start():
    """elapsed_seconds() returns a positive float after starting."""
    import time
    from interview.plan import InterviewRound
    round_ = InterviewRound(round_type=RoundType.TECHNICAL, max_turns=5)
    round_.start()
    time.sleep(0.01)
    elapsed = round_.elapsed_seconds()
    assert elapsed is not None
    assert elapsed >= 0.0


def test_round_is_time_expired_false_when_no_limit():
    """is_time_expired() should be False when no time_limit_minutes is set."""
    from interview.plan import InterviewRound
    round_ = InterviewRound(round_type=RoundType.TECHNICAL, max_turns=5)
    round_.start()
    assert round_.is_time_expired() is False


def test_round_is_time_expired_false_within_limit():
    """is_time_expired() should be False when within the time limit."""
    from interview.plan import InterviewRound
    round_ = InterviewRound(round_type=RoundType.TECHNICAL, time_limit_minutes=60, max_turns=5)
    round_.start()
    assert round_.is_time_expired() is False


def test_round_is_time_expired_true_when_past_limit():
    """is_time_expired() should be True when we inject a past start time."""
    from interview.plan import InterviewRound
    round_ = InterviewRound(round_type=RoundType.TECHNICAL, time_limit_minutes=1, max_turns=5)
    # Artificially set start time to 2 minutes ago
    round_.round_started_at = "2000-01-01T00:00:00+00:00"  # very old timestamp
    assert round_.is_time_expired() is True


def test_round_started_at_serialises():
    """round_started_at should survive to_dict/from_dict."""
    from interview.plan import InterviewRound
    round_ = InterviewRound(round_type=RoundType.TECHNICAL, time_limit_minutes=30, max_turns=5)
    round_.round_started_at = "2025-06-01T10:00:00+00:00"

    d = round_.to_dict()
    assert d["round_started_at"] == "2025-06-01T10:00:00+00:00"

    restored = InterviewRound.from_dict(d)
    assert restored.round_started_at == "2025-06-01T10:00:00+00:00"
