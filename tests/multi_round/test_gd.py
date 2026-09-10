"""
Tests for the GD (Group Discussion) round executor.

These are offline tests — no IBM network calls are made.
All LLM calls are mocked.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from interview.executors import GDRoundExecutor, _GD_PERSONAS, build_executor
from interview.plan import InterviewRound, RoundTurn
from interview.round_types import RoundState, RoundType


# ── Fixtures ───────────────────────────────────────────────────────────

def _make_gd_round(max_turns: int = 4) -> InterviewRound:
    r = InterviewRound(
        round_id   = str(uuid.uuid4()),
        round_type = RoundType.GD,
        max_turns  = max_turns,
        difficulty = "medium",
    )
    r.state = RoundState.IN_PROGRESS
    return r


def _executor():
    return GDRoundExecutor()


# ── Structure tests (no LLM) ───────────────────────────────────────────

def test_gd_executor_has_correct_round_type():
    assert GDRoundExecutor.round_type == RoundType.GD


def test_gd_max_ai_turns_constant():
    """The AI turn limit must be a positive integer — never unbounded."""
    assert GDRoundExecutor.MAX_GD_AI_TURNS > 0
    assert isinstance(GDRoundExecutor.MAX_GD_AI_TURNS, int)


def test_gd_personas_defined():
    """At least 2 distinct AI personas must be defined."""
    assert len(_GD_PERSONAS) >= 2
    names = [p["name"] for p in _GD_PERSONAS]
    assert len(set(names)) == len(names), "Persona names must be unique"
    for p in _GD_PERSONAS:
        assert "name" in p
        assert "style" in p
        assert "position" in p


def test_build_executor_gd_no_agents():
    """build_executor(GD) works without agents."""
    executor = build_executor(RoundType.GD)
    assert isinstance(executor, GDRoundExecutor)


def test_build_executor_gd_with_agents():
    """build_executor(GD) accepts optional agents."""
    mock_interviewer = MagicMock()
    mock_evaluator   = MagicMock()
    executor = build_executor(
        RoundType.GD,
        interviewer_agent=mock_interviewer,
        evaluator_agent=mock_evaluator,
    )
    assert isinstance(executor, GDRoundExecutor)


# ── Topic generation ───────────────────────────────────────────────────

def test_generate_question_first_call_produces_moderator_turn():
    """First call to generate_question should return a moderator/topic turn."""
    executor = _executor()
    round_   = _make_gd_round()

    with patch("utils.llm.generate_text", return_value="Should AI replace human jobs?"):
        turn = executor.generate_question(round_=round_, role="Software Engineer")

    assert turn.gd_role == "moderator"
    assert turn.gd_persona_name == "Moderator"
    assert "Topic" in turn.question or len(turn.question) > 10
    assert turn.time_started_at is not None


def test_generate_question_topic_fallback():
    """generate_question handles LLM failure gracefully with a fallback topic."""
    executor = _executor()
    round_   = _make_gd_round()

    with patch("utils.llm.generate_text", side_effect=Exception("IBM down")):
        turn = executor.generate_question(round_=round_, role="Data Analyst")

    assert turn.gd_role == "moderator"
    assert len(turn.question) > 10  # fallback question is generated


# ── AI participant response ────────────────────────────────────────────

def test_generate_question_after_candidate_turn_produces_ai_participant():
    """After one candidate turn, generate_question should return an AI participant turn."""
    executor = _executor()
    round_   = _make_gd_round(max_turns=3)

    # Simulate first moderator turn already in the round
    moderator_turn = RoundTurn(
        question       = "Group Discussion Topic:\n\nAI in healthcare\n\nPlease share your opening thoughts.",
        gd_role        = "moderator",
        gd_persona_name= "Moderator",
    )
    round_.turns.append(moderator_turn)

    # Simulate one candidate answer already submitted
    candidate_turn = RoundTurn(
        question = "...",
        answer   = "I believe AI in healthcare has great potential.",
        gd_role  = "candidate",
    )
    round_.turns.append(candidate_turn)

    with patch("utils.llm.generate_text", return_value="That's an interesting point about AI."):
        turn = executor.generate_question(round_=round_, role="Data Analyst")

    assert turn.gd_role == "ai_participant"
    assert turn.gd_persona_name in [p["name"] for p in _GD_PERSONAS]
    assert turn.time_started_at is not None


# ── Bounded AI turns ───────────────────────────────────────────────────

def test_ai_turn_limit_produces_moderator_summary():
    """After MAX_GD_AI_TURNS AI turns, the next call should return the moderator summary."""
    executor = _executor()
    max_ai   = GDRoundExecutor.MAX_GD_AI_TURNS
    round_   = _make_gd_round(max_turns=max_ai + 5)

    # Seed the round with alternating candidate+AI turns up to the limit
    round_.turns.append(RoundTurn(
        question="Group Discussion Topic:\n\nTopic here\n\nShare thoughts.",
        gd_role="moderator", gd_persona_name="Moderator",
    ))
    for i in range(max_ai):
        round_.turns.append(RoundTurn(
            question="...", answer=f"Candidate point {i}",
            gd_role="candidate",
        ))
        round_.turns.append(RoundTurn(
            question=f"[AI]: Response {i}",
            gd_role="ai_participant",
            gd_persona_name=_GD_PERSONAS[i % len(_GD_PERSONAS)]["name"],
        ))

    with patch("utils.llm.generate_text", return_value="Discussion summary. Thank you all."):
        turn = executor.generate_question(round_=round_, role="Any Role")

    assert turn.gd_role == "moderator", (
        f"Expected moderator turn after {max_ai} AI turns, got gd_role={turn.gd_role!r}"
    )


# ── evaluate_answer ───────────────────────────────────────────────────

def test_evaluate_answer_sets_candidate_role():
    """evaluate_answer must tag the turn as gd_role='candidate'."""
    executor = _executor()
    round_   = _make_gd_round()

    # Seed a moderator turn so the topic can be extracted
    round_.turns.append(RoundTurn(
        question="Group Discussion Topic:\n\nTech job market\n\nShare thoughts.",
        gd_role="moderator", gd_persona_name="Moderator",
    ))

    turn = RoundTurn(question="...", answer="I think the tech job market is evolving.")

    with patch("utils.llm.generate_text", return_value='{"overall_score": 7, "technical_score": 5, "relevance_score": 8, "clarity_score": 7, "communication_score": 7, "completeness_score": 6, "strengths": ["clear point"], "weaknesses": [], "improvement_suggestions": [], "evaluation": "Good contribution."}'):
        evaluated = executor.evaluate_answer(round_=round_, turn=turn, role="Any")

    assert evaluated.gd_role == "candidate"
    assert evaluated.score is not None
    assert 0 <= evaluated.score <= 10


def test_evaluate_answer_fallback_on_llm_failure():
    """evaluate_answer must return a safe default when IBM Granite fails."""
    executor = _executor()
    round_   = _make_gd_round()
    round_.turns.append(RoundTurn(
        question="Group Discussion Topic:\n\nTopic\n\nPlease share your thoughts.",
        gd_role="moderator", gd_persona_name="Moderator",
    ))

    turn = RoundTurn(question="...", answer="My perspective here.")

    with patch("utils.llm.generate_text", side_effect=Exception("IBM down")):
        evaluated = executor.evaluate_answer(round_=round_, turn=turn, role="Any")

    assert evaluated.gd_role == "candidate"
    assert isinstance(evaluated.evaluation, dict)
    assert evaluated.score is not None


# ── build_round_evaluation ────────────────────────────────────────────

def test_build_round_evaluation_no_candidate_turns():
    """build_round_evaluation returns 0.0 when no candidate turns exist."""
    executor = _executor()
    round_   = _make_gd_round()
    eval_    = executor.build_round_evaluation(round_, "Any")
    assert eval_.score == 0.0


def test_build_round_evaluation_aggregates_candidate_turns():
    """build_round_evaluation averages scores from candidate turns only."""
    executor = _executor()
    round_   = _make_gd_round()

    t1 = RoundTurn(gd_role="candidate", score=8.0, evaluation={"strengths": ["good"], "weaknesses": [], "improvement_suggestions": [], "evaluation": "Good."})
    t2 = RoundTurn(gd_role="candidate", score=6.0, evaluation={"strengths": [], "weaknesses": ["vague"], "improvement_suggestions": ["Be specific"], "evaluation": "OK."})
    # AI participant turn — should NOT affect candidate score
    t3 = RoundTurn(gd_role="ai_participant", score=5.0, evaluation={})

    round_.turns.extend([t1, t2, t3])
    eval_ = executor.build_round_evaluation(round_, "Any")

    assert eval_.score == pytest.approx(7.0, abs=0.1)
    assert "good" in eval_.strengths
    assert "vague" in eval_.weaknesses
    assert "Be specific" in eval_.improvement_suggestions


# ── Topic extraction ──────────────────────────────────────────────────

def test_get_topic_extracts_from_moderator_turn():
    """_get_topic should extract the discussion topic from the moderator's turn."""
    executor = _executor()
    round_   = _make_gd_round()

    topic_text = "Is remote work more productive than office work?"
    round_.turns.append(RoundTurn(
        question=f"Group Discussion Topic:\n\n{topic_text}\n\nPlease share your thoughts.",
        gd_role="moderator",
    ))

    topic = executor._get_topic(round_)
    assert topic == topic_text


def test_get_topic_fallback_when_no_moderator_turn():
    """_get_topic returns a sensible fallback when no moderator turn exists."""
    executor = _executor()
    round_   = _make_gd_round()
    topic    = executor._get_topic(round_)
    assert isinstance(topic, str)
    assert len(topic) > 5


# ── GD round serialisation ────────────────────────────────────────────

def test_gd_turn_serialises_gd_fields():
    """GD-specific fields must round-trip through to_dict / from_dict."""
    turn = RoundTurn(
        question        = "[Arjun]: That's a good point.",
        answer          = "I agree with some caveats.",
        gd_role         = "candidate",
        gd_persona_name = "Arjun",
    )
    d = turn.to_dict()
    assert d["gd_role"] == "candidate"
    assert d["gd_persona_name"] == "Arjun"

    restored = RoundTurn.from_dict(d)
    assert restored.gd_role == "candidate"
    assert restored.gd_persona_name == "Arjun"
