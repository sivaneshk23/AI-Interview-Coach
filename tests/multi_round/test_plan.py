"""
Tests for interview/plan.py

RoundTurn — construction, to_dict/from_dict
RoundEvaluation — construction, to_dict/from_dict
InterviewRound — state transitions, properties, serialisation
InterviewPlan — navigation, progress, scores, serialisation
"""

import pytest
from interview.plan import InterviewPlan, InterviewRound, RoundEvaluation, RoundTurn
from interview.round_types import RoundState, RoundType


# ── RoundTurn ─────────────────────────────────────────────────────────

class TestRoundTurn:
    def test_defaults(self):
        t = RoundTurn()
        assert t.question == ""
        assert t.answer == ""
        assert t.options == []
        assert t.score is None
        assert t.is_correct is None
        assert t.evaluation == {}

    def test_turn_id_auto_generated(self):
        a, b = RoundTurn(), RoundTurn()
        assert a.turn_id != b.turn_id

    def test_to_dict_keys(self):
        t = RoundTurn(question="Q1", answer="A1", score=7.5)
        d = t.to_dict()
        for key in ("turn_id", "question", "answer", "score", "evaluation"):
            assert key in d

    def test_round_trip(self):
        original = RoundTurn(
            question="What is Python?",
            answer="A programming language.",
            score=8.0,
            options=["A", "B"],
            selected_option="A",
            correct_option="A",
            is_correct=True,
        )
        restored = RoundTurn.from_dict(original.to_dict())
        assert restored.question == original.question
        assert restored.answer == original.answer
        assert restored.score == original.score
        assert restored.is_correct == original.is_correct
        assert restored.selected_option == original.selected_option

    def test_from_dict_missing_keys_uses_defaults(self):
        t = RoundTurn.from_dict({})
        assert t.question == ""
        assert t.answer == ""
        assert t.score is None


# ── RoundEvaluation ───────────────────────────────────────────────────

class TestRoundEvaluation:
    def test_defaults(self):
        e = RoundEvaluation()
        assert e.score == 0.0
        assert e.max_score == 10.0
        assert e.strengths == []
        assert e.weaknesses == []
        assert e.improvement_suggestions == []
        assert e.correct_count is None

    def test_to_dict_round_trip(self):
        original = RoundEvaluation(
            score=7.5,
            strengths=["Good"],
            weaknesses=["Needs work"],
            feedback="Decent performance",
            improvement_suggestions=["Practice more"],
            correct_count=8,
            total_questions=10,
        )
        restored = RoundEvaluation.from_dict(original.to_dict())
        assert restored.score == original.score
        assert restored.strengths == original.strengths
        assert restored.correct_count == original.correct_count
        assert restored.total_questions == original.total_questions


# ── InterviewRound ────────────────────────────────────────────────────

class TestInterviewRound:
    def _make_round(self, **kwargs):
        defaults = dict(round_type=RoundType.TECHNICAL, order=1, max_turns=3)
        defaults.update(kwargs)
        return InterviewRound(**defaults)

    def test_default_title_from_type(self):
        r = self._make_round(round_type=RoundType.HR)
        assert r.title == "HR Interview"

    def test_explicit_title_preserved(self):
        r = self._make_round(title="Custom Round")
        assert r.title == "Custom Round"

    def test_initial_state_not_started(self):
        r = self._make_round()
        assert r.state == RoundState.NOT_STARTED

    def test_start_transition(self):
        r = self._make_round()
        r.start()
        assert r.state == RoundState.IN_PROGRESS

    def test_complete_transition(self):
        r = self._make_round()
        r.start()
        r.complete()
        assert r.state == RoundState.COMPLETED

    def test_skip_from_not_started(self):
        r = self._make_round()
        r.skip()
        assert r.state == RoundState.SKIPPED

    def test_skip_from_in_progress(self):
        r = self._make_round()
        r.start()
        r.skip()
        assert r.state == RoundState.SKIPPED

    def test_mark_failed(self):
        r = self._make_round()
        r.start()
        r.mark_failed()
        assert r.state == RoundState.FAILED

    def test_invalid_start_from_completed_raises(self):
        r = self._make_round()
        r.start()
        r.complete()
        with pytest.raises(ValueError):
            r.start()

    def test_invalid_complete_from_not_started_raises(self):
        r = self._make_round()
        with pytest.raises(ValueError):
            r.complete()

    def test_questions_remaining(self):
        r = self._make_round(max_turns=3)
        r.start()
        assert r.questions_remaining == 3
        r.turns.append(RoundTurn(question="Q1", answer="A1"))
        assert r.questions_remaining == 2

    def test_turn_count(self):
        r = self._make_round()
        assert r.turn_count == 0
        r.turns.append(RoundTurn())
        assert r.turn_count == 1

    def test_is_complete_property(self):
        r = self._make_round()
        assert r.is_complete is False
        r.start()
        r.complete()
        assert r.is_complete is True

    def test_is_in_progress_property(self):
        r = self._make_round()
        assert r.is_in_progress is False
        r.start()
        assert r.is_in_progress is True

    def test_to_dict_round_trip(self):
        r = self._make_round(
            round_type=RoundType.HR,
            title="HR Interview",
            order=2,
            max_turns=5,
        )
        r.start()
        r.turns.append(RoundTurn(question="Q", answer="A", score=7.0))
        d = r.to_dict()
        restored = InterviewRound.from_dict(d)
        assert restored.round_id == r.round_id
        assert restored.round_type == r.round_type
        assert restored.state == r.state
        assert len(restored.turns) == 1
        assert restored.turns[0].question == "Q"

    def test_from_dict_with_evaluation(self):
        r = self._make_round()
        r.start()
        r.evaluation = RoundEvaluation(score=8.5, feedback="Great")
        r.complete()
        d = r.to_dict()
        restored = InterviewRound.from_dict(d)
        assert restored.evaluation is not None
        assert restored.evaluation.score == 8.5


# ── InterviewPlan ─────────────────────────────────────────────────────

class TestInterviewPlan:
    def _make_plan(self, n_rounds=3):
        rounds = [
            InterviewRound(round_type=RoundType.TECHNICAL, order=i + 1, max_turns=2)
            for i in range(n_rounds)
        ]
        return InterviewPlan(rounds=rounds)

    def test_current_round_initial(self):
        plan = self._make_plan()
        assert plan.current_round is plan.rounds[0]

    def test_total_rounds(self):
        plan = self._make_plan(3)
        assert plan.total_rounds == 3

    def test_progress_percent_zero_initially(self):
        plan = self._make_plan()
        assert plan.progress_percent == 0

    def test_progress_percent_after_complete(self):
        plan = self._make_plan(2)
        plan.rounds[0].start()
        plan.rounds[0].complete()
        assert plan.progress_percent == 50

    def test_is_complete_false_initially(self):
        plan = self._make_plan()
        assert plan.is_complete is False

    def test_is_complete_when_all_done(self):
        plan = self._make_plan(2)
        for r in plan.rounds:
            r.start()
            r.complete()
        assert plan.is_complete is True

    def test_is_complete_skipped_optional_round(self):
        """Optional round being skipped does not block completion."""
        r1 = InterviewRound(round_type=RoundType.TECHNICAL, order=1, is_required=True)
        r2 = InterviewRound(round_type=RoundType.GD, order=2, is_required=False)
        plan = InterviewPlan(rounds=[r1, r2])
        r1.start()
        r1.complete()
        r2.skip()
        assert plan.is_complete is True

    def test_start_current_round(self):
        plan = self._make_plan()
        r = plan.start_current_round()
        assert r.state == RoundState.IN_PROGRESS

    def test_start_current_round_no_rounds_raises(self):
        plan = InterviewPlan(rounds=[])
        with pytest.raises(ValueError):
            plan.start_current_round()

    def test_advance_to_next_round(self):
        plan = self._make_plan(2)
        plan.rounds[0].start()
        plan.rounds[0].complete()
        next_r = plan.advance_to_next_round()
        assert next_r is plan.rounds[1]

    def test_advance_returns_none_when_all_done(self):
        plan = self._make_plan(1)
        plan.rounds[0].start()
        plan.rounds[0].complete()
        result = plan.advance_to_next_round()
        assert result is None

    def test_advance_raises_if_in_progress(self):
        plan = self._make_plan(2)
        plan.rounds[0].start()
        with pytest.raises(ValueError, match="in progress"):
            plan.advance_to_next_round()

    def test_round_scores_empty_when_no_completions(self):
        plan = self._make_plan()
        assert plan.round_scores() == {}

    def test_round_scores_with_evaluated_rounds(self):
        plan = self._make_plan(2)
        plan.rounds[0].start()
        plan.rounds[0].evaluation = RoundEvaluation(score=8.0)
        plan.rounds[0].complete()
        scores = plan.round_scores()
        assert "technical" in scores
        assert scores["technical"] == 8.0

    def test_overall_score_zero_no_completions(self):
        plan = self._make_plan()
        assert plan.overall_score() == 0.0

    def test_overall_score_average(self):
        # Use two different round types so round_scores dict keeps both entries
        r1 = InterviewRound(round_type=RoundType.TECHNICAL, order=1, max_turns=2)
        r2 = InterviewRound(round_type=RoundType.HR, order=2, max_turns=2)
        plan = InterviewPlan(rounds=[r1, r2])
        r1.start()
        r1.evaluation = RoundEvaluation(score=8.0)
        r1.complete()
        r2.start()
        r2.evaluation = RoundEvaluation(score=6.0)
        r2.complete()
        assert plan.overall_score() == 7.0

    def test_to_dict_round_trip(self):
        plan = self._make_plan(2)
        plan.rounds[0].start()
        d = plan.to_dict()
        restored = InterviewPlan.from_dict(d)
        assert restored.plan_id == plan.plan_id
        assert restored.total_rounds == plan.total_rounds
        assert restored.rounds[0].state == RoundState.IN_PROGRESS

    def test_completed_rounds_list(self):
        plan = self._make_plan(3)
        plan.rounds[0].start()
        plan.rounds[0].complete()
        assert len(plan.completed_rounds) == 1
        assert plan.completed_rounds[0] is plan.rounds[0]

    def test_remaining_rounds_list(self):
        plan = self._make_plan(3)
        plan.rounds[0].start()
        plan.rounds[0].complete()
        # Only not_started rounds are "remaining"
        remaining = plan.remaining_rounds
        assert all(r.state == RoundState.NOT_STARTED for r in remaining)
        assert len(remaining) == 2
