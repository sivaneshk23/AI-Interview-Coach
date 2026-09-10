"""
Tests for server-side timing enforcement in interview rounds.

These are offline tests — no IBM network calls are made.
"""

from __future__ import annotations

import datetime
import time
import uuid

import pytest

from interview.plan import InterviewRound, InterviewPlan, RoundEvaluation
from interview.round_types import RoundState, RoundType


# ── InterviewRound timing ──────────────────────────────────────────────

def _make_round(round_type=RoundType.TECHNICAL, max_turns=5, time_limit_minutes=None):
    return InterviewRound(
        round_id           = str(uuid.uuid4()),
        round_type         = round_type,
        max_turns          = max_turns,
        time_limit_minutes = time_limit_minutes,
    )


class TestRoundTimingFields:
    """round_started_at and is_time_expired() / elapsed_seconds()."""

    def test_round_started_at_initially_none(self):
        r = _make_round()
        assert r.round_started_at is None

    def test_round_start_sets_started_at(self):
        r = _make_round()
        r.start()
        assert r.round_started_at is not None
        # Should be a valid ISO-8601 UTC string
        parsed = datetime.datetime.fromisoformat(r.round_started_at)
        assert parsed is not None

    def test_elapsed_seconds_none_before_start(self):
        r = _make_round()
        assert r.elapsed_seconds() is None

    def test_elapsed_seconds_positive_after_start(self):
        r = _make_round()
        r.start()
        time.sleep(0.01)
        elapsed = r.elapsed_seconds()
        assert elapsed is not None
        assert elapsed >= 0.001

    def test_elapsed_seconds_increases_over_time(self):
        r = _make_round()
        r.start()
        e1 = r.elapsed_seconds()
        time.sleep(0.05)
        e2 = r.elapsed_seconds()
        assert e2 > e1

    def test_is_time_expired_no_limit_always_false(self):
        """Without a time limit, is_time_expired() must always return False."""
        r = _make_round(time_limit_minutes=None)
        r.start()
        assert r.is_time_expired() is False

    def test_is_time_expired_within_limit(self):
        r = _make_round(time_limit_minutes=60)
        r.start()
        assert r.is_time_expired() is False

    def test_is_time_expired_past_limit(self):
        """Inject a very old start time to simulate an expired round."""
        r = _make_round(time_limit_minutes=1)
        r.round_started_at = "1999-01-01T00:00:00+00:00"
        assert r.is_time_expired() is True

    def test_is_time_expired_before_start(self):
        """is_time_expired() must return False if the round hasn't started."""
        r = _make_round(time_limit_minutes=1)
        assert r.is_time_expired() is False

    def test_is_time_expired_handles_malformed_timestamp(self):
        """Malformed timestamp must not crash — should return False."""
        r = _make_round(time_limit_minutes=1)
        r.round_started_at = "not-a-valid-timestamp"
        assert r.is_time_expired() is False


class TestRoundTimingSerialisation:
    """round_started_at must survive dict serialisation."""

    def test_to_dict_includes_round_started_at(self):
        r = _make_round()
        r.start()
        d = r.to_dict()
        assert "round_started_at" in d
        assert d["round_started_at"] == r.round_started_at

    def test_from_dict_restores_round_started_at(self):
        r = _make_round(time_limit_minutes=30)
        r.round_started_at = "2025-06-01T08:00:00+00:00"
        d = r.to_dict()
        restored = InterviewRound.from_dict(d)
        assert restored.round_started_at == "2025-06-01T08:00:00+00:00"
        assert restored.time_limit_minutes == 30

    def test_from_dict_none_started_at(self):
        r = _make_round()
        d = r.to_dict()
        assert d["round_started_at"] is None
        restored = InterviewRound.from_dict(d)
        assert restored.round_started_at is None


class TestRoundTimingBoundary:
    """Edge cases for timing."""

    def test_zero_time_limit_never_expires_before_start(self):
        """time_limit_minutes=0 — effectively no limit when not started."""
        r = _make_round(time_limit_minutes=0)
        # 0 evaluates as falsy → is_time_expired should return False
        assert r.is_time_expired() is False

    def test_very_short_limit_with_old_timestamp(self):
        """1-second equivalent limit with very old start — should expire."""
        r = InterviewRound(
            round_id           = str(uuid.uuid4()),
            round_type         = RoundType.APTITUDE,
            max_turns          = 5,
            time_limit_minutes = 0.001,  # ~0.06 seconds
        )
        r.round_started_at = "2000-01-01T00:00:00+00:00"  # clearly expired
        assert r.is_time_expired() is True

    def test_is_time_expired_not_affected_by_client_state(self):
        """
        is_time_expired() uses only server-side round_started_at.
        Client timer state is irrelevant to this check.
        """
        r = _make_round(time_limit_minutes=1)
        r.round_started_at = "2000-01-01T00:00:00+00:00"

        # Even if client claims the round was just started, server disagrees
        assert r.is_time_expired() is True


class TestRoundTimingMultipleRounds:
    """Timing across a multi-round plan."""

    def test_each_round_gets_independent_timestamp(self):
        """Each round should have its own round_started_at timestamp."""
        from interview.plan import InterviewPlan
        r1 = _make_round(RoundType.APTITUDE, time_limit_minutes=30)
        r2 = _make_round(RoundType.TECHNICAL, time_limit_minutes=45)
        plan = InterviewPlan(rounds=[r1, r2])

        r1.start()
        time.sleep(0.01)
        r2.start()

        assert r1.round_started_at != r2.round_started_at

    def test_plan_timing_does_not_affect_round_scoring(self):
        """Expiry state does not alter existing round scores."""
        r = _make_round(time_limit_minutes=1)
        r.round_started_at = "2000-01-01T00:00:00+00:00"  # expired
        r.evaluation = RoundEvaluation(score=8.5, feedback="Good round.")
        r.state = RoundState.COMPLETED

        assert r.is_time_expired() is True
        assert r.evaluation.score == pytest.approx(8.5)

    def test_plan_serialises_with_timing(self):
        """InterviewPlan serialisation includes round timing fields."""
        r = _make_round(time_limit_minutes=30)
        r.round_started_at = "2025-06-01T10:00:00+00:00"
        plan = InterviewPlan(rounds=[r])

        d = plan.to_dict()
        rounds = d["rounds"]
        assert rounds[0]["round_started_at"] == "2025-06-01T10:00:00+00:00"

        restored = InterviewPlan.from_dict(d)
        assert restored.rounds[0].round_started_at == "2025-06-01T10:00:00+00:00"
