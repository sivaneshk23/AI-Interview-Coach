"""
Tests for interview/round_types.py

RoundType — enum values, str mixin, label()
RoundState — enum values, TERMINAL set, can_transition_to()
"""

import pytest
from interview.round_types import RoundState, RoundType


# ── RoundType ─────────────────────────────────────────────────────────

class TestRoundType:
    def test_all_values_present(self):
        values = {rt.value for rt in RoundType}
        assert values == {"aptitude", "technical", "coding", "gd", "communication", "hr"}

    def test_str_mixin(self):
        """RoundType is a str subclass — values serialize natively to JSON."""
        assert RoundType.TECHNICAL == "technical"
        assert RoundType.HR == "hr"
        assert isinstance(RoundType.CODING, str)

    def test_label_returns_string(self):
        for rt in RoundType:
            lbl = rt.label()
            assert isinstance(lbl, str)
            assert len(lbl) > 0

    def test_known_labels(self):
        assert RoundType.APTITUDE.label() == "Aptitude Test"
        assert RoundType.TECHNICAL.label() == "Technical Interview"
        assert RoundType.CODING.label() == "Coding Challenge"
        assert RoundType.GD.label() == "Group Discussion"
        assert RoundType.COMMUNICATION.label() == "Communication Assessment"
        assert RoundType.HR.label() == "HR Interview"

    def test_round_trip_from_string(self):
        for rt in RoundType:
            assert RoundType(rt.value) is rt


# ── RoundState ────────────────────────────────────────────────────────

class TestRoundState:
    def test_all_values_present(self):
        values = {rs.value for rs in RoundState}
        # TERMINAL is a class attribute, not a member
        expected = {"not_started", "in_progress", "completed", "skipped", "failed"}
        assert expected.issubset(values)

    def test_str_mixin(self):
        assert RoundState.NOT_STARTED == "not_started"
        assert isinstance(RoundState.COMPLETED, str)

    # ── Valid transitions ─────────────────────────────────────────────

    def test_not_started_to_in_progress(self):
        assert RoundState.NOT_STARTED.can_transition_to(RoundState.IN_PROGRESS) is True

    def test_not_started_to_skipped(self):
        assert RoundState.NOT_STARTED.can_transition_to(RoundState.SKIPPED) is True

    def test_in_progress_to_completed(self):
        assert RoundState.IN_PROGRESS.can_transition_to(RoundState.COMPLETED) is True

    def test_in_progress_to_skipped(self):
        assert RoundState.IN_PROGRESS.can_transition_to(RoundState.SKIPPED) is True

    def test_in_progress_to_failed(self):
        assert RoundState.IN_PROGRESS.can_transition_to(RoundState.FAILED) is True

    # ── Invalid transitions ───────────────────────────────────────────

    def test_not_started_to_completed_invalid(self):
        assert RoundState.NOT_STARTED.can_transition_to(RoundState.COMPLETED) is False

    def test_not_started_to_failed_invalid(self):
        assert RoundState.NOT_STARTED.can_transition_to(RoundState.FAILED) is False

    def test_completed_is_terminal(self):
        for target in RoundState:
            if target.value in ("completed", "skipped", "failed",
                                "not_started", "in_progress"):
                assert RoundState.COMPLETED.can_transition_to(target) is False

    def test_skipped_is_terminal(self):
        for target in RoundState:
            assert RoundState.SKIPPED.can_transition_to(target) is False

    def test_failed_is_terminal(self):
        for target in RoundState:
            assert RoundState.FAILED.can_transition_to(target) is False

    def test_in_progress_to_not_started_invalid(self):
        assert RoundState.IN_PROGRESS.can_transition_to(RoundState.NOT_STARTED) is False

    # ── TERMINAL frozenset ────────────────────────────────────────────

    def test_terminal_contains_completed(self):
        assert "completed" in RoundState.TERMINAL

    def test_terminal_contains_skipped(self):
        assert "skipped" in RoundState.TERMINAL

    def test_terminal_contains_failed(self):
        assert "failed" in RoundState.TERMINAL

    def test_terminal_does_not_contain_in_progress(self):
        assert "in_progress" not in RoundState.TERMINAL
