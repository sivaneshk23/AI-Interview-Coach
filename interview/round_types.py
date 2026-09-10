"""
Interview Round Types.

These are round TYPE identifiers — not job-role restrictions.
The job role remains a free-form string throughout the system.

Adding a new round type:
  1. Add a value to RoundType.
  2. Create a subclass of BaseRoundConfig (optional; defaults work).
  3. Add a planner blueprint in interview/planner.py.
  No other files need changing.
"""

from __future__ import annotations

from enum import Enum


class RoundType(str, Enum):
    """
    Supported interview round types.

    New types can be added here without changing any other core logic.
    Using str mixin ensures JSON serialisation works natively.
    """

    APTITUDE = "aptitude"
    TECHNICAL = "technical"
    CODING = "coding"
    GD = "gd"                   # Group Discussion
    COMMUNICATION = "communication"
    HR = "hr"

    def label(self) -> str:
        """Human-readable label for display."""
        _labels = {
            "aptitude": "Aptitude Test",
            "technical": "Technical Interview",
            "coding": "Coding Challenge",
            "gd": "Group Discussion",
            "communication": "Communication Assessment",
            "hr": "HR Interview",
        }
        return _labels.get(self.value, self.value.title())


class RoundState(str, Enum):
    """
    Round lifecycle state machine.

    Valid transitions:
        NOT_STARTED → IN_PROGRESS
        IN_PROGRESS → COMPLETED | SKIPPED
        NOT_STARTED → SKIPPED
        COMPLETED   → (terminal)
        SKIPPED     → (terminal)
        FAILED      → (terminal)  # error state — not a candidate failure

    No other transitions are permitted.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    SKIPPED     = "skipped"
    FAILED      = "failed"   # infrastructure error, not candidate performance

    # Terminal states
    TERMINAL = frozenset({
        "completed",
        "skipped",
        "failed",
    })

    def can_transition_to(self, target: "RoundState") -> bool:
        """Return True if this → target transition is valid."""
        _allowed: dict[str, set[str]] = {
            "not_started": {"in_progress", "skipped"},
            "in_progress":  {"completed", "skipped", "failed"},
            "completed":    set(),
            "skipped":      set(),
            "failed":       set(),
        }
        return target.value in _allowed.get(self.value, set())
