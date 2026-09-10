"""
Interview Round and InterviewPlan dataclasses.

These are the core domain objects for the multi-round interview engine.
They are pure Python dataclasses — no database dependency — so they can
be tested without a DB and serialised/deserialised from the existing
JSON-based SessionStore without changes to the persistence layer.

Persistence contract:
  InterviewPlan.to_dict() / InterviewPlan.from_dict()
  are the canonical serialisation path.  The SessionStore JSON blob
  gains a 'plan' key; old sessions without 'plan' continue loading fine.

Round type hierarchy (for future extension):
  BaseRoundConfig
    AptitudeRoundConfig
    TechnicalRoundConfig
    CodingRoundConfig
    GDRoundConfig
    CommunicationRoundConfig
    HRRoundConfig
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from interview.round_types import RoundState, RoundType


# ── Individual Turn within a Round ────────────────────────────────────

@dataclass
class RoundTurn:
    """
    One question/answer exchange within a round.

    For free-form rounds (technical, HR, communication):
        question = LLM-generated string
        answer   = candidate free-text

    For aptitude MCQ rounds:
        question        = question text
        options         = list of option strings  (A/B/C/D)
        correct_option  = "A" / "B" / "C" / "D"
        selected_option = candidate's choice
        explanation     = correct-answer rationale

    For voice rounds:
        transcript       = speech-to-text transcript of the candidate's spoken answer
        voice_duration_sec = duration of the voice recording in seconds
        voice_input_mode = "voice" | "text" (how the answer was submitted)

    For coding rounds (future):
        question = problem statement
        answer   = candidate code submission (never executed inline)

    For GD rounds:
        gd_role          = "candidate" | "ai_participant_N" | "moderator"
        gd_persona_name  = display name for AI participants
    """

    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""

    # Free-form answer (technical / HR / communication)
    answer: str = ""

    # MCQ fields (aptitude)
    options: List[str] = field(default_factory=list)
    correct_option: Optional[str] = None
    selected_option: Optional[str] = None
    explanation: Optional[str] = None

    # Coding fields — NO execution inside the API process
    code_submission: Optional[str] = None
    execution_result: Optional[str] = None  # sandbox result injected externally

    # Voice fields (Prompt 4)
    transcript: Optional[str] = None           # STT transcript; None = text answer
    voice_duration_sec: Optional[float] = None # recording duration in seconds
    voice_input_mode: str = "text"             # "voice" | "text"

    # GD fields (Prompt 4)
    gd_role: Optional[str] = None             # "candidate" | "ai_participant" | "moderator"
    gd_persona_name: Optional[str] = None     # display name for AI participant

    # Server-side timing (when the turn question was generated)
    time_started_at: Optional[str] = None     # ISO-8601 UTC timestamp

    # Evaluation (may be None for unevaluated MCQ or pending turns)
    evaluation: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    is_correct: Optional[bool] = None   # for MCQ: graded against correct_option
    score: Optional[float] = None       # 0–10 scale; None = not yet scored

    def to_dict(self) -> dict:
        return {
            "turn_id":           self.turn_id,
            "question":          self.question,
            "answer":            self.answer,
            "options":           self.options,
            "correct_option":    self.correct_option,
            "selected_option":   self.selected_option,
            "explanation":       self.explanation,
            "code_submission":   self.code_submission,
            "execution_result":  self.execution_result,
            "transcript":        self.transcript,
            "voice_duration_sec":self.voice_duration_sec,
            "voice_input_mode":  self.voice_input_mode,
            "gd_role":           self.gd_role,
            "gd_persona_name":   self.gd_persona_name,
            "time_started_at":   self.time_started_at,
            "evaluation":        self.evaluation,
            "is_correct":        self.is_correct,
            "score":             self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoundTurn":
        return cls(
            turn_id            = d.get("turn_id", str(uuid.uuid4())),
            question           = d.get("question", ""),
            answer             = d.get("answer", ""),
            options            = d.get("options", []),
            correct_option     = d.get("correct_option"),
            selected_option    = d.get("selected_option"),
            explanation        = d.get("explanation"),
            code_submission    = d.get("code_submission"),
            execution_result   = d.get("execution_result"),
            transcript         = d.get("transcript"),
            voice_duration_sec = d.get("voice_duration_sec"),
            voice_input_mode   = d.get("voice_input_mode", "text"),
            gd_role            = d.get("gd_role"),
            gd_persona_name    = d.get("gd_persona_name"),
            time_started_at    = d.get("time_started_at"),
            evaluation         = d.get("evaluation", {}),
            is_correct         = d.get("is_correct"),
            score              = d.get("score"),
        )


# ── Round Evaluation (aggregate for one round) ────────────────────────

@dataclass
class RoundEvaluation:
    """
    Aggregate evaluation result for a completed round.

    For MCQ aptitude: score is computed from correct_count/total.
    For LLM-evaluated rounds: score is average of per-turn scores.
    """

    score: float = 0.0          # 0–10
    max_score: float = 10.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    feedback: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)

    # MCQ-specific
    correct_count: Optional[int] = None
    total_questions: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "score":                  self.score,
            "max_score":              self.max_score,
            "strengths":              self.strengths,
            "weaknesses":             self.weaknesses,
            "feedback":               self.feedback,
            "improvement_suggestions": self.improvement_suggestions,
            "correct_count":          self.correct_count,
            "total_questions":        self.total_questions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoundEvaluation":
        return cls(
            score                 = d.get("score", 0.0),
            max_score             = d.get("max_score", 10.0),
            strengths             = d.get("strengths", []),
            weaknesses            = d.get("weaknesses", []),
            feedback              = d.get("feedback", ""),
            improvement_suggestions = d.get("improvement_suggestions", []),
            correct_count         = d.get("correct_count"),
            total_questions       = d.get("total_questions"),
        )


# ── InterviewRound ────────────────────────────────────────────────────

@dataclass
class InterviewRound:
    """
    One round in a multi-round interview session.

    Design for extensibility:
    - round_type drives which agent/behaviour is used
    - config holds round-specific configuration (max_turns, time_limit, etc.)
    - turns accumulates all Q/A exchanges
    - evaluation holds the aggregate result after completion
    - state tracks the lifecycle
    """

    round_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    round_type: RoundType = RoundType.TECHNICAL
    order: int = 1                    # 1-based position in the plan
    title: str = ""
    purpose: str = ""
    difficulty: str = "medium"        # easy / medium / hard
    max_turns: int = 5                # questions/exchanges per round
    time_limit_minutes: Optional[int] = None
    is_required: bool = True

    state: RoundState = RoundState.NOT_STARTED
    turns: List[RoundTurn] = field(default_factory=list)
    evaluation: Optional[RoundEvaluation] = None

    # Server-side round timing (Prompt 4)
    round_started_at: Optional[str] = None   # ISO-8601 UTC when round started

    def __post_init__(self):
        # Ensure title has a sensible default
        if not self.title:
            self.title = RoundType(self.round_type).label()

    # ── State transitions ─────────────────────────────────────────────

    def start(self) -> None:
        """Transition NOT_STARTED → IN_PROGRESS. Records server-side start timestamp."""
        import datetime
        if not self.state.can_transition_to(RoundState.IN_PROGRESS):
            raise ValueError(
                f"Round '{self.title}' cannot start from state '{self.state}'."
            )
        self.state = RoundState.IN_PROGRESS
        self.round_started_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def complete(self) -> None:
        """Transition IN_PROGRESS → COMPLETED."""
        if not self.state.can_transition_to(RoundState.COMPLETED):
            raise ValueError(
                f"Round '{self.title}' cannot complete from state '{self.state}'."
            )
        self.state = RoundState.COMPLETED

    def skip(self) -> None:
        """Transition NOT_STARTED|IN_PROGRESS → SKIPPED."""
        if not self.state.can_transition_to(RoundState.SKIPPED):
            raise ValueError(
                f"Round '{self.title}' cannot be skipped from state '{self.state}'."
            )
        self.state = RoundState.SKIPPED

    def mark_failed(self) -> None:
        """Transition IN_PROGRESS → FAILED (infrastructure error)."""
        if not self.state.can_transition_to(RoundState.FAILED):
            raise ValueError(
                f"Round '{self.title}' cannot fail from state '{self.state}'."
            )
        self.state = RoundState.FAILED

    # ── Queries ───────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_complete(self) -> bool:
        return self.state == RoundState.COMPLETED

    @property
    def is_in_progress(self) -> bool:
        return self.state == RoundState.IN_PROGRESS

    @property
    def questions_remaining(self) -> int:
        return max(0, self.max_turns - self.turn_count)

    # ── Serialisation ─────────────────────────────────────────────────

    def is_time_expired(self) -> bool:
        """
        Check whether the round's time limit has been exceeded.

        Returns True only when:
        - time_limit_minutes is set (> 0)
        - round_started_at is recorded
        - elapsed time exceeds the limit

        Designed for server-side enforcement — not dependent on client timer.
        """
        import datetime
        if not self.time_limit_minutes or not self.round_started_at:
            return False
        try:
            start = datetime.datetime.fromisoformat(self.round_started_at)
            elapsed_minutes = (
                datetime.datetime.now(tz=datetime.timezone.utc) - start
            ).total_seconds() / 60.0
            return elapsed_minutes > self.time_limit_minutes
        except Exception:
            return False

    def elapsed_seconds(self) -> Optional[float]:
        """Return elapsed seconds since this round started, or None if not started."""
        import datetime
        if not self.round_started_at:
            return None
        try:
            start = datetime.datetime.fromisoformat(self.round_started_at)
            return (
                datetime.datetime.now(tz=datetime.timezone.utc) - start
            ).total_seconds()
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "round_id":          self.round_id,
            "round_type":        self.round_type.value,
            "order":             self.order,
            "title":             self.title,
            "purpose":           self.purpose,
            "difficulty":        self.difficulty,
            "max_turns":         self.max_turns,
            "time_limit_minutes":self.time_limit_minutes,
            "is_required":       self.is_required,
            "state":             self.state.value,
            "round_started_at":  self.round_started_at,
            "turns":             [t.to_dict() for t in self.turns],
            "evaluation":        self.evaluation.to_dict() if self.evaluation else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewRound":
        obj = cls(
            round_id          = d.get("round_id", str(uuid.uuid4())),
            round_type        = RoundType(d.get("round_type", "technical")),
            order             = d.get("order", 1),
            title             = d.get("title", ""),
            purpose           = d.get("purpose", ""),
            difficulty        = d.get("difficulty", "medium"),
            max_turns         = d.get("max_turns", 5),
            time_limit_minutes= d.get("time_limit_minutes"),
            is_required       = d.get("is_required", True),
            state             = RoundState(d.get("state", "not_started")),
            round_started_at  = d.get("round_started_at"),
            turns             = [RoundTurn.from_dict(t) for t in d.get("turns", [])],
            evaluation        = (
                RoundEvaluation.from_dict(d["evaluation"])
                if d.get("evaluation") else None
            ),
        )
        return obj


# ── InterviewPlan ─────────────────────────────────────────────────────

@dataclass
class InterviewPlan:
    """
    The ordered sequence of rounds for one interview session.

    An InterviewPlan is attached to an InterviewSession and persisted
    as part of the session JSON in the SessionStore.

    Key invariants:
    - rounds are ordered by their 'order' field (1-based)
    - current_round_index points to the active round (0-based into the list)
    - a round must be completed before the next begins
    - the plan knows the overall progress

    Adding a new round type:
    - No changes to InterviewPlan are needed.
    - Add the new RoundType value and a planner blueprint.
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rounds: List[InterviewRound] = field(default_factory=list)
    current_round_index: int = 0

    # ── Navigation ────────────────────────────────────────────────────

    @property
    def current_round(self) -> Optional[InterviewRound]:
        if 0 <= self.current_round_index < len(self.rounds):
            return self.rounds[self.current_round_index]
        return None

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def completed_rounds(self) -> List[InterviewRound]:
        return [r for r in self.rounds if r.is_complete]

    @property
    def remaining_rounds(self) -> List[InterviewRound]:
        return [
            r for r in self.rounds
            if r.state == RoundState.NOT_STARTED
        ]

    @property
    def is_complete(self) -> bool:
        """True when all required rounds are completed or skipped."""
        for r in self.rounds:
            if r.is_required and r.state == RoundState.NOT_STARTED:
                return False
            if r.is_required and r.state == RoundState.IN_PROGRESS:
                return False
        return True

    @property
    def progress_percent(self) -> int:
        if not self.rounds:
            return 0
        done = sum(
            1 for r in self.rounds
            if r.state in (RoundState.COMPLETED, RoundState.SKIPPED)
        )
        return int((done / len(self.rounds)) * 100)

    # ── Round transitions ─────────────────────────────────────────────

    def start_current_round(self) -> InterviewRound:
        """Start the current round. Returns it."""
        r = self.current_round
        if r is None:
            raise ValueError("No current round to start (plan may be complete).")
        r.start()
        return r

    def advance_to_next_round(self) -> Optional[InterviewRound]:
        """
        Move to the next non-completed round.

        Returns the new current round, or None if the plan is complete.
        Raises ValueError if the current round is still in progress.
        """
        current = self.current_round
        if current and current.state == RoundState.IN_PROGRESS:
            raise ValueError(
                f"Cannot advance: current round '{current.title}' is still in progress."
            )
        # Find the next not-started round
        for i, r in enumerate(self.rounds):
            if r.state == RoundState.NOT_STARTED:
                self.current_round_index = i
                return r
        # All rounds done
        return None

    # ── Aggregate scores for final report ────────────────────────────

    def round_scores(self) -> Dict[str, float]:
        """
        Return {round_type: score} for all completed rounds.
        Only includes rounds that have an evaluation.
        Does not fabricate scores for skipped/not-started rounds.
        """
        scores: Dict[str, float] = {}
        for r in self.rounds:
            if r.is_complete and r.evaluation is not None:
                scores[r.round_type.value] = r.evaluation.score
        return scores

    def overall_score(self) -> float:
        """
        Weighted average of completed round scores.
        Returns 0.0 if no rounds have been evaluated.
        """
        scores = list(self.round_scores().values())
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 2)

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "plan_id":             self.plan_id,
            "rounds":              [r.to_dict() for r in self.rounds],
            "current_round_index": self.current_round_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewPlan":
        return cls(
            plan_id             = d.get("plan_id", str(uuid.uuid4())),
            rounds              = [InterviewRound.from_dict(r) for r in d.get("rounds", [])],
            current_round_index = d.get("current_round_index", 0),
        )
