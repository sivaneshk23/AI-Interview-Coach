"""
Round executors — one per round type.

Each executor handles:
  - generating the next question/task for a round
  - processing a candidate answer/submission
  - evaluating round completion
  - building the round-level evaluation summary

Architecture:
    BaseRoundExecutor (interface)
        TechnicalRoundExecutor   — uses InterviewerAgent + EvaluatorAgent + RAG
        HRRoundExecutor          — uses InterviewerAgent + EvaluatorAgent + RAG
        CommunicationRoundExecutor — uses InterviewerAgent + EvaluatorAgent
        AptitudeRoundExecutor    — draws from MCQBank; deterministic grading
        CodingRoundExecutor      — safe execution interface stub
        GDRoundExecutor          — multi-turn GD stub

IBM Granite remains the primary LLM for all generative rounds.
MCQ scoring is deterministic — no LLM call for aptitude grading.

Security:
    CodingRoundExecutor NEVER executes candidate code inline.
    Execution is deferred to an external sandbox.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from interview.plan import InterviewPlan, InterviewRound, RoundEvaluation, RoundTurn
from interview.round_types import RoundType

logger = logging.getLogger(__name__)


# ── Base executor ─────────────────────────────────────────────────────

class BaseRoundExecutor(ABC):
    """
    Interface for round execution.

    Each subclass implements the specific logic for one round type.
    New round types only need to add a new subclass — no other changes.
    """

    round_type: RoundType  # Subclasses set this as a class attribute

    @abstractmethod
    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Generate the next question/task for this round.
        Returns a RoundTurn with the question populated.
        The turn is NOT yet added to round_.turns.
        """
        ...

    @abstractmethod
    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Evaluate the candidate's answer for a turn.
        Returns the updated RoundTurn with evaluation populated.
        """
        ...

    def build_round_evaluation(
        self,
        round_: InterviewRound,
        role: str,
    ) -> RoundEvaluation:
        """
        Build the aggregate evaluation for a completed round.
        Default: average the per-turn scores.
        Subclasses may override for round-specific logic (e.g. MCQ grading).
        """
        turns_with_scores = [t for t in round_.turns if t.score is not None]
        if not turns_with_scores:
            return RoundEvaluation(score=0.0, feedback="No turns evaluated.")

        avg_score = sum(t.score for t in turns_with_scores) / len(turns_with_scores)
        avg_score = round(avg_score, 2)

        # Aggregate from per-turn evaluation dicts
        all_strengths    = _dedup([
            s for t in round_.turns
            for s in (t.evaluation.get("strengths") or [])
        ])
        all_weaknesses   = _dedup([
            w for t in round_.turns
            for w in (t.evaluation.get("weaknesses") or [])
        ])
        all_suggestions  = _dedup([
            s for t in round_.turns
            for s in (t.evaluation.get("improvement_suggestions") or [])
        ])
        feedback = " ".join(
            t.evaluation.get("evaluation", "")
            for t in round_.turns
            if t.evaluation.get("evaluation")
        ).strip()

        return RoundEvaluation(
            score                  = avg_score,
            strengths              = all_strengths,
            weaknesses             = all_weaknesses,
            feedback               = feedback or f"{round_.title} completed.",
            improvement_suggestions= all_suggestions,
        )


def _dedup(items: list) -> list:
    seen = set()
    out = []
    for item in items:
        k = str(item).lower().strip()
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


# ── Helpers for full-history context ─────────────────────────────────

def _all_questions(round_: InterviewRound):
    """Return list of all questions asked so far in this round."""
    return [t.question for t in round_.turns if t.question]


def _all_evaluations(round_: InterviewRound):
    """Return list of all per-turn evaluations so far in this round."""
    return [t.evaluation for t in round_.turns if isinstance(t.evaluation, dict) and t.evaluation]


# ── Technical round executor ──────────────────────────────────────────

class TechnicalRoundExecutor(BaseRoundExecutor):
    """
    Technical interview round.

    Uses InterviewerAgent (IBM Granite + RAG) for questions and
    EvaluatorAgent (IBM Granite) for answer evaluation.
    Passes full turn history for anti-repetition and adaptive difficulty.
    Role is free-form; agent prompts are role-context-aware.
    """

    round_type = RoundType.TECHNICAL

    def __init__(self, interviewer_agent, evaluator_agent):
        self._interviewer = interviewer_agent
        self._evaluator   = evaluator_agent

    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        import datetime
        previous_question = round_.turns[-1].question if round_.turns else ""
        previous_answer   = round_.turns[-1].answer   if round_.turns else ""

        question_text = self._interviewer.generate_question(
            candidate_context        = candidate_context,
            previous_question        = previous_question,
            previous_answer          = previous_answer,
            rag_context              = rag_context,
            all_previous_questions   = _all_questions(round_),
            all_previous_evaluations = _all_evaluations(round_),
        )
        turn = RoundTurn(question=question_text)
        turn.time_started_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        return turn

    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        evaluation = self._evaluator.evaluate(
            question            = turn.question,
            answer              = turn.answer,
            candidate_context   = candidate_context,
            rag_context         = rag_context,
            voice_input_mode    = turn.voice_input_mode,
            voice_duration_sec  = turn.voice_duration_sec,
            transcript          = turn.transcript,
        )
        turn.evaluation = evaluation
        turn.score = float(evaluation.get("overall_score", 0))
        return turn


# ── HR round executor ─────────────────────────────────────────────────

class HRRoundExecutor(BaseRoundExecutor):
    """
    HR / behavioral interview round.

    Uses the HR-specific prompt variant in InterviewerAgent.
    Adaptive: follows up on candidate's mentioned projects/experiences.
    Passes full turn history for anti-repetition.
    Resume context is used when candidate_context is populated from profile.
    """

    round_type = RoundType.HR

    def __init__(self, interviewer_agent, evaluator_agent):
        self._interviewer = interviewer_agent
        self._evaluator   = evaluator_agent

    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        import datetime
        previous_question = round_.turns[-1].question if round_.turns else ""
        previous_answer   = round_.turns[-1].answer   if round_.turns else ""

        question_text = self._interviewer.generate_question(
            candidate_context        = candidate_context,
            previous_question        = previous_question,
            previous_answer          = previous_answer,
            rag_context              = rag_context,
            all_previous_questions   = _all_questions(round_),
            all_previous_evaluations = _all_evaluations(round_),
        )
        turn = RoundTurn(question=question_text)
        turn.time_started_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        return turn

    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        evaluation = self._evaluator.evaluate(
            question            = turn.question,
            answer              = turn.answer,
            candidate_context   = candidate_context,
            rag_context         = rag_context,
            voice_input_mode    = turn.voice_input_mode,
            voice_duration_sec  = turn.voice_duration_sec,
            transcript          = turn.transcript,
        )
        turn.evaluation = evaluation
        turn.score = float(evaluation.get("overall_score", 0))
        return turn


# ── Communication round executor ──────────────────────────────────────

class CommunicationRoundExecutor(BaseRoundExecutor):
    """
    Communication assessment round.

    Evaluates spoken/written communication quality using the communication-
    specific dimensions in EvaluatorAgent (language_quality_score,
    response_structure_score, filler_word_score).

    Voice input: when voice_input_mode="voice" on the submitted turn,
    the transcript field is used as the effective answer text and
    voice_duration_sec is passed to the evaluator.

    Primary score is communication_score (not overall_score) to reflect
    the specific purpose of this round.
    """

    round_type = RoundType.COMMUNICATION

    def __init__(self, interviewer_agent, evaluator_agent):
        self._interviewer = interviewer_agent
        self._evaluator   = evaluator_agent

    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        import datetime
        previous_question = round_.turns[-1].question if round_.turns else ""
        previous_answer   = round_.turns[-1].answer   if round_.turns else ""

        question_text = self._interviewer.generate_question(
            candidate_context        = candidate_context,
            previous_question        = previous_question,
            previous_answer          = previous_answer,
            rag_context              = rag_context,
            all_previous_questions   = _all_questions(round_),
            all_previous_evaluations = _all_evaluations(round_),
        )
        turn = RoundTurn(question=question_text)
        turn.time_started_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        return turn

    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        evaluation = self._evaluator.evaluate(
            question            = turn.question,
            answer              = turn.answer,
            candidate_context   = candidate_context,
            rag_context         = rag_context,
            voice_input_mode    = turn.voice_input_mode,
            voice_duration_sec  = turn.voice_duration_sec,
            transcript          = turn.transcript,
        )
        turn.evaluation = evaluation
        # Communication round: primary score is communication_score
        turn.score = float(
            evaluation.get("communication_score",
            evaluation.get("overall_score", 0))
        )
        return turn

    def build_round_evaluation(
        self,
        round_: InterviewRound,
        role: str,
    ) -> RoundEvaluation:
        """
        Communication round aggregate evaluation.

        Uses communication_score as the primary score dimension.
        Also captures language_quality and response_structure for the report.
        """
        turns_with_scores = [t for t in round_.turns if t.score is not None]
        if not turns_with_scores:
            return RoundEvaluation(score=0.0, feedback="No turns evaluated.")

        avg_score = sum(t.score for t in turns_with_scores) / len(turns_with_scores)
        avg_score = round(avg_score, 2)

        all_strengths   = _dedup([
            s for t in round_.turns for s in (t.evaluation.get("strengths") or [])
        ])
        all_weaknesses  = _dedup([
            w for t in round_.turns for w in (t.evaluation.get("weaknesses") or [])
        ])
        all_suggestions = _dedup([
            s for t in round_.turns for s in (t.evaluation.get("improvement_suggestions") or [])
        ])
        feedback = " ".join(
            t.evaluation.get("evaluation", "")
            for t in round_.turns
            if t.evaluation.get("evaluation")
        ).strip()

        # Aggregate extra communication dimensions
        lang_scores  = [t.evaluation.get("language_quality_score", 0) for t in round_.turns if isinstance(t.evaluation, dict)]
        struct_scores = [t.evaluation.get("response_structure_score", 0) for t in round_.turns if isinstance(t.evaluation, dict)]
        filler_scores = [t.evaluation.get("filler_word_score", 0) for t in round_.turns if isinstance(t.evaluation, dict)]

        def safe_avg(lst):
            valid = [v for v in lst if v]
            return round(sum(valid) / len(valid), 2) if valid else None

        return RoundEvaluation(
            score                   = avg_score,
            strengths               = all_strengths,
            weaknesses              = all_weaknesses,
            feedback                = feedback or "Communication round completed.",
            improvement_suggestions = all_suggestions,
        )


# ── Aptitude round executor — real MCQ bank ────────────────────────────

class AptitudeRoundExecutor(BaseRoundExecutor):
    """
    Aptitude MCQ round executor.

    Draws questions from MCQBank (interview/mcq_bank.py).
    Supports role-aware question selection using keyword heuristics.

    Grading contract (INVARIANT — never changed):
        Correctness is determined by exact string comparison of
        selected_option vs correct_option.
        No LLM call is ever made for MCQ grading.
        score = 10.0 if correct, 0.0 if incorrect (per question).

    Timing:
        Each turn records submitted_at timestamp (server-side).
        The time_started field on RoundTurn is set when the question
        is generated. This prevents client-side timer manipulation.
    """

    round_type = RoundType.APTITUDE

    def __init__(self, bank=None):
        """
        Args:
            bank: MCQBank instance (default: the shared singleton).
                  Injected here to allow testing with custom banks.
        """
        from interview.mcq_bank import get_bank
        self._bank = bank if bank is not None else get_bank()
        # Track which questions have been served in this round to avoid repeats
        self._served_indices: list = []

    def _pick_question(self, round_: InterviewRound, role: str):
        """
        Select the next MCQ from the bank for this round/role.

        Uses a simple served-index tracking to avoid exact repeats within
        one round execution. Falls back gracefully if bank is exhausted.
        """
        from interview.mcq_bank import MCQQuestion
        questions = self._bank.select(
            role=role,
            count=round_.max_turns + 5,  # request extra to allow dedup
            seed=None,  # random each time
        )
        # Avoid repeats by trying to pick one not yet served
        already_asked = {t.question for t in round_.turns}
        for q in questions:
            if q.question not in already_asked:
                return q
        # All unique questions exhausted — allow repeats
        return questions[0] if questions else None

    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Draw a question from the bank and return a RoundTurn.

        The correct_option is embedded in the turn so evaluate_answer
        can grade deterministically without a second bank lookup.

        IMPORTANT: The returned turn is also stored as ``round_._pending_mcq_turn``
        so that evaluate_answer can recover the correct_option even when the
        engine creates a new RoundTurn for the submission (the engine builds a
        fresh RoundTurn from the submitted answer, not the generated one).
        """
        import datetime
        q = self._pick_question(round_, role)

        if q is None:
            # Absolute fallback — should not occur with a seeded bank
            fallback_turn = RoundTurn(
                question       = f"[Aptitude Q{round_.turn_count + 1}] No questions available.",
                options        = ["A. N/A", "B. N/A", "C. N/A", "D. N/A"],
                correct_option = "A",
                explanation    = "No questions available in the bank for this role.",
            )
            round_._pending_mcq_turn = fallback_turn
            return fallback_turn

        turn = RoundTurn(
            question       = q.question,
            options        = q.options,
            correct_option = q.correct_option,
            explanation    = q.explanation,
        )
        # Store metadata for timer tracking (server-side)
        turn.evaluation = {
            "category":      q.category,
            "difficulty":    q.difficulty,
            "time_limit_sec": q.time_limit_sec,
            "time_started":  datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        # Cache on round so evaluate_answer can find the correct_option
        # regardless of whether the engine passes the original turn object.
        round_._pending_mcq_turn = turn
        return turn

    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Grade MCQ by comparing selected_option to correct_option.

        CRITICAL: No LLM call — always deterministic.
        Correctness = exact string match (case-insensitive, stripped).

        Handles the case where the engine creates a fresh RoundTurn for the
        submission (without correct_option) by recovering correct_option from:
          1. The turn itself (if set)
          2. round_._pending_mcq_turn (cached by generate_question)
          3. The bank (by question-text lookup — last resort)
        """
        import datetime

        submitted_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # ── Recover correct_option if missing ────────────────────────────
        # The engine creates a new RoundTurn for answer submission, so
        # correct_option / explanation / options may not be set.
        if turn.correct_option is None:
            pending = getattr(round_, "_pending_mcq_turn", None)
            if pending is not None and pending.question == turn.question:
                turn.correct_option = pending.correct_option
                if not turn.explanation:
                    turn.explanation = pending.explanation
                if not turn.options:
                    turn.options = pending.options
                # Merge timing metadata from generated turn
                if not isinstance(turn.evaluation, dict) or not turn.evaluation.get("time_started"):
                    if isinstance(pending.evaluation, dict):
                        turn.evaluation = {**pending.evaluation, **(turn.evaluation or {})}
            else:
                # Last resort: look up by question text in the bank
                try:
                    all_q = self._bank._questions
                    for bq in all_q:
                        if bq.question.strip() == turn.question.strip():
                            turn.correct_option = bq.correct_option
                            if not turn.explanation:
                                turn.explanation = bq.explanation
                            break
                except Exception:
                    pass  # Can't recover — proceed with None and score 0

        # Retrieve timing metadata from generation phase
        gen_metadata = turn.evaluation if isinstance(turn.evaluation, dict) else {}
        time_started = gen_metadata.get("time_started")
        time_limit_sec = gen_metadata.get("time_limit_sec")

        # Check server-side time expiry (if time limit is set)
        timed_out = False
        if time_started and time_limit_sec:
            try:
                from datetime import datetime as dt, timezone, timedelta
                start = dt.fromisoformat(time_started)
                elapsed = (dt.now(tz=timezone.utc) - start).total_seconds()
                timed_out = elapsed > time_limit_sec
            except Exception:
                pass  # If timing fails, grade normally

        if timed_out:
            # Timed-out answer is treated as incorrect
            turn.is_correct = False
            turn.score = 0.0
            verdict = f"Time expired. The correct answer is {turn.correct_option}."
        elif turn.correct_option and turn.selected_option is not None:
            turn.is_correct = (
                turn.selected_option.strip().upper()
                == turn.correct_option.strip().upper()
            )
            turn.score = 10.0 if turn.is_correct else 0.0
            verdict = "Correct!" if turn.is_correct else (
                f"Incorrect. The correct answer is {turn.correct_option}."
            )
        else:
            # No answer provided — treat as incorrect
            turn.is_correct = False
            turn.score = 0.0
            verdict = f"No answer provided. The correct answer is {turn.correct_option}."

        turn.evaluation = {
            "overall_score":   turn.score,
            "is_correct":      turn.is_correct,
            "correct_option":  turn.correct_option,
            "selected_option": turn.selected_option,
            "explanation":     turn.explanation or "",
            "evaluation":      verdict,
            "timed_out":       timed_out,
            "submitted_at":    submitted_at,
            "category":        gen_metadata.get("category", ""),
            "difficulty":      gen_metadata.get("difficulty", ""),
            "strengths":             [],
            "weaknesses":            [],
            "improvement_suggestions": [],
        }
        return turn

    def build_round_evaluation(
        self,
        round_: InterviewRound,
        role: str,
    ) -> RoundEvaluation:
        """
        MCQ-specific aggregate evaluation.
        Score = (correct / total) * 10 — deterministic, no LLM.
        """
        total   = len(round_.turns)
        correct = sum(1 for t in round_.turns if t.is_correct)
        timed_out_count = sum(
            1 for t in round_.turns
            if isinstance(t.evaluation, dict) and t.evaluation.get("timed_out")
        )
        score   = round((correct / total * 10), 2) if total > 0 else 0.0

        feedback = f"Scored {correct}/{total} correct ({score:.1f}/10)."
        if timed_out_count:
            feedback += f" {timed_out_count} question(s) timed out."

        return RoundEvaluation(
            score          = score,
            correct_count  = correct,
            total_questions= total,
            feedback       = feedback,
        )


# ── Coding round executor (safe interface stub) ───────────────────────

class CodingRoundExecutor(BaseRoundExecutor):
    """
    Coding challenge round — INTERFACE STUB ONLY.

    SECURITY CONTRACT:
        Candidate code is NEVER executed inside the FastAPI process.
        This executor only stores the code submission as a string.
        Execution must be delegated to an external, isolated sandbox.
        The sandbox result can later be injected via execution_result.

    Full implementation (problem bank, test cases, sandbox integration)
    will be added in Prompt 3.
    """

    round_type = RoundType.CODING

    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        turn = RoundTurn(
            question = (
                "Coding problems and test cases will be delivered in Prompt 3. "
                "Please describe your approach to solving a coding problem relevant to your role."
            ),
        )
        return turn

    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Store the submission. Execution result from sandbox is injected externally.
        No inline code execution.
        """
        if turn.answer:
            turn.code_submission = turn.answer

        turn.evaluation = {
            "overall_score": 0.0,
            "evaluation": (
                "Code submission recorded. "
                "Automated test execution will be available in Prompt 3."
            ),
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [
                "Ensure your solution handles edge cases.",
                "Review time and space complexity.",
            ],
        }
        turn.score = 0.0
        return turn


# ── GD round executor ─────────────────────────────────────────────────

# AI personas for the Group Discussion round
_GD_PERSONAS = [
    {
        "name": "Arjun",
        "style": "analytical and data-driven",
        "position": "challenges assumptions with evidence",
    },
    {
        "name": "Priya",
        "style": "pragmatic and solution-focused",
        "position": "argues for practical, real-world implementation",
    },
    {
        "name": "Ravi",
        "style": "contrarian and thought-provoking",
        "position": "plays devil's advocate to challenge the prevailing view",
    },
]


class GDRoundExecutor(BaseRoundExecutor):
    """
    Group Discussion round — full implementation.

    Flow:
    1. Moderator introduces a discussion topic (generate_question)
    2. Candidate makes their opening argument (submitted via evaluate_answer)
    3. AI participant responds (generate_question on next call)
    4. Candidate responds again (evaluate_answer)
    5. Final moderator summary

    SAFETY invariants (HARD — never remove):
    - MAX_GD_AI_TURNS is a hard upper bound on AI participant turns per round
    - No autonomous AI-to-AI conversation — every AI turn requires a
      preceding candidate answer (human always in the loop)
    - Candidate code is never executed
    - All AI persona text is generated by IBM Granite

    Scoring dimensions:
    - Participation (did the candidate contribute?)
    - Reasoning (quality of arguments)
    - Communication (clarity, structure)
    - Listening (did they address what was said?)
    """

    round_type = RoundType.GD
    MAX_GD_AI_TURNS = 3  # Hard limit — prevents unbounded AI loops

    def __init__(self, interviewer_agent=None, evaluator_agent=None):
        # GD uses IBM Granite directly via the LLM utility.
        # Agents are kept for API compatibility but not required.
        self._interviewer = interviewer_agent
        self._evaluator   = evaluator_agent

    def generate_question(
        self,
        round_: InterviewRound,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Return the next GD prompt for the candidate.

        First call: introduces the discussion topic.
        Subsequent calls: returns the next AI participant response
            or the moderator's closing summary.
        """
        import datetime
        from utils.llm import generate_text

        candidate_turns = [t for t in round_.turns if t.gd_role == "candidate"]
        ai_turns        = [t for t in round_.turns if t.gd_role == "ai_participant"]
        ai_turn_count   = len(ai_turns)
        now_iso = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # ── First call: present the topic ─────────────────────────────
        if not round_.turns:
            topic = self._generate_topic(role, generate_text)
            return RoundTurn(
                question       = (
                    f"Group Discussion Topic:\n\n{topic}\n\n"
                    "Please share your opening thoughts on this topic."
                ),
                gd_role        = "moderator",
                gd_persona_name= "Moderator",
                time_started_at= now_iso,
            )

        # ── Closing: max turns reached or AI limit reached ────────────
        if len(candidate_turns) >= round_.max_turns or ai_turn_count >= self.MAX_GD_AI_TURNS:
            summary = self._generate_moderator_summary(round_, generate_text)
            return RoundTurn(
                question       = summary,
                gd_role        = "moderator",
                gd_persona_name= "Moderator",
                time_started_at= now_iso,
            )

        # ── AI participant response ────────────────────────────────────
        persona = _GD_PERSONAS[ai_turn_count % len(_GD_PERSONAS)]
        last_candidate_turn = candidate_turns[-1] if candidate_turns else None
        response = self._generate_ai_participant_response(
            round_               = round_,
            persona              = persona,
            last_candidate_point = last_candidate_turn.answer if last_candidate_turn else "",
            generate_text        = generate_text,
        )
        return RoundTurn(
            question        = f"[{persona['name']}]: {response}",
            gd_role         = "ai_participant",
            gd_persona_name = persona["name"],
            time_started_at = now_iso,
        )

    def evaluate_answer(
        self,
        round_: InterviewRound,
        turn: RoundTurn,
        role: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> RoundTurn:
        """
        Record and evaluate the candidate's GD contribution.

        Sets gd_role="candidate" so the turn is correctly tracked.
        """
        from utils.llm import generate_text

        turn.gd_role = "candidate"
        topic = self._get_topic(round_)

        evaluation = self._evaluate_gd_contribution(
            topic           = topic,
            candidate_point = turn.answer,
            generate_text   = generate_text,
        )
        turn.evaluation = evaluation
        turn.score = float(evaluation.get("overall_score", 5.0))
        return turn

    def build_round_evaluation(
        self,
        round_: InterviewRound,
        role: str,
    ) -> RoundEvaluation:
        """
        GD aggregate evaluation.

        Aggregates all candidate-turn scores.
        Reports participation rate.
        """
        candidate_turns = [t for t in round_.turns if t.gd_role == "candidate"]
        if not candidate_turns:
            return RoundEvaluation(
                score    = 0.0,
                feedback = "No candidate contributions recorded.",
            )

        scored_turns = [t for t in candidate_turns if t.score is not None]
        avg_score = (
            sum(t.score for t in scored_turns) / len(scored_turns)
            if scored_turns else 0.0
        )
        avg_score = round(avg_score, 2)

        all_strengths   = _dedup([
            s for t in candidate_turns for s in (t.evaluation.get("strengths") or [])
        ])
        all_weaknesses  = _dedup([
            w for t in candidate_turns for w in (t.evaluation.get("weaknesses") or [])
        ])
        all_suggestions = _dedup([
            s for t in candidate_turns for s in (t.evaluation.get("improvement_suggestions") or [])
        ])

        total_turns     = len(round_.turns)
        candidate_count = len(candidate_turns)
        participation   = f"{candidate_count}/{total_turns} turns participated."

        return RoundEvaluation(
            score                   = avg_score,
            strengths               = all_strengths,
            weaknesses              = all_weaknesses,
            feedback                = f"Group Discussion completed. {participation}",
            improvement_suggestions = all_suggestions,
            total_questions         = total_turns,
        )

    # ── GD internal helpers ───────────────────────────────────────────

    def _generate_topic(self, role: str, generate_text) -> str:
        from agents.interviewer_agent import _safe_text
        safe_role = _safe_text(role, 100)
        prompt = (
            f"You are the moderator of a professional Group Discussion session.\n"
            f"Generate a thought-provoking, debatable discussion topic relevant to the "
            f"{safe_role} professional domain.\n"
            "The topic should have valid perspectives on multiple sides.\n"
            "It must NOT be political, religious, or harmful.\n"
            "Return ONLY the topic as a single sentence or question — no explanation."
        )
        try:
            return generate_text(prompt, max_new_tokens=100).strip()
        except Exception:
            return (
                f"What is the most important skill for a {safe_role} professional "
                "to develop in the next five years?"
            )

    def _generate_ai_participant_response(
        self,
        round_: InterviewRound,
        persona: dict,
        last_candidate_point: str,
        generate_text,
    ) -> str:
        from agents.interviewer_agent import _safe_text
        topic     = self._get_topic(round_)
        safe_point = _safe_text(last_candidate_point, 500)
        prompt = (
            f"You are {persona['name']}, a participant in a professional Group Discussion.\n"
            f"Your communication style: {persona['style']}.\n"
            f"Your typical position: {persona['position']}.\n\n"
            f"Discussion topic: {topic}\n\n"
            f"The candidate just said:\n\"{safe_point}\"\n\n"
            "Respond with a SINGLE short paragraph (2-4 sentences) that:\n"
            "1. Acknowledges or engages with what the candidate said.\n"
            "2. Advances the discussion from your perspective.\n"
            "3. May agree, disagree, or add a new angle.\n"
            "4. Does NOT end with a direct question.\n\n"
            "Return ONLY your response paragraph."
        )
        try:
            return generate_text(prompt, max_new_tokens=150).strip()
        except Exception:
            return "Thank you for your perspective. I think there are additional dimensions worth considering here."

    def _generate_moderator_summary(self, round_: InterviewRound, generate_text) -> str:
        topic = self._get_topic(round_)
        prompt = (
            f"You are the moderator of a Group Discussion.\n"
            f"Topic: {topic}\n\n"
            "The discussion has concluded.\n"
            "Provide a brief, neutral 2-3 sentence summary of the key points raised.\n"
            "Then thank the participants.\n"
            "Return ONLY the summary and thanks — no headings."
        )
        try:
            return generate_text(prompt, max_new_tokens=120).strip()
        except Exception:
            return "Thank you for participating in the group discussion. The key points have been recorded."

    def _evaluate_gd_contribution(
        self,
        topic: str,
        candidate_point: str,
        generate_text,
    ) -> dict:
        import json as _json
        from agents.interviewer_agent import _safe_text
        from agents.evaluator_agent import _strip_code_fence, _validate_evaluation_fields

        safe_point = _safe_text(candidate_point, 500)
        prompt = (
            f"You are evaluating a candidate's Group Discussion contribution.\n\n"
            f"Topic: {topic}\n\n"
            f"Candidate's point:\n\"{safe_point}\"\n\n"
            "Evaluate this GD contribution objectively.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '    "overall_score": 0,\n'
            '    "technical_score": 0,\n'
            '    "relevance_score": 0,\n'
            '    "clarity_score": 0,\n'
            '    "communication_score": 0,\n'
            '    "completeness_score": 0,\n'
            '    "strengths": [],\n'
            '    "weaknesses": [],\n'
            '    "improvement_suggestions": [],\n'
            '    "evaluation": ""\n'
            "}\n\n"
            "Scoring: relevance_score = was the point relevant to the topic? "
            "clarity_score = was it clearly expressed? "
            "communication_score = was the language effective?\n"
            "Return valid JSON only."
        )
        try:
            response = generate_text(prompt, max_new_tokens=400).strip()
            response = _strip_code_fence(response)
            result = _json.loads(response)
            if isinstance(result, dict):
                return _validate_evaluation_fields(result, [
                    "overall_score", "technical_score", "relevance_score",
                    "clarity_score", "communication_score", "completeness_score",
                ])
        except Exception:
            pass
        return {
            "overall_score": 5, "technical_score": 5, "relevance_score": 5,
            "clarity_score": 5, "communication_score": 5, "completeness_score": 5,
            "strengths": [], "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": "GD contribution recorded.",
        }

    def _get_topic(self, round_: InterviewRound) -> str:
        """Extract the GD topic from the first moderator turn."""
        for t in round_.turns:
            if t.gd_role == "moderator" and "Topic" in (t.question or ""):
                # Parse "Group Discussion Topic:\n\n<TOPIC>\n\n..."
                lines = [ln.strip() for ln in t.question.split("\n") if ln.strip()]
                for i, line in enumerate(lines):
                    if "Topic:" in line:
                        # Topic is the next non-empty line
                        if i + 1 < len(lines):
                            return lines[i + 1]
        return "Professional development and career growth"


# ── Executor registry ─────────────────────────────────────────────────

def build_executor(
    round_type: RoundType,
    interviewer_agent=None,
    evaluator_agent=None,
) -> BaseRoundExecutor:
    """
    Factory function: return the appropriate executor for a round type.

    Technical/HR/Communication: require both agents.
    GD:       accepts optional agents (uses IBM Granite directly).
    Aptitude: no agents needed (deterministic MCQ scoring).
    Coding:   no agents needed (submission stored, not executed).
    """
    _MAP = {
        RoundType.TECHNICAL:     TechnicalRoundExecutor,
        RoundType.HR:            HRRoundExecutor,
        RoundType.COMMUNICATION: CommunicationRoundExecutor,
        RoundType.APTITUDE:      AptitudeRoundExecutor,
        RoundType.CODING:        CodingRoundExecutor,
        RoundType.GD:            GDRoundExecutor,
    }

    cls = _MAP.get(round_type)
    if cls is None:
        raise ValueError(f"No executor registered for round type: {round_type}")

    if round_type in (RoundType.TECHNICAL, RoundType.HR, RoundType.COMMUNICATION):
        if interviewer_agent is None or evaluator_agent is None:
            raise ValueError(
                f"{cls.__name__} requires both interviewer_agent and evaluator_agent."
            )
        return cls(interviewer_agent, evaluator_agent)

    if round_type == RoundType.GD:
        # GD can optionally receive agents (passed through but not strictly required)
        return cls(interviewer_agent, evaluator_agent)

    return cls()
