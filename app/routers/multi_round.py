"""
Multi-round interview session API routes.

These routes are ADDITIVE — they do not replace the existing
/interview/session endpoints (which remain fully operational).

New route prefix: /v2/interview/session

POST /v2/interview/session
    Create a multi-round interview session with an interview plan.
    Returns: session_id, plan summary, first round info, first question.

GET /v2/interview/session/{session_id}
    Return the current state of the session + plan.

POST /v2/interview/session/{session_id}/answer
    Submit an answer for the current question in the current round.
    Returns: evaluation, round_complete, next_question (if any), round_summary (if round done).

POST /v2/interview/session/{session_id}/next-round
    Transition to the next round and return the first question.

POST /v2/interview/session/{session_id}/end-round
    Manually complete the current round early.

GET /v2/interview/session/{session_id}/report
    Return the full multi-round performance report.

Security:
    - All user input is sanitised against prompt-injection patterns
    - No secrets or internal paths are returned in error responses
    - The existing API_KEY bearer auth is applied to all new routes
"""

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.session_store import get_session_store
from interview.engine import MultiRoundEngine
from interview.round_types import RoundType

# Optional JWT auth (for profile auto-population and session linking)
try:
    from app.auth.dependencies import get_current_user
    from app.db.base import get_db
    from app.db.models import CandidateProfile, InterviewSessionLink
    from sqlalchemy.orm import Session
    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/interview", tags=["multi-round-interview"])

# ── Shared injection guard (same pattern as existing routes.py) ───────
_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"you\s+are\s+now|forget\s+everything|"
    r"system\s*:|<\s*/?system\s*>|"
    r"assistant\s*:|<\s*/?assistant\s*>)",
    re.IGNORECASE,
)

def _sanitize(text: str, field_name: str = "input") -> str:
    if _INJECTION_RE.search(text):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content detected in {field_name}.",
        )
    return text


# ── Legacy API_KEY auth (same as existing routes.py) ──────────────────
def _configured_api_key() -> Optional[str]:
    try:
        return get_settings().api_key
    except Exception:
        return None


def _verify_auth(authorization: Optional[str] = Header(default=None)) -> None:
    expected = _configured_api_key()
    if expected is None:
        return
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer scheme required.",
                            headers={"WWW-Authenticate": "Bearer"})
    if parts[1] != expected:
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ── Singleton engine ──────────────────────────────────────────────────
_engine: Optional[MultiRoundEngine] = None

def _get_multi_engine() -> MultiRoundEngine:
    global _engine
    if _engine is None:
        _engine = MultiRoundEngine()
    return _engine


# ── Request/Response schemas ──────────────────────────────────────────

class CreateMultiSessionRequest(BaseModel):
    candidate_name:    str = Field(..., min_length=1, max_length=200)
    role:              str = Field(..., min_length=2, max_length=200,
                                  description="Free-form job role — no role list.")
    experience_level:  str = Field(..., min_length=2, max_length=100)
    interview_type:    str = Field("mixed", max_length=100)
    candidate_context: Optional[str] = Field(None, max_length=2000)
    # If True, auto-populate candidate_context from the confirmed DB profile
    # (requires JWT auth via Authorization header).
    use_profile_context: bool = Field(
        False,
        description="Auto-load confirmed candidate profile as interview context.",
    )


class SubmitRoundAnswerRequest(BaseModel):
    round_id:          str = Field(..., description="ID of the current round.")
    question:          str = Field(..., min_length=5, max_length=2000)
    answer:            str = Field("", max_length=5000)
    candidate_context: Optional[str] = Field(None, max_length=2000)
    selected_option:   Optional[str] = Field(None, max_length=10,
                                             description="MCQ: 'A'/'B'/'C'/'D'.")
    # Voice fields (Prompt 4)
    voice_input_mode:   Optional[str] = Field(
        None, max_length=10,
        description="'voice' or 'text'. Default: 'text'.",
    )
    transcript:         Optional[str] = Field(
        None, max_length=8000,
        description="Speech-to-text transcript of the candidate's spoken answer.",
    )
    voice_duration_sec: Optional[float] = Field(
        None, ge=0, le=600,
        description="Duration of the voice recording in seconds.",
    )


class SubmitGDTurnRequest(BaseModel):
    """Request body for submitting a GD candidate contribution."""
    round_id:          str = Field(..., description="ID of the GD round.")
    answer:            str = Field(..., min_length=1, max_length=5000,
                                   description="Candidate's GD contribution text.")
    candidate_context: Optional[str] = Field(None, max_length=2000)
    # Voice fields
    voice_input_mode:   Optional[str] = Field(None, max_length=10)
    transcript:         Optional[str] = Field(None, max_length=8000)
    voice_duration_sec: Optional[float] = Field(None, ge=0, le=600)


# ── Helpers ───────────────────────────────────────────────────────────

def _plan_summary(session) -> dict:
    """Return a safe summary of the session's plan."""
    plan = getattr(session, "_plan", None)
    if plan is None:
        return {}
    return {
        "plan_id":             plan.plan_id,
        "total_rounds":        plan.total_rounds,
        "completed_rounds":    len(plan.completed_rounds),
        "progress_percent":    plan.progress_percent,
        "plan_complete":       plan.is_complete,
        "rounds": [
            {
                "round_id":   r.round_id,
                "round_type": r.round_type.value,
                "title":      r.title,
                "order":      r.order,
                "state":      r.state.value,
                "max_turns":  r.max_turns,
                "turn_count": r.turn_count,
                "is_required":r.is_required,
                "score":      r.evaluation.score if r.evaluation else None,
            }
            for r in plan.rounds
        ],
    }


def _round_info(round_) -> dict:
    elapsed = round_.elapsed_seconds() if hasattr(round_, "elapsed_seconds") else None
    return {
        "round_id":            round_.round_id,
        "round_type":          round_.round_type.value,
        "title":               round_.title,
        "purpose":             round_.purpose,
        "order":               round_.order,
        "state":               round_.state.value,
        "max_turns":           round_.max_turns,
        "turn_count":          round_.turn_count,
        "is_required":         round_.is_required,
        "time_limit_minutes":  round_.time_limit_minutes,
        "round_started_at":    getattr(round_, "round_started_at", None),
        "elapsed_seconds":     round(elapsed, 1) if elapsed is not None else None,
        "time_expired":        round_.is_time_expired() if hasattr(round_, "is_time_expired") else False,
    }


# ── Routes ────────────────────────────────────────────────────────────

def _link_session_to_user(
    session_id: str,
    role: str,
    experience_level: str,
    interview_type: str,
    authorization: Optional[str],
) -> None:
    """
    If a valid JWT Bearer token is present, record an InterviewSessionLink
    row that ties this session UUID to the authenticated user.

    Failures are swallowed so a bad/missing token never blocks session creation.
    """
    if not _AUTH_AVAILABLE or not authorization:
        return
    try:
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return
        from app.auth.tokens import verify_access_token
        token_data = verify_access_token(parts[1])
        if token_data is None:
            return
        db = next(get_db())
        try:
            link = InterviewSessionLink(
                user_id          = token_data.user_id,
                session_id       = session_id,
                role             = role[:200],
                experience_level = experience_level[:100],
                interview_type   = interview_type[:100],
            )
            db.add(link)
            db.commit()
        except Exception:
            db.rollback()
            logger.debug("InterviewSessionLink insert failed (duplicate or DB error).")
        finally:
            db.close()
    except Exception as exc:
        logger.debug("Session-to-user linking failed: %s", type(exc).__name__)


def _build_candidate_context(
    request: "CreateMultiSessionRequest",
    authorization: Optional[str],
) -> str:
    """
    Determine the candidate_context string for the session.

    Priority:
    1. Explicit candidate_context from request
    2. If use_profile_context=True + JWT token present → load from DB profile
    3. Empty string
    """
    if request.candidate_context:
        return request.candidate_context

    if request.use_profile_context and authorization and _AUTH_AVAILABLE:
        try:
            parts = authorization.split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                from app.auth.tokens import verify_access_token
                token_data = verify_access_token(parts[1])
                if token_data:
                    db = next(get_db())
                    try:
                        profile = (
                            db.query(CandidateProfile)
                            .filter(
                                CandidateProfile.user_id == token_data.user_id,
                                CandidateProfile.is_confirmed == True,
                            )
                            .first()
                        )
                        if profile and profile.interview_context:
                            from app.resume.file_handler import sanitize_resume_text_for_context
                            return sanitize_resume_text_for_context(profile.interview_context)
                    finally:
                        db.close()
        except Exception as exc:
            logger.debug("Profile context auto-load failed: %s", type(exc).__name__)

    return ""


@router.post("/session", status_code=201)
def create_multi_session(
    request: CreateMultiSessionRequest,
    _: None = Depends(_verify_auth),
    authorization: Optional[str] = Header(default=None),
):
    """
    Create a new multi-round interview session.

    Returns the session ID, the generated interview plan,
    the first round details, and the first question.

    If use_profile_context=True and the user is authenticated (JWT Bearer),
    the confirmed candidate profile is automatically used as candidate_context.
    """
    request.role = _sanitize(request.role, "role")
    if request.candidate_context:
        request.candidate_context = _sanitize(request.candidate_context, "candidate_context")

    candidate_context = _build_candidate_context(request, authorization)

    try:
        engine = _get_multi_engine()
        store  = get_session_store()

        session = engine.create_session(
            candidate_name    = request.candidate_name,
            role              = request.role,
            experience_level  = request.experience_level,
            interview_type    = request.interview_type,
            candidate_context = candidate_context,
        )

        # Start first round and get first question
        round_, first_turn = engine.start_next_round(
            session           = session,
            candidate_context = candidate_context,
        )

        store.save(session)

        # Link session to authenticated user (best-effort — never blocks response)
        _link_session_to_user(
            session_id       = session.session_id,
            role             = request.role,
            experience_level = request.experience_level,
            interview_type   = request.interview_type,
            authorization    = authorization,
        )

        return {
            "session_id":      session.session_id,
            "candidate_name":  session.candidate_name,
            "role":            session.role,
            "experience_level":session.experience_level,
            "interview_type":  session.interview_type,
            "plan":            _plan_summary(session),
            "current_round":   _round_info(round_),
            "first_question":  first_turn.question,
            # MCQ fields — only populated for APTITUDE rounds
            "options":         first_turn.options if first_turn.options else None,
            # GD fields — populated when first round is GD
            "gd_role":         getattr(first_turn, "gd_role", None),
            "gd_persona_name": getattr(first_turn, "gd_persona_name", None),
            "round_id":        round_.round_id,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Multi-round session creation failed")
        raise HTTPException(status_code=500, detail="Could not create session. Please try again.")


@router.get("/session/{session_id}")
def get_session_state(
    session_id: str,
    _: None = Depends(_verify_auth),
):
    """Return the current state of the multi-round session."""
    store   = get_session_store()
    session = store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    plan    = getattr(session, "_plan", None)
    current = plan.current_round if plan else None

    return {
        "session_id":       session.session_id,
        "candidate_name":   session.candidate_name,
        "role":             session.role,
        "experience_level": session.experience_level,
        "interview_type":   session.interview_type,
        "turn_count":       session.question_count,
        "plan":             _plan_summary(session),
        "current_round":    _round_info(current) if current else None,
    }


@router.post("/session/{session_id}/answer")
def submit_round_answer(
    session_id: str,
    request: SubmitRoundAnswerRequest,
    _: None = Depends(_verify_auth),
):
    """
    Submit an answer for the current question in the current round.

    Returns the evaluation, whether the round is complete, and the
    next question (if the round is not yet complete).
    """
    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")
    # Sanitise free-text answer for injection patterns (defense-in-depth:
    # evaluator agent also sanitises, but we catch it earlier here)
    if request.answer:
        _sanitize(request.answer, "answer")

    store   = get_session_store()
    session = store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not hasattr(session, "_plan") or session._plan is None:
        raise HTTPException(
            status_code=422,
            detail="This session does not have a multi-round plan. "
                   "Use the v1 /interview/session endpoint.",
        )

    try:
        engine = _get_multi_engine()

        # Attach voice fields to the turn BEFORE calling submit_answer.
        # The engine's submit_answer builds a RoundTurn; we need to pass
        # voice metadata through. We do this via a thin wrapper approach:
        # engine.submit_answer creates the turn then passes to executor —
        # so we inject voice data into the request-level state and let
        # the engine carry it through via new keyword args.
        voice_mode = (request.voice_input_mode or "text").lower()
        transcript = None
        if voice_mode == "voice" and request.transcript:
            transcript = _sanitize(request.transcript[:4000], "transcript")

        turn, round_complete = engine.submit_answer(
            session           = session,
            round_id          = request.round_id,
            question          = request.question,
            answer            = request.answer,
            candidate_context = request.candidate_context or "",
            selected_option   = request.selected_option,
            voice_input_mode  = voice_mode,
            transcript        = transcript,
            voice_duration_sec= request.voice_duration_sec,
        )

        plan    = session._plan
        round_  = engine._find_round(plan, request.round_id)

        # Check server-side round time expiry
        time_expired = round_.is_time_expired()

        # Prepare next question (if round continues and not time-expired)
        next_question = None
        next_options  = None
        next_gd_role  = None
        if not round_complete and not time_expired and round_.questions_remaining > 0:
            next_turn  = engine._generate_question(
                session           = session,
                round_            = round_,
                candidate_context = request.candidate_context or "",
            )
            next_question = next_turn.question
            next_options  = next_turn.options if next_turn.options else None
            next_gd_role  = next_turn.gd_role if hasattr(next_turn, "gd_role") else None

        # MCQ-specific immediate feedback
        mcq_feedback = None
        if round_.round_type.value == "aptitude" and isinstance(turn.evaluation, dict):
            mcq_feedback = {
                "is_correct":     turn.is_correct,
                "correct_option": turn.evaluation.get("correct_option"),
                "explanation":    turn.evaluation.get("explanation", ""),
                "timed_out":      turn.evaluation.get("timed_out", False),
            }

        # Round summary if complete
        round_summary = None
        if round_complete and round_.evaluation:
            round_summary = {
                "round_id":       round_.round_id,
                "title":          round_.title,
                "score":          round_.evaluation.score,
                "feedback":       round_.evaluation.feedback,
                "correct_count":  round_.evaluation.correct_count,
                "total_questions":round_.evaluation.total_questions,
                "strengths":      round_.evaluation.strengths,
                "weaknesses":     round_.evaluation.weaknesses,
                "improvement_suggestions": round_.evaluation.improvement_suggestions,
            }

        store.save(session)

        return {
            "turn_id":          turn.turn_id,
            "evaluation":       turn.evaluation,
            "score":            turn.score,
            "is_correct":       turn.is_correct,
            "mcq_feedback":     mcq_feedback,
            "round_id":         request.round_id,
            "round_complete":   round_complete,
            "next_question":    next_question,
            "next_options":     next_options,
            "next_gd_role":     next_gd_role,
            "round_summary":    round_summary,
            "plan":             _plan_summary(session),
            "plan_complete":    plan.is_complete,
            "time_expired":     time_expired,
            "current_round":    _round_info(round_),
            # Voice response metadata (for frontend display)
            "voice_input_mode": turn.voice_input_mode,
            "voice_duration_sec": turn.voice_duration_sec,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Answer submission failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Answer submission failed. Please try again.")


class AdvanceRoundRequest(BaseModel):
    """Optional body for advancing to the next round."""
    candidate_context: Optional[str] = Field(None, max_length=2000)


@router.post("/session/{session_id}/next-round")
def advance_to_next_round(
    session_id: str,
    request: AdvanceRoundRequest = AdvanceRoundRequest(),
    _: None = Depends(_verify_auth),
):
    """
    Advance to the next round (after the current round is complete).
    Returns the new round's details and its first question.

    The engine's start_next_round() handles plan advancement internally;
    this endpoint does NOT manually advance the plan index.
    """
    store   = get_session_store()
    session = store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not hasattr(session, "_plan") or session._plan is None:
        raise HTTPException(status_code=422, detail="No multi-round plan found.")

    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")

    try:
        engine = _get_multi_engine()
        ctx    = request.candidate_context or ""

        # Check if the plan is already complete before trying to advance
        if session._plan.is_complete:
            return {
                "message":       "All rounds completed. Retrieve the final report.",
                "plan_complete": True,
                "plan":          _plan_summary(session),
            }

        # engine.start_next_round advances the plan index internally
        round_, first_turn = engine.start_next_round(
            session           = session,
            candidate_context = ctx,
        )

        store.save(session)

        return {
            "current_round":   _round_info(round_),
            "first_question":  first_turn.question,
            # MCQ fields — only populated for APTITUDE rounds
            "options":         first_turn.options if first_turn.options else None,
            # GD fields — only populated for GD rounds
            "gd_role":         getattr(first_turn, "gd_role", None),
            "gd_persona_name": getattr(first_turn, "gd_persona_name", None),
            "round_id":        round_.round_id,
            "plan":            _plan_summary(session),
            "plan_complete":   session._plan.is_complete,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Round advance failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Could not advance to next round.")


@router.post("/session/{session_id}/end-round")
def end_current_round(
    session_id: str,
    round_id: str,
    _: None = Depends(_verify_auth),
):
    """
    Manually complete the current round (candidate ends early).
    Builds the round evaluation from turns completed so far.
    """
    store   = get_session_store()
    session = store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not hasattr(session, "_plan") or session._plan is None:
        raise HTTPException(status_code=422, detail="No multi-round plan found.")

    try:
        engine = _get_multi_engine()
        plan   = session._plan
        round_ = engine._find_round(plan, round_id)

        # Tolerate already-completed rounds (idempotent for report generation)
        if round_.is_in_progress:
            engine.complete_round_manually(session=session, round_id=round_id)
            store.save(session)

        return {
            "round_id":       round_id,
            "round_complete": True,
            "score":          round_.evaluation.score if round_.evaluation else None,
            "plan":           _plan_summary(session),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("End-round failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Could not end round.")


@router.get("/session/{session_id}/report")
def get_final_report(
    session_id: str,
    _: None = Depends(_verify_auth),
):
    """Return the full multi-round performance report."""
    store   = get_session_store()
    session = store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not hasattr(session, "_plan") or session._plan is None:
        raise HTTPException(status_code=422, detail="No multi-round plan found.")

    try:
        engine = _get_multi_engine()
        return engine.build_final_report(session)
    except Exception:
        logger.exception("Report generation failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Report generation failed.")


@router.post("/session/{session_id}/gd-turn")
def submit_gd_turn(
    session_id: str,
    request: SubmitGDTurnRequest,
    _: None = Depends(_verify_auth),
):
    """
    Submit a candidate's Group Discussion contribution and get the next AI response.

    This endpoint is specific to GD rounds.  The flow:
    1. Candidate submits their contribution via this endpoint.
    2. The contribution is evaluated by IBM Granite.
    3. The next AI participant response (or moderator closing summary)
       is generated and returned as next_question.

    SAFETY: AI participant turns are bounded by MAX_GD_AI_TURNS.
    No unbounded autonomous AI conversation is possible.
    """
    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")
    _sanitize(request.answer, "answer")

    store   = get_session_store()
    session = store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not hasattr(session, "_plan") or session._plan is None:
        raise HTTPException(status_code=422, detail="No multi-round plan found.")

    try:
        engine = _get_multi_engine()
        plan   = session._plan
        round_ = engine._find_round(plan, request.round_id)

        if round_.round_type.value != "gd":
            raise HTTPException(status_code=422, detail="This endpoint is only for GD rounds.")

        if not round_.is_in_progress:
            raise HTTPException(status_code=422, detail="GD round is not in progress.")

        voice_mode = (request.voice_input_mode or "text").lower()
        transcript = None
        if voice_mode == "voice" and request.transcript:
            transcript = _sanitize(request.transcript[:4000], "transcript")

        turn, round_complete = engine.submit_answer(
            session           = session,
            round_id          = request.round_id,
            question          = round_.turns[-1].question if round_.turns else "GD Opening",
            answer            = request.answer,
            candidate_context = request.candidate_context or "",
            voice_input_mode  = voice_mode,
            transcript        = transcript,
            voice_duration_sec= request.voice_duration_sec,
        )

        # Get the next AI participant response / moderator summary
        next_question = None
        next_gd_role  = None
        next_persona  = None
        if not round_complete:
            next_turn    = engine._generate_question(
                session           = session,
                round_            = round_,
                candidate_context = request.candidate_context or "",
            )
            # Add AI turn to the round's turns so it's tracked
            round_.turns.append(next_turn)
            next_question = next_turn.question
            next_gd_role  = next_turn.gd_role
            next_persona  = next_turn.gd_persona_name

        store.save(session)

        return {
            "turn_id":          turn.turn_id,
            "evaluation":       turn.evaluation,
            "score":            turn.score,
            "round_id":         request.round_id,
            "round_complete":   round_complete,
            "next_question":    next_question,
            "next_gd_role":     next_gd_role,
            "next_persona":     next_persona,
            "plan":             _plan_summary(session),
            "plan_complete":    plan.is_complete,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("GD turn submission failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="GD turn submission failed.")
