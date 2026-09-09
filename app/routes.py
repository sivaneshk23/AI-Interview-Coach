"""
API routes for the AI Interview Trainer.

Error handling ensures internal details (including any credential
information) are never returned to callers.

Authentication
--------------
If the API_KEY environment variable is set, all endpoints require
the caller to supply it as a Bearer token:

    Authorization: Bearer <API_KEY>

Leave API_KEY unset (or empty) to disable authentication — useful for
local development.
"""

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from app.config import get_settings
from app.engine_provider import get_engine
from app.schemas import (
    AnswerRequest,
    InterviewRequest,
    SessionSummaryResponse,
    StartSessionRequest,
    SubmitAnswerRequest,
)
from app.session_store import get_session_store
from app.services.evaluation_service import EvaluationService
from app.services.interview_service import InterviewService
from evaluation.performance_report import build_report

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Service singletons ───────────────────────────────────────────────
interview_service = InterviewService()
evaluation_service = EvaluationService()

# ── Auth dependency ──────────────────────────────────────────────────
_MAX_TURNS = 10          # Maximum questions per session
_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"you\s+are\s+now|forget\s+everything|"
    r"system\s*:|<\s*/?system\s*>|"
    r"assistant\s*:|<\s*/?assistant\s*>)",
    re.IGNORECASE,
)


def _configured_api_key() -> Optional[str]:
    """Return the configured API key (reads config each time)."""
    try:
        return get_settings().api_key
    except Exception:
        return None


def _verify_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency that enforces bearer-token auth when API_KEY is set.

    If API_KEY is not configured, all requests pass through.
    Never reveals the actual key in error responses.
    """
    expected = _configured_api_key()
    if expected is None:
        return  # Auth disabled

    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authorization header must use Bearer scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if parts[1] != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )


def _sanitize(text: str, field_name: str = "input") -> str:
    """Raise 422 if text contains a prompt-injection pattern."""
    if _INJECTION_RE.search(text):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content detected in {field_name}.",
        )
    return text


# ── Health endpoint ──────────────────────────────────────────────────

@router.get("/health")
def health():
    """Service liveness check. Returns model and region — never secrets."""
    try:
        settings = get_settings()
        model_info = {
            "model_id": settings.model_id,
            "region": settings.ibm_region,
            "auth_enabled": settings.api_key is not None,
        }
    except Exception:
        model_info = {}

    return {
        "status": "healthy",
        "service": "AI Interview Trainer",
        "version": "1.0.0",
        **model_info,
    }


# ── Stateless question / evaluate endpoints (legacy) ─────────────────

@router.post("/interview/question")
def generate_question(
    request: InterviewRequest,
    _: None = Depends(_verify_auth),
):

    _sanitize(request.role, "role")
    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")

    try:
        return interview_service.generate_question(
            role=request.role,
            experience_level=request.experience_level,
            interview_type=request.interview_type,
            topic=request.topic,
            candidate_context=request.candidate_context or "",
        )

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        logger.exception("Question generation failed")
        raise HTTPException(
            status_code=500,
            detail="Question generation failed. Please try again.",
        )


@router.post("/interview/evaluate")
def evaluate_answer(
    request: AnswerRequest,
    _: None = Depends(_verify_auth),
):

    _sanitize(request.role, "role")
    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")

    try:
        return evaluation_service.evaluate(
            question=request.question,
            answer=request.answer,
            role=request.role,
            experience_level=request.experience_level,
            interview_type=request.interview_type or "technical",
            candidate_context=request.candidate_context or "",
        )

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        logger.exception("Answer evaluation failed")
        raise HTTPException(
            status_code=500,
            detail="Answer evaluation failed. Please try again.",
        )


# ── Session API ───────────────────────────────────────────────────────

@router.post("/interview/session", status_code=201)
def start_session(
    request: StartSessionRequest,
    _: None = Depends(_verify_auth),
):
    """
    Create a persistent interview session and return the first question.

    The returned session_id must be supplied to subsequent
    /interview/session/{id}/answer calls.
    """

    _sanitize(request.role, "role")
    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")

    try:
        engine = get_engine()
        store = get_session_store()

        session = engine.create_session(
            candidate_name=request.candidate_name,
            role=request.role,
            experience_level=request.experience_level,
            interview_type=request.interview_type,
        )

        first_question = engine.get_next_question(
            session=session,
            candidate_context=request.candidate_context or "",
        )

        # Persist session (agents are re-attached on load)
        store.save(session)

        return {
            "session_id": session.session_id,
            "candidate_name": session.candidate_name,
            "role": session.role,
            "experience_level": session.experience_level,
            "interview_type": session.interview_type,
            "first_question": first_question,
            "turn_number": 1,
        }

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        logger.exception("Session creation failed")
        raise HTTPException(
            status_code=500,
            detail="Could not create interview session. Please try again.",
        )


@router.post("/interview/session/{session_id}/answer")
def submit_session_answer(
    session_id: str,
    request: SubmitAnswerRequest,
    _: None = Depends(_verify_auth),
):
    """
    Submit an answer for the current question in a session.

    Returns the evaluation plus the next adaptive question (if the
    session has not yet reached _MAX_TURNS).
    """

    if request.candidate_context:
        _sanitize(request.candidate_context, "candidate_context")

    store = get_session_store()
    session = store.load(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        engine = get_engine()

        # Re-attach agents (not serialised to SQLite)
        from agents.evaluator_agent import EvaluatorAgent
        from agents.interviewer_agent import InterviewerAgent

        if not hasattr(session, "_interviewer"):
            session._interviewer = InterviewerAgent(
                role=session.role,
                interview_type=session.interview_type,
                difficulty="medium",
            )
        if not hasattr(session, "_evaluator"):
            session._evaluator = EvaluatorAgent(
                role=session.role,
                interview_type=session.interview_type,
            )

        evaluation = engine.submit_answer(
            session=session,
            question=request.question,
            answer=request.answer,
            candidate_context=request.candidate_context or "",
        )

        turn_number = session.question_count
        interview_complete = turn_number >= _MAX_TURNS
        next_question = None

        if not interview_complete:
            next_question = engine.get_next_question(
                session=session,
                candidate_context=request.candidate_context or "",
            )

        # Persist updated session
        store.save(session)

        return {
            "turn_number": turn_number,
            "evaluation": evaluation,
            "next_question": next_question,
            "interview_complete": interview_complete,
        }

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        logger.exception("Answer submission failed for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Answer submission failed. Please try again.",
        )


@router.get("/interview/session/{session_id}/summary")
def session_summary(
    session_id: str,
    _: None = Depends(_verify_auth),
):
    """
    Return the aggregate performance report for a completed session.
    """

    store = get_session_store()
    session = store.load(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        report = build_report(session)
        return report

    except Exception:
        logger.exception("Report generation failed for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Report generation failed. Please try again.",
        )
