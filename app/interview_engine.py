"""
Interview Engine.

Coordinates the InterviewerAgent and EvaluatorAgent for a complete
interview session, with RAG context injected into every LLM call.
"""

import logging
from uuid import uuid4

from agents.evaluator_agent import EvaluatorAgent
from agents.interviewer_agent import InterviewerAgent
from app.models import InterviewSession

logger = logging.getLogger(__name__)


def _get_rag_engine():
    """
    Lazy-load the RAGEngine to avoid importing sentence-transformers
    at module level (which is slow and unnecessary during tests).
    Returns None if the RAG index or knowledge documents are missing.
    """
    try:
        from rag.rag_engine import RAGEngine
        from app.config import get_settings

        settings = get_settings()
        return RAGEngine(
            knowledge_directory=settings.rag_documents_dir,
            vector_directory=settings.rag_vector_dir,
        )
    except Exception as exc:
        logger.warning(
            "RAGEngine could not be initialised (%s). "
            "Continuing without RAG context.",
            type(exc).__name__,
        )
        return None


class InterviewEngine:
    """
    Coordinates the interviewer and evaluator agents.

    A new pair of agents is created per session so that concurrent
    sessions do not share agent state. RAG context is retrieved once
    per question/evaluation call and injected into the agent prompts.
    """

    def __init__(self):
        self._rag = _get_rag_engine()

    def create_session(
        self,
        candidate_name: str,
        role: str,
        experience_level: str,
        interview_type: str,
    ) -> InterviewSession:

        session = InterviewSession(
            session_id=str(uuid4()),
            candidate_name=candidate_name,
            role=role,
            experience_level=experience_level,
            interview_type=interview_type,
        )

        # Agents are stored on the session object so that concurrent
        # sessions do not overwrite each other's agents.
        session._interviewer = InterviewerAgent(
            role=role,
            interview_type=interview_type,
            difficulty="medium",
        )

        session._evaluator = EvaluatorAgent(
            role=role,
            interview_type=interview_type,
        )

        return session

    def _retrieve_context(
        self,
        session: InterviewSession,
        candidate_context: str = "",
    ) -> str:
        """Return RAG context for the given session, or empty string."""

        if self._rag is None:
            return ""

        try:
            return self._rag.retrieve_context(
                role=session.role,
                experience_level=session.experience_level,
                interview_type=session.interview_type,
                candidate_context=candidate_context,
                top_k=3,
            )
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed (%s). Continuing without context.",
                type(exc).__name__,
            )
            return ""

    def get_next_question(
        self,
        session: InterviewSession,
        candidate_context: str = "",
    ) -> str:

        if not hasattr(session, "_interviewer"):
            raise RuntimeError(
                "Session has no interviewer agent. "
                "Call create_session() first."
            )

        previous_questions = session.questions()
        previous_answers = session.answers()

        previous_question = (
            previous_questions[-1] if previous_questions else ""
        )
        previous_answer = (
            previous_answers[-1] if previous_answers else ""
        )

        rag_context = self._retrieve_context(session, candidate_context)

        return session._interviewer.generate_question(
            candidate_context=candidate_context,
            previous_question=previous_question,
            previous_answer=previous_answer,
            rag_context=rag_context,
        )

    def submit_answer(
        self,
        session: InterviewSession,
        question: str,
        answer: str,
        candidate_context: str = "",
    ) -> dict:

        if not hasattr(session, "_evaluator"):
            raise RuntimeError(
                "Session has no evaluator agent. "
                "Call create_session() first."
            )

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        rag_context = self._retrieve_context(session, candidate_context)

        evaluation = session._evaluator.evaluate(
            question=question,
            answer=answer,
            candidate_context=candidate_context,
            rag_context=rag_context,
        )

        session.add_turn(
            question=question,
            answer=answer,
            evaluation=evaluation,
        )

        return evaluation
