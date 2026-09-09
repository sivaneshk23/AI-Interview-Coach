"""
Evaluation service.

Delegates answer evaluation to InterviewEngine so the live API
uses EvaluatorAgent with RAG context.
"""

import logging

from app.engine_provider import get_engine

logger = logging.getLogger(__name__)


class EvaluationService:

    def evaluate(
        self,
        question: str,
        answer: str,
        role: str,
        experience_level: str,
        interview_type: str = "technical",
        candidate_context: str = "",
    ) -> dict:
        """
        Evaluate a candidate's answer using EvaluatorAgent with RAG context.

        Returns a structured evaluation dict with scores, strengths,
        weaknesses, and improvement suggestions.
        """

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        engine = get_engine()
        session = engine.create_session(
            candidate_name="candidate",
            role=role,
            experience_level=experience_level,
            interview_type=interview_type,
        )

        return engine.submit_answer(
            session=session,
            question=question,
            answer=answer,
            candidate_context=candidate_context,
        )
