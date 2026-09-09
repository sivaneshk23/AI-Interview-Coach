"""
Evaluation service.

Delegates answer evaluation to InterviewEngine so the live API
uses EvaluatorAgent with RAG context.
"""

import logging

from app.interview_engine import InterviewEngine

logger = logging.getLogger(__name__)

# Module-level engine so the RAG index is loaded once per process.
_engine = InterviewEngine()


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

        session = _engine.create_session(
            candidate_name="candidate",
            role=role,
            experience_level=experience_level,
            interview_type=interview_type,
        )

        return _engine.submit_answer(
            session=session,
            question=question,
            answer=answer,
            candidate_context=candidate_context,
        )
