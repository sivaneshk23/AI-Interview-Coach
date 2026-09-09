"""
Interview service.

Delegates question generation to InterviewEngine so the live API
uses InterviewerAgent with RAG context.
"""

from app.engine_provider import get_engine


class InterviewService:

    def generate_question(
        self,
        role: str,
        experience_level: str,
        interview_type: str,
        topic: str | None = None,
        candidate_context: str = "",
    ) -> dict:
        """
        Generate a single interview question for the given role and type.

        The question is produced by InterviewerAgent with RAG context
        injected from the knowledge base.
        """

        engine = get_engine()
        session = engine.create_session(
            candidate_name="candidate",
            role=role,
            experience_level=experience_level,
            interview_type=interview_type,
        )

        question = engine.get_next_question(
            session=session,
            candidate_context=candidate_context,
        )

        return {
            "question": question,
            "category": topic if topic else interview_type,
            "difficulty": experience_level,
        }
