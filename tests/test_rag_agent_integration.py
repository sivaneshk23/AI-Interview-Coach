"""
Tests: RAG → agent context injection.

Verifies that when RAGEngine returns context, it reaches the
InterviewerAgent and EvaluatorAgent prompts.
No live IBM API or real RAG index is required.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestRAGAgentContextInjection:

    def test_interviewer_agent_receives_rag_context(self):
        """
        InterviewerAgent.generate_question builds a prompt that includes
        the rag_context string when one is provided.
        """
        from agents.interviewer_agent import InterviewerAgent

        captured_prompts = []

        def mock_generate(prompt, **kwargs):
            captured_prompts.append(prompt)
            return "Tell me about your Kubernetes experience."

        with patch("agents.interviewer_agent.generate_text", side_effect=mock_generate):
            agent = InterviewerAgent(
                role="Cloud Engineer",
                interview_type="technical",
                difficulty="medium",
            )
            agent.generate_question(
                candidate_context="5 years cloud experience",
                rag_context="[Retrieved Source 1]\nKubernetes is an orchestration platform.",
            )

        assert len(captured_prompts) == 1
        assert "Kubernetes is an orchestration platform" in captured_prompts[0]
        assert "Cloud Engineer" in captured_prompts[0]

    def test_interviewer_agent_works_without_rag_context(self):
        """generate_question works correctly when rag_context is empty."""
        from agents.interviewer_agent import InterviewerAgent

        with patch(
            "agents.interviewer_agent.generate_text",
            return_value="What is a container?",
        ):
            agent = InterviewerAgent(role="Cloud Engineer")
            question = agent.generate_question(rag_context="")

        assert isinstance(question, str)
        assert len(question.strip()) > 0

    def test_evaluator_agent_receives_rag_context(self):
        """
        EvaluatorAgent.evaluate builds a prompt that includes
        the rag_context string when one is provided.
        """
        from agents.evaluator_agent import EvaluatorAgent

        captured_prompts = []

        good_json = """{
            "overall_score": 7,
            "technical_score": 7,
            "relevance_score": 7,
            "clarity_score": 7,
            "communication_score": 7,
            "completeness_score": 7,
            "strengths": ["Good"],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": "Solid answer."
        }"""

        def mock_generate(prompt, **kwargs):
            captured_prompts.append(prompt)
            return good_json

        with patch("agents.evaluator_agent.generate_text", side_effect=mock_generate):
            agent = EvaluatorAgent(role="Cloud Engineer")
            agent.evaluate(
                question="What is a pod?",
                answer="A pod is the smallest deployable unit in Kubernetes.",
                rag_context="[Retrieved Source 1]\nPod definition context.",
            )

        assert len(captured_prompts) == 1
        assert "Pod definition context" in captured_prompts[0]

    def test_engine_passes_rag_context_to_interviewer(self):
        """
        InterviewEngine._retrieve_context result is forwarded to
        InterviewerAgent.generate_question as rag_context.
        """
        from app.interview_engine import InterviewEngine

        fixed_context = "[Retrieved Source 1]\nRelevant knowledge."
        captured = {}

        def mock_generate_question(
            candidate_context="",
            previous_question="",
            previous_answer="",
            rag_context="",
        ):
            captured["rag_context"] = rag_context
            return "A question?"

        with patch("app.interview_engine._get_rag_engine", return_value=None):
            engine = InterviewEngine()

        session = engine.create_session(
            candidate_name="Test",
            role="ML Engineer",
            experience_level="Junior",
            interview_type="technical",
        )
        session._interviewer.generate_question = mock_generate_question

        # Patch _retrieve_context to return our fixed context
        engine._retrieve_context = MagicMock(return_value=fixed_context)

        engine.get_next_question(session, candidate_context="2 years ML exp")

        assert captured.get("rag_context") == fixed_context

    def test_engine_passes_rag_context_to_evaluator(self):
        """
        InterviewEngine._retrieve_context result is forwarded to
        EvaluatorAgent.evaluate as rag_context.
        """
        from app.interview_engine import InterviewEngine

        fixed_context = "[Retrieved Source 1]\nEvaluation knowledge."
        captured = {}

        fixed_eval = {
            "overall_score": 8,
            "technical_score": 8,
            "relevance_score": 8,
            "clarity_score": 8,
            "communication_score": 8,
            "completeness_score": 8,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": "",
        }

        def mock_evaluate(question="", answer="", candidate_context="", rag_context=""):
            captured["rag_context"] = rag_context
            return fixed_eval

        with patch("app.interview_engine._get_rag_engine", return_value=None):
            engine = InterviewEngine()

        session = engine.create_session(
            candidate_name="Test",
            role="ML Engineer",
            experience_level="Junior",
            interview_type="technical",
        )
        session._evaluator.evaluate = mock_evaluate

        engine._retrieve_context = MagicMock(return_value=fixed_context)

        engine.submit_answer(
            session=session,
            question="What is gradient descent?",
            answer="It's an optimization algorithm.",
        )

        assert captured.get("rag_context") == fixed_context
