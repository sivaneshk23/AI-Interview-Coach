"""
Tests: EvaluatorAgent malformed LLM JSON fallback.

Verifies that the agent's _parse_evaluation handles all edge cases
without raising exceptions and always returns a valid dict.
"""

import pytest

from agents.evaluator_agent import EvaluatorAgent


@pytest.fixture
def evaluator():
    return EvaluatorAgent(role="Software Engineer", interview_type="technical")


class TestEvaluatorFallbackParsing:

    def test_valid_json_is_parsed_correctly(self, evaluator):
        raw = """{
            "overall_score": 7,
            "technical_score": 8,
            "relevance_score": 6,
            "clarity_score": 7,
            "communication_score": 7,
            "completeness_score": 6,
            "strengths": ["Good depth"],
            "weaknesses": ["Missing examples"],
            "improvement_suggestions": ["Add code examples"],
            "evaluation": "Solid understanding."
        }"""
        result = evaluator._parse_evaluation(raw)

        assert result["overall_score"] == 7
        assert result["technical_score"] == 8
        assert result["strengths"] == ["Good depth"]
        assert result["weaknesses"] == ["Missing examples"]

    def test_json_inside_markdown_fence_is_parsed(self, evaluator):
        raw = """```json
{
    "overall_score": 5,
    "technical_score": 5,
    "relevance_score": 5,
    "clarity_score": 5,
    "communication_score": 5,
    "completeness_score": 5,
    "strengths": [],
    "weaknesses": [],
    "improvement_suggestions": [],
    "evaluation": "Average."
}
```"""
        result = evaluator._parse_evaluation(raw)
        assert result["overall_score"] == 5

    def test_json_embedded_in_prose_is_extracted(self, evaluator):
        raw = """Here is my evaluation:
{
    "overall_score": 6,
    "technical_score": 6,
    "relevance_score": 6,
    "clarity_score": 6,
    "communication_score": 6,
    "completeness_score": 6,
    "strengths": ["Relevant"],
    "weaknesses": ["Shallow"],
    "improvement_suggestions": ["Go deeper"],
    "evaluation": "Acceptable."
}
Thank you."""
        result = evaluator._parse_evaluation(raw)
        assert result["overall_score"] == 6

    def test_completely_invalid_response_returns_fallback(self, evaluator):
        raw = "I cannot evaluate this answer because it is unclear."
        result = evaluator._parse_evaluation(raw)

        # Fallback must return a valid dict with all expected keys.
        assert isinstance(result, dict)
        assert result["overall_score"] == 0
        assert result["technical_score"] == 0
        assert isinstance(result["strengths"], list)
        assert isinstance(result["weaknesses"], list)
        assert isinstance(result["improvement_suggestions"], list)
        # Raw response is preserved in evaluation field for debugging.
        assert raw in result["evaluation"]

    def test_empty_response_returns_fallback(self, evaluator):
        result = evaluator._parse_evaluation("")
        assert isinstance(result, dict)
        assert result["overall_score"] == 0

    def test_scores_are_clamped_to_0_10(self, evaluator):
        raw = """{
            "overall_score": 15,
            "technical_score": -3,
            "relevance_score": 100,
            "clarity_score": 0,
            "communication_score": 5,
            "completeness_score": 5,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": ""
        }"""
        result = evaluator._parse_evaluation(raw)
        assert result["overall_score"] == 10
        assert result["technical_score"] == 0
        assert result["relevance_score"] == 10

    def test_non_list_fields_are_normalised(self, evaluator):
        raw = """{
            "overall_score": 5,
            "technical_score": 5,
            "relevance_score": 5,
            "clarity_score": 5,
            "communication_score": 5,
            "completeness_score": 5,
            "strengths": "Great answer",
            "weaknesses": null,
            "improvement_suggestions": 42,
            "evaluation": ""
        }"""
        result = evaluator._parse_evaluation(raw)
        assert isinstance(result["strengths"], list)
        assert isinstance(result["weaknesses"], list)
        assert isinstance(result["improvement_suggestions"], list)
