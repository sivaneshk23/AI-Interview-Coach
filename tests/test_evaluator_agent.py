from agents.evaluator_agent import EvaluatorAgent


def test_evaluator_agent():

    evaluator = EvaluatorAgent(
        role="Data Analyst",
        interview_type="technical",
    )

    result = evaluator.evaluate(
        question="What is a Pandas DataFrame?",
        answer="A DataFrame is a two-dimensional labeled data structure in Pandas.",
    )

    assert isinstance(result, dict)
    assert "overall_score" in result
    assert "strengths" in result
    assert "weaknesses" in result
    assert "improvement_suggestions" in result