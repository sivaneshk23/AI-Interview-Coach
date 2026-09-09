from agents.interviewer_agent import InterviewerAgent


def test_interviewer_agent():
    agent = InterviewerAgent(
        role="Data Analyst",
        interview_type="technical",
        difficulty="medium",
    )

    question = agent.generate_question(
        candidate_context="Python, Pandas, SQL and Power BI",
    )

    assert isinstance(question, str)
    assert len(question.strip()) > 0