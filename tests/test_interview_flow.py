from app.interview_engine import InterviewEngine


def test_complete_interview_flow():

    engine = InterviewEngine()

    # Create an interview session
    session = engine.create_session(
        candidate_name="Test Candidate",
        role="Data Analyst",
        experience_level="Fresher",
        interview_type="technical",
    )

    assert session.session_id
    assert session.role == "Data Analyst"
    assert session.experience_level == "Fresher"
    assert session.interview_type == "technical"

    # Generate first question
    question = engine.get_next_question(session)

    assert isinstance(question, str)
    assert len(question.strip()) > 0

    # Simulate candidate answer
    answer = """
    Pandas is a Python library commonly used for data manipulation
    and analysis. It provides DataFrame and Series structures and
    supports filtering, grouping, merging and aggregation.
    """

    # Evaluate answer and store the turn
    evaluation = engine.submit_answer(
        session=session,
        question=question,
        answer=answer,
        candidate_context="Candidate is a B.Tech AI and Data Science fresher.",
    )

    assert isinstance(evaluation, dict)

    # The turn should now be stored
    assert len(session.questions()) == 1
    assert len(session.answers()) == 1

    # Engine should be able to generate another question
    next_question = engine.get_next_question(session)

    assert isinstance(next_question, str)
    assert len(next_question.strip()) > 0