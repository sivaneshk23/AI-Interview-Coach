from app.interview_engine import InterviewEngine


def test_session_creation():

    engine = InterviewEngine()

    session = engine.create_session(
        candidate_name="Test Candidate",
        role="Data Analyst",
        experience_level="Entry Level",
        interview_type="Technical",
    )

    assert session.candidate_name == "Test Candidate"
    assert session.role == "Data Analyst"
    assert session.question_count == 0