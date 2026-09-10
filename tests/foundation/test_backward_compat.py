"""
Tests: Backward compatibility — all existing legacy endpoints must remain functional.

Verifies that the production foundation phase did not break:
- POST /interview/question (legacy stateless)
- POST /interview/evaluate (legacy stateless)
- POST /interview/session
- POST /interview/session/{id}/answer
- GET  /interview/session/{id}/summary
- GET  /health

All IBM API calls and RAG are mocked.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base, get_db
from app.main import app
from app.session_store import SessionStore


MOCK_QUESTION = "Describe your experience with distributed systems."

MOCK_EVALUATION = {
    "overall_score": 7,
    "technical_score": 7,
    "relevance_score": 7,
    "clarity_score": 7,
    "communication_score": 7,
    "completeness_score": 7,
    "strengths": ["Good structure"],
    "weaknesses": [],
    "improvement_suggestions": [],
    "evaluation": "Well answered.",
}


@pytest.fixture(autouse=True)
def mock_llm():
    with patch("utils.llm.IBMWatsonxService.generate", return_value=MOCK_QUESTION):
        yield


@pytest.fixture(autouse=True)
def mock_rag():
    with patch("app.interview_engine._get_rag_engine", return_value=None):
        yield


@pytest.fixture()
def client(tmp_path):
    # New SQLAlchemy DB
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_compat.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    # Legacy session store
    store = SessionStore(db_path=tmp_path / "sessions.db")
    import app.session_store as ss_module
    import app.routes as routes_module
    from unittest.mock import patch as _patch

    app.dependency_overrides[get_db] = override_get_db

    with _patch.object(ss_module, "_store", store):
        with _patch.object(routes_module, "get_session_store", lambda: store):
            yield TestClient(app)

    app.dependency_overrides.pop(get_db, None)


# ── Health check ──────────────────────────────────────────────────────

class TestLegacyHealthCheck:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# ── Legacy stateless endpoints ────────────────────────────────────────

class TestLegacyStatelessEndpoints:

    def test_generate_question_still_works(self, client):
        resp = client.post("/interview/question", json={
            "role": "Backend Developer",
            "experience_level": "Mid-level",
            "interview_type": "technical",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "question" in data

    def test_evaluate_answer_still_works(self, client):
        with patch(
            "agents.evaluator_agent.EvaluatorAgent.evaluate",
            return_value=MOCK_EVALUATION,
        ):
            resp = client.post("/interview/evaluate", json={
                "role": "Backend Developer",
                "experience_level": "Mid-level",
                "interview_type": "technical",
                "question": "What is a REST API?",
                "answer": "REST is an architectural style for distributed systems.",
            })
        assert resp.status_code == 200
        assert "overall_score" in resp.json()


# ── Session API backward compatibility ───────────────────────────────

class TestLegacySessionAPI:

    def test_create_session_still_works(self, client):
        resp = client.post("/interview/session", json={
            "candidate_name": "Legacy Tester",
            "role": "DevOps Engineer",
            "experience_level": "Senior",
            "interview_type": "technical",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert "first_question" in data

    def test_submit_answer_still_works(self, client):
        # Create session
        sess_resp = client.post("/interview/session", json={
            "candidate_name": "Legacy Tester",
            "role": "DevOps Engineer",
            "experience_level": "Senior",
            "interview_type": "technical",
        })
        session_id = sess_resp.json()["session_id"]

        with patch(
            "agents.evaluator_agent.EvaluatorAgent.evaluate",
            return_value=MOCK_EVALUATION,
        ):
            ans_resp = client.post(
                f"/interview/session/{session_id}/answer",
                json={
                    "question": MOCK_QUESTION,
                    "answer": "I use Kubernetes and Terraform.",
                },
            )

        assert ans_resp.status_code == 200
        data = ans_resp.json()
        assert "evaluation" in data
        assert data["evaluation"]["overall_score"] == 7

    def test_session_summary_still_works(self, client):
        sess_resp = client.post("/interview/session", json={
            "candidate_name": "Legacy Tester",
            "role": "Data Analyst",
            "experience_level": "Fresher",
            "interview_type": "technical",
        })
        session_id = sess_resp.json()["session_id"]

        with patch(
            "agents.evaluator_agent.EvaluatorAgent.evaluate",
            return_value=MOCK_EVALUATION,
        ):
            client.post(
                f"/interview/session/{session_id}/answer",
                json={"question": MOCK_QUESTION, "answer": "SQL and Python."},
            )

        summary = client.get(f"/interview/session/{session_id}/summary")
        assert summary.status_code == 200
        data = summary.json()
        assert "overall_score" in data
        assert "question_count" in data

    def test_unknown_session_returns_404(self, client):
        resp = client.post(
            "/interview/session/nonexistent-xyz/answer",
            json={"question": "What is Docker?", "answer": "A container platform."},
        )
        assert resp.status_code == 404

    def test_role_is_free_form_string_in_session_api(self, client):
        """Legacy session API still accepts any free-form role."""
        arbitrary_roles = [
            "Quantum Computing Researcher",
            "Healthcare Informatics Specialist",
            "Digital Marketing Lead",
            "Supply Chain Data Analyst",
        ]
        for role in arbitrary_roles:
            resp = client.post("/interview/session", json={
                "candidate_name": "Tester",
                "role": role,
                "experience_level": "Mid-level",
                "interview_type": "technical",
            })
            assert resp.status_code == 201, f"Role {role!r} rejected"
            assert resp.json()["role"] == role
