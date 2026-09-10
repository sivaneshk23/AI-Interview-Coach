"""
Tests: Candidate profile — creation, retrieval, update, confirmation, free-form role.

All tests are fully offline — no IBM API calls.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base, get_db
from app.main import app


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def client(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_candidate.db",
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

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def _register_and_token(client, email="cand@example.com", password="securepassword123"):
    r = client.post("/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── Profile CRUD ──────────────────────────────────────────────────────

class TestCandidateProfile:

    def test_get_profile_returns_404_when_no_profile(self, client):
        token = _register_and_token(client)
        resp = client.get("/candidate/profile", headers=_auth(token))
        assert resp.status_code == 404

    def test_upsert_profile_creates_profile(self, client):
        token = _register_and_token(client)
        resp = client.put("/candidate/profile", headers=_auth(token), json={
            "full_name": "Alice Smith",
            "job_role": "Data Scientist",
            "college": "IIT Delhi",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_name"] == "Alice Smith"
        assert data["job_role"] == "Data Scientist"
        assert data["college"] == "IIT Delhi"
        assert data["is_confirmed"] is False  # not confirmed until explicit confirmation

    def test_get_profile_after_upsert(self, client):
        token = _register_and_token(client)
        client.put("/candidate/profile", headers=_auth(token), json={"full_name": "Bob Jones"})
        resp = client.get("/candidate/profile", headers=_auth(token))
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Bob Jones"

    def test_upsert_is_idempotent(self, client):
        token = _register_and_token(client)
        for _ in range(3):
            resp = client.put("/candidate/profile", headers=_auth(token), json={
                "full_name": "Carol",
                "job_role": "ML Engineer",
            })
            assert resp.status_code == 200
        resp = client.get("/candidate/profile", headers=_auth(token))
        assert resp.json()["full_name"] == "Carol"

    def test_profile_requires_auth(self, client):
        resp = client.get("/candidate/profile")
        assert resp.status_code == 401

    def test_upsert_requires_auth(self, client):
        resp = client.put("/candidate/profile", json={"full_name": "Eve"})
        assert resp.status_code == 401


# ── Role is free-form ─────────────────────────────────────────────────

class TestFreeFormRole:
    """
    The job_role MUST remain a free-form string.
    Any legitimate professional role must be accepted.
    """

    @pytest.mark.parametrize("role", [
        "Data Scientist",
        "AI Engineer",
        "ML Engineer",
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Cloud Engineer",
        "Cybersecurity Analyst",
        "DevOps Engineer",
        "Business Analyst",
        "Product Manager",
        "Financial Analyst",
        "HR Business Partner",
        "Marketing Analyst",
        "Healthcare Data Analyst",
        "Quantitative Researcher",
        "Embedded Systems Engineer",
        "Blockchain Developer",
        "Site Reliability Engineer",
        "Data Engineer",
        "Research Scientist",
        "UX Designer",
        "全栈工程师",   # non-ASCII: should be accepted
        "A" * 200,  # max length
    ])
    def test_any_role_accepted(self, client, role):
        token = _register_and_token(client, email=f"role_{hash(role) % 99999}@test.com")
        resp = client.put("/candidate/profile", headers=_auth(token), json={"job_role": role})
        assert resp.status_code == 200, f"Role {role!r} was rejected: {resp.json()}"

    def test_role_too_long_returns_422(self, client):
        token = _register_and_token(client)
        resp = client.put("/candidate/profile", headers=_auth(token), json={
            "job_role": "X" * 201  # exceeds max_length=200
        })
        assert resp.status_code == 422


# ── Confirmation flow ─────────────────────────────────────────────────

class TestConfirmation:

    def test_confirm_sets_is_confirmed_true(self, client):
        token = _register_and_token(client)
        client.put("/candidate/profile", headers=_auth(token), json={"full_name": "Dana"})
        resp = client.post("/candidate/profile/confirm", headers=_auth(token))
        assert resp.status_code == 200
        assert resp.json()["is_confirmed"] is True

    def test_confirm_without_profile_returns_404(self, client):
        token = _register_and_token(client)
        resp = client.post("/candidate/profile/confirm", headers=_auth(token))
        assert resp.status_code == 404

    def test_update_resets_confirmation(self, client):
        token = _register_and_token(client)
        client.put("/candidate/profile", headers=_auth(token), json={"full_name": "Eve"})
        client.post("/candidate/profile/confirm", headers=_auth(token))
        # Verify confirmed
        r = client.get("/candidate/profile", headers=_auth(token))
        assert r.json()["is_confirmed"] is True
        # Update — should reset confirmation
        client.put("/candidate/profile", headers=_auth(token), json={"full_name": "Eve (updated)"})
        r2 = client.get("/candidate/profile", headers=_auth(token))
        assert r2.json()["is_confirmed"] is False

    def test_candidates_are_isolated(self, client):
        """One candidate cannot access another's profile."""
        t1 = _register_and_token(client, "user1@test.com")
        t2 = _register_and_token(client, "user2@test.com")
        client.put("/candidate/profile", headers=_auth(t1), json={"full_name": "User One"})
        # User 2 has no profile
        resp = client.get("/candidate/profile", headers=_auth(t2))
        assert resp.status_code == 404
