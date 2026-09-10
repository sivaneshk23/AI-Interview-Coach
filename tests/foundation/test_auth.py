"""
Tests: Authentication — registration, login, duplicate prevention, password security.

All tests are fully offline — no IBM API calls, no file I/O beyond a temp DB.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base, get_db
from app.main import app


# ── Test DB fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_engine(tmp_path):
    """Create a fresh in-memory SQLite engine for each test function."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_auth.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def client(db_engine):
    """TestClient with the test DB injected."""
    TestingSession = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


# ── Registration ──────────────────────────────────────────────────────

class TestRegistration:

    def test_register_returns_token(self, client):
        resp = client.post("/auth/register", json={
            "email": "alice@example.com",
            "password": "securepassword123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["email"] == "alice@example.com"
        assert "user_id" in data
        assert isinstance(data["user_id"], int)

    def test_register_does_not_return_password(self, client):
        resp = client.post("/auth/register", json={
            "email": "bob@example.com",
            "password": "securepassword123",
        })
        assert resp.status_code == 201
        response_text = resp.text
        # Password must not appear in response in any form
        assert "securepassword123" not in response_text
        assert "password" not in resp.json()

    def test_register_duplicate_email_returns_409(self, client):
        payload = {"email": "carol@example.com", "password": "securepassword123"}
        r1 = client.post("/auth/register", json=payload)
        assert r1.status_code == 201
        r2 = client.post("/auth/register", json=payload)
        assert r2.status_code == 409

    def test_register_email_is_case_insensitive(self, client):
        """Emails are normalised to lowercase on registration."""
        r1 = client.post("/auth/register", json={
            "email": "Dave@Example.COM",
            "password": "securepassword123",
        })
        assert r1.status_code == 201
        assert r1.json()["email"] == "dave@example.com"

    def test_register_short_password_returns_422(self, client):
        resp = client.post("/auth/register", json={
            "email": "eve@example.com",
            "password": "short",  # < 8 chars
        })
        assert resp.status_code == 422

    def test_register_invalid_email_returns_422(self, client):
        resp = client.post("/auth/register", json={
            "email": "not-an-email",
            "password": "securepassword123",
        })
        assert resp.status_code == 422

    def test_register_missing_password_returns_422(self, client):
        resp = client.post("/auth/register", json={"email": "frank@example.com"})
        assert resp.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────

class TestLogin:

    def _register(self, client, email="grace@example.com", password="securepassword123"):
        r = client.post("/auth/register", json={"email": email, "password": password})
        assert r.status_code == 201
        return r.json()

    def test_login_returns_token(self, client):
        self._register(client)
        resp = client.post("/auth/login", json={
            "email": "grace@example.com",
            "password": "securepassword123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password_returns_401(self, client):
        self._register(client)
        resp = client.post("/auth/login", json={
            "email": "grace@example.com",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    def test_login_wrong_email_returns_401(self, client):
        resp = client.post("/auth/login", json={
            "email": "nobody@example.com",
            "password": "securepassword123",
        })
        assert resp.status_code == 401

    def test_login_does_not_reveal_whether_email_or_password_wrong(self, client):
        """Both wrong-email and wrong-password must return the same message."""
        r1 = client.post("/auth/login", json={
            "email": "nobody@example.com",
            "password": "anypassword123",
        })
        self._register(client)
        r2 = client.post("/auth/login", json={
            "email": "grace@example.com",
            "password": "wrongpassword",
        })
        # Both must be 401 and both must have the same detail message
        assert r1.status_code == 401
        assert r2.status_code == 401
        assert r1.json()["detail"] == r2.json()["detail"]

    def test_login_does_not_return_password(self, client):
        self._register(client)
        resp = client.post("/auth/login", json={
            "email": "grace@example.com",
            "password": "securepassword123",
        })
        assert "securepassword123" not in resp.text
        assert "hashed_password" not in resp.text


# ── /auth/me endpoint ─────────────────────────────────────────────────

class TestMe:

    def test_me_returns_user_info(self, client):
        r = client.post("/auth/register", json={
            "email": "hank@example.com",
            "password": "securepassword123",
        })
        token = r.json()["access_token"]
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "hank@example.com"
        assert data["is_active"] is True
        assert "user_id" in data

    def test_me_without_token_returns_401(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_with_invalid_token_returns_401(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalidtoken"})
        assert resp.status_code == 401


# ── Password hashing ──────────────────────────────────────────────────

class TestPasswordHashing:

    def test_hash_is_not_plaintext(self):
        from app.auth.passwords import hash_password
        h = hash_password("mypassword")
        assert h != "mypassword"
        assert len(h) > 20

    def test_verify_correct_password(self):
        from app.auth.passwords import hash_password, verify_password
        h = hash_password("correct")
        assert verify_password("correct", h) is True

    def test_verify_wrong_password(self):
        from app.auth.passwords import hash_password, verify_password
        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses unique salts — same password yields different hashes."""
        from app.auth.passwords import hash_password
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
