"""
Tests: Resume integration — new endpoints added in Prompt 3.

New endpoints tested:
  - POST /resume/{id}/extract-profile  (with mocked LLM)
  - DELETE /resume/{id}                (soft-delete + authorization)

All tests are fully offline — no IBM API calls, no network I/O.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base, get_db
from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────

def _real_docx_bytes(text: str = "Python SQL Machine Learning") -> bytes:
    import docx
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def client(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_resume_p3.db",
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


def _register_and_token(client, email="user@example.com"):
    r = client.post("/auth/register", json={"email": email, "password": "securepassword123"})
    assert r.status_code == 201
    return r.json()["access_token"]


def _upload_docx(client, token, text="Python SQL Machine Learning"):
    docx_bytes = _real_docx_bytes(text)
    r = client.post(
        "/resume/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                "resume.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert r.status_code == 201
    return r.json()["resume_id"]


# ── extract-profile endpoint ──────────────────────────────────────────

class TestExtractProfileEndpoint:

    def test_requires_auth(self, client):
        r = client.post("/resume/999/extract-profile")
        assert r.status_code == 401

    def test_returns_404_for_nonexistent_resume(self, client):
        token = _register_and_token(client, "noresume@test.com")
        r = client.post(
            "/resume/99999/extract-profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404

    def test_returns_404_for_other_users_resume(self, client):
        t1 = _register_and_token(client, "owner@test.com")
        t2 = _register_and_token(client, "other@test.com")
        rid = _upload_docx(client, t1)
        r = client.post(
            f"/resume/{rid}/extract-profile",
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert r.status_code == 404

    def test_extract_profile_unavailable_when_llm_down(self, client):
        """When LLM is unavailable, endpoint returns 200 with status='unavailable'."""
        token = _register_and_token(client, "extract@test.com")
        rid = _upload_docx(client, token, text="Alice Smith | Python | IIT Delhi")

        with patch(
            "app.services.profile_extraction_service._call_llm",
            return_value=None,
        ):
            r = client.post(
                f"/resume/{rid}/extract-profile",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "unavailable"
        assert data["resume_id"] == rid

    def test_extract_profile_success_with_mocked_llm(self, client):
        """When LLM returns valid JSON, endpoint returns extracted fields."""
        token = _register_and_token(client, "extractok@test.com")
        rid = _upload_docx(client, token, text="Alice Smith Python Developer IIT Delhi")

        mock_json = """{
            "full_name": "Alice Smith",
            "college": "IIT Delhi",
            "degree": "B.Tech",
            "branch": "Computer Science",
            "graduation_year": 2024,
            "skills": "Python, Django, REST APIs",
            "experience": "Intern at TCS",
            "suggested_role": "Backend Developer"
        }"""

        with patch(
            "app.services.profile_extraction_service._call_llm",
            return_value=mock_json,
        ):
            r = client.post(
                f"/resume/{rid}/extract-profile",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("success", "partial")
        assert data["full_name"] == "Alice Smith"
        assert data["college"] == "IIT Delhi"
        assert data["graduation_year"] == 2024
        assert "Python" in data["skills"]
        # storage_path must NOT appear
        assert "storage_path" not in data
        assert "hashed_password" not in data

    def test_extract_profile_response_schema(self, client):
        """Response must include required fields."""
        token = _register_and_token(client, "schema@test.com")
        rid = _upload_docx(client, token)

        with patch(
            "app.services.profile_extraction_service._call_llm",
            return_value=None,
        ):
            r = client.post(
                f"/resume/{rid}/extract-profile",
                headers={"Authorization": f"Bearer {token}"},
            )

        data = r.json()
        required = {"resume_id", "status", "message"}
        for key in required:
            assert key in data, f"Missing key: {key}"


# ── DELETE /resume/{id} ───────────────────────────────────────────────

class TestResumeDelete:

    def test_requires_auth(self, client):
        r = client.delete("/resume/999")
        assert r.status_code == 401

    def test_delete_own_resume(self, client):
        token = _register_and_token(client, "del@test.com")
        rid = _upload_docx(client, token)

        r = client.delete(
            f"/resume/{rid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 204

    def test_deleted_resume_not_in_list(self, client):
        token = _register_and_token(client, "dellst@test.com")
        rid = _upload_docx(client, token)

        # Verify it's listed before deletion
        list_r = client.get("/resume/", headers={"Authorization": f"Bearer {token}"})
        assert any(r["resume_id"] == rid for r in list_r.json())

        # Delete
        client.delete(f"/resume/{rid}", headers={"Authorization": f"Bearer {token}"})

        # Should no longer appear in list
        list_r2 = client.get("/resume/", headers={"Authorization": f"Bearer {token}"})
        assert not any(r["resume_id"] == rid for r in list_r2.json())

    def test_deleted_resume_extracted_text_inaccessible(self, client):
        token = _register_and_token(client, "delext@test.com")
        rid = _upload_docx(client, token)

        client.delete(f"/resume/{rid}", headers={"Authorization": f"Bearer {token}"})

        r = client.get(
            f"/resume/{rid}/extracted",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404

    def test_cannot_delete_other_users_resume(self, client):
        t1 = _register_and_token(client, "owner2@test.com")
        t2 = _register_and_token(client, "thief2@test.com")
        rid = _upload_docx(client, t1)

        r = client.delete(
            f"/resume/{rid}",
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert r.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        token = _register_and_token(client, "dne@test.com")
        r = client.delete(
            "/resume/99999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404

    def test_double_delete_returns_404(self, client):
        """Soft-deleting an already-deleted resume returns 404."""
        token = _register_and_token(client, "dbl@test.com")
        rid = _upload_docx(client, token)

        client.delete(f"/resume/{rid}", headers={"Authorization": f"Bearer {token}"})
        r = client.delete(f"/resume/{rid}", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 404

    def test_delete_extract_profile_returns_404(self, client):
        """extract-profile on deleted resume returns 404."""
        token = _register_and_token(client, "delprof@test.com")
        rid = _upload_docx(client, token)

        client.delete(f"/resume/{rid}", headers={"Authorization": f"Bearer {token}"})

        r = client.post(
            f"/resume/{rid}/extract-profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404
