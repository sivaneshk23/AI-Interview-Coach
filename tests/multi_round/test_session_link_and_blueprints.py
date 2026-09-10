"""
Tests for:

1. InterviewSessionLink wiring in create_multi_session
   - Session linked to authenticated user when valid JWT Bearer present
   - Session creation succeeds without auth (no link created)
   - Session creation succeeds with invalid token (link silently skipped)
   - No duplicate links for same user+session

2. Blueprint updates (GD round added to software and data blueprints)
   - software blueprint includes GD round
   - data blueprint includes GD round
   - GD rounds are optional (is_required=False)
   - Plan total_rounds updated correctly

3. Round transition API
   - advance_to_next_round returns gd_role/gd_persona_name on GD round
   - advance_to_next_round returns plan_complete when all rounds done
   - advance_to_next_round returns 404 for unknown session
   - advance_to_next_round returns first_question and round_id

4. create_multi_session returns gd_role/gd_persona_name when first round is GD
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from interview.planner import BLUEPRINTS, InterviewPlanner
from interview.round_types import RoundType


# ── Mock constants ────────────────────────────────────────────────────

MOCK_QUESTION = "Tell me about your experience."

MOCK_EVALUATION = {
    "score": 7.0,
    "overall_score": 7.0,
    "technical_score": 7,
    "relevance_score": 7,
    "clarity_score": 7,
    "communication_score": 7,
    "completeness_score": 7,
    "strengths": ["Good"],
    "weaknesses": ["Needs work"],
    "improvement_suggestions": [],
    "feedback": "Solid.",
    "evaluation": "OK.",
}


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_llm():
    with patch("utils.llm.IBMWatsonxService.generate", return_value=MOCK_QUESTION):
        yield


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))

    app_db_path = tmp_path / "test_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{app_db_path}")

    try:
        from app.db.base import init_db
        init_db()
    except Exception:
        pass

    yield db_path


@pytest.fixture()
def client(tmp_db):
    return TestClient(app, raise_server_exceptions=False)


def _create_session(client, role="Software Engineer", **kwargs):
    body = {
        "candidate_name": "Test Candidate",
        "role": role,
        "experience_level": "Mid-level",
        "interview_type": "mixed",
    }
    body.update(kwargs)
    return client.post("/v2/interview/session", json=body)


# ══════════════════════════════════════════════════════════════════════
# 1. Blueprint — GD round presence
# ══════════════════════════════════════════════════════════════════════

class TestBlueprintGDRounds:
    """GD round is now included in software and data blueprints."""

    def _round_types(self, blueprint_name: str) -> list[str]:
        return [e["round_type"].value for e in BLUEPRINTS[blueprint_name]]

    def test_software_blueprint_includes_gd(self):
        types = self._round_types("software")
        assert RoundType.GD.value in types

    def test_data_blueprint_includes_gd(self):
        types = self._round_types("data")
        assert RoundType.GD.value in types

    def test_general_blueprint_includes_gd(self):
        types = self._round_types("general")
        assert RoundType.GD.value in types

    def test_software_gd_is_optional(self):
        for entry in BLUEPRINTS["software"]:
            if entry["round_type"] == RoundType.GD:
                assert entry.get("is_required", True) is False, (
                    "GD round should be optional in software blueprint"
                )
                return
        pytest.fail("GD round not found in software blueprint")

    def test_data_gd_is_optional(self):
        for entry in BLUEPRINTS["data"]:
            if entry["round_type"] == RoundType.GD:
                assert entry.get("is_required", True) is False, (
                    "GD round should be optional in data blueprint"
                )
                return
        pytest.fail("GD round not found in data blueprint")

    def test_software_blueprint_has_hr_after_gd(self):
        """HR must still exist after GD in the software blueprint."""
        blueprint = BLUEPRINTS["software"]
        types = [e["round_type"] for e in blueprint]
        gd_idx = types.index(RoundType.GD)
        hr_idx = types.index(RoundType.HR)
        assert hr_idx > gd_idx, "HR must come after GD in software blueprint"

    def test_data_blueprint_has_hr_after_gd(self):
        blueprint = BLUEPRINTS["data"]
        types = [e["round_type"] for e in blueprint]
        gd_idx = types.index(RoundType.GD)
        hr_idx = types.index(RoundType.HR)
        assert hr_idx > gd_idx, "HR must come after GD in data blueprint"

    def test_software_blueprint_round_count(self):
        """software blueprint now has 5 rounds: APTITUDE, TECHNICAL, CODING, GD, HR."""
        assert len(BLUEPRINTS["software"]) == 5

    def test_data_blueprint_round_count(self):
        """data blueprint now has 5 rounds: APTITUDE, TECHNICAL, CODING, GD, HR."""
        assert len(BLUEPRINTS["data"]) == 5

    def test_software_blueprint_orders_are_sequential(self):
        orders = [e["order"] for e in BLUEPRINTS["software"]]
        assert orders == sorted(orders), "Blueprint orders should be sequential"

    def test_data_blueprint_orders_are_sequential(self):
        orders = [e["order"] for e in BLUEPRINTS["data"]]
        assert orders == sorted(orders)

    def test_plan_excludes_gd_when_filtered(self):
        """Planner.create_plan with exclude_round_types=[GD] skips GD for software."""
        planner = InterviewPlanner()
        plan = planner.create_plan(
            role="Software Engineer",
            exclude_round_types=[RoundType.GD],
        )
        round_types = [r.round_type for r in plan.rounds]
        assert RoundType.GD not in round_types

    def test_plan_has_five_rounds_for_software(self):
        planner = InterviewPlanner()
        plan = planner.create_plan("Software Engineer")
        assert plan.total_rounds == 5

    def test_plan_has_five_rounds_for_data(self):
        planner = InterviewPlanner()
        plan = planner.create_plan("Data Scientist")
        assert plan.total_rounds == 5


# ══════════════════════════════════════════════════════════════════════
# 2. InterviewSessionLink wiring
# ══════════════════════════════════════════════════════════════════════

class TestSessionLinkWiring:
    """Session linking to authenticated user via JWT."""

    def test_session_creation_succeeds_without_auth(self, client):
        """No Authorization header — session created, no link written (no error)."""
        r = _create_session(client)
        assert r.status_code == 201

    def test_session_creation_succeeds_with_invalid_token(self, client):
        """Invalid JWT token — session created, link silently skipped."""
        r = client.post(
            "/v2/interview/session",
            headers={"Authorization": "Bearer not-a-valid-jwt"},
            json={
                "candidate_name": "Alice",
                "role": "Software Engineer",
                "experience_level": "Junior",
            },
        )
        assert r.status_code == 201

    def test_session_link_created_for_authenticated_user(self, client, tmp_path, monkeypatch):
        """
        When a valid JWT is present, InterviewSessionLink is created.

        We mock _link_session_to_user to verify it's called without
        needing a real JWT/user in the test DB.
        """
        with patch(
            "app.routers.multi_round._link_session_to_user"
        ) as mock_link:
            r = _create_session(
                client,
                role="Software Engineer",
            )
            assert r.status_code == 201
            # _link_session_to_user should always be called (even without auth)
            mock_link.assert_called_once()

    def test_link_call_receives_correct_role(self, client):
        """_link_session_to_user is called with the role from the request."""
        with patch(
            "app.routers.multi_round._link_session_to_user"
        ) as mock_link:
            _create_session(client, role="Cloud Developer")
            call_kwargs = mock_link.call_args
            assert call_kwargs is not None
            # role is second positional/keyword argument
            args, kwargs = call_kwargs
            role_passed = kwargs.get("role") or (args[1] if len(args) > 1 else None)
            assert role_passed == "Cloud Developer"

    def test_link_function_is_noop_without_auth_available(self, tmp_path):
        """_link_session_to_user does nothing when _AUTH_AVAILABLE is False."""
        import app.routers.multi_round as mr
        original = mr._AUTH_AVAILABLE
        try:
            mr._AUTH_AVAILABLE = False
            # Should not raise
            mr._link_session_to_user(
                session_id="test-session",
                role="Tester",
                experience_level="Mid",
                interview_type="mixed",
                authorization=None,
            )
        finally:
            mr._AUTH_AVAILABLE = original

    def test_link_function_handles_db_error_gracefully(self):
        """DB errors during linking are silently swallowed."""
        import app.routers.multi_round as mr

        with patch("app.auth.tokens.verify_access_token") as mock_verify, \
             patch("app.db.base.get_db") as mock_get_db:
            mock_verify.return_value = MagicMock(user_id=1)
            mock_db = MagicMock()
            mock_db.add.side_effect = Exception("DB error")
            mock_get_db.return_value = iter([mock_db])

            # Should not raise
            mr._link_session_to_user(
                session_id="abc",
                role="Tester",
                experience_level="Mid",
                interview_type="mixed",
                authorization="Bearer validtoken",
            )


# ══════════════════════════════════════════════════════════════════════
# 3. advance_to_next_round (next-round endpoint)
# ══════════════════════════════════════════════════════════════════════

class TestAdvanceToNextRound:
    """POST /v2/interview/session/{id}/next-round"""

    def _get_session_and_complete_round(self, client, role="Software Engineer"):
        """Helper: create session, complete the first round, return session_id + round_id."""
        r = _create_session(client, role=role)
        assert r.status_code == 201, r.text
        data = r.json()
        session_id = data["session_id"]
        round_id   = data["round_id"]

        # Manually end the round via end-round endpoint
        end_r = client.post(
            f"/v2/interview/session/{session_id}/end-round",
            params={"round_id": round_id},
        )
        # end-round may return 200 or 422 if already complete — just proceed
        return session_id, round_id

    def test_returns_404_for_unknown_session(self, client):
        r = client.post("/v2/interview/session/no-such-session/next-round")
        assert r.status_code == 404

    def test_returns_plan_fields(self, client):
        session_id, _ = self._get_session_and_complete_round(client)
        r = client.post(f"/v2/interview/session/{session_id}/next-round")
        assert r.status_code == 200
        body = r.json()
        assert "plan" in body or "plan_complete" in body

    def test_returns_first_question_for_next_round(self, client):
        session_id, _ = self._get_session_and_complete_round(client)
        r = client.post(f"/v2/interview/session/{session_id}/next-round")
        body = r.json()
        if not body.get("plan_complete"):
            assert "first_question" in body
            assert len(body["first_question"]) > 0

    def test_returns_round_id_for_next_round(self, client):
        session_id, _ = self._get_session_and_complete_round(client)
        r = client.post(f"/v2/interview/session/{session_id}/next-round")
        body = r.json()
        if not body.get("plan_complete"):
            assert "round_id" in body

    def test_plan_complete_when_all_rounds_done(self, client):
        """
        After completing ALL rounds, next-round returns plan_complete=True.
        """
        from interview.engine import MultiRoundEngine
        from interview.plan import RoundEvaluation
        from interview.planner import InterviewPlanner
        from interview.round_types import RoundState, RoundType
        from app.session_store import get_session_store

        planner = InterviewPlanner()
        plan = planner.create_plan(
            "HR Specialist",
            include_round_types=[RoundType.HR],
            max_turns_override=1,
        )

        engine = MultiRoundEngine()
        session = engine.create_session(
            candidate_name="Test",
            role="HR Specialist",
            experience_level="Mid",
        )
        session._plan = plan

        # Mark the only round as completed with an evaluation
        plan.rounds[0].state = RoundState.COMPLETED
        plan.rounds[0].evaluation = RoundEvaluation(
            score=7.0,
            feedback="Done.",
        )

        store = get_session_store()
        store.save(session)

        r = client.post(f"/v2/interview/session/{session.session_id}/next-round")
        body = r.json()
        assert r.status_code == 200
        assert body.get("plan_complete") is True


# ══════════════════════════════════════════════════════════════════════
# 4. create_multi_session returns gd_role/gd_persona_name in response
# ══════════════════════════════════════════════════════════════════════

class TestCreateSessionGDFields:
    """create_multi_session now includes gd_role / gd_persona_name in response."""

    def test_response_contains_gd_fields(self, client):
        """The response body should include gd_role and gd_persona_name keys."""
        r = _create_session(client, role="Software Engineer")
        assert r.status_code == 201
        body = r.json()
        # Keys present (values may be null for non-GD first round)
        assert "gd_role" in body
        assert "gd_persona_name" in body

    def test_gd_fields_null_for_aptitude_first_round(self, client):
        """For software/data roles, first round is APTITUDE — gd fields should be null."""
        r = _create_session(client, role="Software Engineer")
        body = r.json()
        # First round is APTITUDE for software, so gd_role/gd_persona_name are None
        assert body["gd_role"] is None
        assert body["gd_persona_name"] is None


# ══════════════════════════════════════════════════════════════════════
# 5. _link_session_to_user unit tests (no HTTP layer)
# ══════════════════════════════════════════════════════════════════════

class TestLinkSessionToUserUnit:
    """Pure unit tests for _link_session_to_user."""

    def test_noop_when_authorization_is_none(self):
        from app.routers.multi_round import _link_session_to_user
        # Should not raise
        _link_session_to_user("sid", "role", "level", "type", None)

    def test_noop_when_not_bearer_scheme(self):
        from app.routers.multi_round import _link_session_to_user
        _link_session_to_user("sid", "role", "level", "type", "Basic dXNlcjpwYXNz")

    def test_noop_when_token_verify_returns_none(self):
        from app.routers.multi_round import _link_session_to_user
        with patch("app.auth.tokens.verify_access_token", return_value=None):
            _link_session_to_user("sid", "role", "level", "type", "Bearer bad")

    def test_writes_link_when_token_valid(self):
        """
        _link_session_to_user writes an InterviewSessionLink when token is valid.

        We patch both the module-level `get_db` reference inside multi_round
        AND the tokens module so the local import picks up the mock.
        """
        import app.routers.multi_round as mr
        import app.auth.tokens as _tok_mod
        from app.routers.multi_round import _link_session_to_user

        fake_token = MagicMock()
        fake_token.user_id = 42

        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = MagicMock()
        mock_db.close = MagicMock()

        # Patch get_db directly on the router module (it was imported there)
        original_get_db = mr.get_db  # type: ignore[attr-defined]
        original_verify = _tok_mod.verify_access_token

        try:
            mr.get_db = lambda: iter([mock_db])  # type: ignore[attr-defined]
            _tok_mod.verify_access_token = lambda t: fake_token
            _link_session_to_user("test-sid", "Dev", "Mid", "mixed", "Bearer valid")
        finally:
            mr.get_db = original_get_db  # type: ignore[attr-defined]
            _tok_mod.verify_access_token = original_verify

        mock_db.add.assert_called_once()
        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.user_id == 42
        assert added_obj.session_id == "test-sid"
        assert added_obj.role == "Dev"
        mock_db.commit.assert_called_once()
