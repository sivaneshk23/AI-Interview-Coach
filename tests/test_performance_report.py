"""
Tests: performance report builder.

No IBM API calls — purely local computation.
"""

import pytest

from app.models import InterviewSession, InterviewTurn
from evaluation.performance_report import build_report


def _session_with_turns(turns: list[dict]) -> InterviewSession:
    session = InterviewSession(
        session_id="test-report-session",
        candidate_name="Carol",
        role="Product Manager",
        experience_level="Senior",
        interview_type="behavioral",
    )
    for t in turns:
        session.add_turn(
            question=t["question"],
            answer=t["answer"],
            evaluation=t["evaluation"],
        )
    return session


class TestBuildReport:

    def test_empty_session_returns_zeros(self):
        session = InterviewSession(
            session_id="empty",
            candidate_name="Dan",
            role="Analyst",
            experience_level="Fresher",
            interview_type="HR",
        )
        report = build_report(session)

        assert report["question_count"] == 0
        assert report["overall_score"] == 0.0
        assert report["strengths"] == []
        assert report["weaknesses"] == []
        assert report["improvement_suggestions"] == []
        assert report["turns"] == []

    def test_single_turn_report(self):
        session = _session_with_turns([
            {
                "question": "Tell me about a challenge.",
                "answer": "I led a migration project.",
                "evaluation": {
                    "overall_score": 7,
                    "technical_score": 6,
                    "relevance_score": 8,
                    "clarity_score": 7,
                    "communication_score": 8,
                    "completeness_score": 6,
                    "strengths": ["Clear communication"],
                    "weaknesses": ["Lacked metrics"],
                    "improvement_suggestions": ["Add data to support claims"],
                    "evaluation": "Solid answer.",
                },
            }
        ])

        report = build_report(session)

        assert report["question_count"] == 1
        assert report["overall_score"] == 7.0
        assert "Clear communication" in report["strengths"]
        assert "Lacked metrics" in report["weaknesses"]
        assert "Add data to support claims" in report["improvement_suggestions"]
        assert len(report["turns"]) == 1
        assert report["turns"][0]["turn"] == 1

    def test_multi_turn_averages(self):
        session = _session_with_turns([
            {
                "question": "Q1",
                "answer": "A1",
                "evaluation": {
                    "overall_score": 6,
                    "technical_score": 5,
                    "relevance_score": 6,
                    "clarity_score": 7,
                    "communication_score": 6,
                    "completeness_score": 5,
                    "strengths": ["Honest"],
                    "weaknesses": ["Vague"],
                    "improvement_suggestions": ["Be more specific"],
                    "evaluation": "OK.",
                },
            },
            {
                "question": "Q2",
                "answer": "A2",
                "evaluation": {
                    "overall_score": 8,
                    "technical_score": 9,
                    "relevance_score": 8,
                    "clarity_score": 7,
                    "communication_score": 8,
                    "completeness_score": 9,
                    "strengths": ["Technical depth"],
                    "weaknesses": [],
                    "improvement_suggestions": [],
                    "evaluation": "Great.",
                },
            },
        ])

        report = build_report(session)

        assert report["question_count"] == 2
        assert report["overall_score"] == 7.0  # (6 + 8) / 2
        # Dimension averages
        assert report["dimension_scores"]["technical_score"] == 7.0  # (5+9)/2
        # Strengths should be deduplicated union
        assert "Honest" in report["strengths"]
        assert "Technical depth" in report["strengths"]

    def test_deduplication_of_strengths(self):
        session = _session_with_turns([
            {
                "question": "Q1", "answer": "A1",
                "evaluation": {
                    "overall_score": 7,
                    "strengths": ["Good teamwork"],
                    "weaknesses": [],
                    "improvement_suggestions": [],
                    "evaluation": "",
                },
            },
            {
                "question": "Q2", "answer": "A2",
                "evaluation": {
                    "overall_score": 7,
                    "strengths": ["Good teamwork"],   # duplicate
                    "weaknesses": [],
                    "improvement_suggestions": [],
                    "evaluation": "",
                },
            },
        ])

        report = build_report(session)
        assert report["strengths"].count("Good teamwork") == 1

    def test_report_contains_required_keys(self):
        session = _session_with_turns([
            {
                "question": "Q", "answer": "A",
                "evaluation": {"overall_score": 5, "evaluation": "Fine."},
            }
        ])
        report = build_report(session)

        required = [
            "session_id", "candidate_name", "role", "experience_level",
            "interview_type", "question_count", "overall_score",
            "dimension_scores", "strengths", "weaknesses",
            "improvement_suggestions", "turns",
        ]
        for key in required:
            assert key in report, f"Missing key: {key}"
