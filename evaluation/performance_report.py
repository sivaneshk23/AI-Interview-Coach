"""
Performance report generator.

Aggregates per-turn evaluation scores from a completed InterviewSession
into a single structured report.
"""

from __future__ import annotations

from typing import Any

from app.models import InterviewSession


def build_report(session: InterviewSession) -> dict[str, Any]:
    """
    Build an aggregate performance report for a completed session.

    Returns a dict with:
        session_id, candidate_name, role, experience_level, interview_type,
        question_count, overall_score, dimension_scores (dict),
        strengths, weaknesses, improvement_suggestions, turns (list)
    """

    turns = session.turns

    if not turns:
        return {
            "session_id": session.session_id,
            "candidate_name": session.candidate_name,
            "role": session.role,
            "experience_level": session.experience_level,
            "interview_type": session.interview_type,
            "question_count": 0,
            "overall_score": 0.0,
            "dimension_scores": {},
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "turns": [],
        }

    score_keys = [
        "overall_score",
        "technical_score",
        "relevance_score",
        "clarity_score",
        "communication_score",
        "completeness_score",
    ]

    def _avg(key: str) -> float:
        values = [
            float(t.evaluation.get(key, 0))
            for t in turns
            if t.evaluation.get(key) is not None
        ]
        return round(sum(values) / len(values), 2) if values else 0.0

    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    # Aggregate list fields across all turns
    all_strengths = _dedup(
        [s for t in turns for s in (t.evaluation.get("strengths") or [])]
    )
    all_weaknesses = _dedup(
        [w for t in turns for w in (t.evaluation.get("weaknesses") or [])]
    )
    all_suggestions = _dedup(
        [
            s
            for t in turns
            for s in (
                t.evaluation.get("improvement_suggestions")
                or t.evaluation.get("improvement_plan")
                or []
            )
        ]
    )

    dimension_scores = {
        key: _avg(key)
        for key in score_keys
        if key != "overall_score"
    }

    turn_summaries = [
        {
            "turn": i + 1,
            "question": t.question,
            "answer": t.answer,
            "overall_score": t.evaluation.get("overall_score", 0),
            "evaluation": t.evaluation.get("evaluation", ""),
        }
        for i, t in enumerate(turns)
    ]

    return {
        "session_id": session.session_id,
        "candidate_name": session.candidate_name,
        "role": session.role,
        "experience_level": session.experience_level,
        "interview_type": session.interview_type,
        "question_count": len(turns),
        "overall_score": _avg("overall_score"),
        "dimension_scores": dimension_scores,
        "strengths": all_strengths,
        "weaknesses": all_weaknesses,
        "improvement_suggestions": all_suggestions,
        "turns": turn_summaries,
    }
