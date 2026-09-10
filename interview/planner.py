"""
Interview Planner — role-aware interview plan generation.

The planner selects a round blueprint based on the candidate's role.
Role matching uses keyword heuristics on the FREE-FORM role string.

CRITICAL DESIGN CONSTRAINT:
    The job role MUST remain a free-form string.
    There is NO closed role list.
    There are NO enum values for roles.
    There is NO role-specific conditional spaghetti.

    The blueprints below are EXAMPLES only.
    Unrecognised roles fall through to a sensible default.
    The blueprint system is open — new blueprints can be registered
    without changing InterviewEngine or any existing code.

Architecture:
    Blueprint = a named ordered list of RoundConfig objects
    Planner   = maps a free-form role string → Blueprint name → Plan

Adding a new blueprint:
    1. Define a list[RoundConfig] below (or load from config file).
    2. Register it with BLUEPRINTS[name] = [...].
    3. Add keyword hints in ROLE_KEYWORDS[name] = [...].
    No other changes needed.
"""

from __future__ import annotations

import re
from typing import Dict, List

from interview.plan import InterviewPlan, InterviewRound
from interview.round_types import RoundType


# ── Round configuration blueprints ────────────────────────────────────

# Each blueprint entry is a dict describing one round's configuration.
# Keys match InterviewRound constructor parameters.

_BlueprintEntry = Dict


BLUEPRINTS: Dict[str, List[_BlueprintEntry]] = {

    # Software / backend / frontend / full-stack / mobile / devops roles
    "software": [
        {
            "round_type": RoundType.APTITUDE,
            "order": 1,
            "title": "Aptitude Test",
            "purpose": "Assess logical reasoning, quantitative aptitude, and problem-solving speed.",
            "difficulty": "medium",
            "max_turns": 10,
            "time_limit_minutes": 30,
            "is_required": True,
        },
        {
            "round_type": RoundType.TECHNICAL,
            "order": 2,
            "title": "Technical Interview",
            "purpose": "Assess CS fundamentals, system design, and domain knowledge.",
            "difficulty": "medium",
            "max_turns": 8,
            "time_limit_minutes": 45,
            "is_required": True,
        },
        {
            "round_type": RoundType.CODING,
            "order": 3,
            "title": "Coding Challenge",
            "purpose": "Evaluate problem-solving, algorithmic thinking, and code quality.",
            "difficulty": "medium",
            "max_turns": 3,
            "time_limit_minutes": 60,
            "is_required": True,
        },
        {
            "round_type": RoundType.GD,
            "order": 4,
            "title": "Group Discussion",
            "purpose": "Assess collaborative communication, leadership, and critical thinking.",
            "difficulty": "medium",
            "max_turns": 6,
            "time_limit_minutes": 20,
            "is_required": False,  # GD is optional for software roles
        },
        {
            "round_type": RoundType.HR,
            "order": 5,
            "title": "HR Interview",
            "purpose": "Assess cultural fit, motivation, and soft skills.",
            "difficulty": "easy",
            "max_turns": 6,
            "time_limit_minutes": 30,
            "is_required": True,
        },
    ],

    # Data science / ML / AI / analytics / data engineering
    "data": [
        {
            "round_type": RoundType.APTITUDE,
            "order": 1,
            "title": "Aptitude Test",
            "purpose": "Assess quantitative aptitude, statistics, and logical reasoning.",
            "difficulty": "medium",
            "max_turns": 10,
            "time_limit_minutes": 30,
            "is_required": True,
        },
        {
            "round_type": RoundType.TECHNICAL,
            "order": 2,
            "title": "Technical / Domain Interview",
            "purpose": "Assess data science concepts, ML fundamentals, statistics, and tools.",
            "difficulty": "medium",
            "max_turns": 8,
            "time_limit_minutes": 45,
            "is_required": True,
        },
        {
            "round_type": RoundType.CODING,
            "order": 3,
            "title": "Coding / SQL Challenge",
            "purpose": "Evaluate data manipulation, SQL, and Python/R coding ability.",
            "difficulty": "medium",
            "max_turns": 3,
            "time_limit_minutes": 60,
            "is_required": True,
        },
        {
            "round_type": RoundType.GD,
            "order": 4,
            "title": "Group Discussion",
            "purpose": "Discuss data-driven insights, ML ethics, or business problem framing.",
            "difficulty": "medium",
            "max_turns": 6,
            "time_limit_minutes": 20,
            "is_required": False,  # GD is optional for data roles
        },
        {
            "round_type": RoundType.HR,
            "order": 5,
            "title": "HR Interview",
            "purpose": "Assess communication, motivation, and team fit.",
            "difficulty": "easy",
            "max_turns": 6,
            "time_limit_minutes": 30,
            "is_required": True,
        },
    ],

    # Non-technical / business / product / marketing / HR / finance roles
    "general": [
        {
            "round_type": RoundType.APTITUDE,
            "order": 1,
            "title": "Aptitude Test",
            "purpose": "Assess verbal reasoning, numerical ability, and critical thinking.",
            "difficulty": "easy",
            "max_turns": 10,
            "time_limit_minutes": 30,
            "is_required": True,
        },
        {
            "round_type": RoundType.COMMUNICATION,
            "order": 2,
            "title": "Communication Assessment",
            "purpose": "Assess verbal communication, clarity, and articulation.",
            "difficulty": "easy",
            "max_turns": 5,
            "time_limit_minutes": 20,
            "is_required": True,
        },
        {
            "round_type": RoundType.GD,
            "order": 3,
            "title": "Group Discussion",
            "purpose": "Assess critical thinking, leadership, and collaborative communication.",
            "difficulty": "medium",
            "max_turns": 6,
            "time_limit_minutes": 20,
            "is_required": False,  # GD is often optional
        },
        {
            "round_type": RoundType.HR,
            "order": 4,
            "title": "HR Interview",
            "purpose": "Assess cultural fit, motivation, domain interest, and interpersonal skills.",
            "difficulty": "easy",
            "max_turns": 8,
            "time_limit_minutes": 30,
            "is_required": True,
        },
    ],
}


# ── Role keyword heuristics ────────────────────────────────────────────
# Maps blueprint name → list of keyword patterns (case-insensitive regex)
# These are HINTS only — not a closed role list.
# A role that matches no keyword falls to "general".

ROLE_KEYWORDS: Dict[str, List[str]] = {
    "software": [
        r"\bsoftware\b", r"\bdeveloper\b", r"\bengineer\b",
        r"\bbackend\b", r"\bfrontend\b", r"\bfull.?stack\b",
        r"\bmobile\b", r"\bdevops\b", r"\bsre\b",
        r"\bsecurity\b", r"\bsystems?\b", r"\bembedded\b",
        r"\bcloud\b", r"\bplatform\b", r"\binfrastructure\b",
        r"\bblockchain\b", r"\bweb\b",
    ],
    "data": [
        r"\bdata\b", r"\bml\b", r"\bai\b",
        r"\bmachine.?learning\b", r"\bdeep.?learning\b",
        r"\bscientist\b", r"\banalyst\b", r"\banalytic\b",
        r"\bstatistic\b", r"\bsql\b", r"\bquant\b",
        r"\bnlp\b", r"\bvision\b", r"\bresearch\b",
    ],
    # "general" is the catch-all — no keywords needed
}


# ── Planner ────────────────────────────────────────────────────────────

class InterviewPlanner:
    """
    Generates an InterviewPlan for a given (free-form) role and experience level.

    The role is matched against keyword heuristics to select the most
    appropriate blueprint. No role list is hardcoded. An unrecognised role
    simply uses the "general" blueprint.

    Custom blueprints can be injected at construction time for testing
    or future configuration-file-driven overrides.
    """

    def __init__(
        self,
        blueprints: Dict[str, List[_BlueprintEntry]] | None = None,
        role_keywords: Dict[str, List[str]] | None = None,
    ):
        self._blueprints   = blueprints  or BLUEPRINTS
        self._role_keywords = role_keywords or ROLE_KEYWORDS

    def _select_blueprint(self, role: str) -> str:
        """
        Select a blueprint name for the given free-form role string.

        Returns the name of the best-matching blueprint,
        or "general" if no keywords match.
        """
        role_lower = role.lower()
        for blueprint_name, patterns in self._role_keywords.items():
            for pattern in patterns:
                if re.search(pattern, role_lower):
                    return blueprint_name
        return "general"

    def create_plan(
        self,
        role: str,
        experience_level: str = "Mid-level",
        include_round_types: List[RoundType] | None = None,
        exclude_round_types: List[RoundType] | None = None,
        max_turns_override: int | None = None,
    ) -> InterviewPlan:
        """
        Create an InterviewPlan for the given candidate.

        Args:
            role:               Free-form job role string.
            experience_level:   Candidate level (for future difficulty scaling).
            include_round_types: If set, only include these round types.
            exclude_round_types: If set, skip these round types.
            max_turns_override:  Override max_turns on all rounds (useful for testing).

        Returns:
            A fully configured InterviewPlan ready to attach to a session.
        """
        blueprint_name = self._select_blueprint(role)
        blueprint      = self._blueprints.get(blueprint_name, self._blueprints["general"])

        rounds = []
        for entry in blueprint:
            rt = entry["round_type"]

            if include_round_types and rt not in include_round_types:
                continue
            if exclude_round_types and rt in exclude_round_types:
                continue

            max_turns = max_turns_override if max_turns_override is not None \
                        else entry.get("max_turns", 5)

            r = InterviewRound(
                round_type        = rt,
                order             = entry.get("order", len(rounds) + 1),
                title             = entry.get("title", rt.label()),
                purpose           = entry.get("purpose", ""),
                difficulty        = entry.get("difficulty", "medium"),
                max_turns         = max_turns,
                time_limit_minutes= entry.get("time_limit_minutes"),
                is_required       = entry.get("is_required", True),
            )
            rounds.append(r)

        # Re-sequence orders to be contiguous after any filtering
        for i, r in enumerate(rounds):
            r.order = i + 1

        return InterviewPlan(rounds=rounds)
