from __future__ import annotations

import re
from typing import List, Optional

from utils.llm import generate_text


# ── Injection guard (minimal — full sanitisation lives in routes.py) ──
_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"you\s+are\s+now|forget\s+everything|"
    r"system\s*:|<\s*/?system\s*>)",
    re.IGNORECASE,
)


def _safe_text(text: str, max_len: int = 2000) -> str:
    """Strip obvious injection patterns and truncate."""
    if _INJECTION_RE.search(text):
        return "[content removed]"
    return str(text)[:max_len]


class InterviewerAgent:
    """
    AI Interviewer Agent.

    Generates interview questions based on:
    - job role
    - interview type (technical / HR / communication / gd)
    - candidate profile / resume context
    - FULL turn history (all previous questions — for anti-repetition)
    - previous answer (most recent — for adaptive follow-up)
    - difficulty
    - RAG knowledge base context

    Round-type-specific prompts are selected via _PROMPT_BUILDERS.
    All resume/profile text is sanitised before being included in prompts.
    """

    def __init__(
        self,
        role: str = "General",
        interview_type: str = "technical",
        difficulty: str = "medium",
    ):
        self.role = role
        self.interview_type = interview_type
        self.difficulty = difficulty

    # ── Public interface ──────────────────────────────────────────────

    def generate_question(
        self,
        candidate_context: str = "",
        previous_question: str = "",
        previous_answer: str = "",
        rag_context: str = "",
        # Extended context for Prompt 4 adaptive question generation
        all_previous_questions: Optional[List[str]] = None,
        all_previous_evaluations: Optional[List[dict]] = None,
        voice_input_mode: str = "text",  # "voice" | "text"
    ) -> str:
        """
        Generate the next interview question.

        Args:
            candidate_context:        Resume/profile summary (sanitised).
            previous_question:        Last question asked (may be empty for Q1).
            previous_answer:          Last answer given (may be empty for Q1).
            rag_context:              RAG knowledge base snippets.
            all_previous_questions:   Full list of questions asked so far (anti-repetition).
            all_previous_evaluations: Per-turn evaluation dicts (adaptive difficulty).
            voice_input_mode:         Whether the candidate is using voice input.
        """
        # Sanitise all untrusted inputs
        safe_context   = _safe_text(candidate_context, 2000)
        safe_prev_q    = _safe_text(previous_question, 500)
        safe_prev_a    = _safe_text(previous_answer, 1000)
        safe_rag       = _safe_text(rag_context, 1000)

        builder = self._get_prompt_builder()
        prompt  = builder(
            role                     = self.role,
            difficulty               = self.difficulty,
            candidate_context        = safe_context,
            previous_question        = safe_prev_q,
            previous_answer          = safe_prev_a,
            rag_context              = safe_rag,
            all_previous_questions   = all_previous_questions or [],
            all_previous_evaluations = all_previous_evaluations or [],
            voice_input_mode         = voice_input_mode,
        )

        response = generate_text(prompt, max_new_tokens=180)
        return response.strip()

    # ── Prompt builders per round type ────────────────────────────────

    def _get_prompt_builder(self):
        builders = {
            "hr":            _build_hr_prompt,
            "behavioral":    _build_hr_prompt,
            "communication": _build_communication_prompt,
            "gd":            _build_gd_prompt,
        }
        return builders.get(self.interview_type.lower(), _build_technical_prompt)


# ── Generic anti-repetition section ──────────────────────────────────

def _questions_asked_section(all_previous_questions: List[str]) -> str:
    if not all_previous_questions:
        return ""
    truncated = [q[:120] for q in all_previous_questions[-15:]]
    lines = "\n".join(f"  - {q}" for q in truncated)
    return f"\nQuestions already asked (DO NOT repeat or paraphrase these):\n{lines}\n"


def _rag_section(rag_context: str) -> str:
    if rag_context and rag_context.strip():
        return f"\nRelevant interview knowledge:\n{rag_context}\n"
    return ""


def _adaptive_note(all_previous_evaluations: List[dict]) -> str:
    """Generate a difficulty-adjustment hint based on recent performance."""
    if not all_previous_evaluations:
        return ""
    recent = all_previous_evaluations[-3:]
    scores = [e.get("overall_score", 5) for e in recent if isinstance(e, dict)]
    if not scores:
        return ""
    avg = sum(scores) / len(scores)
    if avg >= 7.5:
        return "\nThe candidate is performing well — increase difficulty slightly.\n"
    if avg <= 3.5:
        return "\nThe candidate is struggling — ask a simpler clarifying follow-up.\n"
    return ""


# ── Technical prompt ──────────────────────────────────────────────────

def _build_technical_prompt(
    role, difficulty, candidate_context, previous_question,
    previous_answer, rag_context, all_previous_questions,
    all_previous_evaluations, voice_input_mode, **_
) -> str:
    voice_note = (
        "\nThe candidate is answering via voice — keep the question concise and clear.\n"
        if voice_input_mode == "voice" else ""
    )
    return f"""You are an expert technical interviewer conducting a technical interview.

Target role: {role}
Difficulty: {difficulty}

Candidate background:
{candidate_context or "Not provided."}
{_rag_section(rag_context)}{_questions_asked_section(all_previous_questions)}{_adaptive_note(all_previous_evaluations)}{voice_note}
Most recent question: {previous_question or "None — this is the first question."}
Most recent answer: {previous_answer or "None."}

Instructions:
1. Ask exactly ONE technical interview question.
2. Make it relevant to the role and candidate background.
3. Do NOT repeat or closely paraphrase any question in the list above.
4. Do NOT provide the answer or hints.
5. Do NOT ask multiple questions at once.
6. If a previous answer is provided, make the next question adaptive to that answer.
7. Keep the question professional and realistic.
8. If the candidate mentioned specific projects or skills in their background, you may ask about them directly.

Return ONLY the interview question — no preamble, no explanation.
"""


# ── HR prompt ─────────────────────────────────────────────────────────

def _build_hr_prompt(
    role, difficulty, candidate_context, previous_question,
    previous_answer, rag_context, all_previous_questions,
    all_previous_evaluations, voice_input_mode, **_
) -> str:
    voice_note = (
        "\nThe candidate is answering via voice — keep the question conversational.\n"
        if voice_input_mode == "voice" else ""
    )
    return f"""You are an experienced HR interviewer conducting a behavioral/HR interview.

Target role: {role}

Candidate background:
{candidate_context or "Not provided."}
{_questions_asked_section(all_previous_questions)}{voice_note}
Most recent question: {previous_question or "None — this is the opening question."}
Most recent answer: {previous_answer or "None."}

Instructions:
1. Ask exactly ONE HR or behavioral question.
2. Cover themes like: introduction, strengths/weaknesses, motivation, teamwork, conflict resolution, goals, cultural fit, and resume-specific follow-ups.
3. If the candidate mentioned a specific project or experience in their background, ask a behavioral follow-up about it.
4. Do NOT repeat any question already asked (see list above).
5. Do NOT ask technical questions — focus on soft skills, motivation, and character.
6. Do NOT provide answers or hints.
7. If a previous answer reveals something interesting (e.g. a project, a challenge), ask a natural follow-up.
8. Start with "Tell me about yourself" only if this is the first question.

Return ONLY the interview question.
"""


# ── Communication prompt ──────────────────────────────────────────────

def _build_communication_prompt(
    role, difficulty, candidate_context, previous_question,
    previous_answer, rag_context, all_previous_questions,
    all_previous_evaluations, voice_input_mode, **_
) -> str:
    return f"""You are a communication skills assessor evaluating a candidate for a {role} role.

Candidate background:
{candidate_context or "Not provided."}
{_questions_asked_section(all_previous_questions)}
Most recent question: {previous_question or "None — this is the first question."}
Most recent answer: {previous_answer or "None."}

Instructions:
1. Ask exactly ONE question designed to assess communication clarity, structure, and articulation.
2. Questions can be: opinion questions, describe a situation, explain a concept to a non-expert, give a short presentation, describe a project.
3. Do NOT repeat any previously asked question.
4. Keep the question open-ended so the candidate can demonstrate their speaking ability.
5. Do NOT ask technical questions.

Return ONLY the question.
"""


# ── GD topic/prompt ───────────────────────────────────────────────────

def _build_gd_prompt(
    role, difficulty, candidate_context, previous_question,
    previous_answer, rag_context, all_previous_questions,
    all_previous_evaluations, voice_input_mode, **_
) -> str:
    return f"""You are the moderator of a Group Discussion session.

Target role context: {role}
{_questions_asked_section(all_previous_questions)}

Your task: Introduce a thought-provoking discussion topic relevant to the professional domain of the target role.

Instructions:
1. Present exactly ONE discussion topic.
2. The topic should be debatable — it should have valid arguments on multiple sides.
3. Frame it as a statement or question for the group to discuss.
4. Keep it relevant to business, technology, society, or the professional field.
5. Do NOT pick a political or religious controversy.
6. Do NOT repeat topics from the list above.

Return ONLY the discussion topic statement or question.
"""
