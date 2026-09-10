from __future__ import annotations

import json
import re
from typing import List, Optional

from utils.llm import generate_text


# ── Injection guard ───────────────────────────────────────────────────
_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"you\s+are\s+now|forget\s+everything|"
    r"system\s*:|<\s*/?system\s*>)",
    re.IGNORECASE,
)


def _safe_text(text: str, max_len: int = 2000) -> str:
    if _INJECTION_RE.search(text):
        return "[content removed]"
    return str(text)[:max_len]


class EvaluatorAgent:
    """
    Evaluates candidate interview answers using IBM watsonx.ai.

    Supports two evaluation modes:
    - Standard technical/HR evaluation (all dimensions)
    - Communication-specific evaluation (spoken answer quality dimensions)

    Voice answers are evaluated on the same dimensions as text answers;
    additional communication-specific scores (filler_word_score,
    response_structure_score) are added when voice_input_mode="voice".
    """

    def __init__(self, role: str = "General", interview_type: str = "technical"):
        self.role = role
        self.interview_type = interview_type

    def evaluate(
        self,
        question: str,
        answer: str,
        candidate_context: str = "",
        rag_context: str = "",
        # Prompt 4: voice/communication context
        voice_input_mode: str = "text",    # "voice" | "text"
        voice_duration_sec: Optional[float] = None,
        transcript: Optional[str] = None,  # STT transcript (may differ from typed)
    ) -> dict:
        """
        Evaluate a candidate answer.

        When voice_input_mode="voice", the transcript is used as the answer
        text, and additional voice-quality dimensions are evaluated.
        """
        safe_question = _safe_text(question, 500)
        safe_context  = _safe_text(candidate_context, 1500)
        safe_rag      = _safe_text(rag_context, 800)

        # For voice, use transcript if available; otherwise use the answer string
        effective_answer = answer
        if voice_input_mode == "voice" and transcript:
            effective_answer = transcript
        safe_answer = _safe_text(effective_answer, 2000)

        is_communication = (
            self.interview_type.lower() in ("communication",)
            or voice_input_mode == "voice"
        )

        if is_communication:
            result = self._evaluate_communication(
                question           = safe_question,
                answer             = safe_answer,
                candidate_context  = safe_context,
                rag_context        = safe_rag,
                voice_input_mode   = voice_input_mode,
                voice_duration_sec = voice_duration_sec,
            )
        else:
            result = self._evaluate_standard(
                question           = safe_question,
                answer             = safe_answer,
                candidate_context  = safe_context,
                rag_context        = safe_rag,
            )

        return result

    # ── Standard evaluation ───────────────────────────────────────────

    def _evaluate_standard(
        self,
        question: str,
        answer: str,
        candidate_context: str = "",
        rag_context: str = "",
    ) -> dict:
        rag_section = (
            f"\nRelevant interview knowledge:\n{rag_context}\n"
            if rag_context and rag_context.strip() else ""
        )

        prompt = f"""You are an expert interview evaluator.

Target role: {self.role}
Interview type: {self.interview_type}

Candidate context:
{candidate_context or "Not provided."}
{rag_section}
Interview question:
{question}

Candidate answer:
{answer}

Evaluate the candidate objectively.

Return ONLY valid JSON using exactly this structure:

{{
    "overall_score": 0,
    "technical_score": 0,
    "relevance_score": 0,
    "clarity_score": 0,
    "communication_score": 0,
    "completeness_score": 0,
    "strengths": [],
    "weaknesses": [],
    "improvement_suggestions": [],
    "evaluation": ""
}}

Scoring:
0 = extremely poor
1-2 = poor
3-4 = below average
5-6 = average
7-8 = good
9 = excellent
10 = exceptional

Rules:
- Be objective.
- Do not give a high score merely because the answer sounds confident.
- Evaluate technical correctness separately from communication quality.
- Identify missing concepts.
- Give practical improvement suggestions.
- Do not invent candidate experience.
- Return valid JSON only.
"""

        response = generate_text(prompt, max_new_tokens=500).strip()
        return self._parse_evaluation(response)

    # ── Communication-specific evaluation ─────────────────────────────

    def _evaluate_communication(
        self,
        question: str,
        answer: str,
        candidate_context: str = "",
        rag_context: str = "",
        voice_input_mode: str = "text",
        voice_duration_sec: Optional[float] = None,
    ) -> dict:
        duration_note = ""
        if voice_duration_sec is not None:
            duration_note = f"\nVoice response duration: {voice_duration_sec:.1f} seconds."
            if voice_duration_sec < 10:
                duration_note += " (Very short response)"
            elif voice_duration_sec > 180:
                duration_note += " (Very long response — check for rambling)"

        # Count simple filler words as an objective measure
        filler_count = _count_filler_words(answer)
        filler_note = ""
        if voice_input_mode == "voice":
            filler_note = f"\nFiller word count in transcript: {filler_count} (um, uh, like, you know, basically, actually, literally, right, so)."

        prompt = f"""You are a communication skills evaluator.

Target role: {self.role}
Evaluation focus: Communication quality, verbal clarity, structure, and articulation.{duration_note}{filler_note}

Candidate context:
{candidate_context or "Not provided."}

Question:
{question}

Candidate answer (transcript if voice):
{answer}

Evaluate the candidate's COMMUNICATION QUALITY objectively.

Return ONLY valid JSON using exactly this structure:

{{
    "overall_score": 0,
    "technical_score": 0,
    "relevance_score": 0,
    "clarity_score": 0,
    "communication_score": 0,
    "completeness_score": 0,
    "language_quality_score": 0,
    "response_structure_score": 0,
    "filler_word_score": 0,
    "strengths": [],
    "weaknesses": [],
    "improvement_suggestions": [],
    "evaluation": ""
}}

Scoring guide:
- overall_score: holistic assessment of the full answer
- clarity_score: how clearly the candidate expressed their ideas
- communication_score: overall verbal communication effectiveness
- language_quality_score: grammar, vocabulary, sentence structure
- response_structure_score: opening/body/conclusion clarity (1=rambling, 10=structured)
- filler_word_score: 10 = no filler words; 1 = excessive fillers (um, uh, like, you know)
- completeness_score: did the answer fully address the question?

Rules:
- Focus on HOW the candidate communicated, not just WHAT they said.
- A technically correct answer delivered poorly should score lower on communication dimensions.
- Identify specific communication weaknesses (e.g. "started many sentences with 'like'").
- Give actionable verbal communication improvement tips.
- Return valid JSON only.
"""

        response = generate_text(prompt, max_new_tokens=600).strip()
        result = self._parse_communication_evaluation(response)

        # Inject objective filler word count into result for reference
        result["_filler_word_count"] = filler_count
        result["_voice_duration_sec"] = voice_duration_sec
        return result

    # ── Parsing helpers ───────────────────────────────────────────────

    def _parse_evaluation(self, response: str) -> dict:
        """Safely parse standard evaluation JSON."""
        response = _strip_code_fence(response)
        try:
            result = json.loads(response)
            if isinstance(result, dict):
                return self._validate_standard_evaluation(result)
        except json.JSONDecodeError:
            pass

        start = response.find("{")
        end   = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(response[start:end + 1])
                if isinstance(result, dict):
                    return self._validate_standard_evaluation(result)
            except json.JSONDecodeError:
                pass

        return {
            "overall_score":          0,
            "technical_score":        0,
            "relevance_score":        0,
            "clarity_score":          0,
            "communication_score":    0,
            "completeness_score":     0,
            "strengths":              [],
            "weaknesses":             [],
            "improvement_suggestions":[],
            "evaluation":             response,
        }

    def _parse_communication_evaluation(self, response: str) -> dict:
        """Safely parse communication-mode evaluation JSON."""
        response = _strip_code_fence(response)
        try:
            result = json.loads(response)
            if isinstance(result, dict):
                return self._validate_communication_evaluation(result)
        except json.JSONDecodeError:
            pass

        start = response.find("{")
        end   = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(response[start:end + 1])
                if isinstance(result, dict):
                    return self._validate_communication_evaluation(result)
            except json.JSONDecodeError:
                pass

        return {
            "overall_score":           0,
            "technical_score":         0,
            "relevance_score":         0,
            "clarity_score":           0,
            "communication_score":     0,
            "completeness_score":      0,
            "language_quality_score":  0,
            "response_structure_score":0,
            "filler_word_score":       0,
            "strengths":               [],
            "weaknesses":              [],
            "improvement_suggestions": [],
            "evaluation":              response,
        }

    def _validate_standard_evaluation(self, result: dict) -> dict:
        """Validate and normalise standard evaluation output."""
        score_fields = [
            "overall_score", "technical_score", "relevance_score",
            "clarity_score", "communication_score", "completeness_score",
        ]
        return _validate_evaluation_fields(result, score_fields)

    def _validate_communication_evaluation(self, result: dict) -> dict:
        """Validate and normalise communication evaluation output."""
        score_fields = [
            "overall_score", "technical_score", "relevance_score",
            "clarity_score", "communication_score", "completeness_score",
            "language_quality_score", "response_structure_score", "filler_word_score",
        ]
        return _validate_evaluation_fields(result, score_fields)


# ── Shared validation helper ──────────────────────────────────────────

def _validate_evaluation_fields(result: dict, score_fields: List[str]) -> dict:
    """Clamp score fields to [0, 10] and ensure list fields are lists."""
    for field in score_fields:
        value = result.get(field, 0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(10.0, value))
        result[field] = int(value) if value.is_integer() else value

    for field in ("strengths", "weaknesses", "improvement_suggestions"):
        value = result.get(field, [])
        if not isinstance(value, list):
            value = [str(value)] if value else []
        result[field] = value

    result["evaluation"] = str(result.get("evaluation", ""))
    return result


def _strip_code_fence(text: str) -> str:
    """Remove Markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _count_filler_words(text: str) -> int:
    """Count common spoken filler words in a transcript."""
    fillers = [
        r"\bum\b", r"\buh\b", r"\blike\b", r"\byou\s+know\b",
        r"\bbasically\b", r"\bactually\b", r"\bliterally\b",
        r"\bright\b", r"\bso\b", r"\bkinda\b", r"\bsort\s+of\b",
        r"\bkind\s+of\b", r"\bmean\b",
    ]
    count = 0
    text_lower = text.lower()
    for pattern in fillers:
        count += len(re.findall(pattern, text_lower))
    return count
