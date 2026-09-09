import json
from utils.llm import generate_text


class EvaluatorAgent:
    """
    Evaluates candidate interview answers using IBM watsonx.ai.
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
    ) -> dict:

        rag_section = (
            f"\nRelevant interview knowledge:\n{rag_context}\n"
            if rag_context and rag_context.strip()
            else ""
        )

        prompt = f"""
You are an expert interview evaluator.

Target role:
{self.role}

Interview type:
{self.interview_type}

Candidate context:
{candidate_context}
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

        response = generate_text(
            prompt,
            max_new_tokens=500,
        ).strip()

        return self._parse_evaluation(response)
    def _parse_evaluation(self, response: str) -> dict:
        """
        Safely parse the LLM evaluation response.

        Handles:
        - Normal JSON
        - JSON inside ```json ... ```
        - Extra text around JSON
        - Invalid/truncated responses
        """

        response = response.strip()

        # Remove Markdown code fences.
        if response.startswith("```"):
            lines = response.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response = "\n".join(lines).strip()

        # Try to parse the complete response.
        try:
            result = json.loads(response)

            if isinstance(result, dict):
                return self._validate_evaluation(result)

        except json.JSONDecodeError:
            pass

        # Try to find a JSON object inside the response.
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            candidate = response[start:end + 1]

            try:
                result = json.loads(candidate)

                if isinstance(result, dict):
                    return self._validate_evaluation(result)

            except json.JSONDecodeError:
                pass

        # Safe fallback.
        return {
            "overall_score": 0,
            "technical_score": 0,
            "relevance_score": 0,
            "clarity_score": 0,
            "communication_score": 0,
            "completeness_score": 0,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation": response,
        }

    def _validate_evaluation(self, result: dict) -> dict:
        """
        Validate and normalize the evaluator output.
        """

        score_fields = [
            "overall_score",
            "technical_score",
            "relevance_score",
            "clarity_score",
            "communication_score",
            "completeness_score",
        ]

        list_fields = [
            "strengths",
            "weaknesses",
            "improvement_suggestions",
        ]

        # Validate scores.
        for field in score_fields:
            value = result.get(field, 0)

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0

            value = max(0, min(10, value))

            if value.is_integer():
                value = int(value)

            result[field] = value

        # Validate list fields.
        for field in list_fields:
            value = result.get(field, [])

            if not isinstance(value, list):
                value = [str(value)] if value else []

            result[field] = value

        # Always provide evaluation text.
        result["evaluation"] = str(
            result.get("evaluation", "")
        )

        return result