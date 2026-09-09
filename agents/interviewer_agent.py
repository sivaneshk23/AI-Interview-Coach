from utils.llm import generate_text


class InterviewerAgent:
    """
    AI Interviewer Agent.

    Generates interview questions based on:
    - job role
    - interview type
    - candidate profile
    - previous conversation
    - difficulty
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

    def generate_question(
        self,
        candidate_context: str = "",
        previous_question: str = "",
        previous_answer: str = "",
        rag_context: str = "",
    ) -> str:

        rag_section = (
            f"\nRelevant interview knowledge:\n{rag_context}\n"
            if rag_context and rag_context.strip()
            else ""
        )

        prompt = f"""
You are an expert AI interview trainer.

Your task is to conduct a realistic interview.

Candidate target role:
{self.role}

Interview type:
{self.interview_type}

Difficulty:
{self.difficulty}

Candidate context:
{candidate_context}
{rag_section}
Previous question:
{previous_question}

Previous answer:
{previous_answer}

Instructions:
1. Ask exactly ONE interview question.
2. Make the question relevant to the target role.
3. Do not provide the answer.
4. Do not ask multiple questions at once.
5. If a previous answer is provided, make the next question adaptive to that answer.
6. Gradually increase difficulty when the candidate performs well.
7. If the answer is weak, ask a simpler follow-up question.
8. Keep the question professional and realistic.
9. Avoid repeating previous questions.

Return ONLY the interview question.
"""

        response = generate_text(
            prompt,
            max_new_tokens=150,
        )

        return response.strip()