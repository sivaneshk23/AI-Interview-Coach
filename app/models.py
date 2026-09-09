from dataclasses import dataclass, field
from typing import Any


@dataclass
class InterviewTurn:
    question: str
    answer: str
    evaluation: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterviewSession:
    session_id: str
    candidate_name: str
    role: str
    experience_level: str
    interview_type: str
    turns: list[InterviewTurn] = field(default_factory=list)

    def add_turn(
        self,
        question: str,
        answer: str,
        evaluation: dict[str, Any],
    ) -> None:

        self.turns.append(
            InterviewTurn(
                question=question,
                answer=answer,
                evaluation=evaluation,
            )
        )

    @property
    def question_count(self) -> int:
        return len(self.turns)

    def questions(self) -> list[str]:
        return [turn.question for turn in self.turns]

    def answers(self) -> list[str]:
        return [turn.answer for turn in self.turns]