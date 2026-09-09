from typing import List, Optional

from pydantic import BaseModel, Field


class InterviewRequest(BaseModel):
    role: str = Field(..., min_length=2, max_length=200)
    experience_level: str = Field(..., min_length=2, max_length=100)
    interview_type: str = Field(..., min_length=2, max_length=100)
    topic: Optional[str] = Field(None, max_length=200)
    candidate_context: Optional[str] = Field(None, max_length=2000)


class InterviewQuestion(BaseModel):
    question: str
    category: str
    difficulty: str


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=5000)
    role: str = Field(..., min_length=2, max_length=200)
    experience_level: str = Field(..., min_length=2, max_length=100)
    interview_type: Optional[str] = Field("technical", max_length=100)
    candidate_context: Optional[str] = Field(None, max_length=2000)


class EvaluationResult(BaseModel):
    overall_score: float
    technical_score: float
    communication_score: float
    relevance_score: float
    strengths: List[str]
    weaknesses: List[str]
    feedback: str
    improvement_plan: List[str]
