from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Existing stateless API schemas ────────────────────────────────────

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
    """
    Matches the dict returned by EvaluatorAgent._validate_evaluation().
    Previously had mismatched field names (feedback/improvement_plan);
    now aligned with the actual agent output.
    """

    overall_score: float
    technical_score: float
    relevance_score: float
    clarity_score: float
    communication_score: float
    completeness_score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    evaluation: str


# ── Session API schemas ───────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    """Request body for POST /interview/session."""

    candidate_name: str = Field(..., min_length=1, max_length=200)
    role: str = Field(..., min_length=2, max_length=200)
    experience_level: str = Field(..., min_length=2, max_length=100)
    interview_type: str = Field(..., min_length=2, max_length=100)
    candidate_context: Optional[str] = Field(None, max_length=2000)


class SessionResponse(BaseModel):
    """Response body for POST /interview/session."""

    session_id: str
    candidate_name: str
    role: str
    experience_level: str
    interview_type: str
    first_question: str
    turn_number: int


class SubmitAnswerRequest(BaseModel):
    """Request body for POST /interview/session/{session_id}/answer."""

    question: str = Field(..., min_length=5, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=5000)
    candidate_context: Optional[str] = Field(None, max_length=2000)


class TurnResponse(BaseModel):
    """Response body for POST /interview/session/{session_id}/answer."""

    turn_number: int
    evaluation: Dict[str, Any]
    next_question: Optional[str] = None
    interview_complete: bool = False


class SessionSummaryResponse(BaseModel):
    """Response body for GET /interview/session/{session_id}/summary."""

    session_id: str
    candidate_name: str
    role: str
    experience_level: str
    interview_type: str
    question_count: int
    overall_score: float
    dimension_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    turns: List[Dict[str, Any]]
