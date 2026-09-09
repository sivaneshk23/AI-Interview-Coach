"""
API routes for the AI Interview Trainer.

Error handling ensures internal details (including any credential
information) are never returned to callers.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import (
    InterviewRequest,
    AnswerRequest,
)
from app.services.interview_service import InterviewService
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)

router = APIRouter()

interview_service = InterviewService()
evaluation_service = EvaluationService()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "AI Interview Trainer",
    }


@router.post("/interview/question")
def generate_question(request: InterviewRequest):

    try:
        return interview_service.generate_question(
            role=request.role,
            experience_level=request.experience_level,
            interview_type=request.interview_type,
            topic=request.topic,
            candidate_context=request.candidate_context or "",
        )

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        logger.exception("Question generation failed")
        raise HTTPException(
            status_code=500,
            detail="Question generation failed. Please try again.",
        )


@router.post("/interview/evaluate")
def evaluate_answer(request: AnswerRequest):

    try:
        return evaluation_service.evaluate(
            question=request.question,
            answer=request.answer,
            role=request.role,
            experience_level=request.experience_level,
            interview_type=request.interview_type or "technical",
            candidate_context=request.candidate_context or "",
        )

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    except Exception:
        logger.exception("Answer evaluation failed")
        raise HTTPException(
            status_code=500,
            detail="Answer evaluation failed. Please try again.",
        )
