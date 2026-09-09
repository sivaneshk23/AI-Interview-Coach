"""
Shared InterviewEngine singleton.

Both InterviewService and EvaluationService import the engine from here so
that only one RAGEngine (and therefore one FAISS index) is loaded per process.
"""

import logging

from app.interview_engine import InterviewEngine

logger = logging.getLogger(__name__)

_engine: InterviewEngine | None = None


def get_engine() -> InterviewEngine:
    """Return the process-level InterviewEngine, creating it on first call."""
    global _engine
    if _engine is None:
        logger.info("Initialising shared InterviewEngine (FAISS index will load now).")
        _engine = InterviewEngine()
    return _engine
