"""
IBM watsonx.ai LLM service.

This module is the single concrete LLM call path for the application.
Both the service layer and utils/llm.py delegate here so there is one
implementation of the IBM API interaction.
"""

import logging

from app.ibm_client import get_model
from app.config import get_settings

logger = logging.getLogger(__name__)


class IBMWatsonxService:
    """
    Thin wrapper around the IBM watsonx.ai SDK model client.

    Uses the chat interface (ModelInference.chat) which is
    compatible with IBM Granite and Llama-family models.
    """

    _SYSTEM_PROMPT = (
        "You are an expert AI assistant for an intelligent interview "
        "training platform. Follow instructions precisely and produce "
        "accurate, professional, useful responses."
    )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 700,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text from the given prompt via IBM watsonx.ai.

        Returns the generated text string.
        Raises RuntimeError with a safe message on IBM API failure.
        Never propagates raw credentials in exceptions.
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": self._SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        params = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        try:
            model = get_model()
            response = model.chat(messages=messages, params=params)
        except Exception as exc:
            # Log the real error server-side; never expose credentials.
            logger.error("IBM watsonx.ai call failed: %s", type(exc).__name__)
            raise RuntimeError(
                "IBM watsonx.ai is currently unavailable. "
                "Please try again later."
            ) from None

        if not isinstance(response, dict):
            raise RuntimeError(
                "IBM watsonx.ai returned an unexpected response format."
            )

        try:
            content = response["choices"][0]["message"]["content"]

            if content:
                return str(content).strip()

        except (KeyError, IndexError, TypeError):
            pass

        raise RuntimeError(
            "IBM watsonx.ai returned an empty or malformed response."
        )
