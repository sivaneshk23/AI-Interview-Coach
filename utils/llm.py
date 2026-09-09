"""
LLM utility for agents.

Delegates to IBMWatsonxService so there is a single IBM watsonx.ai
call path across the entire application.
"""

from app.services.llm_service import IBMWatsonxService

_service = IBMWatsonxService()


def generate_text(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    Generate text using IBM watsonx.ai.

    This function preserves the public interface used by all agents,
    while delegating to the shared IBMWatsonxService implementation.
    """

    return _service.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
