from functools import lru_cache

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from app.config import get_settings


@lru_cache(maxsize=1)
def get_model() -> ModelInference:
    """
    Create and cache the IBM watsonx.ai model client.

    The model client is created only once and reused
    throughout the application lifetime.
    """

    settings = get_settings()

    credentials = Credentials(
        url=settings.watsonx_url,
        api_key=settings.ibm_api_key,
    )

    model = ModelInference(
        model_id=settings.model_id,
        credentials=credentials,
        project_id=settings.ibm_project_id,
    )

    return model


def clear_model_cache() -> None:
    """
    Clear the cached model instance.

    Used in tests to reset state between runs with different
    environment configurations.
    """
    get_model.cache_clear()
