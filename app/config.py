import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Central configuration for the AI Interview Trainer Agent.
    """

    ibm_api_key: str
    ibm_project_id: str
    ibm_region: str
    model_id: str
    rag_documents_dir: str
    rag_vector_dir: str

    @property
    def watsonx_url(self) -> str:
        """
        Build the IBM watsonx.ai endpoint from the configured region.
        """

        return f"https://{self.ibm_region}.ml.cloud.ibm.com"


def _required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Check your .env file."
        )

    return value.strip()


def get_settings() -> Settings:
    return Settings(
        ibm_api_key=_required_env("IBM_API_KEY"),
        ibm_project_id=_required_env("IBM_PROJECT_ID"),
        ibm_region=os.getenv("IBM_REGION", "jp-tok").strip(),
        model_id=os.getenv(
            "MODEL_ID",
            "ibm/granite-4-h-small",
        ).strip(),
        rag_documents_dir=os.getenv(
            "RAG_DOCUMENTS_DIR",
            "data/knowledge",
        ).strip(),
        rag_vector_dir=os.getenv(
            "RAG_VECTOR_DIR",
            "data/vector_store",
        ).strip(),
    )