from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer embedding model once.
    """

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def create_embeddings(texts: list[str]):
    """
    Convert text documents into vector embeddings.
    """

    if not texts:
        raise ValueError("Cannot create embeddings from empty text.")

    model = get_embedding_model()

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )