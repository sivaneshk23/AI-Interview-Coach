from pathlib import Path
import json

import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector store for interview knowledge.
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(self, embeddings, documents: list[dict]) -> None:

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {vectors.shape[1]}."
            )

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ) -> list[dict]:

        if not self.documents:
            return []

        top_k = max(1, min(top_k, len(self.documents)))

        query_vector = np.asarray(
            query_embedding,
            dtype="float32",
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_vector,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            document = dict(self.documents[index])

            document["score"] = float(score)

            results.append(document)

        return results

    def save(self, directory: str = "data/vector_store") -> None:

        directory_path = Path(directory)
        directory_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(directory_path / "index.faiss"),
        )

        with open(
            directory_path / "documents.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, directory: str = "data/vector_store"):

        directory_path = Path(directory)

        index_path = directory_path / "index.faiss"
        documents_path = directory_path / "documents.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not documents_path.exists():
            raise FileNotFoundError(
                f"Document metadata not found: {documents_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            documents_path,
            "r",
            encoding="utf-8",
        ) as file:

            documents = json.load(file)

        store = cls(index.d)

        store.index = index
        store.documents = documents

        return store