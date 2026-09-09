from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class InterviewRAG:
    """
    Lightweight local RAG engine for the Interview Trainer Agent.

    Uses:
    - Sentence Transformers for embeddings
    - FAISS for vector similarity search
    - Local Markdown/text knowledge documents
    """

    def __init__(
        self,
        documents_dir: str = "rag/documents",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.documents_dir = Path(documents_dir)
        self.model = SentenceTransformer(model_name)

        self.chunks: List[str] = []
        self.metadata: List[Dict[str, str]] = []
        self.index = None

        self._load_documents()

    def _load_documents(self) -> None:
        """Load and chunk supported knowledge-base files."""

        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Knowledge directory not found: {self.documents_dir}"
            )

        files = list(self.documents_dir.glob("*.md"))
        files += list(self.documents_dir.glob("*.txt"))

        if not files:
            raise FileNotFoundError(
                f"No knowledge documents found in {self.documents_dir}"
            )

        for file_path in files:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            file_chunks = self._chunk_text(text)

            for chunk in file_chunks:
                self.chunks.append(chunk)
                self.metadata.append(
                    {
                        "source": file_path.name,
                    }
                )

        if not self.chunks:
            raise ValueError("Knowledge documents contain no usable text.")

        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> List[str]:
        """Split text into overlapping chunks."""

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - overlap, start + 1)

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant knowledge chunks.
        """

        if not query or not query.strip():
            raise ValueError("RAG query cannot be empty.")

        if self.index is None:
            raise RuntimeError("RAG index has not been initialized.")

        top_k = max(1, min(top_k, len(self.chunks)))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                {
                    "content": self.chunks[index],
                    "source": self.metadata[index]["source"],
                    "score": float(score),
                }
            )

        return results

    def build_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Build a clean context block for an LLM prompt.
        """

        results = self.search(
            query=query,
            top_k=top_k,
        )

        if not results:
            return ""

        context_parts = []

        for number, result in enumerate(results, start=1):
            context_parts.append(
                f"[Retrieved Source {number}]\n"
                f"Source: {result['source']}\n"
                f"Relevance: {result['score']:.4f}\n"
                f"{result['content']}"
            )

        return "\n\n".join(context_parts)


class InterviewRetriever:
    """
    Adapter that gives RAGEngine a build_context() interface
    over an existing VectorStore + embeddings.

    RAGEngine builds and owns the VectorStore; this class
    handles query embedding and result formatting.
    """

    def __init__(self, vector_store) -> None:
        from rag.embeddings import get_embedding_model
        self._store = vector_store
        self._model = get_embedding_model()

    def build_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Embed the query, search the vector store, and return a
        formatted context block ready for injection into a prompt.
        """

        import numpy as np

        if not query or not query.strip():
            return ""

        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_embedding, dtype="float32")

        results = self._store.search(query_vector, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for number, result in enumerate(results, start=1):
            source = result.get("source", "unknown")
            score = result.get("score", 0.0)
            text = result.get("text", result.get("content", ""))
            context_parts.append(
                f"[Retrieved Source {number}]\n"
                f"Source: {source}\n"
                f"Relevance: {score:.4f}\n"
                f"{text}"
            )

        return "\n\n".join(context_parts)
