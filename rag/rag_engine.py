from pathlib import Path

from rag.document_loader import load_documents
from rag.embeddings import create_embeddings
from rag.vector_store import VectorStore
from rag.retriever import InterviewRetriever


class RAGEngine:
    """
    Complete retrieval pipeline for the Interview Trainer.
    """

    def __init__(
        self,
        knowledge_directory: str = "data/knowledge",
        vector_directory: str = "data/vector_store",
    ):

        self.knowledge_directory = knowledge_directory
        self.vector_directory = vector_directory

        vector_path = Path(vector_directory)

        if (
            vector_path.exists()
            and (vector_path / "index.faiss").exists()
            and (vector_path / "documents.json").exists()
        ):

            self.vector_store = VectorStore.load(
                vector_directory
            )

        else:

            documents = load_documents(
                knowledge_directory
            )

            texts = [
                document["text"]
                for document in documents
            ]

            embeddings = create_embeddings(
                texts
            )

            dimension = embeddings.shape[1]

            self.vector_store = VectorStore(
                dimension=dimension
            )

            self.vector_store.add(
                embeddings,
                documents,
            )

            self.vector_store.save(
                vector_directory
            )

        self.retriever = InterviewRetriever(
            self.vector_store
        )

    def retrieve_context(
        self,
        role: str,
        experience_level: str,
        interview_type: str,
        candidate_context: str = "",
        top_k: int = 5,
    ) -> str:

        query = f"""
Target job role: {role}

Experience level: {experience_level}

Interview type: {interview_type}

Candidate context:
{candidate_context}

Find the most relevant interview preparation knowledge
for this candidate and target role.
"""

        return self.retriever.build_context(
            query=query,
            top_k=top_k,
        )