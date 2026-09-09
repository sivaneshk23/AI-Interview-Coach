from rag.retriever import InterviewRAG


def test_rag_retrieval():
    rag = InterviewRAG()

    results = rag.search(
        "What should a Data Analyst know about Pandas and SQL?",
        top_k=3,
    )

    assert results
    assert len(results) <= 3

    for result in results:
        assert "content" in result
        assert "source" in result
        assert "score" in result


def test_rag_context():
    rag = InterviewRAG()

    context = rag.build_context(
        "How should behavioral interview answers be structured?",
        top_k=3,
    )

    assert context
    assert "Retrieved Source" in context