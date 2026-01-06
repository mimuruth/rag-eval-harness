from rag_eval.retrieve import load_documents_jsonl, TfidfRetriever


def test_retrieve_returns_results():
    docs = load_documents_jsonl("data/documents.jsonl")
    retriever = TfidfRetriever(docs)
    hits = retriever.retrieve("HTTP 429 quota retry backoff", top_k=3)
    assert len(hits) == 3
    assert all(h.score >= 0.0 for h in hits)
