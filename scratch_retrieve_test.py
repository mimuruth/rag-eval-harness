from rag_eval.retrieve import load_documents_jsonl, TfidfRetriever

docs = load_documents_jsonl("data/documents.jsonl")
retriever = TfidfRetriever(docs)

query = "Why do I get HTTP 429 from Azure OpenAI and what should I do?"
hits = retriever.retrieve(query, top_k=3)

print("Query:", query)
print("\nTop hits:")
for h in hits:
    print("-", h.doc_id, f"(score={h.score:.3f})", "=>", h.title)
