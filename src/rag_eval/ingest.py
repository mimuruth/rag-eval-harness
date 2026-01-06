from __future__ import annotations

import argparse

from rag_eval.retrieve import load_documents_jsonl, TfidfRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick TF-IDF retrieval test")
    parser.add_argument("--docs", default="data/documents.jsonl", help="Path to documents.jsonl")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of docs to retrieve")
    args = parser.parse_args()

    docs = load_documents_jsonl(args.docs)
    retriever = TfidfRetriever(docs)
    hits = retriever.retrieve(args.query, top_k=args.top_k)

    print("Query:", args.query)
    print("\nTop hits:")
    for h in hits:
        print(f"- {h.doc_id} (score={h.score:.3f}) — {h.title}")


if __name__ == "__main__":
    main()
