from __future__ import annotations
from rag_eval.llm import MockLLMClient


import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from rag_eval.retrieve import load_documents_jsonl, TfidfRetriever


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {p} line {line_no}: {e}") from e
    if not rows:
        raise ValueError(f"No rows found in {p}")
    return rows


def build_context(retrieved: List[Dict[str, Any]]) -> str:
    # Keep a simple, readable context format for now.
    blocks = []
    for i, d in enumerate(retrieved, start=1):
        blocks.append(f"[Doc {i}] {d['doc_id']} — {d['title']}\n{d['text']}")
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval over eval_set and save results.")
    parser.add_argument("--docs", default="data/documents.jsonl", help="Path to documents.jsonl")
    parser.add_argument("--eval", default="data/eval_set.jsonl", help="Path to eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=3, help="Number of docs to retrieve per question")
    parser.add_argument("--outdir", default="results", help="Output directory for run artifacts")
    args = parser.parse_args()

    docs = load_documents_jsonl(args.docs)
    retriever = TfidfRetriever(docs)
    llm = MockLLMClient()

    eval_rows = load_jsonl(args.eval)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outdir / f"run_{ts}.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        for row in eval_rows:
            qid = row.get("id", "")
            question = row["question"]

            t0 = time.perf_counter()
            hits = retriever.retrieve(question, top_k=args.top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            retrieved_docs = [
                {"doc_id": h.doc_id, "title": h.title, "text": h.text, "score": h.score}
                for h in hits
            ]
            
            system = (
                "You are a helpful assistant. "
                "Answer using ONLY the provided context. "
                "If the context is insufficient, say you don't know."
            )
            
            user = f"Question:\n{question}\n\nContext:\n{build_context(retrieved_docs)}"
            
            t1 = time.perf_counter()
            resp = llm.generate(system=system, user=user)
            llm_latency_ms = (time.perf_counter() - t1) * 1000.0


            record = {
                "id": qid,
                "question": question,
                "expected": row.get("expected", ""),
                "must_include": row.get("must_include", []),
                "must_not_include": row.get("must_not_include", []),
                "retrieved": retrieved_docs,
                "context": build_context(retrieved_docs),
                "retrieval_latency_ms": round(latency_ms, 2),
                # placeholder for step 3B:
                "answer": resp.text,
                "llm_latency_ms": round(llm_latency_ms, 2),
                "model": resp.model,
                "tokens": resp.tokens,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
