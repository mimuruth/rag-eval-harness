from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def must_include_score(answer: str, must_include: List[str]) -> float:
    """
    Score 0..1 based on the fraction of required phrases present.
    If must_include is empty, return 1.0 (no requirements).
    """
    if not must_include:
        return 1.0
    a = _normalize(answer)
    hits = 0
    for phrase in must_include:
        p = _normalize(phrase)
        if p and p in a:
            hits += 1
    return hits / len(must_include)


def must_not_include_violations(answer: str, must_not_include: List[str]) -> int:
    """
    Count how many forbidden phrases appear in the answer.
    """
    if not must_not_include:
        return 0
    a = _normalize(answer)
    violations = 0
    for phrase in must_not_include:
        p = _normalize(phrase)
        if p and p in a:
            violations += 1
    return violations


def grounding_score(answer: str, context: str) -> float:
    """
    Heuristic grounding score 0..1.
    We compute TF-IDF cosine similarity between answer and context.
    - 0.0 means no overlap
    - closer to 1.0 means strong overlap
    """
    answer = answer.strip()
    context = context.strip()
    if not answer or not context:
        return 0.0

    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vect.fit_transform([answer, context])
    sim = cosine_similarity(mat[0:1], mat[1:2]).flatten()[0]
    # clamp just in case
    if sim < 0:
        return 0.0
    if sim > 1:
        return 1.0
    return float(sim)


def evaluate_run(run_path: str | Path) -> List[Dict[str, Any]]:
    """
    Read a run_*.jsonl file and produce per-question metrics.
    """
    rows = load_jsonl(run_path)
    out: List[Dict[str, Any]] = []

    for row in rows:
        qid = row.get("id", "")
        question = row.get("question", "")
        expected = row.get("expected", "")
        answer = row.get("answer") or ""  # mock may be used
        context = row.get("context") or ""

        mi = row.get("must_include", []) or []
        mni = row.get("must_not_include", []) or []

        mi_score = must_include_score(answer, list(mi))
        violations = must_not_include_violations(answer, list(mni))
        g_score = grounding_score(answer, context)

        out.append(
            {
                "id": qid,
                "question": question,
                "expected": expected,
                "answer": answer,
                "must_include_score": round(mi_score, 3),
                "must_not_include_violations": violations,
                "grounding_score": round(g_score, 3),
                "retrieval_latency_ms": row.get("retrieval_latency_ms"),
                "llm_latency_ms": row.get("llm_latency_ms"),
                "model": row.get("model"),
                "top_doc_ids": ",".join([d.get("doc_id", "") for d in (row.get("retrieved") or [])]),
            }
        )

    return out


def write_eval_csv(rows: List[Dict[str, Any]], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # stable column order
    fieldnames = [
        "id",
        "question",
        "expected",
        "answer",
        "must_include_score",
        "must_not_include_violations",
        "grounding_score",
        "retrieval_latency_ms",
        "llm_latency_ms",
        "model",
        "top_doc_ids",
    ]

    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a run_*.jsonl file and write eval CSV.")
    parser.add_argument("--run", required=True, help="Path to results/run_*.jsonl")
    parser.add_argument("--out", default=None, help="Output CSV path (default: results/eval_<ts>.csv)")
    args = parser.parse_args()

    run_path = Path(args.run)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else Path("results") / f"eval_{ts}.csv"

    rows = evaluate_run(run_path)
    write_eval_csv(rows, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
