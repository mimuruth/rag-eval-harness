from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    score: float


def load_documents_jsonl(path: str | Path) -> List[Document]:
    """Load documents from a JSONL file (one JSON object per line)."""
    p = Path(path)
    docs: List[Document] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docs.append(
                    Document(
                        doc_id=str(obj["doc_id"]),
                        title=str(obj.get("title", "")),
                        text=str(obj.get("text", "")),
                    )
                )
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {p} line {line_no}: {e}") from e
    if not docs:
        raise ValueError(f"No documents found in {p}")
    return docs


class TfidfRetriever:
    """Small-corpus TF-IDF retriever with cosine similarity ranking."""

    def __init__(self, documents: Sequence[Document], *, max_features: int = 5000):
        if not documents:
            raise ValueError("documents must not be empty")
        self._docs = list(documents)

        corpus = [f"{d.title}\n{d.text}" for d in self._docs]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
        )
        self._doc_matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, *, top_k: int = 3) -> List[RetrievedDoc]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._doc_matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]

        results: List[RetrievedDoc] = []
        for idx in ranked_idx:
            d = self._docs[int(idx)]
            results.append(
                RetrievedDoc(
                    doc_id=d.doc_id,
                    title=d.title,
                    text=d.text,
                    score=float(sims[int(idx)]),
                )
            )
        return results
