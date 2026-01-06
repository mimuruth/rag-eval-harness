from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    tokens: Optional[dict] = None


class BaseLLMClient:
    def generate(self, *, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Offline client so the pipeline runs without any API keys."""

    def generate(self, *, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        # A simple deterministic placeholder "answer"
        # Useful to validate end-to-end flow before real LLM integration.
        answer = (
            "MOCK ANSWER: I will answer using the provided context only.\n\n"
            "Summary: Based on the retrieved context, the most likely causes and mitigations are described above."
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(text=answer, model="mock", latency_ms=latency_ms, tokens=None)
