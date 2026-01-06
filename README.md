\# RAG Evaluation Harness (TF-IDF Baseline → LLM → Regression)

A small, high-impact project that demonstrates \*\*end-to-end AI engineering practices\*\* for Retrieval-Augmented Generation (RAG) systems, including \*\*data curation, retrieval, evaluation, and regression detection\*\*.

This repository is intentionally \*\*model-agnostic\*\* and \*\*cloud-agnostic\*\*, with a clean path to integrate \*\*Azure OpenAI\*\* or \*\*Azure AI Search\*\*.

\---

\## Project Goals

This project showcases how real AI teams should answer:

\- Are my RAG answers \*\*correct\*\*?

\- Are they \*\*grounded\*\* in retrieved context?

\- Did quality \*\*improve or regress\*\* after a change?

\- Can I measure this \*\*repeatably\*\*?

Rather than focusing on model training, the emphasis is on \*\*system quality, evaluation, and engineering discipline\*\*.

\---

\## What This Demonstrates

\- Enterprise-style dataset curation (no scraping)

\- TF-IDF retrieval baseline (cheap, explainable)

\- End-to-end RAG execution pipeline

\- Evaluation metrics (correctness, grounding, policy violations)

\- Regression comparison across runs

\- Reproducible artifacts for CI / review

This mirrors how AI systems are built and reviewed in production.

\---

\## Repository Structure

rag-eval-harness/

├── README.md

├── pyproject.toml

├── .env.example

├── data/

│ ├── documents.jsonl # Knowledge base (enterprise-style docs)

│ └── eval_set.jsonl # Gold evaluation questions

├── src/

│ └── rag_eval/

│ ├── init.py

│ ├── retrieve.py # TF-IDF retriever

│ ├── llm.py # Mock / pluggable LLM client

│ ├── run.py # End-to-end RAG pipeline

│ ├── evaluate.py # Scoring + metrics

│ ├── report.py # Regression comparison

│ └── summary.py # Aggregate metrics

├── tests/

│ └── test_retrieve.py

└── results/

├── run_.jsonl

├── eval_.csv

└── regression_\*.md

\---

\## Environment Setup (Windows)

\### 1) Create and activate virtual environment

\*\*Git Bash\*\*

\`\`\`bash

python -m venv .venv

source .venv/Scripts/activate

Verify:

python -c "import sys; print(sys.executable)"

**2) Install dependencies**

python -m pip install -U pip

python -m pip install scikit-learn pytest python-dotenv httpx

python -m pip install -e .

**Step 1 - Create the Dataset**

data/documents.jsonl

Enterprise-style knowledge base (one JSON object per line):

{

"doc_id": "ai-001",

"title": "Azure OpenAI quota limits",

"text": "Azure OpenAI enforces token-per-minute and request-per-minute quotas..."

}

data/eval_set.jsonl

Gold evaluation set with explicit constraints:

{

"id": "q-001",

"question": "Why might an Azure OpenAI request return HTTP 429 errors?",

"expected": "HTTP 429 errors occur when quota limits are exceeded...",

"must_include": \["quota", "retry"\],

"must_not_include": \["increase temperature"\]

}

These fields enable automated evaluation later.

**Step 2 - TF-IDF Retrieval Baseline**

Retrieve top-K documents using TF-IDF + cosine similarity.

**Quick retrieval test**

python -m rag_eval.ingest \\

\--query "Why do I get HTTP 429 from Azure OpenAI?" \\

\--top-k 3

Why TF-IDF first?

- Cheap
- Explainable
- Perfect baseline before embeddings / Azure AI Search

**Step 3 - Run End-to-End RAG Pipeline**

**3A: Retrieval + context (no LLM)**

python -m rag_eval.run --top-k 3

This writes:

results/run_&lt;timestamp&gt;.jsonl

Each record contains:

- retrieved documents
- constructed context
- latency metrics

**3B: Add LLM (Mock by default)**

The pipeline uses a pluggable LLM client.

By default it runs offline using MockLLMClient.

Later you can replace it with:

- Azure OpenAI
- OpenAI
- Any HTTP LLM endpoint

**Step 4 - Evaluation Metrics**

Convert a run into measurable quality signals.

**Run evaluation**

python -m rag_eval.evaluate \\

\--run results/run_&lt;timestamp&gt;.jsonl

Outputs:

results/eval_&lt;timestamp&gt;.csv

**Metrics Included**

must_include_score (0-1)

must_not_include_violations (count)

grounding_score (0-1)

latency metadata

**Optional summary**

python -m rag_eval.summary --csv results/eval_&lt;timestamp&gt;.csv

**Step 5 - Regression Comparison**

Compare two evaluation runs to detect quality changes.

python -m rag_eval.report \\

\--old results/eval_&lt;old&gt;.csv \\

\--new results/eval_&lt;new&gt;.csv

**Outputs:**

results/regression_&lt;timestamp&gt;.csv

results/regression_&lt;timestamp&gt;.md

The Markdown report highlights:

- average metric deltas
- newly introduced violations
- worst-regressing questions

**Environment Variables (Optional)**

See .env.example for Azure OpenAI configuration:

AZURE_OPENAI_ENDPOINT=https://&lt;resource&gt;.openai.azure.com/

AZURE_OPENAI_API_KEY=...

AZURE_OPENAI_DEPLOYMENT=...

AZURE_OPENAI_API_VERSION=2024-10-21

**Testing**

pytest -q

**Extensions (Future Work)**

Replace TF-IDF with Azure AI Search

Add embedding retriever comparison

Add CI guardrails (fail build on regression)

Add cost / token tracking

Add Azure OpenAI client