# README

# NyayaGraph: A **Multimodal Dual Knowledge GraphRAG for Consumer Legal Assistance in India**

A legal GraphRAG backend (FastAPI + Neo4j + Groq) that answers questions about a user’s own consumer case. It retrieves facts from the user’s private case documents, finds the governing provisions of the Consumer Protection Act, 2019 and its rules, and returns a citation-checked JSON answer.

## The one rule

`docs/` is the specification and the source of truth. The code implements it. Every rule has an id (R-09, E-05, P-06, SC-22 and so on), and those ids appear in code comments, so you can go from a doc rule to the code that enforces it and back.

## Reading order (about 2 hours for the docs)

1. **`docs/00-overview.md`** (5 min). What the product is, what is in and out of scope, and the fixed technology choices.
2. **`docs/01-architecture-and-data-flow.md`** (20 min). The most important file. It covers the architecture diagram, the configuration variables, and the three pipelines step by step:
    - **A1 to A8:** ingesting a statute PDF into the Legal KB.
    - **B1 to B10:** ingesting a user’s case documents (PDF, image, audio, text).
    - **C1 to C11:** answering a question (planner, retrieval, reranking, answer, verifier).
3. **`docs/06-file-structure.md`** (10 min). Where every file lives, and the layering rule: `api -> agents/ingestion -> services -> schemas`, never upward.
4. **`docs/05-roadmap.md`** (10 min). Phases 0 to 6, what each delivers, and each phase’s exit criteria.
5. **Reference files, read as needed.** You do not need to read these end to end:
    - **`docs/02-data-models.md`:** node ids, Neo4j indexes, node properties, relationships, the extraction models and the agent state.
    - **`docs/03-api-contracts.md`:** the endpoints, error codes and the JSON response.
    - **`docs/04-rules-and-edge-cases.md`:** every rule (R-01 to R-27), the retrieval settings, the statute-parsing rules (P-01 to P-21) and failure handling (E-01 to E-15).
6. **`docs/07-ingestion-integration.md`** (about 30 min; essential for whoever built the ingestion prototype). How the prototype (`New-Ingestion.zip`, kept read-only in `reference/ingestion-prototype/`) was brought into this project:

Never import `reference/` from `app/` or `tests/`.
    - what was kept, what was rewritten, and what was dropped (Qdrant, OpenAI, WhisperX and others are not allowed by the spec);
    - the defects found (IN-01 to IN-13);
    - which proposals were approved (SC-1 to SC-4).
7. **`docs/README.md`.** The index plus the project’s decision log. It is long because every decision is recorded:

Skim the Assumptions section once, then search it by id when the code mentions something like “A-35” or “SC-22”.
    - **D-x:** deviations the team approved, such as the Groq model and the RTX 3050 laptop GPU.
    - **A-x:** assumptions made where the spec was silent.
    - **SC-x:** proposed spec changes and whether they were approved (SC-5 to SC-23 so far).
    - **O-x:** observations from testing, such as the Groq 200,000-tokens-a-day limit and the reranker findings.
    - **Phase exit evidence:** the actual test results for each phase.

**`AGENTS.md`** (in the project root) and **`.agents/rules/`** are the instructions for the AI coding agent. They are also a good one-page summary of the hard rules: all database queries in one file, no logging of case text, the verifier is authoritative, and so on.

## Then the code (`app/`), in pipeline order

- **`app/ingestion/`:** legal ingestion (`legal/`), case ingestion (`loaders/`, `chunking/`, `case/`) and the job queue (`jobs.py`). This is where the prototype’s work ended up.
- **`app/services/`:** Neo4j access (`retrieval/graph_queries.py` holds *all* the database queries), embeddings, search, reranking, Groq, OCR and Whisper.
- **`app/agents/`:** the question-answering pipeline. `graph.py` wires it, and `nodes/` has one file per step: planner, case_retriever, issue_builder, legal_retriever, graph_expander, legal_reranker, responder, verifier.
- **`app/prompts/`:** every prompt sent to the LLM.
- **`app/api/`:** the FastAPI endpoints, which only validate input and hand off.
- **`tests/`:**
    - `unit/` (fast, no services needed);
    - `integration/` (needs Neo4j; some tests need Groq);
    - `golden/` (statute parsing checks G1 to G12);
    - `eval/` (31 labelled questions and `run_eval.py`).
- **`scripts/`:** `smoke_check.py`, `init_neo4j.py`, `ingest_legal.py`.

## Where the project stands (23 November 2026)

- Phases 0 to 6 are done: environment, foundations, Legal KB, case ingestion, question answering, API hardening, and verification and operations.
- Phase 4 exit: recall at 8 0.865 on 31 labelled questions, zero unverified citations, p95 latency 8.2 s, user-isolation tests pass, and `pytest -q` 514 passed.
- Phase 5 (API hardening) is done: rate limits, `/health`, the provision endpoint, a contract test for every error code, and `python -m app.main` binding to `APP_HOST`. `pytest -q`: 559 passed.
- Phase 6 is done: the soak test (no 500s under load), restart recovery after a forced kill, the log-scrubbing test, and `SETUP.md`. `pytest -q`: 565 passed.
- Still open for the v1 definition of done: the CI workflow (`.github/workflows/ci.yml`, A-48) is written but has not run yet, because the repository is not on GitHub yet; and `SETUP.md` has not been tried on a clean machine.

## Running it

Follow “Commands” in `AGENTS.md`. You need:

- Neo4j Desktop 2026.10.2;
- Ollama with `nomic-embed-text`;
- the CUDA build of PyTorch (D-3 in `docs/README.md`);
- a `.env` copied from `.env.example`.

Run `python -m scripts.smoke_check` first; all 4 checks should print OK. Groq’s free tier allows 200,000 tokens a day, so do not run the evaluation (`tests/eval/run_eval.py`) or the live-model integration tests casually. The full step-by-step setup is in `SETUP.md`.