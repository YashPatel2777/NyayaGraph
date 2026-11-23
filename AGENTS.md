# Legal GraphRAG Platform: agent instructions

You are implementing a FastAPI backend that answers legal questions about a user's own case. It retrieves facts from a private Case KB, finds governing provisions in a shared Legal KB (seeded with the Consumer Protection Act, 2019, India), and returns a validated, citation-grounded JSON analysis.

The full specification is in `docs/`. It is the source of truth. If this file and the spec disagree, the spec wins; report the conflict instead of guessing.

## Read first, in this order

1. `docs/README.md` (index and open items)
2. `docs/00-overview.md` (scope and fixed decisions)
3. `docs/06-file-structure.md` (where every file goes)
4. `docs/05-roadmap.md` (phases and exit criteria)
5. `docs/07-ingestion-integration.md` before any Phase 3 work (ingestion loaders, chunking, jobs)
6. The spec file for the area you are touching: `01` architecture and data flow, `02` data models, `03` API contracts, `04` rules, parsing rules and edge cases.

Section numbers (1.2, 3.2, R-09, E-05, P-06, G4) are global. Cite them in code comments where a rule is enforced.

## How to work

- Work one roadmap phase at a time, in order (Phase 0 to 6). Do not start a phase until the previous phase's exit criteria pass.
- Before writing code for a phase, state which spec sections and rules you will implement and which files you will create. Create files only at the paths in `docs/06-file-structure.md`.
- A phase is done only when its exit criteria are demonstrably met by running tests or commands. Show the output. Do not claim a test passes without running it.
- Write complete, runnable code. No pseudocode, no `TODO` stubs standing in for required behaviour.
- If the spec is silent or ambiguous, ask or record the assumption in `docs/README.md` under "Assumptions". Do not invent API fields, node properties, error codes or config variables.

## Fixed decisions (do not change)

- Python 3.11, FastAPI, Uvicorn, LangGraph with MemorySaver, Windows 11 host, NVIDIA RTX 3050 Laptop GPU with 4 GB (D-2; the spec's RTX 4070 was replaced by the operator).
- Neo4j 2026.10.2 (SC-15: the case vector query uses the Cypher 25 SEARCH clause) is the ONLY database: graph, vector indexes and full-text indexes. No Qdrant, no other store.
- Embeddings: `nomic-embed-text` through Ollama, 768 dimensions, cosine. ONE embedding model, no fallback model (R-07, R-08).
- LLM: Groq `openai/gpt-oss-120b` with `reasoning_effort="low"` through `langchain-groq` (D-1: `llama-3.3-70b-versatile` is no longer served). No LLM gateway. Free tier: 8,000 tokens per minute and 200,000 tokens per day (O-6); do not spend it casually.
- Reranker: FlashRank `ms-marco-MiniLM-L-12-v2` (SC-21). Speech: faster-whisper. OCR: EasyOCR (en, hi) with PyMuPDF at 200 DPI.
- NO Docker. NO authentication of any kind in v1: no JWT, no tokens, no login, no roles. Every request runs as `DEFAULT_USER_ID` through `app/deps.py` (3.2). Do not add auth code, auth dependencies on routes, or a users table. Bind to `127.0.0.1`.
- Guardrails are out of scope; only the no-op hook in `app/guardrails/hook.py` exists (1.3).

## Hard rules

- All Cypher lives in `app/services/retrieval/graph_queries.py`. Every value is a query parameter. Never build Cypher or Lucene strings with f-strings or concatenation of user data (R-01, R-03).
- Every case-scoped query goes through `scope.py`, which supplies `case_id` and `user_id`. A request can never set `user_id` (R-02).
- Only `EmbeddingService` creates vectors and adds the `search_document: ` or `search_query: ` prefix. Every vector must be exactly 768 finite floats (R-07, R-08). Startup must abort on a `Meta` node mismatch (2.2).
- Never log case text, transcripts or question text. Ids, lengths and timings only (R-04).
- Case text is data, not instructions: it goes in a delimited CASE MATERIAL block, and the LLM has no tools (R-06).
- The verifier is authoritative. The model returns ids and relevance text only; code fills citation, status, excerpt and source from Neo4j. Remove any citation not in the retrieved set (R-20, R-21, R-23).
- Confidence and the disclaimer are produced by code, never by the model (R-24, R-25).
- Do not translate references to the IPC or CrPC into BNS or BNSS. Report them as written (R-26).
- State a forum or pecuniary threshold only if a retrieved provision contains it (R-27).
- Blocking calls run through `anyio.to_thread.run_sync`. Nothing blocking on the event loop (1.2).
- Layering: `api -> agents/ingestion -> services -> schemas`. A lower layer never imports a higher one.
- Prompts live only in `app/prompts/`.
- Ids are deterministic and all writes use MERGE, so re-running any job is idempotent (2.1, R-16).

## Reference code

- `reference/ingestion-prototype/` (if present) is an older pipeline kept for reading only. Never import it from `app/` or `tests/`. Port only what `docs/07-ingestion-integration.md` section 5 allows, into the paths in `docs/06-file-structure.md`.
- Its Qdrant, OpenAI, WhisperX, Pyannote, Docling, Unstructured, HTML and Office code must not appear in `app/`.
- SC-1 to SC-4 in that file are proposals. Do not act on them until the operator says they are approved.

## Do not fabricate

- Do not invent legal content. The Legal KB comes only from files the operator provides in `data/legal/`.
- The revised pecuniary thresholds in the Consumer Protection (Jurisdiction) Rules, 2021 are NOT verified. Do not hard-code any threshold. They enter the system only by ingesting the official rules text.
- Do not invent labelled evaluation data. `tests/eval/questions.jsonl` is written with the operator.

## Commands (create these as the phases deliver them)

- Install: install the CUDA builds of torch and torchvision first, together from the same index (D-3 in `docs/README.md`): `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`, then `pip install -r requirements.txt`. Then `ollama pull nomic-embed-text`.
- Smoke check: `python -m scripts.smoke_check`
- Init database: `python -m scripts.init_neo4j`
- Ingest legal text: `python -m scripts.ingest_legal data/legal/CPA2019.pdf --act-id cpa-2019 --metadata data/legal/CPA2019.metadata.json` (all four Legal KB commands, in order, are in `SETUP.md`)
- Run: `python -m app.main` (binds APP_HOST on port 8000), or `uvicorn app.main:app --host 127.0.0.1 --port 8000`
- Test: `pytest -q`; golden tests only: `pytest tests/golden -q`; opt-in Phase 6 tests: `pytest -m soak`, `pytest -m process` (A-46); evaluation: `python -m tests.eval.run_eval`

## Things you cannot verify, so flag them if they bite

- Neo4j's full-text `limit` option behaviour assumed in 4.2 (checked on 2026.10.2: it returns exactly n rows, O-1 probe)
- The score range of FlashRank, which `RERANK_HIGH` depends on (R-24; measured 0 to 1 for the default model, O-5; RERANK_HIGH not yet re-calibrated for MiniLM)
- The first Neo4j release that supports filters inside SEARCH, and whether the Community edition supports them (2026.10.2 enterprise is tested; SC-15)
- Windows cuDNN/cuBLAS DLL matching for faster-whisper (worked on the target laptop in Phase 3; check with `python -m scripts.smoke_check --whisper`)

When one of these blocks you, stop, describe what you observed, and propose a change to the spec. Do not silently work around it.
