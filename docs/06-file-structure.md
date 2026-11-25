# File and folder structure

> Part of the Legal GraphRAG Platform spec (v1.1). Section numbers refer to the other files in `docs/`.

This is the target layout. Create files at exactly these paths. Each file lists the spec section that defines its behaviour and the roadmap phase (05-roadmap.md) in which it is built.

## 1. Full tree

```
legal-graphrag/
├── AGENTS.md                          # rules for the coding agent (always active)
├── README.md                          # how to run the project (written in Phase 6 as SETUP.md content)
├── SETUP.md                           # Phase 6: Neo4j, Ollama, CUDA PyTorch, env vars
├── requirements.txt                   # Phase 0 (torch is NOT listed; install the CUDA build first)
├── .env.example                       # Phase 1: every variable in 1.3, no secrets
├── .gitignore                         # .env, data/uploads/, __pycache__/, .venv/
├── pytest.ini                         # Phase 1 (asyncio_mode = auto)
│
├── .agents/
│   └── rules/                         # Antigravity rules (see AGENTS.md); not application code
│
├── docs/
│   ├── README.md                      # index, reading order, changelog
│   ├── 00-overview.md                 # scope, fixed decisions
│   ├── 01-architecture-and-data-flow.md
│   ├── 02-data-models.md
│   ├── 03-api-contracts.md
│   ├── 04-rules-and-edge-cases.md
│   ├── 05-roadmap.md
│   └── 06-file-structure.md           # this file
│
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app + lifespan: load models once, startup checks, exit code 2 (1.2)      [Ph 1, 5]
│   ├── config.py                      # pydantic-settings, every variable in 1.3                                        [Ph 1]
│   ├── deps.py                        # get_current_user() returns DEFAULT_USER_ID; the ONLY place identity is decided (3.2) [Ph 1]
│   ├── errors.py                      # AppError, error codes, exception handlers, error envelope (3.3)                  [Ph 1]
│   ├── middleware.py                  # X-Request-ID, per-IP sliding-window rate limit (3.1, 3.2)                        [Ph 1, 5]
│   ├── observability.py               # Logfire setup with scrubbing on (R-04)                                           [Ph 1]
│   │
│   ├── api/                           # routers only: validate, call a service, return a model
│   │   ├── health.py                  # GET /health (3.4)                                                                [Ph 5]
│   │   ├── cases.py                   # POST/GET/DELETE /v1/cases                                                        [Ph 3]
│   │   ├── documents.py               # POST/GET /v1/cases/{case_id}/documents                                           [Ph 3]
│   │   ├── jobs.py                    # GET /v1/jobs/{job_id}                                                            [Ph 3]
│   │   ├── query.py                   # POST /v1/query                                                                   [Ph 4]
│   │   ├── legal.py                   # GET /v1/legal/provisions/{provision_id}                                          [Ph 5]
│   │   └── admin_legal.py             # POST /v1/admin/legal/ingest, GET /v1/admin/legal/acts (open in v1, see 3.2)      [Ph 2]
│   │
│   ├── schemas/
│   │   ├── graph_schema.py            # constants: labels, relationship types, index names, id builders (2.1)            [Ph 1]
│   │   ├── extraction.py              # ExtractionResult and Ext* models (2.5)                                           [Ph 1]
│   │   ├── output.py                  # AnalysisResponse, Provision, Fact, Confidence, Trace (3.5)                       [Ph 1]
│   │   └── api.py                     # request and response models for every endpoint (3.4)                             [Ph 1]
│   │
│   ├── prompts/                       # one module per prompt; plain string constants with named placeholders
│   │   ├── extraction.py              # case entity extraction                                                           [Ph 3]
│   │   ├── link_verify.py             # claim-to-provision yes/no check                                                  [Ph 3]
│   │   ├── entity_adjudicate.py       # R-14 ambiguous merge check                                                       [Ph 3]
│   │   ├── planner.py                 # intent + search_query                                                            [Ph 4]
│   │   ├── issue_statement.py         # unused since SC-22 (the issue statement equals the query); kept for reference   [Ph 4]
│   │   ├── responder.py               # must follow R-06, R-20, R-26, R-27                                               [Ph 4]
│   │   └── repair.py                  # one-shot "fix this JSON" retry (R-22)                                            [Ph 1]
│   │
│   ├── agents/
│   │   ├── state.py                   # AgentState (2.5)                                                                 [Ph 4]
│   │   ├── graph.py                   # LangGraph wiring and routing (1.4 C)                                             [Ph 4]
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── case_retriever.py
│   │       ├── issue_builder.py
│   │       ├── legal_retriever.py
│   │       ├── graph_expander.py
│   │       ├── legal_reranker.py
│   │       ├── responder.py
│   │       └── verifier.py            # authoritative citation check (R-21, R-23, R-24, R-25)
│   │
│   ├── ingestion/
│   │   ├── jobs.py                    # asyncio.Queue worker, GPU semaphore(1), restart recovery (1.2, E-11)             [Ph 3]
│   │   ├── loaders/
│   │   │   ├── router.py              # MIME and signature to loader (3.4 table)
│   │   │   ├── pdf.py                 # pypdf per page; pages under 40 chars go to OCR
│   │   │   ├── image.py               # EasyOCR
│   │   │   ├── audio.py               # faster-whisper, vad_filter, timestamped segments
│   │   │   ├── chat.py                # WhatsApp and Telegram exports inside text uploads (SC-1)
│   │   │   └── text.py                # UTF-8
│   │   ├── chunking/
│   │   │   ├── splitter.py            # case chunks: 1,200 target / 1,500 max / 150 overlap (B4)
│   │   │   └── legal_splitter.py      # retrieval-unit rule (2.3)                                                        [Ph 2]
│   │   ├── legal/
│   │   │   ├── parser.py              # rules P-01 to P-09, P-12                                                         [Ph 2]
│   │   │   ├── reference_extractor.py # rules P-10, P-11
│   │   │   ├── concept_extractor.py   # optional LLM ABOUT edges (deferred; leave the file out until needed)
│   │   │   └── legal_processor.py     # orchestrates A1 to A8
│   │   └── case/
│   │       ├── extractor.py           # B6
│   │       ├── entity_resolver.py     # R-14
│   │       ├── legal_linker.py        # B9
│   │       └── case_processor.py      # orchestrates B1 to B10
│   │
│   ├── services/
│   │   ├── llm/
│   │   │   └── groq_client.py         # ChatGroq, JSON mode, tenacity retry, retry-after (1.2)                           [Ph 1]
│   │   ├── retrieval/
│   │   │   ├── neo4j_service.py       # driver, sessions, batched UNWIND writes, retry (E-04)                            [Ph 1]
│   │   │   ├── scope.py               # injects case_id and user_id as parameters (R-01)                                 [Ph 1]
│   │   │   ├── embeddings.py          # EmbeddingService: prefixes, 768 check, Meta check (R-07, R-08)                   [Ph 1]
│   │   │   ├── vector_search.py       # over-fetch loop (R-09)                                                           [Ph 4]
│   │   │   ├── fulltext_search.py     # Lucene escaping (R-03)                                                           [Ph 4]
│   │   │   ├── hybrid.py              # RRF, constant 60 (R-10)                                                          [Ph 4]
│   │   │   ├── graph_queries.py       # ALL Cypher lives here (4.2)                                                      [Ph 2, 4]
│   │   │   └── ranking_service.py     # FlashRank wrapper                                                                [Ph 4]
│   │   └── media/
│   │       ├── ocr_service.py         # EasyOCR + PyMuPDF at 200 DPI                                                     [Ph 3]
│   │       └── transcription_service.py  # faster-whisper medium, CUDA float16, CPU fallback (E-08)                      [Ph 3]
│   │
│   └── guardrails/
│       └── hook.py                    # guard(text, user_id) -> GuardResult; no-op while ENABLE_GUARDRAILS=false (1.3)
│
├── scripts/
│   ├── smoke_check.py                 # Phase 0 exit: Neo4j, Ollama 768 floats, CUDA, Groq
│   ├── init_neo4j.py                  # DDL from 2.2, idempotent                                                         [Ph 1]
│   └── ingest_legal.py                # CLI wrapper: python -m scripts.ingest_legal data/legal/CPA2019.pdf --act-id cpa-2019 ... [Ph 2]
│
├── ui/                                # SC-24: Streamlit web UI; an HTTP client of 3.4 only, never imports app/
│   ├── api_client.py                  # httpx wrapper over the 3.4 endpoints; 3.3 envelope -> ApiError
│   ├── rate_budget.py                 # Groq tokens-per-minute cooldown estimate (O-6), pure functions
│   └── streamlit_app.py               # sidebar (health, cases), Ask, Documents, Case tabs; provision dialog
├── .streamlit/config.toml             # SC-24: binds 127.0.0.1:8501, usage statistics off, 200 MB uploads
├── requirements-ui.txt                # SC-24: streamlit (installed on top of requirements.txt)
│
├── tests/
│   ├── conftest.py                    # throwaway Neo4j database fixture
│   ├── unit/                          # scope, embeddings, Lucene escaping, RRF, id builders, verifier
│   ├── golden/
│   │   └── test_cpa_2019.py           # G1 to G9 (05-roadmap.md, Phase 2)
│   ├── integration/                   # ingestion jobs, E-07 to E-11, API contract tests for every code in 3.3
│   └── eval/
│       ├── questions.jsonl            # labelled evaluation set, at least 30 (Phase 4)
│       ├── run_eval.py                # provision recall at 8, unverified-citation count, p95 latency
│       └── simulate_h3_h6.py          # hypothesis tests H3 (verifier) and H6 (latency); simulated data now, real CSVs later
│
└── data/
    ├── legal/                         # source statutes (CPA2019.pdf, Jurisdiction Rules 2021 once obtained)
    └── uploads/                       # raw case uploads: {user_id}/{case_id}/{doc_id}/original.{ext}; gitignored
```

## 2. Layering rules

Dependencies point downward only. A lower layer never imports from a higher one.

```
api  ->  agents / ingestion  ->  services  ->  schemas
```

- `api/` contains no business logic and no Cypher.
- `agents/nodes/*` and `ingestion/*` call `services/*`; they never open a Neo4j session or an HTTP client themselves.
- All Cypher is in `services/retrieval/graph_queries.py`. All case-scoped queries go through `scope.py` (R-01). A unit test scans `app/` for f-strings passed to `session.run` and fails the build if it finds one.
- Prompts live only in `app/prompts/`. Nodes import them; nodes do not contain prompt text.
- `schemas/graph_schema.py` is the single source of label, relationship and index names. Do not type a label string anywhere else.
- Identity comes from `deps.get_current_user()` only. When authentication is added later, that one function changes and nothing else does (3.2).

## 3. What is deliberately absent

| Not present | Why |
| --- | --- |
| `docker-compose.yml`, `Dockerfile` | Decision: Neo4j Desktop or Community and a native Ollama install |
| Qdrant client or service | Neo4j holds vectors and full-text indexes (00-overview.md) |
| Portkey or any LLM gateway | Groq is called directly through langchain-groq |
| Gemini or sentence-transformers fallback | One embedding model only, nomic-embed-text (R-08) |
| `auth/`, JWT code, role checks, user table | No authentication in v1 (3.2) |
| `frontend/`, a UI inside `app/` | Out of scope for v1. SC-24 adds `ui/` (Streamlit, an HTTP client of the API only; see the tree) |
| Office and HTML loaders | Allowed upload types are PDF, image, audio and text only (3.4) |
| `concept_extractor.py`, judgment ingestion | Deferred (05-roadmap.md, Deferred) |

## 4. Naming and conventions

- Python 3.11, type hints everywhere, Pydantic v2 models for every boundary.
- Module names are lowercase snake_case; one public class or one cohesive set of functions per file.
- Blocking calls (Neo4j driver, Whisper, OCR, FlashRank) run through `anyio.to_thread.run_sync` (1.2).
- Every error raised to the API is an `AppError` carrying a code from 3.3. Never return a raw exception message.
- Logs and spans carry ids, lengths and timings only (R-04).
- Test file names mirror the module under test: `app/services/retrieval/scope.py` is tested by `tests/unit/test_scope.py`.

## 5. Build order mapped to the tree

| Phase | Create |
| --- | --- |
| 0 | `requirements.txt`, `scripts/smoke_check.py`, `.env.example` (draft) |
| 1 | `app/config.py`, `deps.py`, `errors.py`, `middleware.py` (request id), `observability.py`, `schemas/*`, `prompts/repair.py`, `services/llm/groq_client.py`, `services/retrieval/{neo4j_service,scope,embeddings}.py`, `scripts/init_neo4j.py`, `pytest.ini`, unit tests |
| 2 | `ingestion/legal/*`, `ingestion/chunking/legal_splitter.py`, `api/admin_legal.py`, `scripts/ingest_legal.py`, `tests/golden/test_cpa_2019.py` |
| 3 | `ingestion/loaders/*` (including `chat.py`, SC-1), `ingestion/chunking/splitter.py`, `ingestion/case/*`, `ingestion/jobs.py`, `services/media/*`, `prompts/{extraction,link_verify,entity_adjudicate}.py`, `api/{cases,documents,jobs}.py` |
| 4 | `services/retrieval/{vector_search,fulltext_search,hybrid,ranking_service}.py`, `agents/*`, `prompts/{planner,issue_statement,responder}.py`, `api/query.py`, `tests/eval/*` |
| 5 | `api/{health,legal}.py`, rate limit in `middleware.py`, upload validation, contract tests |
| 6 | `SETUP.md`, soak, recovery and log-scrubbing tests |
