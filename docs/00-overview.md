# Legal GraphRAG Platform: Technical Specification

> Part of the Legal GraphRAG Platform spec (v1.1). Section numbers (1.2, 3.2, R-09, E-05, P-06, G4...) are global across all files in `docs/`; see `docs/README.md` for the index. Do not renumber them: code comments and tests refer to them.

Version 1.1 | 15 October 2026 | Status: ready for implementation

Audience: backend, data and integration engineers.

## 0. Scope and fixed decisions

**Product.** A FastAPI service that answers legal questions about a user's own case. It retrieves facts from the user's private Case KB, finds the governing provisions in a shared Legal KB (seeded with the Consumer Protection Act, 2019, India), and returns a validated, citation-grounded JSON analysis.

**In scope:** legal ingestion, case ingestion (PDF, image, audio, text), hybrid vector + full-text + graph retrieval, structured answer generation, asynchronous jobs, per-case data scoping.

**Out of scope for v1:** authentication and role-based access (v1 runs as one fixed local user, see 3.2), guardrails (a no-op hook is specified in 1.3), a web UI, multi-region deployment, fine-tuning, jurisdictions other than India.

| Concern | Decision |
| --- | --- |
| Runtime | Python 3.11, FastAPI, Uvicorn. Windows 11 host, NVIDIA RTX 4070 (8 GB VRAM) |
| Orchestration | LangGraph StateGraph with MemorySaver checkpointer |
| Store | Neo4j 2026.10.2, the tested version (SC-15: the case vector query uses the Cypher 25 SEARCH clause with an in-index filter; the first release supporting it is not verified). Graph, vector indexes and full-text indexes. No other database |
| Embeddings | nomic-embed-text served by Ollama, 768 dimensions, cosine similarity. The only embedding model; there is no fallback model |
| LLM | Groq, llama-3.3-70b-versatile, via langchain-groq |
| Reranker | FlashRank (local ONNX), model name set by config |
| Speech-to-text | faster-whisper, model medium, CUDA float16 |
| OCR | EasyOCR (languages en, hi); PyMuPDF renders pages at 200 DPI |
| Auth | None in v1: every request runs as one fixed local user (DEFAULT\_USER\_ID). There is no login, token or role check anywhere in v1 |
| Deployment | No Docker. Neo4j runs as Neo4j Desktop or the Community zip; Ollama runs as a native Windows app |
| Observability | Logfire |

