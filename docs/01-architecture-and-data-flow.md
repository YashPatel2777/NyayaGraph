# Architecture, configuration and data flow

> Part of the Legal GraphRAG Platform spec (v1.1). Section numbers (1.2, 3.2, R-09, E-05, P-06, G4...) are global across all files in `docs/`; see `docs/README.md` for the index. Do not renumber them: code comments and tests refer to them.

## 1\. Architectural blueprint and data flow

### 1.1 System architecture

```
 Client (web UI or API consumer)
        |  HTTP (loopback only), no authentication in v1
        v
+-------------------- FastAPI app (one Uvicorn process) --------------------+
| routers: /v1/cases /v1/query /v1/legal /v1/admin /v1/jobs /health         |
| rate limit | request id | error envelope                       |
|        |                               |                                  |
|        v                               v                                  |
|  LangGraph query agent          Job worker (asyncio.Queue,                |
|  planner > case_retriever       GPU semaphore = 1)                        |
|  > issue_builder > legal_       case ingestion | legal ingestion         |
|  retriever > graph_expander                                               |
|  > legal_reranker > responder                                             |
|  > verifier                                                               |
+---+-----------+-------------+--------------+------------+---------+------+
    v           v             v              v            v         v
  Neo4j       Ollama        Groq API      FlashRank   faster-whisper EasyOCR
  graph +     nomic-embed-  llama-3.3-    (local      (CUDA)       (CUDA)
  vectors +   text          70b-versatile  ONNX)
  full-text
```

| Component | Responsibility | Interface |
| --------- | -------------- | --------- |
| API layer | Validation, rate limiting, error envelope, request IDs | HTTP/JSON, port 8000 |
| Query agent | LangGraph graph, one invocation per POST /v1/query | In process |
| Job worker | Runs ingestion jobs from an asyncio.Queue; GPU stages guarded by a semaphore of 1 | In process |
| Neo4j | System of record for graph, text, vectors and job state | Bolt, port 7687 |
| Ollama | Produces nomic embeddings | HTTP, port 11434 |
| Groq | Planning, entity extraction, link verification, answer generation | HTTPS |
| FlashRank | Cross-encoder rerank of text candidates | In process |
| Whisper, EasyOCR | Media to text | In process, CUDA |

### 1.2 Resource and concurrency rules

* One Uvicorn process. Ingestion runs on a worker task in the same process. At most one job may be in a GPU stage (OCR or Whisper) at a time. Embedding calls go to Ollama and are not GPU-guarded.
* Models (FlashRank, Whisper, EasyOCR reader) load once in the FastAPI lifespan handler. Startup aborts with exit code 2 if Neo4j is unreachable, Ollama does not serve EMBED\_MODEL, any required index is not ONLINE, or the Meta node check in 2.2 fails.
* Blocking calls (Neo4j driver, Whisper, OCR, FlashRank) run through anyio.to\_thread.run\_sync. Nothing blocking runs on the event loop.
* Every Groq call is wrapped by tenacity: exponential backoff 2, 4, 8, 16, 32 seconds with full jitter, maximum 5 attempts, and the retry-after header is honoured on HTTP 429.

### 1.3 Configuration (environment variables)

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| APP\_ENV | dev | dev or prod. In prod, docs endpoints are disabled |
| APP\_HOST | 127.0.0.1 | Interface Uvicorn binds to. Keep it loopback while the service has no authentication |
| DEFAULT\_USER\_ID | local | The single user id used for every request in v1; stored on all case-scoped nodes |
| NEO4J\_URI, NEO4J\_USER, NEO4J\_PASSWORD, NEO4J\_DATABASE | bolt://localhost:7687, neo4j, required, neo4j | Connection |
| OLLAMA\_URL | http://localhost:11434 | Embedding server |
| EMBED\_MODEL, EMBED\_DIM | nomic-embed-text, 768 | Locked values, checked against the Meta node |
| GROQ\_API\_KEY, GROQ\_MODEL | required, llama-3.3-70b-versatile | LLM access |
| FLASHRANK\_MODEL | library default | Reranker model name |
| WHISPER\_MODEL, OCR\_LANGS | medium, en,hi | Local media models |
| UPLOAD\_DIR | ./data/uploads | Raw uploaded files |
| ENABLE\_GUARDRAILS | false | When true, guard(text, user\_id) runs before the agent; it returns GuardResult(blocked: bool, reply: str or None) |
| LINK\_MIN\_SCORE | 0.60 | Minimum vector score for a claim to legal-unit link candidate. Initial value, calibrate on the evaluation set |
| RERANK\_HIGH | 0.5 | Minimum best legal rerank score for confidence 'high' (R-24). Initial value. The FlashRank score range has not been verified, so calibrate on the evaluation set |
| CORS\_ORIGINS | empty | Comma-separated allowed origins. Empty means CORS is off. No wildcard when APP\_ENV is prod |
| LOGFIRE\_TOKEN | none | Tracing |

### 1.4 Data flow

**A. Legal ingestion (operator, via the open /v1/admin endpoint)**

1. A1. The operator submits a file with metadata: act\_id, title, short\_title, year, number, as\_on\_date, jurisdiction, kind (act, rule or regulation), and for rules issued\_under (act\_id and section).
2. A2. A Job node is created with status queued; the API returns 202.
3. A3. Text is extracted page by page with pypdf. A page with fewer than 40 non-space characters is sent to OCR.
4. A4. The cleaning and parsing rules in 4.3 run: strip watermark and page numbers, locate the body start, split into Chapter, Section and Clause, and collect footnotes.
5. A5. In memory the parser builds nodes, hierarchy edges, REFERS\_TO and REFERS\_TO\_ACT edges, DEFINES edges (from definitions), amendment notes and in\_force\_from dates. A parse report (counts per label) is produced.
6. A6. Retrieval units are chosen (2.3), embed\_text is built, and vectors are computed in batches of 32 with the prefix 'search\_document: '.
7. A7. Writes go to Neo4j in one transaction per Act using UNWIND batches of 200 and MERGE on id. Re-ingesting an act\_id first deletes that Act's subtree inside the same transaction.
8. A8. Verification compares node counts with the parse report and checks that indexes are ONLINE. The job ends succeeded or failed.

**B. Case ingestion (user, per document)**

1. B1. The API validates MIME type and size, checks that case\_id exists, and computes sha256. If the same hash already exists in the case, the existing document is returned with 200 and no job is created.
2. B2. The file is saved to UPLOAD\_DIR/{user\_id}/{case\_id}/{doc\_id}/original.{ext}. A CaseDocument (processing\_status queued) and a Job are created in one transaction. The API returns 202.
3. B3. Media routing: pdf uses pypdf per page with OCR for pages under 40 characters; image uses EasyOCR; audio is decoded directly by faster-whisper (PyAV decodes it; the ffmpeg binary is needed only if a format fails to decode) and transcribed with vad\_filter=True, returning segments with start and end seconds; text is read as UTF-8.
4. B4. Chunking: paragraph splitter, target 1,200 characters, hard maximum 1,500, overlap 150. Each chunk keeps page (pdf, image) or start\_sec and end\_sec (audio).
5. B5. Chunks are embedded with the prefix 'search\_document: ' and written as CaseChunk nodes (MERGE on id). Vector search over this document works from this point.
6. B6. Extraction: batches of 6 consecutive chunks go to Groq (temperature 0, JSON mode) and are validated against ExtractionResult (2.5). On validation failure the call is retried once with the validation error appended; a second failure marks that batch failed and the job continues.
7. B7. Entity resolution inside the case (rule R-14).
8. B8. Entities, MENTIONS and entity relationships are written with MERGE on deterministic ids. BEFORE edges are derived in code by sorting events on date, not by the LLM.
9. B9. Legal linking: regex citations create INVOKES edges. Each Claim is embedded, the top 5 LegalUnits with score of at least LINK\_MIN\_SCORE go to Groq for a yes/no check with a rationale, and each yes creates an ALLEGES\_VIOLATION\_OF edge.
10. B10. The document becomes ready, or ready\_with\_warnings if any batch failed. The job ends accordingly.

**C. Query**

1. C1. Rate limit, then verify that the case exists (404 if not).
2. C2. If ENABLE\_GUARDRAILS is true, run guard(); return the block reply if blocked.
3. C3. planner: Groq classifies intent as CONVERSATIONAL, CASE\_ONLY, LEGAL\_ONLY or CASE\_AND\_LEGAL and writes a rewritten search\_query. The original query is kept in state.
4. C4. case\_retriever: embed 'search\_query: ' plus the query, run vector and full-text search with scope, fuse with reciprocal rank fusion, rerank with FlashRank, keep the top 5.
5. C5. issue\_builder: collect entities mentioned in those chunks, their one-hop neighbours, and any INVOKES or ALLEGES\_VIOLATION\_OF links. The issue statement equals the original query; there is no LLM call (SC-22, option b: an LLM-written whole-case issue statement pulled retrieval away from the question).
6. C6. legal\_retriever: hybrid search over LegalUnit using the original query (the issue statement equals it, SC-22); fuse vector and full-text results with reciprocal rank fusion; add seed units reached through case-graph links.
7. C7. graph\_expander: run the expansion query in 4.2 around the legal candidates.
8. C8. legal\_reranker: FlashRank against the original query (SC-22); keep the top 8.
9. C9. responder: Groq with the LegalAnalysis schema, temperature 0.1.
10. C10. verifier: drop any citation whose source id was not retrieved, recompute confidence, attach the fixed disclaimer and the trace.
11. C11. Persist thread state and return 200.

Routing: CONVERSATIONAL goes straight to responder. LEGAL\_ONLY skips case\_retriever and issue\_builder (the issue statement equals the query). CASE\_ONLY ends after case\_retriever and goes to responder.