# Implementation roadmap

> Part of the Legal GraphRAG Platform spec (v1.1). Section numbers (1.2, 3.2, R-09, E-05, P-06, G4...) are global across all files in `docs/`; see `docs/README.md` for the index. Do not renumber them: code comments and tests refer to them.

## 5. Implementation roadmap

Estimates are working days for one developer and total about 19 days. Each phase ends only when its exit criteria pass.

File and folder layout for every phase is in 06-file-structure.md. Create files at those paths.

### Phase 0. Environment (0.5 day)

- Install Neo4j 2026.10.2 (Desktop or Community; SC-15), Ollama, and the CUDA build of PyTorch. Run `ollama pull nomic-embed-text`. faster-whisper decodes audio through PyAV; install the ffmpeg binary only if a format fails to decode.
- Add to requirements.txt: pydantic-settings, rapidfuzz, tenacity, slowapi, filetype, pytest, pytest-asyncio.
- Write scripts/smoke\_check.py.
- Exit: smoke\_check prints OK for four checks: Neo4j connect, Ollama embed returns 768 floats, torch.cuda.is\_available() is True, Groq returns a completion.

### Phase 1. Foundations and models (2 days)

- config.py (pydantic-settings), Logfire setup with scrubbing, request-id middleware, error envelope and exception handlers.
- schemas/graph\_schema.py (labels, relationship types, index names as constants), schemas/extraction.py, schemas/output.py, schemas/api.py.
- services/retrieval/neo4j\_service.py, scope.py, embeddings.py (with the Meta check), services/llm/groq\_client.py (tenacity, JSON mode).
- scripts/init\_neo4j.py exactly as in 2.2.
- Exit: init\_neo4j runs twice with no error and no change; unit tests show scope.py always passes case\_id and user\_id as parameters; embedding tests confirm the two prefixes, the 768 check, and that a model mismatch aborts startup.

### Phase 2. Legal KB (3 days)

- ingestion/legal/parser.py, reference\_extractor.py, legal\_processor.py, ingestion/chunking/legal\_splitter.py, scripts/ingest\_legal.py, and the admin ingest endpoint wiring.
- Ingest CPA2019.pdf with metadata act\_id cpa-2019, as\_on\_date 2026-09-01, expected\_sections 107.
- Obtain the Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National Commission) Rules, 2021 from the official gazette, ingest them as kind rule issued under sections 34, 47 and 58, and create the PRESCRIBES\_FOR edges. Confirm the thresholds against the official text before relying on R-27.
- Exit: all of these golden tests pass.
  - G1. 8 Chapter nodes and 107 Section nodes; sec:cpa-2019:1 through sec:cpa-2019:107 each exactly once.
  - G2. Section 2 has 47 definition Clauses and 47 LegalConcept nodes; con:consumer exists and its definition text begins with the words 'consumer' means any person who.
  - G3. REFERS\_TO edges exist for 24 to 20, 24 to 21, 49 to each of 35, 36, 37, 38 and 39, 73 to 72, 92 to 88 and 89, and 97 to 21 and 96.
  - G4. Section 38 has a REFERS\_TO\_ACT edge to act:indian-penal-code and no REFERS\_TO edge to any section numbered 193 or 228.
  - G5. Section 55 has an amendment note and an AMENDS edge from act:33-of-2021 with effective\_on 2021-04-04.
  - G6. in\_force\_from is 2020-07-20 for sections 34 and 90, 2020-07-24 for sections 10, 88 and 94, and null for section 1.
  - G7. Act.objects\_text is set and no LegalUnit contains the text STATEMENT OF OBJECTS.
  - G8. No node text contains the string IndiaCode.
  - G9. Every LegalUnit has a 768-float embedding, all four indexes are ONLINE, and the query 'limitation period for filing a consumer complaint' returns sec:cpa-2019:69 in the top 3 results.
  - G10 (SC-5, SC-6). The Jurisdiction Rules, 2021 give exactly 5 Section nodes; PRESCRIBES\_FOR edges run from rules 3, 4 and 5 to sec:cpa-2019:34, 47 and 58; no node text contains a Devanagari character; rule 3's text contains 'fifty lakh rupees'.
  - G11 (SC-5, D-4; Gazette original with a text layer, SC-8 option a). The E-Commerce Rules, 2020 give exactly 8 Section nodes, and rule 5 has REFERS\_TO\_ACT to act:information-technology-act-2000.
  - G12 (SC-9). After G.S.R. 328(E) of 17 May 2021 is ingested, cl:cp-e-commerce-rules-2020:4:1 begins 'where an e-commerce entity is a company incorporated' and contains 'nodal officer'. The 2020 wording is gone from rule 4. Rule 4 carries the substitution note and an AMENDS edge with effective\_on 2021-05-17 and the previous text. The principal's as\_on\_date is 2021-05-17. A replace re-ingest of the principal and a re-ingest of the notification both leave all of this unchanged.

### Phase 3. Case ingestion (4 days)

- Loaders (pdf, image, audio, text) and router, services/media/ocr\_service.py and transcription\_service.py, ingestion/case/ extractor, entity\_resolver, legal\_linker, case\_processor, ingestion/jobs.py (queue, GPU semaphore, restart recovery), and the cases, documents and jobs endpoints.
- Exit: a sample case with one digital PDF, one scanned image and one audio file reaches ready; re-running a job leaves node counts unchanged; deleting the case leaves zero nodes with that case\_id and removes the stored files; each E-07 to E-11 case has an automated test.

### Phase 4. Query agent (4 days)

- vector\_search, fulltext\_search, hybrid (RRF), ranking\_service, graph\_queries (4.2), each LangGraph node in 1.1 order, agents/graph.py with the routing in 1.4, responder and verifier, and POST /v1/query.
- Build an evaluation set of at least 30 labelled questions (20 case plus legal, 5 legal only, 5 conversational or out of scope), each with the expected provision ids.
- Exit: provision recall at 8 is at least 0.80; zero unverified citations in any response (hard gate); p95 end-to-end latency at most 20 seconds on the target laptop, measured per query with the Groq token budget full: run\_eval.py waits for the budget to refill between questions and leaves the wait out of the latency (SC-14 c1); the isolation test passes at service level (two user\_id values with overlapping content: queries scoped to user B never return user A's chunks; API-level isolation returns when authentication is added).

### Phase 5. API and hardening (2 days)

- Per-IP rate limits, upload validation, /health, remaining endpoints in 3.4, OpenAPI review.
- Exit: a contract test exists for every error code in 3.3; the service listens on APP\_HOST, default 127.0.0.1.

### Phase 6. Verification and operations (2 days)

- Soak test with 20 concurrent queries and one concurrent ingestion; behaviour under Groq rate limiting; restart recovery test (kill the process mid-job); log-scrubbing test proving no case text appears in logs or spans; SETUP.md covering Neo4j, Ollama, CUDA and environment variables.
- Exit: no 500 responses in the soak test, and every soak response is 200, 502 LLM\_UPSTREAM\_ERROR or 504 QUERY\_TIMEOUT with the error envelope, with the count of 200s recorded (SC-14 c2; on the Groq free tier most of the 20 queries are expected to time out); the recovery test resumes and finishes the job; the scrubbing test passes.

### Deferred

Guardrails (enable through ENABLE\_GUARDRAILS), judgment ingestion with INTERPRETS, CITES and OVERRULES, a durable LangGraph checkpointer in place of MemorySaver, and a web UI.

### Definition of done for v1

- Every exit criterion above is met, and the golden tests G1 to G9 run in CI against a throwaway Neo4j database.
- No Cypher or Lucene string contains interpolated user data (checked by a unit test that scans services and agents for f-strings passed to session.run).
- A new developer can follow SETUP.md from a clean Windows machine to a passing smoke\_check and a first successful POST /v1/query.
