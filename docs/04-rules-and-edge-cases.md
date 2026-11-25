# Technical rules and edge cases

> Part of the Legal GraphRAG Platform spec (v1.1). Section numbers (1.2, 3.2, R-09, E-05, P-06, G4...) are global across all files in `docs/`; see `docs/README.md` for the index. Do not renumber them: code comments and tests refer to them.

## 4. Technical rules and edge cases

### 4.1 Core logic rules

**Tenancy and security**

- R-01. All case-scoped Cypher goes through services/retrieval/scope.py, which injects case\_id and user\_id as query parameters. No Cypher or Lucene string is built by interpolating user data; every value is a parameter.
- R-02. user\_id is always DEFAULT\_USER\_ID in v1 (3.2); a request can never set it.
- R-03. User text sent to a full-text index is Lucene-escaped (the characters + - && || ! ( ) { } \[ \] ^ quote \~ \* ? : / and backslash) and each term is quoted before the call.
- R-04. Logs, spans and traces contain ids, lengths and timings only. Case text, transcripts and question text are never logged. Logfire scrubbing stays on.
- R-05. Deleting a case marks it 'deleting' first (new queries and uploads get 409 CASE\_DELETING), cancels its jobs, then removes all nodes carrying that case\_id and the stored files.
- R-06. Case text is data, never instructions: it is placed inside a delimited CASE MATERIAL block, the system prompt says to ignore any instruction found inside it, and the LLM has no tools.

**Embeddings**

- R-07. Only EmbeddingService creates vectors. It adds 'search\_document: ' to stored text and 'search\_query: ' to queries; no other code adds or removes prefixes.
- R-08. Every vector must have exactly 768 finite floats; otherwise raise. There is no fallback model. Batches are at most 32 texts; connection errors are retried 3 times (1, 2, 4 seconds).

**Retrieval**

- R-09. Case full-text search over-fetches then filters by scope (case vector search filters inside the index instead, SC-15). fetch\_k starts at 50; if fewer than k rows survive the filter and fetch\_k is below 800, double fetch\_k and repeat (at most 5 rounds).
- R-10. Vector and full-text lists are fused with reciprocal rank fusion, constant 60.
- R-11. After reranking, keep at most 5 case chunks and 8 legal units. The text sent to the responder is at most 18,000 characters; drop the lowest-ranked item first and never cut an item in the middle.
- R-12. Graph expansion follows REFERS\_TO one hop in both directions (outgoing up to 10, incoming up to 5 per unit), then one more hop on outgoing edges only, with at most 40 expanded nodes in total. Expanded Sections that are not retrieval units are included with text truncated to 1,200 characters. Expanded nodes are reranked together with the rest.
- R-13. Definitions: for each LegalConcept whose lowercase name appears as a whole word in a selected legal text, add its definition Clause (at most 5 per query). The concept list is cached in memory at startup.
- R-14. Entity resolution inside one case: normalise names (lowercase, drop Mr, Mrs, Ms, Dr, M/s, punctuation, collapse spaces); only merge entities of the same kind; rapidfuzz token\_sort\_ratio of 92 or more merges automatically; 80 to 91 triggers one Groq yes/no adjudication; below 80 stays separate. For organisations below 80, the names are compared again without company suffixes (pvt, private, ltd, limited, llp, inc, incorporated, co, company, corp, corporation, plc, retail); 80 or more there triggers the Groq adjudication and never merges automatically (SC-13). A merged entity keeps all aliases and all MENTIONS edges.

**Ingestion**

- R-15. Every extracted entity must cite at least one chunk\_id from its batch. Entities without a valid chunk\_id are discarded and counted in the job warnings.
- R-16. All ids are deterministic and all writes use MERGE, so re-running a job never duplicates nodes. Before re-running entity extraction for a document, delete that document's MENTIONS edges and any entity left with no MENTIONS.
- R-17. Dates must be ISO 8601 (partial allowed). An unparseable date is stored as null with a job warning; it is never guessed.
- R-18. Legal nodes are written only by the admin ingest job. No endpoint writes Legal KB nodes from user input.

**Generation**

- R-19. Temperatures: planner 0, extraction 0, link verification 0, responder 0.1. Token caps: planner 200, extraction 4,000 (SC-11; was 1,500), responder 1,800. The issue-statement call (temperature 0.2, cap 300 under SC-19) was removed by SC-22.
- R-20. The responder may cite only chunk ids and provision ids present in the context it was given.
- R-21. The verifier is authoritative. It removes any fact whose source\_chunk\_id, or any provision whose provision\_id, is not in the retrieved set, and it fills citation, status, act\_as\_on, excerpt and source from Neo4j. It also removes a fact that its cited chunk does not support: at least 60 % of the fact's content words and every number in it must appear in that chunk's text (SC-16). The count of removed items goes to trace.counts.dropped\_citations. Chunk and provision labels left in summary, analysis or gaps\_or\_conflicts are replaced by the item's file name and page or its citation, and unknown labels are removed (SC-17). The same replacement applies to a provision's relevance; if the named text would exceed 500 characters, its labels are removed instead. Labels inside a fact are removed, because the fact's source is shown with it, and the fact is checked for support after removal (SC-17, extended 2026-11-25).
- R-22. Structured output is validated with Pydantic. On failure, retry once with the validation error appended; a second failure returns 502 LLM\_UPSTREAM\_ERROR with details.reason 'invalid\_structured\_output'.
- R-23. If no provision survives verification: applicable\_provisions is empty, confidence.level is 'low', and, when the intent is LEGAL\_ONLY or CASE\_AND\_LEGAL, gaps\_or\_conflicts includes 'No supporting provision was retrieved.' (SC-18). The system never gives a legal conclusion without a retrieved provision behind it.
- R-24. Confidence is computed by code: high when at least one direct provision and at least one case fact exist and the best legal rerank score is at least RERANK\_HIGH (config, initial value 0.5, calibrate on the evaluation set); medium when at least one provision exists; otherwise low.
- R-25. The disclaimer is a constant string appended by code, never written by the LLM.
- R-26. References to other Acts (for example the Indian Penal Code or the Code of Criminal Procedure) are reported exactly as written in the retrieved text. The system does not map them to successor statutes.
- R-27. Forum and pecuniary limits (Act sections 34, 47 and 58): the responder may state a threshold only if a retrieved provision contains it. When one of those sections is retrieved and a rule linked by PRESCRIBES\_FOR exists, the rule text takes precedence and both are cited.

### 4.2 Retrieval parameters and Cypher

| Parameter | Value |
| --- | --- |
| Case full-text fetch\_k (initial) | 50, doubled up to 800 (R-09). Case vector search has no fetch\_k: one filtered SEARCH with LIMIT k (SC-15) |
| Case candidates before rerank | 15 |
| Case chunks kept after rerank | 5 |
| Legal vector and full-text fetch each | 30 |
| Legal pool before expansion | 20, plus up to 10 seed units from case-graph links |
| Legal units kept after rerank | 8 |
| RRF constant | 60 |
| Extraction batch | 6 chunks |
| Embedding batch | 32 texts |
| Case chunk target, maximum, overlap | 1,200, 1,500, 150 characters |
| Per-call timeouts | Groq 30 s, Neo4j transaction 15 s, whole query 45 s |

Case vector search (SC-15). The scope filter runs inside the index, which declares case\_id and user\_id (2.2); $k is the 15 case candidates before rerank:

```cypher
CYPHER 25
MATCH (node:CaseChunk)
  SEARCH node IN (VECTOR INDEX case_chunk_embedding FOR $query_vec
                  WHERE node.case_id = $case_id AND node.user_id = $user_id
                  LIMIT $k) SCORE AS score
RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id,
       node.page AS page, node.start_sec AS start_sec, node.end_sec AS end_sec, score
ORDER BY score DESC
```

Case full-text search keeps the procedure, because full-text SEARCH cannot filter (O-1). It runs inside the R-09 over-fetch loop:

```cypher
CALL db.index.fulltext.queryNodes('case_text_ft', $lucene, {limit: $fetch_k})
YIELD node, score
WHERE node.case_id = $case_id AND node.user_id = $user_id
RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id,
       node.page AS page, node.start_sec AS start_sec, node.end_sec AS end_sec, score
ORDER BY score DESC LIMIT $k
```

Legal graph expansion (structural part; definitions are added in code, R-13):

```cypher
UNWIND $unit_ids AS uid
MATCH (u:LegalUnit {id: uid})
OPTIONAL MATCH (p:Section)-[:HAS_CLAUSE]->(u)
WITH u, coalesce(p, u) AS sec
OPTIONAL MATCH (sec)-[:REFERS_TO]->(out) WHERE out:Section OR out:Clause
OPTIONAL MATCH (inc)-[:REFERS_TO]->(sec) WHERE inc:Section OR inc:Clause
OPTIONAL MATCH (rule:Section)-[:PRESCRIBES_FOR]->(sec)
RETURN u.id AS unit_id, sec.id AS section_id,
       collect(DISTINCT out.id)[0..10] AS refers_to,
       collect(DISTINCT inc.id)[0..5] AS referenced_by,
       collect(DISTINCT rule.id) AS prescribed_by
```

### 4.3 Statute parsing rules (IndiaCode PDFs, validated against the Consumer Protection Act, 2019 file)

Extraction is per page, and page boundaries are kept until P-05 has run. Apply these rules in order. Regexes are written with character classes instead of backslashes.

- P-01 Clean. Delete every token matching `IndiaCode[0-9]*` (the watermark fused to the page number) and every line that consists only of digits.
- P-02 Body start. Discard everything before the first line that begins with 'An Act to'. If there is no such line, fail with PARSE\_BODY\_START\_NOT\_FOUND. This drops the title page, the abbreviations list and the Arrangement of Sections, which repeats every section title and would otherwise create duplicate Sections.
- P-03 Body end. Stop at the first line that starts with 'STATEMENT OF OBJECTS AND REASONS' (case-insensitive). Store the remainder in Act.objects\_text. It is never a LegalUnit and is never embedded.
- P-04 Chapters. A chapter heading is a line matching `^CHAPTER [IVXLC]+$`. The upper-case line or lines after it, up to the next section heading, are the title, stored exactly as written. Do not correct source typos (the body of the CPA file prints DISPUTER in the Chapter IV title). Chapter ids use the roman numeral only.
- P-05 Footnotes first. A footnote starts at a line matching `^([*]|[0-9]{1,2})[.][ ]+(Ins[.]|Subs[.]|Omitted|Vide|[0-9]{1,2}(st|nd|rd|th)[ ]+[A-Z][a-z]+,[ ]+[0-9]{4})`. The footnote block runs from that line to the end of the same page. Remove the block from the body and keep it for P-09.
- P-06 Section headings. Inside the body, a line is a section heading when it matches `^([0-9]{1,3}[A-Z]?)[.][ ]+([A-Z].{2,300}?)[.]—(.*)$`. If a heading wraps onto the next line, join up to 2 following lines until '.—' appears; if it does not appear within 3 lines, the line is body text. Number guard: accept a heading only when its number is the previous section number plus 1, or the previous number with the next letter suffix (55 then 55A). Anything else is body text. Group 2 without a trailing period is the title; group 3 starts the section text.
- P-07 Subsections. Inside a section, a line matching `^[(][0-9]+[A-Z]?[)][ ]` starts a Clause of kind 'subsection', labelled with the number and any letter suffix (for example 1A). Lines starting (a) or (i), and provisos and explanations, stay inside the current Clause. A section with no numbered subsections gets no Clause nodes.
- P-08 Definitions. In the section titled 'Definitions', a line matching `^[(][0-9]+[)][ ][“]` starts a Clause of kind 'definition' with label equal to the number and term equal to the text between the curly quotes. Create LegalConcept con:{slug(term)}, a DEFINES edge, and set definition\_text to the Clause text. The Definitions section itself has is\_unit false.
- P-09 Footnote meaning. Lines starting Ins., Subs. or Omitted create an amendment note on the section where the marker appears. The marker is a digit immediately before a square bracket, as in 1\[(1A); if none is found, use the last section heading on that page. Store each note as a string in Section.amendment\_notes. When the note names an Act number and an effective date (for example Act 33 of 2021 and w.e.f. 04-04-2021), also create the stub Act act:33-of-2021 (external true) and an AMENDS edge with effective\_on. Vide and dated commencement footnotes set in\_force\_from: each paragraph starting with a date such as 20th July, 2020 lists items like 's. 3 to 9 (both inclusive)', 's. 82 to 87' and 'ss. 90 and 91'; an item followed by \[Except ...\] excludes the sections named in the brackets; a section that no paragraph lists gets null. Clause-level exceptions (for example definitions (4) and (13)) are recorded only in the Section's amendment\_notes in v1.
- P-10 References, run on every Section and Clause text:
  - Section references: `(?i)sections?[ ]+([0-9]{1,3}[A-Z]?(?:[ ]*(?:,|and|to)[ ]*[0-9]{1,3}[A-Z]?)*)`. Expand 'N to M' into every number in the range and split on commas and 'and'.
  - Other-Act test on the text right after the match: `^[ ]*of[ ]+(?:the[ ]+)?([A-Z][A-Za-z&' ]+?(?:Act|Code)(?:,[ ]*[0-9]{4})?)`. If it matches and the captured name is not this Act's title, create REFERS\_TO\_ACT to the stub act:{slug(name)} (external true) and create no REFERS\_TO edge. This stops 'section 193 and 228 of the Indian Penal Code' from linking to this Act's sections 193 and 228.
  - Otherwise create REFERS\_TO to sec:{act}:{n} for each number that exists. A number that does not exist adds a job warning and no edge.
  - Qualifiers: 'clause (N) of section 2' links to the definition Clause cl:{act}:2:N. 'sub-section (N) of section M' links to cl:{act}:M:N when that Clause exists, else to the Section; accept both sub-section and subsection because the PDF drops the hyphen in places. 'Chapter V' links to ch:{act}:V. 'this Act' creates no edge. Self-references are ignored.
- P-11 Stub Act ids keep the year as written: 'Code of Criminal Procedure, 1973' becomes act:code-of-criminal-procedure-1973 and 'Indian Penal Code' becomes act:indian-penal-code.
- P-12 Validation. Section numbers must form a continuous sequence (letter suffixes allowed). A gap, a duplicate, or a count that differs from expected\_sections fails the job with PARSE\_VALIDATION\_FAILED and a report listing the problem numbers. Unknown reference targets are warnings only.

**Rules documents (kind rule or regulation; SC-5, approved 2026-11-20).** P-01, P-05, P-06, P-07, P-09, P-10, P-11 and P-12 apply as written except where P-13 to P-18 replace them. The document is an Act node with kind 'rule'; each rule is a Section (sec:{act\_slug}:{rule number}); each sub-rule is a Clause of kind 'subsection'.

- P-13 Body start (replaces P-02). Discard everything before the first line that P-06 accepts as a heading numbered 1. Text in other scripts and the Gazette header before it go with it; a Hindi line starting '3.' fails P-06's title test and is discarded.
- P-14 Enabling text (replaces P-03). The paragraph starting 'In exercise of the powers conferred by' up to rule 1 is stored in Act.enabling\_text; it is never a LegalUnit and never embedded. A missing Statement of Objects is not an error.
- P-15 No chapters. Without CHAPTER headings there are no Chapter nodes; Sections hang off the Act through HAS\_SECTION and chapter\_number is null.
- P-16 Definitions. In the rule titled 'Definitions', a line matching `^[(][a-z]{1,3}[)][ ]+[“"]` starts a definition Clause labelled with the letter(s). Rules create no LegalConcept nodes (concept ids are global); their definition Clauses are still retrieval units.
- P-17 References (extends P-10). 'section N' and 'section N of the Act' refer to the parent Act in issued\_under: REFERS\_TO sec:{parent}:N, with a missing target as a warning. 'clause (N) of section 2 of the Act' links to cl:{parent}:2:N. 'rule N' and 'sub-rule (N)' refer to this document. The P-10 other-Act test still applies.
- P-18 Amendments by notification (extends P-09). A footnote such as 'Subs. by G.S.R. 328(E), dated 17-5-2021 (w.e.f. 17-5-2021)' goes into Section.amendment\_notes and creates no AMENDS edge. 'Prior to substitution it read as:' text belongs to the footnote and is removed from the body. Exception (SC-9): when the amending notification itself is ingested, P-20 creates AMENDS from it.

**Amendment notifications (metadata key amends; SC-9, approved 2026-11-20).**

- P-19 Parse and apply. The English part of the notification is a list of numbered paragraphs without titles. The body starts at the first '1.' after the English enabling text and ends at '[F. No.'. Every paragraph after the short title (1) must be exactly one instruction of the form 'In the <principal rules>, in rule N, for sub-rule (M), the following sub-rule shall be substituted, namely:— “(M) …”' or 'In the <principal rules>, for rule N, the following rule shall be substituted, namely:— “N. Title.—…”'. The principal is named correctly and the quoted text starts with its label. Anything else (insertion, omission, several instructions, a missing target) fails the job with PARSE\_VALIDATION\_FAILED and writes nothing. The notification is an Act node (kind 'rule') whose Sections are stored but are never retrieval units.
- P-20 Record. The principal rules are rebuilt from their own source file with all substitutions applied. Each substitution gives an AMENDS edge from the notification to the principal Section (effective\_on, previous\_text) and the Section note 'Subs. by <number>, dated <d-m-yyyy> (w.e.f. <d-m-yyyy>)'. The principal's as\_on\_date becomes the latest amendment's date. Changed units are re-embedded, and the notification and the rebuilt principal are written in one transaction.
- P-21 Re-application. Ingesting the principal again re-applies the substitutions stored on its amending notifications, so a re-ingest never reverts an amendment. Re-ingesting a notification changes nothing (R-16).

### 4.4 Error handling and edge cases

| ID | Failure | Detection | Required handling |
| --- | --- | --- | --- |
| E-01 | Groq rate limit (429, tokens-per-minute) or 5xx | HTTP status or exception | Retry per 1.2 and honour retry-after. In a job, the job stays running and the stage resumes. In a query, if waiting would pass the 45 s budget return 504 QUERY\_TIMEOUT; if retries are exhausted return 502 LLM\_UPSTREAM\_ERROR. Never return a partial answer |
| E-02 | LLM returns invalid JSON or breaks the schema | Pydantic ValidationError | One repair retry with the error text appended (R-22). Then: extraction batch marked failed with a job warning and the job continues; query returns 502 with reason invalid\_structured\_output |
| E-03 | Ollama down or model not pulled | Connection error or 404 on embed; checked at startup | Startup aborts with exit code 2. At runtime retry 3 times (1, 2, 4 s), then 503 DEPENDENCY\_UNAVAILABLE for queries, or job status failed with error\_code OLLAMA\_UNAVAILABLE for jobs (resumable, E-11) |
| E-04 | Neo4j unavailable or transient error | ServiceUnavailable or TransientError | Retry 3 times (0.5, 1, 2 s). Then 503 for queries, or job failed with error\_code NEO4J\_UNAVAILABLE. Writes are idempotent, so a retry is safe |
| E-05 | Vector search returns fewer than k rows after the scope filter | Row count | Run the over-fetch loop (R-09). If still empty, set case\_chunks to empty, continue as LEGAL\_ONLY, set trace.downgraded true and add 'No case material matched the question.' to gaps\_or\_conflicts |
| E-06 | Case has no ready documents | Count of documents with status ready or ready\_with\_warnings is 0 | mode case\_only returns 409 NO\_READY\_DOCUMENTS. Other modes continue as LEGAL\_ONLY with a gaps\_or\_conflicts note |
| E-07 | PDF encrypted, corrupt or image-only | pypdf error or encrypted flag; fewer than 40 characters per page | Encrypted: job failed, PDF\_ENCRYPTED. Corrupt: PDF\_UNREADABLE. Pages under 40 characters go to OCR; if OCR output for the whole file is under 40 characters, job failed with OCR\_EMPTY; individual blank pages are skipped with a warning |
| E-08 | Whisper out of memory or audio cannot be decoded | CUDA out-of-memory error; decode exception | Retry once on CPU with compute\_type int8 and model small. Decode failure: job failed, AUDIO\_DECODE\_FAILED. If language is not given, auto-detect and store it; below 0.5 detection probability add a warning |
| E-09 | Same file uploaded twice to a case | sha256 lookup within the case | Return the existing document with 200 and duplicate true; create no job |
| E-10 | Case deleted while a job is running | Case status is deleting or the node is missing, checked at every stage boundary and before each write | Stop, set the job to cancelled, remove partial nodes for that document |
| E-11 | Process restarts mid-job | On startup, Jobs with status running | Set to queued and add 1 to attempts; at more than 3 attempts set failed with JOB\_ATTEMPTS\_EXCEEDED. The upload is still on disk and all writes are MERGE-based, so resuming is safe. Note: MemorySaver thread memory is lost on restart in v1 |
| E-12 | Query takes too long | asyncio timeout of 45 s | Cancel the graph run and return 504 QUERY\_TIMEOUT |
| E-13 | Missing fields, oversized file, wrong type | Pydantic validation and file-signature check | 422 VALIDATION\_ERROR with details.fields; 413 FILE\_TOO\_LARGE; 415 UNSUPPORTED\_MEDIA\_TYPE. Nothing is written before validation passes |
| E-14 | Legal parser anomaly | P-12 validation | Fail the job with PARSE\_VALIDATION\_FAILED and the gap report; write nothing to Neo4j. Unknown reference targets only add warnings |
| E-15 | Prompt injection inside an uploaded document | Not detectable reliably | Contained by design (R-06, R-20, R-21): case text is delimited data, the LLM has no tools, and any id it invents is removed by the verifier |

