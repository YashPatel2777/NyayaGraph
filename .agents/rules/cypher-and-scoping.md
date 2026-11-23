---
trigger: glob
globs: app/services/**/*.py, app/agents/**/*.py, app/ingestion/**/*.py
description: Cypher, Lucene and tenant-scoping safety for Legal GraphRAG data access code
---

# Data access rules

- Cypher exists only in `app/services/retrieval/graph_queries.py`. Other modules call functions from it.
- Pass every value as a query parameter. Never use f-strings, `%` or `+` to put user data into Cypher or Lucene text (R-01, R-03).
- Case-scoped reads and writes go through `scope.py`; it supplies `case_id` and `user_id`. Never hand-write those filters.
- Escape user text for full-text search with the Lucene escaper, then quote each term (R-03).
- Vector and full-text search over-fetch, then filter by scope, using the doubling loop in R-09. Fuse with RRF, constant 60 (R-10).
- All writes use MERGE on the deterministic id from `schemas/graph_schema.py` (2.1, R-16).
- Do not log text content. Ids, lengths and timings only (R-04).
