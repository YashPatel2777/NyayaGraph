"""ALL Cypher lives here (R-01, 06 section 2).

Every value is a query parameter ($name). No query in this module is built from user data
by f-strings, % or concatenation (R-01, R-03). Retrieval and expansion queries from 4.2 are
added in Phases 2 and 4.
"""

from __future__ import annotations

from typing import Final

# 2.2 DDL, exactly as specified, one statement per entry (schema commands run in auto-commit
# transactions). Every statement is idempotent: IF NOT EXISTS, or MERGE ... ON CREATE.
SCHEMA_DDL: Final[tuple[str, ...]] = (
    "CREATE CONSTRAINT act_id IF NOT EXISTS FOR (n:Act) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (n:Clause) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT legalunit_id IF NOT EXISTS FOR (n:LegalUnit) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:LegalConcept) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT judgment_id IF NOT EXISTS FOR (n:Judgment) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT casedoc_id IF NOT EXISTS FOR (n:CaseDocument) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT casechunk_id IF NOT EXISTS FOR (n:CaseChunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT party_id IF NOT EXISTS FOR (n:Party) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (n:Job) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT meta_key IF NOT EXISTS FOR (n:Meta) REQUIRE n.key IS UNIQUE",
    """CREATE VECTOR INDEX legal_unit_embedding IF NOT EXISTS
FOR (n:LegalUnit) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}""",
    # SC-15: case_id and user_id are filter properties, for the in-index scope filter (4.2).
    """CREATE VECTOR INDEX case_chunk_embedding IF NOT EXISTS
FOR (n:CaseChunk) ON (n.embedding) WITH [n.case_id, n.user_id]
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}""",
    "CREATE FULLTEXT INDEX legal_text_ft IF NOT EXISTS FOR (n:LegalUnit) ON EACH [n.search_text]",
    "CREATE FULLTEXT INDEX case_text_ft IF NOT EXISTS FOR (n:CaseChunk) ON EACH [n.text]",
    "CREATE INDEX casechunk_scope IF NOT EXISTS FOR (n:CaseChunk) ON (n.case_id, n.user_id)",
    "CREATE INDEX case_owner IF NOT EXISTS FOR (n:Case) ON (n.user_id)",
    "CREATE INDEX job_owner IF NOT EXISTS FOR (n:Job) ON (n.user_id, n.status)",
    """MERGE (m:Meta {key: 'embedding'})
ON CREATE SET m.model = 'nomic-embed-text', m.dim = 768,
  m.doc_prefix = 'search_document: ', m.query_prefix = 'search_query: ', m.schema_version = 1""",
)

AWAIT_INDEXES: Final = "CALL db.awaitIndexes($timeout_s)"

# Startup and /health checks (1.2, 2.2, 3.4)
PING: Final = "RETURN 1 AS ok"

GET_META: Final = """
MATCH (m:Meta {key: $key})
RETURN m.model AS model, m.dim AS dim, m.doc_prefix AS doc_prefix,
       m.query_prefix AS query_prefix, m.schema_version AS schema_version
"""

INDEX_STATES: Final = """
SHOW INDEXES YIELD name, state, properties
WHERE name IN $names
RETURN name, state, properties
"""

# SC-15 migration in init_neo4j: a case vector index created before SC-15 lacks the filter
# properties and is dropped so the 2.2 DDL can create it again. Vectors stay on the nodes.
DROP_CASE_CHUNK_EMBEDDING_INDEX: Final = "DROP INDEX case_chunk_embedding IF EXISTS"

# Schema inventory, used by init_neo4j to show that a second run changes nothing.
SCHEMA_INVENTORY: Final = """
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
RETURN name, type, labelsOrTypes, properties, options
ORDER BY name
"""

CONSTRAINT_INVENTORY: Final = """
SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties
RETURN name, type, labelsOrTypes, properties
ORDER BY name
"""

META_NODES: Final = """
MATCH (m:Meta)
RETURN m.key AS key, m.model AS model, m.dim AS dim, m.doc_prefix AS doc_prefix,
       m.query_prefix AS query_prefix, m.schema_version AS schema_version
ORDER BY key
"""

# Legal KB writes (1.4 A7). Run in one transaction per Act, rows in UNWIND batches of 200,
# MERGE on the deterministic id (2.1, R-16). The act_id property holds the act slug.

LEGAL_DELETE_DERIVED_EDGES: Final = """
MATCH (n) WHERE n.act_id = $act_id
  AND (n:Chapter OR n:Section OR n:Clause OR n:LegalUnit OR n:LegalConcept)
OPTIONAL MATCH (n)-[r:REFERS_TO|REFERS_TO_ACT|HAS_SECTION|HAS_CLAUSE|DEFINES|PRESCRIBES_FOR]->()
DELETE r
"""

LEGAL_DELETE_ACT_EDGES: Final = """
MATCH (a:Act {id: $act_node_id})
OPTIONAL MATCH (a)-[r:HAS_CHAPTER|HAS_SECTION]->()
DELETE r
"""

LEGAL_DELETE_AMENDS: Final = """
MATCH (s:Section {act_id: $act_id})<-[r:AMENDS]-()
DELETE r
"""

LEGAL_DELETE_STALE: Final = """
MATCH (n) WHERE n.act_id = $act_id
  AND (n:Chapter OR n:Section OR n:Clause OR n:LegalUnit OR n:LegalConcept)
  AND NOT n.id IN $keep_ids
DETACH DELETE n
"""

LEGAL_MERGE_ACT: Final = """
MERGE (a:Act {id: $act_node_id})
SET a += $props
"""

LEGAL_MERGE_CHAPTERS: Final = """
UNWIND $rows AS row
MERGE (c:Chapter {id: row.id})
SET c += row.props
WITH c
MATCH (a:Act {id: $act_node_id})
MERGE (a)-[:HAS_CHAPTER]->(c)
"""

LEGAL_MERGE_SECTIONS: Final = """
UNWIND $rows AS row
MERGE (s:Section {id: row.id})
SET s += row.props
FOREACH (_ IN CASE WHEN row.is_unit THEN [1] ELSE [] END | SET s:LegalUnit)
FOREACH (_ IN CASE WHEN row.is_unit THEN [] ELSE [1] END | REMOVE s:LegalUnit)
WITH s, row
MATCH (a:Act {id: $act_node_id})
OPTIONAL MATCH (c:Chapter {id: row.chapter_id})
FOREACH (_ IN CASE WHEN c IS NULL THEN [1] ELSE [] END | MERGE (a)-[:HAS_SECTION]->(s))
FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | MERGE (c)-[:HAS_SECTION]->(s))
"""

LEGAL_MERGE_CLAUSES: Final = """
UNWIND $rows AS row
MERGE (c:Clause {id: row.id})
SET c += row.props
FOREACH (_ IN CASE WHEN row.is_unit THEN [1] ELSE [] END | SET c:LegalUnit)
FOREACH (_ IN CASE WHEN row.is_unit THEN [] ELSE [1] END | REMOVE c:LegalUnit)
WITH c, row
MATCH (s:Section {id: row.section_id})
MERGE (s)-[:HAS_CLAUSE]->(c)
"""

LEGAL_MERGE_PARTS: Final = """
UNWIND $rows AS row
MERGE (u:LegalUnit {id: row.id})
SET u += row.props
WITH u, row
MATCH (s:Section {id: row.section_id})
MERGE (s)-[:HAS_CLAUSE]->(u)
"""

LEGAL_MERGE_CONCEPTS: Final = """
UNWIND $rows AS row
MERGE (k:LegalConcept {id: row.id})
SET k += row.props
WITH k, row
MATCH (c:Clause {id: row.clause_id})
MERGE (c)-[:DEFINES]->(k)
"""

LEGAL_MERGE_STUB_ACTS: Final = """
UNWIND $rows AS row
MERGE (a:Act {id: row.id})
ON CREATE SET a.external = true, a.title = row.title, a.short_title = row.title,
              a.kind = 'act', a.number = row.number, a.jurisdiction = 'IN'
"""

LEGAL_MERGE_REFERS_TO: Final = """
UNWIND $rows AS row
MATCH (s:Section|Clause {id: row.source})
MATCH (t:Section|Clause|Chapter {id: row.target})
MERGE (s)-[r:REFERS_TO]->(t)
SET r.qualifier = row.qualifier, r.source = 'regex'
"""

LEGAL_MERGE_REFERS_TO_ACT: Final = """
UNWIND $rows AS row
MATCH (s:Section|Clause {id: row.source})
MATCH (a:Act {id: row.target})
MERGE (s)-[r:REFERS_TO_ACT]->(a)
SET r.section_ref = row.section_ref
"""

LEGAL_MERGE_AMENDS: Final = """
UNWIND $rows AS row
MATCH (a:Act {id: row.act_id})
MATCH (s:Section {id: row.section_id})
MERGE (a)-[r:AMENDS]->(s)
SET r.note = row.note, r.effective_on = row.effective_on, r.previous_text = row.previous_text
"""

LEGAL_MERGE_PRESCRIBES_FOR: Final = """
UNWIND $rows AS row
MATCH (rule:Section {id: row.rule_section_id})
MATCH (s:Section {id: row.section_id})
MERGE (rule)-[r:PRESCRIBES_FOR]->(s)
SET r.note = row.note
"""

# Lookups used by legal ingestion
LEGAL_ACT: Final = """
MATCH (a:Act {id: $act_node_id})
RETURN a.id AS id, a.external AS external, a.title AS title, a.short_title AS short_title,
       a.ingest_metadata AS ingest_metadata, a.source_path AS source_path
"""

# SC-9 (P-21): substitutions stored on the notifications that amend an act.
AMENDMENT_SUBSTITUTIONS: Final = """
MATCH (a:Act)
WHERE a.amends_act_id = $act_id AND ($exclude IS NULL OR a.id <> $exclude)
RETURN a.id AS id, a.substitutions AS substitutions
ORDER BY a.id
"""

LEGAL_PROVISION_IDS: Final = """
MATCH (n) WHERE n.act_id = $act_id AND (n:Section OR n:Clause)
RETURN n.id AS id
"""

# A8 verification: node counts per label for one act.
LEGAL_COUNTS: Final = """
MATCH (n) WHERE n.act_id = $act_id
RETURN sum(CASE WHEN n:Chapter THEN 1 ELSE 0 END) AS chapters,
       sum(CASE WHEN n:Section THEN 1 ELSE 0 END) AS sections,
       sum(CASE WHEN n:Clause THEN 1 ELSE 0 END) AS clauses,
       sum(CASE WHEN n:LegalUnit THEN 1 ELSE 0 END) AS units,
       sum(CASE WHEN n:LegalUnit AND size(n.embedding) = $dim THEN 1 ELSE 0 END) AS embedded,
       sum(CASE WHEN n:LegalConcept THEN 1 ELSE 0 END) AS concepts
"""

# Legal vector search (shared KB, no scope). Hybrid retrieval arrives in Phase 4.
LEGAL_VECTOR_SEARCH: Final = """
CALL db.index.vector.queryNodes('legal_unit_embedding', $k, $query_vec)
YIELD node, score
RETURN node.id AS id, score
ORDER BY score DESC
"""

# Legal full-text search (shared KB, no scope, so no over-fetch loop; 4.2 fetch 30). $lucene
# must already be escaped and quoted (R-03, fulltext_search.py).
LEGAL_FULLTEXT_SEARCH: Final = """
CALL db.index.fulltext.queryNodes('legal_text_ft', $lucene, {limit: $k})
YIELD node, score
RETURN node.id AS id, score
ORDER BY score DESC
"""

# Legal KB reads for case legal linking (B9). Shared KB, no case scope.
LEGAL_ACT_TITLES: Final = """
MATCH (a:Act)
RETURN a.id AS id, a.kind AS kind, a.short_title AS short_title, a.title AS title,
       coalesce(a.external, false) AS external
ORDER BY a.id
"""

LEGAL_PROVISIONS_EXIST: Final = """
UNWIND $ids AS pid
MATCH (n:Section|Clause {id: pid})
RETURN n.id AS id
"""

LEGAL_UNIT_TEXTS: Final = """
UNWIND $ids AS uid
MATCH (u:LegalUnit {id: uid})
RETURN u.id AS id, u.search_text AS search_text
"""

LIST_ACTS: Final = """
MATCH (a:Act)
WHERE $cursor IS NULL OR a.id > $cursor
RETURN a.id AS id, a.kind AS kind, a.title AS title, a.short_title AS short_title,
       a.year AS year, a.number AS number, a.jurisdiction AS jurisdiction,
       a.as_on_date AS as_on_date, a.status AS status, a.external AS external,
       a.issued_under_act_id AS issued_under_act_id,
       a.issued_under_section AS issued_under_section
ORDER BY a.id
LIMIT $limit
"""

# Jobs (2.3 system nodes)
CREATE_JOB: Final = """
MERGE (j:Job {id: $id})
ON CREATE SET j.type = $type, j.user_id = $user_id, j.case_id = $case_id, j.doc_id = $doc_id,
              j.status = 'queued', j.stage = null, j.progress = 0, j.attempts = 0,
              j.warnings = [], j.created_at = $created_at
"""

UPDATE_JOB: Final = """
MATCH (j:Job {id: $id})
SET j += $props
"""

JOB_FOR_WORKER: Final = """
MATCH (j:Job {id: $id})
RETURN j.id AS id, j.type AS type, j.status AS status, j.user_id AS user_id,
       j.case_id AS case_id, j.doc_id AS doc_id, j.attempts AS attempts
"""

# GET /v1/jobs/{job_id}: only the caller's own jobs (R-02).
JOB_FOR_USER: Final = """
MATCH (j:Job {id: $id})
WHERE j.user_id = $user_id
RETURN j.id AS id, j.type AS type, j.status AS status, j.stage AS stage, j.progress AS progress,
       j.warnings AS warnings, j.error_code AS error_code, j.error_message AS error_message,
       j.created_at AS created_at, j.started_at AS started_at, j.finished_at AS finished_at
"""

# E-11: on startup, running jobs go back to queued with one more attempt; past 3 attempts
# they fail with JOB_ATTEMPTS_EXCEEDED.
JOBS_RECOVER: Final = """
MATCH (j:Job {status: 'running'})
SET j.attempts = coalesce(j.attempts, 0) + 1
WITH j, j.attempts > $max_attempts AS exceeded
SET j.status = CASE WHEN exceeded THEN 'failed' ELSE 'queued' END,
    j.error_code = CASE WHEN exceeded THEN 'JOB_ATTEMPTS_EXCEEDED' ELSE j.error_code END,
    j.error_message = CASE WHEN exceeded THEN 'The job was interrupted too many times.'
                           ELSE j.error_message END,
    j.finished_at = CASE WHEN exceeded THEN $now ELSE j.finished_at END
RETURN j.id AS id, j.type AS type, j.status AS status, j.user_id AS user_id,
       j.case_id AS case_id, j.doc_id AS doc_id, j.attempts AS attempts
"""

JOBS_QUEUED: Final = """
MATCH (j:Job {status: 'queued'})
RETURN j.id AS id
ORDER BY j.created_at
"""

# R-05: deleting a case cancels its queued jobs; a running one stops at its next E-10 check.
CASE_JOBS_CANCEL: Final = """
MATCH (j:Job)
WHERE j.case_id = $case_id AND j.user_id = $user_id AND j.status = 'queued'
SET j.status = 'cancelled', j.finished_at = $now
RETURN count(j) AS cancelled
"""

JOB_STATUS: Final = """
MATCH (j:Job {id: $id})
RETURN j.status AS status, j.stage AS stage, j.progress AS progress,
       j.error_code AS code, j.error_message AS msg, j.warnings AS warnings
"""

# Case KB (1.4 B). Every query below is case-scoped: it references $case_id and $user_id
# and runs through scope.ScopedGraph (R-01, R-02). case_id and doc_id properties hold bare
# UUIDs (A-8); node ids carry the 2.1 prefix.

CASE_CREATE: Final = """
MERGE (c:Case {id: $case_node_id})
ON CREATE SET c.case_id = $case_id, c.user_id = $user_id, c.title = $title,
              c.status = 'active', c.created_at = $created_at
WITH c
WHERE c.user_id = $user_id
RETURN c.case_id AS case_id, c.title AS title, c.status AS status, c.created_at AS created_at
"""

CASE_STATUS: Final = """
MATCH (c:Case {case_id: $case_id, user_id: $user_id})
RETURN c.status AS status
"""

CASE_GET: Final = """
MATCH (c:Case {case_id: $case_id, user_id: $user_id})
RETURN c.case_id AS case_id, c.title AS title, c.status AS status, c.created_at AS created_at
"""

# GET /v1/cases: the caller's cases, paged by case_id (3.1, A-30). User-scoped, so it does
# not go through ScopedGraph, which needs a case.
CASE_LIST: Final = """
MATCH (c:Case)
WHERE c.user_id = $user_id AND ($cursor IS NULL OR c.case_id > $cursor)
RETURN c.case_id AS case_id, c.title AS title, c.status AS status, c.created_at AS created_at
ORDER BY c.case_id
LIMIT $limit
"""

# R-05: mark first, so new uploads and queries get 409 CASE_DELETING.
CASE_MARK_DELETING: Final = """
MATCH (c:Case {case_id: $case_id, user_id: $user_id})
SET c.status = 'deleting'
RETURN c.case_id AS case_id, c.title AS title, c.status AS status, c.created_at AS created_at
"""

# R-05: every node carrying the case_id, in batches.
CASE_DELETE_BATCH: Final = """
MATCH (n)
WHERE n.case_id = $case_id AND n.user_id = $user_id
WITH n LIMIT $batch
DETACH DELETE n
RETURN count(*) AS deleted
"""

# Startup: finish deletions a restart interrupted.
CASES_DELETING: Final = """
MATCH (c:Case {status: 'deleting'})
RETURN c.case_id AS case_id, c.user_id AS user_id
"""

DOC_COUNT: Final = """
MATCH (d:CaseDocument)
WHERE d.case_id = $case_id AND d.user_id = $user_id
RETURN count(d) AS n
"""

# E-09: the same file twice in one case.
DOC_BY_SHA256: Final = """
MATCH (d:CaseDocument)
WHERE d.case_id = $case_id AND d.user_id = $user_id AND d.sha256 = $sha256
RETURN d.id AS id, d.processing_status AS processing_status
ORDER BY d.uploaded_at
LIMIT 1
"""

# A-6: the CaseDocument fields the API returns.
DOC_GET: Final = """
MATCH (d:CaseDocument {id: $doc_node_id, case_id: $case_id, user_id: $user_id})
RETURN d.id AS id, d.case_id AS case_id, d.filename AS filename, d.media_type AS media_type,
       d.mime AS mime, d.size_bytes AS size_bytes, d.sha256 AS sha256,
       d.processing_status AS processing_status, d.page_count AS page_count,
       d.duration_sec AS duration_sec, d.language AS language, d.error_code AS error_code,
       d.uploaded_at AS uploaded_at
"""

DOC_LIST: Final = """
MATCH (d:CaseDocument)
WHERE d.case_id = $case_id AND d.user_id = $user_id AND ($cursor IS NULL OR d.id > $cursor)
WITH d ORDER BY d.id LIMIT $limit
RETURN d.id AS id, d.case_id AS case_id, d.filename AS filename, d.media_type AS media_type,
       d.mime AS mime, d.size_bytes AS size_bytes, d.sha256 AS sha256,
       d.processing_status AS processing_status, d.page_count AS page_count,
       d.duration_sec AS duration_sec, d.language AS language, d.error_code AS error_code,
       d.uploaded_at AS uploaded_at
"""

# B2: the CaseDocument and its Job in one transaction, only while the case is active.
DOC_CREATE_WITH_JOB: Final = """
MATCH (c:Case {case_id: $case_id, user_id: $user_id})
WHERE c.status = 'active'
MERGE (d:CaseDocument {id: $doc_node_id})
ON CREATE SET d += $doc_props, d.case_id = $case_id, d.user_id = $user_id,
              d.processing_status = 'queued'
MERGE (c)-[:HAS_DOCUMENT]->(d)
MERGE (j:Job {id: $job_node_id})
ON CREATE SET j.type = 'case_document', j.user_id = $user_id, j.case_id = $case_id,
              j.doc_id = $doc_id, j.status = 'queued', j.stage = 'stored', j.progress = 0,
              j.attempts = 0, j.warnings = [], j.created_at = $created_at
RETURN d.id AS doc_node_id, j.id AS job_node_id
"""

DOC_FOR_PROCESSING: Final = """
MATCH (d:CaseDocument {id: $doc_node_id, case_id: $case_id, user_id: $user_id})
RETURN d.stored_path AS stored_path, d.mime AS mime, d.media_type AS media_type,
       d.language AS language
"""

DOC_UPDATE: Final = """
MATCH (d:CaseDocument {id: $doc_node_id, case_id: $case_id, user_id: $user_id})
SET d += $props
"""

# B5: chunks MERGE on their deterministic id (R-16) and hang off their document.
CHUNKS_MERGE: Final = """
UNWIND $rows AS row
MATCH (d:CaseDocument {id: $doc_node_id, case_id: $case_id, user_id: $user_id})
MERGE (k:CaseChunk {id: row.id})
SET k += row.props, k.case_id = $case_id, k.user_id = $user_id, k.doc_id = $doc_id
MERGE (d)-[:HAS_CHUNK]->(k)
"""

# Chunks of a document that a re-run no longer produces.
CHUNKS_DELETE_STALE: Final = """
MATCH (k:CaseChunk {doc_id: $doc_id, case_id: $case_id, user_id: $user_id})
WHERE NOT k.id IN $keep_ids
DETACH DELETE k
"""

# E-10: partial nodes of a cancelled document.
CHUNKS_DELETE_FOR_DOC: Final = """
MATCH (k:CaseChunk {doc_id: $doc_id, case_id: $case_id, user_id: $user_id})
DETACH DELETE k
"""

# R-16: before extraction re-runs for a document, its MENTIONS go, then any entity of the
# case left with no MENTIONS (also used for E-10 clean-up).
DOC_MENTIONS_DELETE: Final = """
MATCH (k:CaseChunk {doc_id: $doc_id, case_id: $case_id, user_id: $user_id})-[m:MENTIONS]->()
DELETE m
"""

CASE_ORPHAN_ENTITIES_DELETE: Final = """
MATCH (e:Party|Event|Evidence|Claim)
WHERE e.case_id = $case_id AND e.user_id = $user_id AND NOT ()-[:MENTIONS]->(e)
DETACH DELETE e
"""

# B7: parties already in the case (from other documents), for R-14 resolution.
CASE_PARTIES: Final = """
MATCH (p:Party)
WHERE p.case_id = $case_id AND p.user_id = $user_id
RETURN p.id AS id, p.name AS name, p.aliases AS aliases, p.kind AS kind, p.role AS role
ORDER BY p.id
"""

# B8: entities MERGE on their deterministic id (R-16) and get MENTIONS from their chunks.
ENTITY_MERGE_PARTIES: Final = """
UNWIND $rows AS row
MERGE (n:Party {id: row.id})
SET n += row.props, n.case_id = $case_id, n.user_id = $user_id
WITH n, row
UNWIND row.chunk_ids AS cid
MATCH (k:CaseChunk {id: cid, case_id: $case_id, user_id: $user_id})
MERGE (k)-[:MENTIONS]->(n)
"""

ENTITY_MERGE_EVENTS: Final = """
UNWIND $rows AS row
MERGE (n:Event {id: row.id})
SET n += row.props, n.case_id = $case_id, n.user_id = $user_id
WITH n, row
UNWIND row.chunk_ids AS cid
MATCH (k:CaseChunk {id: cid, case_id: $case_id, user_id: $user_id})
MERGE (k)-[:MENTIONS]->(n)
"""

ENTITY_MERGE_CLAIMS: Final = """
UNWIND $rows AS row
MERGE (n:Claim {id: row.id})
SET n += row.props, n.case_id = $case_id, n.user_id = $user_id
WITH n, row
UNWIND row.chunk_ids AS cid
MATCH (k:CaseChunk {id: cid, case_id: $case_id, user_id: $user_id})
MERGE (k)-[:MENTIONS]->(n)
"""

ENTITY_MERGE_EVIDENCE: Final = """
UNWIND $rows AS row
MERGE (n:Evidence {id: row.id})
SET n += row.props, n.case_id = $case_id, n.user_id = $user_id
WITH n, row
UNWIND row.chunk_ids AS cid
MATCH (k:CaseChunk {id: cid, case_id: $case_id, user_id: $user_id})
MERGE (k)-[:MENTIONS]->(n)
"""

PARTICIPATED_IN_MERGE: Final = """
UNWIND $rows AS row
MATCH (p:Party {id: row.source}) WHERE p.case_id = $case_id AND p.user_id = $user_id
MATCH (e:Event {id: row.target}) WHERE e.case_id = $case_id AND e.user_id = $user_id
MERGE (p)-[:PARTICIPATED_IN]->(e)
"""

SUPPORTS_MERGE: Final = """
UNWIND $rows AS row
MATCH (v:Evidence {id: row.source}) WHERE v.case_id = $case_id AND v.user_id = $user_id
MATCH (c:Claim {id: row.target}) WHERE c.case_id = $case_id AND c.user_id = $user_id
MERGE (v)-[:SUPPORTS]->(c)
"""

CONTRADICTS_MERGE: Final = """
UNWIND $rows AS row
MATCH (v:Evidence {id: row.source}) WHERE v.case_id = $case_id AND v.user_id = $user_id
MATCH (c:Claim {id: row.target}) WHERE c.case_id = $case_id AND c.user_id = $user_id
MERGE (v)-[:CONTRADICTS]->(c)
"""

# BEFORE edges are derived in code from event dates (B8) and rebuilt for the whole case.
CASE_EVENT_DATES: Final = """
MATCH (e:Event)
WHERE e.case_id = $case_id AND e.user_id = $user_id AND e.date IS NOT NULL
RETURN e.id AS id, e.date AS date
"""

CASE_BEFORE_DELETE: Final = """
MATCH (a:Event)-[r:BEFORE]->(:Event)
WHERE a.case_id = $case_id AND a.user_id = $user_id
DELETE r
"""

BEFORE_MERGE: Final = """
UNWIND $rows AS row
MATCH (a:Event {id: row.source}) WHERE a.case_id = $case_id AND a.user_id = $user_id
MATCH (b:Event {id: row.target}) WHERE b.case_id = $case_id AND b.user_id = $user_id
MERGE (a)-[:BEFORE]->(b)
"""

# B9: regex citations (INVOKES) and verified claim links (ALLEGES_VIOLATION_OF).
DOC_INVOKES_DELETE: Final = """
MATCH (k:CaseChunk {doc_id: $doc_id, case_id: $case_id, user_id: $user_id})-[r:INVOKES]->()
DELETE r
"""

INVOKES_MERGE: Final = """
UNWIND $rows AS row
MATCH (k:CaseChunk {id: row.chunk_id, case_id: $case_id, user_id: $user_id})
MATCH (t:Section|Clause {id: row.target})
MERGE (k)-[r:INVOKES]->(t)
SET r.text_span = row.text_span
"""

CLAIM_ALLEGES_DELETE: Final = """
MATCH (c:Claim)-[r:ALLEGES_VIOLATION_OF]->()
WHERE c.id IN $claim_ids AND c.case_id = $case_id AND c.user_id = $user_id
DELETE r
"""

ALLEGES_MERGE: Final = """
UNWIND $rows AS row
MATCH (c:Claim {id: row.claim_id}) WHERE c.case_id = $case_id AND c.user_id = $user_id
MATCH (t:Section|Clause {id: row.target})
MERGE (c)-[r:ALLEGES_VIOLATION_OF]->(t)
SET r.confidence = row.confidence, r.method = 'vector+llm', r.rationale = row.rationale
"""

CASE_NODE_COUNTS: Final = """
MATCH (n {case_id: $case_id, user_id: $user_id})
RETURN labels(n) AS labels, count(*) AS n
ORDER BY n DESC
"""

# 4.2 case vector search (SC-15): the scope filter runs inside the index, which declares
# case_id and user_id (2.2), so it always returns up to k in-scope rows. No R-09 loop.
CASE_VECTOR_SEARCH: Final = """
CYPHER 25
MATCH (node:CaseChunk)
  SEARCH node IN (VECTOR INDEX case_chunk_embedding FOR $query_vec
                  WHERE node.case_id = $case_id AND node.user_id = $user_id
                  LIMIT $k) SCORE AS score
RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id,
       node.page AS page, node.start_sec AS start_sec, node.end_sec AS end_sec, score
ORDER BY score DESC
"""

# 4.2 case full-text search, run inside the R-09 over-fetch loop (full-text SEARCH cannot
# filter, O-1). $lucene must already be escaped and quoted (R-03, fulltext_search.py).
CASE_FULLTEXT_SEARCH: Final = """
CALL db.index.fulltext.queryNodes('case_text_ft', $lucene, {limit: $fetch_k})
YIELD node, score
WHERE node.case_id = $case_id AND node.user_id = $user_id
RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id,
       node.page AS page, node.start_sec AS start_sec, node.end_sec AS end_sec, score
ORDER BY score DESC LIMIT $k
"""

# ---------------------------------------------------------------------------------------
# Query agent (Phase 4, 1.4 C)

# E-06: documents of the case that are ready for retrieval.
CASE_READY_DOC_COUNT: Final = """
MATCH (d:CaseDocument)
WHERE d.case_id = $case_id AND d.user_id = $user_id
  AND d.processing_status IN ['ready', 'ready_with_warnings']
RETURN count(d) AS n
"""

# C5: entities mentioned in the selected chunks plus their one-hop case neighbours, with the
# legal units their claims are linked to. Every node is checked against the scope.
CASE_ENTITIES_FOR_CHUNKS: Final = """
MATCH (c:CaseChunk)-[:MENTIONS]->(e)
WHERE c.id IN $chunk_ids AND c.case_id = $case_id AND c.user_id = $user_id
  AND e.case_id = $case_id AND e.user_id = $user_id
WITH collect(DISTINCT e) AS mentioned
UNWIND mentioned AS e
OPTIONAL MATCH (e)-[:PARTICIPATED_IN|SUPPORTS|CONTRADICTS]-(n)
WHERE n.case_id = $case_id AND n.user_id = $user_id
WITH mentioned, collect(DISTINCT n) AS neighbours
UNWIND mentioned + neighbours AS e
WITH DISTINCT e, e IN mentioned AS is_mentioned
OPTIONAL MATCH (e)-[:ALLEGES_VIOLATION_OF]->(u)
RETURN e.id AS id, labels(e) AS labels, e.name AS name, e.kind AS kind, e.role AS role,
       e.description AS description, e.date AS date, e.text AS text,
       e.claim_type AS claim_type, is_mentioned, collect(DISTINCT u.id) AS legal_ids
ORDER BY is_mentioned DESC, id
LIMIT $limit
"""

# C5/C6: regex citations in the selected chunks (INVOKES), seeds for legal retrieval.
CASE_CHUNK_INVOKES: Final = """
MATCH (c:CaseChunk)-[:INVOKES]->(t)
WHERE c.id IN $chunk_ids AND c.case_id = $case_id AND c.user_id = $user_id
RETURN c.id AS chunk_id, collect(DISTINCT t.id) AS legal_ids
ORDER BY chunk_id
"""

# R-21 hydration of fact sources: doc_id on CaseChunk is the bare UUID (A-23).
CASE_CHUNK_SOURCES: Final = """
MATCH (c:CaseChunk)
WHERE c.id IN $chunk_ids AND c.case_id = $case_id AND c.user_id = $user_id
OPTIONAL MATCH (d:CaseDocument {case_id: $case_id, user_id: $user_id})-[:HAS_CHUNK]->(c)
RETURN c.id AS id, c.doc_id AS doc_id, d.filename AS filename, c.page AS page,
       c.start_sec AS start_sec, c.end_sec AS end_sec
"""

# Hydration of legal provisions by id (C6 to C10, R-21): any Section, Clause or split part
# (A-11), with its parent Section and Act. Shared KB, no case scope.
LEGAL_PROVISIONS_HYDRATE: Final = """
UNWIND $ids AS pid
MATCH (n:Section|Clause|LegalUnit {id: pid})
OPTIONAL MATCH (p:Section)-[:HAS_CLAUSE]->(n)
WITH n, coalesce(p, n) AS sec
OPTIONAL MATCH (a:Act)-[:HAS_SECTION|HAS_CHAPTER*1..2]->(sec)
RETURN n.id AS id, n:Section AS is_section, n:LegalUnit AS is_unit, n.kind AS kind,
       n.label AS label, n.text AS text, n.search_text AS search_text,
       sec.id AS section_id, sec.number AS section_number, sec.title AS section_title,
       sec.status AS status, sec.text AS section_text,
       a.id AS act_node_id, a.short_title AS act_short_title, a.kind AS act_kind,
       a.as_on_date AS act_as_on
"""

# 4.2 legal graph expansion, exactly as specified (structural part; R-12 caps are applied in
# code, and definitions are added in code, R-13).
LEGAL_EXPANSION: Final = """
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
"""

# R-12 second hop: outgoing REFERS_TO only, from first-hop Sections or Clauses (which need
# not be LegalUnits).
LEGAL_EXPANSION_OUTGOING: Final = """
UNWIND $ids AS pid
MATCH (n:Section|Clause {id: pid})
OPTIONAL MATCH (p:Section)-[:HAS_CLAUSE]->(n)
WITH n, coalesce(p, n) AS sec
OPTIONAL MATCH (sec)-[:REFERS_TO]->(out) WHERE out:Section OR out:Clause
RETURN n.id AS id, collect(DISTINCT out.id)[0..10] AS refers_to
"""

# R-13: the concept list, cached in memory at startup.
LEGAL_CONCEPTS: Final = """
MATCH (c:Clause)-[:DEFINES]->(k:LegalConcept)
RETURN k.id AS id, k.name AS name, c.id AS clause_id
ORDER BY k.id
"""

# GET /v1/legal/provisions/{provision_id} (3.4, A-45): one Section or Clause with its parent
# Section and Act, and the ids it refers to and is referred to by. A split part (LegalUnit
# only, A-11) does not match, so it returns 404.
PROVISION_GET: Final = """
MATCH (n:Section|Clause {id: $id})
OPTIONAL MATCH (p:Section)-[:HAS_CLAUSE]->(n)
WITH n, p, coalesce(p, n) AS sec
OPTIONAL MATCH (a:Act)-[:HAS_SECTION|HAS_CHAPTER*1..2]->(sec)
OPTIONAL MATCH (n)-[:REFERS_TO]->(out) WHERE out:Section OR out:Clause
WITH n, p, sec, a, collect(DISTINCT out.id) AS refers_to
OPTIONAL MATCH (inc)-[:REFERS_TO]->(n) WHERE inc:Section OR inc:Clause
RETURN n.id AS id, n:Section AS is_section, n.label AS label, n.text AS text,
       sec.number AS section_number, sec.status AS status, sec.in_force_from AS in_force_from,
       sec.amendment_notes AS amendment_notes, p.id AS parent_id,
       a.short_title AS act_short_title, a.kind AS act_kind,
       refers_to, collect(DISTINCT inc.id) AS referenced_by
"""

# Evaluation labelling aid (tests/eval/run_eval.py --list): every Section of the ingested
# Acts and Rules with its Clauses. External stubs and amendment notifications (SC-9) are
# left out, because their provisions are never retrieval targets. Split parts (A-11) are not
# Clauses, so they never appear (A-35).
LEGAL_PROVISION_INDEX: Final = """
MATCH (a:Act)-[:HAS_SECTION|HAS_CHAPTER*1..2]->(s:Section)
WHERE coalesce(a.external, false) = false AND a.amends_act_id IS NULL
OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(c:Clause)
WITH a, s, c ORDER BY c.id
RETURN a.id AS act_node_id, a.short_title AS act_short_title, a.kind AS act_kind,
       s.id AS id, s.number AS number, s.title AS title,
       [x IN collect(c) | {id: x.id, label: x.label, kind: x.kind, term: x.term,
                           text: left(x.text, 110)}] AS clauses
"""

# Evaluation check: whether a labelled id can come out of retrieval directly, as a LegalUnit
# or as the base id of split parts (A-11, A-35). Other ids appear only via graph expansion.
LEGAL_RETRIEVABLE: Final = """
UNWIND $ids AS pid
MATCH (n:Section|Clause {id: pid})
OPTIONAL MATCH (:Section)-[:HAS_CLAUSE]->(p:LegalUnit)
WHERE NOT p:Clause AND p.id STARTS WITH pid + '-p'
RETURN n.id AS id, n:LegalUnit AS unit, count(p) > 0 AS split
"""

# Throwaway test databases (tests/conftest.py). Run against the system database.
CREATE_DATABASE: Final = "CREATE DATABASE $name IF NOT EXISTS WAIT"
DROP_DATABASE: Final = "DROP DATABASE $name IF EXISTS WAIT"
