# Data models and schemas

> Part of the Legal GraphRAG Platform spec (v1.1). Section numbers (1.2, 3.2, R-09, E-05, P-06, G4...) are global across all files in `docs/`; see `docs/README.md` for the index. Do not renumber them: code comments and tests refer to them.

## 2. Data models

### 2.1 Identifier grammar

Every node has a string property id, unique per label. Ids are deterministic so re-ingestion is idempotent. A slug is lowercase with every run of non-alphanumeric characters replaced by a single hyphen.

| Node | id format | Example |
| --- | --- | --- |
| Act | act:{slug} | act:cpa-2019 |
| Chapter | ch:{act\_slug}:{roman} | ch:cpa-2019:IV |
| Section | sec:{act\_slug}:{number} | sec:cpa-2019:38 |
| Clause | cl:{act\_slug}:{section}:{label} | cl:cpa-2019:2:7 and cl:cpa-2019:38:2; in rules documents a definition label may be lowercase letters, for example cl:cp-jurisdiction-rules-2021:2:a (SC-5, P-16) |
| LegalConcept | con:{slug of term} | con:consumer |
| Judgment, JudgmentChunk | jdg:{slug}, jch:{slug}:{order} | jdg:xyz-v-abc-2020 |
| Case | case:{uuid4} |  |
| CaseDocument | doc:{uuid4} |  |
| CaseChunk | chk:{doc\_uuid}:{order, 4 digits} | chk:9f1c...:0007 |
| Party, Event, Evidence, Claim | pty, evt, evd, clm : {case\_uuid}:{first 12 hex of sha1 of normalised key} |  |
| Job | job:{uuid4} |  |

### 2.2 Neo4j DDL (scripts/init\_neo4j.py, idempotent)

```cypher
CREATE CONSTRAINT act_id IF NOT EXISTS FOR (n:Act) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (n:Clause) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT legalunit_id IF NOT EXISTS FOR (n:LegalUnit) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:LegalConcept) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT judgment_id IF NOT EXISTS FOR (n:Judgment) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT casedoc_id IF NOT EXISTS FOR (n:CaseDocument) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT casechunk_id IF NOT EXISTS FOR (n:CaseChunk) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT party_id IF NOT EXISTS FOR (n:Party) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT job_id IF NOT EXISTS FOR (n:Job) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT meta_key IF NOT EXISTS FOR (n:Meta) REQUIRE n.key IS UNIQUE;

CREATE VECTOR INDEX legal_unit_embedding IF NOT EXISTS
FOR (n:LegalUnit) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX case_chunk_embedding IF NOT EXISTS
FOR (n:CaseChunk) ON (n.embedding) WITH [n.case_id, n.user_id]
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

CREATE FULLTEXT INDEX legal_text_ft IF NOT EXISTS FOR (n:LegalUnit) ON EACH [n.search_text];
CREATE FULLTEXT INDEX case_text_ft IF NOT EXISTS FOR (n:CaseChunk) ON EACH [n.text];

CREATE INDEX casechunk_scope IF NOT EXISTS FOR (n:CaseChunk) ON (n.case_id, n.user_id);
CREATE INDEX case_owner IF NOT EXISTS FOR (n:Case) ON (n.user_id);
CREATE INDEX job_owner IF NOT EXISTS FOR (n:Job) ON (n.user_id, n.status);

MERGE (m:Meta {key: 'embedding'})
ON CREATE SET m.model = 'nomic-embed-text', m.dim = 768,
  m.doc_prefix = 'search_document: ', m.query_prefix = 'search_query: ', m.schema_version = 1;
```

Case vector index filter properties (SC-15): `WITH [n.case_id, n.user_id]` lets the 4.2 case vector query filter by scope inside the index. When `case_chunk_embedding` exists without those properties (a database created before SC-15), init\_neo4j drops it and creates it again. The vectors stay on the nodes, so nothing is re-embedded. Startup (1.2) and GET /health treat the index as not ready unless it is ONLINE and declares both properties.

Startup check: the Meta node with key 'embedding' must have model equal to EMBED\_MODEL and dim equal to EMBED\_DIM. Any mismatch aborts startup. Changing the embedding model requires a new database or a full re-embed; there is no runtime switching.

### 2.3 Node properties

**Legal KB (shared, no user\_id)**

| Node | Properties |
| --- | --- |
| Act | id, kind ('act', 'rule', 'regulation'), title, short\_title, year (int), number (string, for example '35 of 2019'), jurisdiction ('IN'), as\_on\_date (date), status ('in\_force', 'repealed'), external (bool; true for stub nodes created from references), issued\_under\_act\_id, issued\_under\_section (list of string; SC-6), objects\_text (P-03), enabling\_text (rules only; P-14, SC-5), source\_file, source\_path and ingest\_metadata (the file and metadata of the last ingest, used to rebuild rules when an amendment arrives; SC-9), amends\_act\_id and substitutions (amending notifications only: the principal's act\_id and the P-19 substitutions as JSON; SC-9) |
| Chapter | id, act\_id, number (roman string), title |
| Section | id, act\_id, number (string, may be '1A'), title, chapter\_number, text, status ('in\_force', 'repealed', 'omitted'), in\_force\_from (date or null), amendment\_notes (list of string), is\_unit (bool) |
| Clause | id, act\_id, section\_number, label (string), kind ('subsection', 'definition'), term (definitions only), text |
| LegalConcept | id, name, definition\_text, act\_id |
| Judgment | id, title, court, decided\_on (date), citation, status ('good\_law', 'overruled') |
| JudgmentChunk | id, judgment\_id, order (int), text |

LegalUnit is an additional label on every node that is embedded for retrieval (Section where is\_unit is true, Clause, JudgmentChunk). Each LegalUnit also has embedding (list of 768 floats), embed\_model, and search\_text.

**Retrieval-unit rule.** A Section is a unit when its text is at most 2,000 characters. Otherwise the Section keeps is\_unit false (no embedding, no LegalUnit label) and its subsection Clauses are the units. A Clause over 2,000 characters is split at proviso or sentence boundaries into parts with ids suffixed -p1, -p2. Each definition in section 2 is always its own unit.

**Embedded and searched text** (search\_text is identical; the embedding adds the prefix 'search\_document: '):

```
{act.short_title} | Chapter {roman}: {chapter title} | Section {number}. {section title} | ({label})
{text}
```

**Case KB (private; every node has case\_id and user\_id)**

| Node | Properties |
| --- | --- |
| Case | id, user\_id, title, created\_at (datetime), status ('active', 'deleting') |
| CaseDocument | id, case\_id, user\_id, filename, media\_type ('pdf', 'image', 'audio', 'text'), mime, size\_bytes, sha256, processing\_status ('queued', 'processing', 'ready', 'ready\_with\_warnings', 'failed'), page\_count, duration\_sec, language, error\_code, uploaded\_at, stored\_path |
| CaseChunk | id, case\_id, user\_id, doc\_id, order (int), text, embedding, embed\_model, page (int or null), start\_sec (float or null), end\_sec (float or null) |
| Party | id, name, aliases (list), kind ('person', 'organisation'), role ('complainant', 'opposite\_party', 'witness', 'other') |
| Event | id, description, date (ISO 8601, partial allowed), location |
| Evidence | id, kind ('document', 'receipt', 'message', 'photo', 'testimony', 'other'), description |
| Claim | id, text, claim\_type ('defect', 'deficiency', 'unfair\_trade\_practice', 'overcharging', 'product\_liability', 'unfair\_contract', 'other'), asserted\_by (party id or null) |

**System nodes**

| Node | Properties |
| --- | --- |
| Job | id, type ('case\_document', 'legal\_ingest'), user\_id, case\_id, doc\_id, status ('queued', 'running', 'succeeded', 'succeeded\_with\_warnings', 'failed', 'cancelled'), stage (string), progress (0 to 100), attempts (int), error\_code, error\_message, warnings (list of string), created\_at, started\_at, finished\_at |
| Meta | key, model, dim, doc\_prefix, query\_prefix, schema\_version |

### 2.4 Relationships

| Type | From to | Properties | Created by |
| --- | --- | --- | --- |
| HAS\_CHAPTER | Act to Chapter |  | parser |
| HAS\_SECTION | Chapter to Section; Act to Section when the document has no chapters (P-15, SC-5) |  | parser |
| HAS\_CLAUSE | Section to Clause |  | parser |
| REFERS\_TO | Section or Clause to Section, Clause or Chapter | qualifier (string or null), source ('regex') | reference extractor |
| REFERS\_TO\_ACT | Section or Clause to Act | section\_ref (string or null) | reference extractor |
| AMENDS | Act to Section | note, effective\_on (date or null), previous\_text (SC-9 substitutions only) | footnote parser; amendment notifications (P-20, SC-9) |
| DEFINES | Clause to LegalConcept |  | parser (section 2 definitions) |
| ABOUT | Section to LegalConcept | source ('llm') | optional concept pass |
| PRESCRIBES\_FOR | Section (of a rule) to Section | note | admin metadata (prescribes\_for, 3.4; SC-6) |
| INTERPRETS | Judgment to Section or Clause |  | judgment ingestion |
| CITES, OVERRULES | Judgment to Judgment |  | judgment ingestion |
| HAS\_CHUNK | Judgment to JudgmentChunk; CaseDocument to CaseChunk |  | ingestion |
| HAS\_DOCUMENT | Case to CaseDocument |  | API |
| MENTIONS | CaseChunk to Party, Event, Evidence or Claim |  | extraction |
| PARTICIPATED\_IN | Party to Event | role | extraction |
| BEFORE | Event to Event |  | code, sorted by date |
| SUPPORTS, CONTRADICTS | Evidence to Claim |  | extraction |
| INVOKES | CaseChunk to Section or Clause | text\_span | regex |
| ALLEGES\_VIOLATION\_OF | Claim to Section or Clause | confidence (0 to 1), method ('vector+llm'), rationale (at most 300 characters) | legal linker |

### 2.5 Extraction models and agent state (Python)

```python
from typing import Annotated, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
import operator

class ExtParty(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal['person', 'organisation']
    role: Literal['complainant', 'opposite_party', 'witness', 'other']
    aliases: list[str] = []

class ExtEvent(BaseModel):
    description: str = Field(max_length=400)
    date: Optional[str] = None          # ISO 8601, partial allowed: YYYY, YYYY-MM, YYYY-MM-DD
    location: Optional[str] = None
    participants: list[str] = []        # names exactly as they appear in parties[]
    chunk_ids: list[str]                # must be ids from the input batch

class ExtClaim(BaseModel):
    text: str = Field(max_length=500)
    claim_type: Literal['defect', 'deficiency', 'unfair_trade_practice', 'overcharging',
                        'product_liability', 'unfair_contract', 'other']
    asserted_by: Optional[str] = None   # a party name from parties[]
    chunk_ids: list[str]

class ExtEvidence(BaseModel):
    kind: Literal['document', 'receipt', 'message', 'photo', 'testimony', 'other']
    description: str = Field(max_length=400)
    supports: list[int] = []            # indexes into claims[]
    contradicts: list[int] = []
    chunk_ids: list[str]

class ExtractionResult(BaseModel):
    parties: list[ExtParty] = []
    events: list[ExtEvent] = []
    claims: list[ExtClaim] = []
    evidence: list[ExtEvidence] = []

class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    user_id: str
    case_id: Optional[str]
    original_query: str
    search_query: str
    intent: Literal['CONVERSATIONAL', 'CASE_ONLY', 'LEGAL_ONLY', 'CASE_AND_LEGAL']
    case_chunks: list[dict]        # id, text, doc_id, page, start_sec, end_sec, score
    case_entities: list[dict]
    issue_statement: str
    legal_candidates: list[dict]   # id, kind, text, act_id, number, score, via
    graph_context: list[dict]
    legal_final: list[dict]
    analysis: dict
    trace: dict
```

