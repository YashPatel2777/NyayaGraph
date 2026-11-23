# Ingestion prototype: integration and build plan

> Part of the Legal GraphRAG Platform spec (proposed v1.2). Section numbers (1.2, 3.4, R-04, E-08, B4, P-06...) refer to the other files in `docs/`. This file adds three new id families: `IN-xx` (defects found in the prototype), `SC-x` (proposed spec changes that need the operator's approval) and `T3.x` (Phase 3 tasks).

## 1. What this file is for

The operator wrote an earlier ingestion pipeline (`New-Ingestion.zip`, 2,238 lines of Python in `ingestion/`). It was built for a different stack than the one fixed in `00-overview.md`. This file says, file by file, what to keep, what to change and what to throw away, lists the bugs found in it, and turns the porting work into Phase 3 tasks that the coding agent can follow.

**The prototype is reference material, not a starting codebase.** Do not copy it into `app/` and patch it. Port the pieces listed in section 4, one task at a time, into the paths in `06-file-structure.md`.

How this was checked: every file was read in full, all files compile, and the behaviours in section 4 were run against the code with no API keys set. Docling, Unstructured, WhisperX, Pyannote, PyMuPDF and Qdrant were not installed in the test environment, so statements about those libraries come from reading the prototype's code and are marked "by reading".

## 2. What the zip contains

| File | Lines | What it does |
| --- | --- | --- |
| `ingestion/processor.py` | 526 | Routes a file to a loader, chunks it, extracts graph triples, embeds, upserts to **Qdrant**, saves JSON to `processed_data/`. Also a folder scanner with `--wipe` and `--kb-type`. |
| `ingestion/chunking/splitter.py` | 445 | Paragraph packer (1,500 chars) plus LLM or regex "triple" extraction with its own entity and edge vocabulary. |
| `ingestion/loaders/pdf.py` | 414 | Docling, then Unstructured, then PyMuPDF with an OpenAI vision call for stamps and signatures, then pypdf. |
| `ingestion/loaders/audio.py` | 344 | WhisperX transcription, word alignment, Pyannote speaker diarization, merge by speaker. OpenAI Whisper API fallback. |
| `ingestion/loaders/chat.py` | 338 | WhatsApp (Android and iOS) and Telegram (text and JSON) export parser; strips system messages; handles multi-line messages. |
| `ingestion/loaders/{html,office,text}.py` | 42, 32, 23 | BeautifulSoup HTML, Unstructured DOCX and PPTX, plain text. |
| `ingestion/requirements.txt` | 36 | Qdrant, OpenAI, Docling, Unstructured, WhisperX, Pyannote, sentence-transformers, Logfire and more. |

## 3. Conflicts with the fixed decisions

These are not bugs. They are places where the prototype contradicts `AGENTS.md` and `00-overview.md`. The spec wins in every row.

| Prototype does | Spec says | Resolution |
| --- | --- | --- |
| Stores vectors in Qdrant (`QdrantClient`, `upsert`) | Neo4j is the only database (R-07, 00-overview) | Delete. Chunks are written as `CaseChunk` nodes (B5). |
| Embeds with `app.services.retrieval.embeddings`, else a **fake 1,536-dim hash vector** | One model, `nomic-embed-text`, 768 floats, no fallback (R-07, R-08) | Only `EmbeddingService` makes vectors. A missing model aborts startup (2.2). |
| OpenAI for graph extraction and vision (`gpt-4o-mini`) | Groq `llama-3.3-70b-versatile`; text only | Delete the OpenAI paths. Extraction is B6 on Groq. Stamp and signature "vision summaries" are dropped (OCR gives the text on them). |
| WhisperX plus Pyannote diarization | faster-whisper, `vad_filter=True`, timestamped segments (B3) | Replace the engine. Diarization needs a Hugging Face token and gated models; it is deferred (SC-4). |
| Docling and Unstructured for PDFs | pypdf per page; pages under 40 characters go to EasyOCR (A3, B3, E-07) | Delete both. They are heavy, and Docling output carries no page numbers (IN-05). |
| HTML, DOCX and PPTX loaders | Allowed uploads are PDF, image, audio and text only (3.4, 415 UNSUPPORTED_MEDIA_TYPE) | Delete `html.py` and `office.py`. |
| Decides case vs legal from the file name or folder | Case uploads and legal ingestion are different endpoints (3.4) | Delete `resolve_kb_type` (IN-01). |
| Generic `EntityNode` and `RelationalEdge` (`filed_claim_against`, `testified_on`, `contradicts_exhibit`, `occurred_on`) | `ExtractionResult` with Party, Event, Claim, Evidence and the relationships in 2.4 | Rewrite the extraction prompt against 2.5. The vocabulary does not carry over. |
| Regex "heuristic" extractor when no API key | LLM only; a failed batch is retried once, then marked failed (B6) | Delete (IN-06). |
| Writes chunk text to `processed_data/<source>/<file>.json` | Stored files live under `UPLOAD_DIR/{user_id}/{case_id}/{doc_id}/` and are removed with the case (R-05) | Delete (IN-10). |
| `logfire.configure(...)` at import, mock logfire copy-pasted into 8 files, file names in span attributes | One `observability.py` with scrubbing on; ids, lengths and timings only (R-04) | Delete the copies. Import one `span()` helper. |
| Folder scanner, `--wipe`, `true`/`noisy` source types | Jobs through the API (A2, B2); legal CLI is `scripts/ingest_legal.py` | Delete the scanner. Keep the `true`/`noisy` idea only as test fixture folders (section 6). |

## 4. Defects found (run against the code)

Each row was reproduced unless it says "by reading". `IN-xx` ids are for commit messages and tests.

| Id | Defect | Evidence | Fix in the port |
| --- | --- | --- | --- |
| IN-01 | `resolve_kb_type` looks for the substring `act` in the path. `contract_receipt.pdf`, `exact_invoice.pdf`, `fact_sheet.txt` and `impact.pdf` all resolve to **legal**; `DATA/CPA2019.pdf` resolves to **case**. | Ran 7 paths; 5 wrong | Function deleted. The endpoint decides the KB. |
| IN-02 | Chat timestamps are never normalised. The format list expects `"12/01/2026, 14:33"` but the code joins date and time with a space, so every message keeps the raw string. | Output was `[12/01/2026 14:33]`, not ISO | Join with `", "` for the first formats and try the space formats too; test all four WhatsApp and Telegram layouts. |
| IN-03 | The Act-title regex spans whitespace and captures a whole phrase. | Returned `The Central Authority shall act under the Consumer Protection Act, 2019` | Not ported. Act titles come from operator metadata (A1), not from text. |
| IN-04 | The court regex has no pattern for consumer commissions. District, State and National Commission text all return `None`. | 4 of 4 returned `None` | Not ported. `court` is not a `CaseChunk` property in 2.3. |
| IN-05 | A PDF chunk with no `--- Page N ---` marker gets `page = chunk_index + 1`. That is wrong for any chunk after the first on a page, and Docling output never has markers (by reading). | Chunk index 7 returned page 8 | Page is carried by the loader as structure (T3.3, T3.6). It is never guessed from text. |
| IN-06 | The heuristic case extractor invents edges. The condition `\bcontradict\|inconsistent\|dispute\|deny\|denies\b` binds `\b` to the first and last words only, so "denies" anywhere creates `contradicts_exhibit`. A witness plus any date creates `testified_on`. The party regex misses "Complainant" and "Opposite Party", the two roles consumer cases actually use. | `PW-1 contradicts_exhibit Exhibit P-1` produced from text that never links them | Whole extractor deleted. |
| IN-07 | The chunker packs paragraphs only. A 6,000-character paragraph stays one 5,999-character chunk. There is no overlap. A heading can end one chunk while its body starts the next. | Ran all three (overlap checked with distinct paragraphs: the tail of chunk 1 does not reappear in chunk 2) | New splitter per B4: 1,200 target, 1,500 hard maximum, 150 overlap, sentence-level split of long paragraphs (T3.6). |
| IN-08 | The fake embedding fallback returns 1,536 floats. If it ever reached Neo4j it would break the 768-dimension index silently. | `len(embed_texts(['a'])[0])` was 1,536 | No fallback exists (R-08). |
| IN-09 | Dates like `03/04/2026` are read day-first without saying so, and only the first date in a chunk is used. India is day-first, so the default is right, but US-style exports will be wrong. | `03/04/2026` became `2026-04-03` | Infer from the whole file: if any first number exceeds 12 it is day-first, if any second number exceeds 12 it is month-first, else day-first (T3.2). No new config variable. |
| IN-10 | Case text is written to `processed_data/` outside the case's upload folder, so deleting a case (R-05) would leave it behind. | By reading | Not ported. |
| IN-11 | `route_file_to_loader` has no image branch, so PNG, JPEG and TIFF raise "Unsupported file type". Routing is by extension, not file signature. | By reading | Router checks signature and MIME (3.4, E-13) and has an image branch (T3.1). |
| IN-12 | Imports are wrapped in `try/except ImportError` with fallbacks (`app.config` else a local `Settings`). A real broken import silently switches the app to defaults. | By reading | One import path. A missing module fails loudly. |
| IN-13 | `text.py` and `html.py` open files with `errors="ignore"`. Non-UTF-8 Hindi or smart-quote text loses bytes without a warning. | By reading | Decode UTF-8 (with BOM); on failure decode with replacement and add a job warning (assumption A-1). |

## 5. Disposition of every file

| Prototype file | Verdict | Goes to | What is kept |
| --- | --- | --- | --- |
| `processor.py` | **Replace** | `ingestion/jobs.py`, `ingestion/case/case_processor.py`, `ingestion/loaders/router.py` | Only the idea of one function that drives load, chunk, embed, write. The step order is now B1 to B10. |
| `chunking/splitter.py` | **Adapt** (about 20%) | `ingestion/chunking/splitter.py`, `app/prompts/extraction.py` | Paragraph packing. The two system prompts are a starting point for the extraction prompt, rewritten for 2.5 and R-06. |
| `loaders/chat.py` | **Keep, with fixes** | `ingestion/loaders/chat.py` (needs SC-1) | The whole parser: four regexes, system-message filter, multi-line handling, Telegram JSON. Fix IN-02 and IN-09. |
| `loaders/audio.py` | **Adapt** | `ingestion/loaders/audio.py` and `services/media/transcription_service.py` | `format_seconds`, the segment dataclass, merge of neighbouring segments by pause. Engine replaced by faster-whisper. |
| `loaders/pdf.py` | **Adapt** | `ingestion/loaders/pdf.py` and `services/media/ocr_service.py` | The `--- Page N ---` idea becomes a structured per-page result. `_table_to_markdown` is kept for a later enhancement (SC-3). |
| `loaders/text.py` | **Keep** (trivial) | `ingestion/loaders/text.py` | Read file; fix IN-13. |
| `loaders/html.py`, `loaders/office.py` | **Drop** | none | none |
| `requirements.txt` | **Replace** | `requirements.txt` | Pydantic, pypdf, PyMuPDF, Pillow, Logfire. See section 7. |

### 5.1 Proposed spec changes (need the operator's yes before the agent acts)

**Decisions (2026-11-20):** SC-1 and SC-2 approved; `chat.py` added to `06-file-structure.md`, and A-1 to A-3 (section 11) copied into `docs/README.md`. SC-3 stays optional until after the Phase 3 exit. SC-4 stays deferred.

| Id | Change | Why | Cost |
| --- | --- | --- | --- |
| SC-1 | Add `app/ingestion/loaders/chat.py` to `06-file-structure.md`. Chat exports are uploaded as `.txt` (media type `text`), so no new upload type and no new node property. The text loader calls the chat parser when `is_chat_export` is true. | Consumer disputes are mostly WhatsApp threads with the seller. The spec's text loader would index raw export noise (system lines, `<Media omitted>`). | About 340 lines already written; one new file |
| SC-2 | Chat chunking groups whole messages, never splitting inside one. Sender and timestamp stay inside the chunk text as `[2026-01-12 14:33:00] Ravi: ...`. | `CaseChunk` has no speaker or time property (2.3). Keeping them in the text makes them searchable and citable without a schema change. | None |
| SC-3 | Optional, after Phase 3 exit: render PDF tables as Markdown (the prototype's `_table_to_markdown`). | Invoices and receipts are tables; pypdf flattens them. | Needs PyMuPDF `find_tables`; not in the v1 exit criteria |
| SC-4 | Deferred: speaker diarization for audio. | Needs a Hugging Face token and gated models; the spec's `CaseChunk` has no speaker field. | Out of v1 |

If SC-1 is approved, patch `06-file-structure.md` by adding this line under `loaders/`, and add `chat.py` to Phase 3 in section 5 of that file:

```
│   │   │   ├── chat.py                # WhatsApp and Telegram exports inside text uploads (SC-1)
```

## 6. Phase 3 plan (4 days)

Phases 0 to 2 do not use the prototype. Do not start these tasks until the Phase 2 exit criteria pass (the Legal KB must exist for B9). Task order matters: each task is independently testable, and the first seven give a working "upload, chunk, embed, search this document" path before any LLM extraction is added.

### T3.0 Stage the reference (do once, any time before Phase 3)

1. Unzip into `reference/ingestion-prototype/` at the workspace root. Delete every `__pycache__/` and `.pyc` (the zip contains Python 3.11 and 3.12 bytecode).
2. Add `reference/` to `.gitignore` only if you do not want it in version control; otherwise leave it in. Add `norecursedirs = reference` to `pytest.ini` so its imports are never collected.
3. Nothing under `reference/` may be imported by `app/` or `tests/`. It is read-only.

### T3.1 `loaders/router.py` (3.4, E-13, IN-11)

- Input: path, declared MIME. Output: a loader and a media type from `pdf`, `image`, `audio`, `text`.
- Validate with the file signature using `filetype`. Signature and declared MIME must agree, otherwise 415.
- Allowed types and size limits are the table in 3.4. Do not copy the prototype's extension lists.
- Tests: each allowed type; a PNG renamed `.pdf`; a DOCX (rejected); an empty file.

### T3.2 `loaders/text.py` and `loaders/chat.py` (SC-1, SC-2, IN-02, IN-09, IN-13)

- Port `parse_chat_content`, `parse_chat_json`, `is_chat_export`, the four regexes and `SYSTEM_PATTERNS` from `reference/`.
- Fix IN-02 and IN-09 as in section 4.
- Return a list of messages, not one joined string, so the chunker can group whole messages.
- Fixture `tests/fixtures/case/chat_whatsapp_android.txt` with exactly these lines must produce two messages (the media line and the encryption line are dropped, the wrapped line is joined):

```
12/01/2026, 14:32 - Messages and calls are end-to-end encrypted.
12/01/2026, 14:33 - Ravi: Sir my phone screen broke in 2 days
and seller refuses refund
12/01/2026, 14:40 - Seller: No refund policy
13/01/2026, 09:00 - Ravi: <Media omitted>
```

- Expected: `[2026-01-12 14:33:00] Ravi: Sir my phone screen broke in 2 days\nand seller refuses refund` and `[2026-01-12 14:40:00] Seller: No refund policy`.
- Add iOS bracket format, Telegram bracket, Telegram two-line, Telegram JSON, a month-first file (`1/15/26, 2:32 PM`) and a plain note that is not a chat.

### T3.3 `loaders/pdf.py` and `services/media/ocr_service.py` (A3, B3, E-07, IN-05)

- pypdf per page. Result is a list of `PageText(page, text, ocr: bool)`.
- A page with fewer than 40 non-space characters is rendered with PyMuPDF at 200 DPI and sent to EasyOCR (`en`, `hi`).
- Encrypted: fail `PDF_ENCRYPTED`. Corrupt: `PDF_UNREADABLE`. Whole file under 40 characters after OCR: `OCR_EMPTY`. Blank individual pages are skipped.
- Tests: a digital PDF, a scanned PDF, a mixed PDF, an encrypted PDF, a truncated PDF.

### T3.4 `loaders/image.py` (B3)

- EasyOCR on PNG, JPEG or TIFF. Page is 1. Empty OCR output is a failed job with `OCR_EMPTY`.

### T3.5 `loaders/audio.py` and `services/media/transcription_service.py` (B3, E-08, SC-4)

- faster-whisper `medium`, CUDA `float16`, `vad_filter=True`. Segments carry `start_sec` and `end_sec`.
- E-08: out-of-memory retries once on CPU with `int8` and the `small` model; decode failure is `AUDIO_DECODE_FAILED`; store detected language and add a warning if detection probability is under 0.5.
- Port from the prototype: `format_seconds`, the segment dataclass, and a merge of adjacent segments when the gap is under 2 seconds. Remove every `speaker` field and the WhisperX and Pyannote calls.
- Verify the faster-whisper call shape in `smoke_check` before relying on it; the library was not installed when this file was written. Also see the open item on Windows cuDNN and cuBLAS DLLs in `README.md`.

### T3.6 `chunking/splitter.py` (B4, IN-07)

- Input: a list of `(text, page, start_sec, end_sec)` units from a loader. Output: chunks with `order`, `text`, `page`, `start_sec`, `end_sec`.
- Target 1,200 characters, hard maximum 1,500, overlap 150. Split on blank lines first, then on sentence ends for any paragraph over 1,500. A chunk never crosses a page (assumption A-2).
- A heading line (short, ends without a full stop, followed by body text) is attached to the text after it, not left at the end of a chunk.
- Chat: pack whole messages (SC-2). A single message over 1,500 characters is split at sentence ends.
- Audio: pack consecutive segments; the chunk's `start_sec` is the first segment's start and `end_sec` the last segment's end.
- Ids: `chk:{doc_uuid}:{order, 4 digits}` (2.1).
- Tests: 6,000-character paragraph gives chunks of at most 1,500 with 150 overlap; the heading case from IN-07; no chunk contains text from two pages; ids are stable on a second run.

### T3.7 `ingestion/case/case_processor.py` steps B5 and B10 (R-07, R-08, R-16)

- Embed chunk texts with `EmbeddingService` in batches, prefix `search_document: `, then write `CaseChunk` nodes with `case_id` and `user_id` through `neo4j_service`, using `MERGE` on `id` and batched `UNWIND`.
- No file or log contains chunk text (R-04). Spans carry `doc_id`, chunk counts and timings.
- Checkpoint: vector and full-text search over this document works. Run it by hand before continuing.

### T3.8 Extraction, entity resolution and writes: B6, B7, B8 (2.5, R-14, R-16)

- `prompts/extraction.py`: start from the two prototype prompts but target `ExtractionResult` exactly. Case text goes inside a delimited CASE MATERIAL block and the model has no tools (R-06). The model returns `chunk_ids` only from the batch it was given.
- Batches of 6 consecutive chunks, temperature 0, JSON mode, validated against `ExtractionResult`. On a validation failure retry once with the error appended; a second failure marks that batch failed and the job continues (`ready_with_warnings`).
- `entity_resolver.py`: R-14 thresholds (92 auto-merge, 80 to 91 one Groq yes/no). `BEFORE` edges are computed in code from event dates.
- Tests: a batch with an invented chunk id is rejected; two spellings of one party merge; re-running leaves node counts unchanged.

### T3.9 `ingestion/case/legal_linker.py`: B9

- Regex citations (`Section 2`, `section 35(1)`, `Sec. 38`) create `INVOKES` edges only when the section exists in the Legal KB. Do not translate IPC or CrPC references (R-26). Port nothing from the prototype's `Section\s+\d+` regex without testing it on `Section 2(1)(d)`, `S. 38` and `sections 35 and 36`.
- Each `Claim`: embed, take the top 5 `LegalUnit`s with score at least `LINK_MIN_SCORE`, one Groq yes/no with a rationale of at most 300 characters, each yes creates `ALLEGES_VIOLATION_OF`.

### T3.10 `ingestion/jobs.py` (1.2, E-10, E-11)

- `asyncio.Queue` worker, GPU semaphore of 1, `anyio.to_thread.run_sync` for every blocking call.
- Check the case status at every stage boundary and before each write (E-10). On startup, set `running` jobs back to `queued` and increment `attempts`; over 3 fails with `JOB_ATTEMPTS_EXCEEDED` (E-11).

### T3.11 API: `api/cases.py`, `api/documents.py`, `api/jobs.py` (3.4, B1, B2, E-09, R-05)

- sha256 duplicate check inside the case returns the existing document with 200 and `duplicate: true` and creates no job.
- Delete a case: mark `deleting`, cancel jobs, remove nodes and the stored files.

### T3.12 Phase 3 exit (05-roadmap.md)

Run, and show the output of:

- A sample case with one digital PDF, one scanned image, one audio file **and one WhatsApp `.txt`** reaches `ready`.
- Re-running a job leaves node counts unchanged.
- Deleting the case leaves zero nodes with that `case_id` and removes the stored files.
- Each of E-07, E-08, E-09, E-10, E-11 has an automated test.
- A log-scrubbing check shows no chunk text in logs or spans.

## 7. Requirements after the port

Remove: `qdrant-client`, `openai`, `tiktoken`, `docling`, `unstructured[all-docs]`, `python-docx`, `python-pptx`, `beautifulsoup4`, `whisperx`, `pyannote.audio`, `transformers`, `accelerate`, `sentence-transformers`, `torchaudio`.

Keep: `pydantic`, `pydantic-settings`, `pypdf`, `pymupdf`, `pillow`, `logfire`, `numpy`.

Add (per `AGENTS.md` and the Phase 0 list): `fastapi`, `uvicorn`, `neo4j`, `langgraph`, `langchain-groq`, `easyocr`, `faster-whisper`, `flashrank`, `rapidfuzz`, `tenacity`, `slowapi`, `filetype`, `pytest`, `pytest-asyncio`. `torch` is installed first as the CUDA build and is **not** listed in `requirements.txt`. Let `pip` resolve versions; do not copy the lower bounds from the prototype.

## 8. Test fixtures

The prototype treats sub-folders named `true` and `noisy` as source types. Keep that as fixture layout, not as pipeline logic:

```
tests/fixtures/case/
├── clean/     # digital invoice PDF, clean chat export, clear audio
└── noisy/     # scanned receipt photo, skewed scan, chat with media and system lines, audio with background noise
```

Use `noisy/` to confirm OCR fallback (E-07), chat filtering and `ready_with_warnings`. Do not commit real people's chats or receipts; anonymise names, phone numbers and order ids first.

## 9. Prompts to paste into Antigravity

**Stage the reference (T3.0)**

> Read AGENTS.md and docs/07-ingestion-integration.md. Do only T3.0: unzip the ingestion prototype I attached into reference/ingestion-prototype/, delete all __pycache__ and .pyc files, add `norecursedirs = reference` to pytest.ini. Do not edit any other file. List what you changed.

**Start Phase 3** (only after Phase 2 exit passes)

> Read AGENTS.md, docs/07-ingestion-integration.md, docs/06-file-structure.md and docs/05-roadmap.md. Implement T3.1 only. Before writing code, list the spec sections, rules and defect ids (IN-xx) it covers and the files you will create. Port from reference/ingestion-prototype/ only what section 5 of the integration doc allows. Run the tests for T3.1 and show the output. Stop and wait before T3.2.

Repeat with the next task id. If the agent proposes adding a file that is not in `06-file-structure.md`, it should stop and ask; SC-1 is the only approved-pending exception.

## 10. Not verified

- Behaviour of Docling, Unstructured, WhisperX, Pyannote and PyMuPDF inside the prototype (not installed here; findings marked "by reading").
- The exact faster-whisper and EasyOCR call signatures on the operator's Windows machine (T3.3 to T3.5 must be proved by `smoke_check` and the tests).
- Whether the prototype's chat regexes cover every export your teammates produce. Add a failing fixture for each new layout before changing the regexes.

## 11. Assumptions to record in `docs/README.md`

- **A-1 (2.3, IN-13).** Text uploads decode as UTF-8 with BOM; on failure they decode with replacement characters and the job gets a warning.
- **A-2 (B4).** A chunk never spans two pages. Short pages produce short chunks rather than merged ones.
- **A-3 (SC-1, SC-2).** Chat exports are a kind of `text` upload; sender and timestamp stay inside the chunk text.
