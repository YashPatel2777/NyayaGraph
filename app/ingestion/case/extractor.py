"""Case entity extraction (B6, 2.5).

- Batches of 6 consecutive chunks go to Groq at temperature 0, JSON mode, at most 4,000
  output tokens (4.2, R-19 as changed by SC-11), validated against ExtractionResult. GroqClient retries once with
  the validation error appended (R-22); a second failure marks the batch failed with a job
  warning and extraction continues (E-02). An exhausted Groq retry budget makes the stage
  wait and resume (E-01, A-28).
- R-06: chunk text goes inside the CASE MATERIAL block; any delimiter string inside it is
  replaced first, so case text cannot close the block.
- R-15: chunks are shown as labels C1 to C6 (A-29) and mapped back in code; a label that
  is not in the batch is removed, and an item left with none is discarded and counted in
  the job warnings. ExtParty has no chunk_ids in 2.5, so a party's chunks are the
  batch chunks whose normalised text contains its normalised name or an alias (A-25); a party
  found in none is discarded the same way.
- R-17: a date that is not ISO 8601 (partial allowed) is stored as null with a warning.
- SC-12: CachedReplies keeps every validated Groq reply of a document in
  UPLOAD_DIR/{user_id}/{case_id}/{doc_id}/llm_cache.json, so a re-run or an E-11 resume
  replays the same replies and leaves node counts unchanged. The file is deleted with the
  case (R-05) and never logged.
- R-04: logs carry counts and timings only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import anyio
from pydantic import BaseModel, ValidationError

from app.errors import AppError
from app.ingestion.case.entity_resolver import (
    BatchEntities,
    ClaimMention,
    EventMention,
    EvidenceMention,
    PartyMention,
    iso_date,
    normalise_name,
)
from app.prompts.extraction import (
    CASE_MATERIAL_END,
    CASE_MATERIAL_START,
    CHUNK_BLOCK,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
)
from app.schemas.extraction import ExtractionResult
from app.services.llm.groq_client import GroqClient, LLMResult, Message

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

CACHE_FILE = "llm_cache.json"
CACHE_VERSION = 1


class CachedReplies:
    """SC-12: a per-document cache of validated Groq replies in front of GroqClient.

    The key is sha256 of the model, schema, temperature, token cap and messages, so a change
    to the chunks or a prompt is a miss. Each new reply is written at once (atomic replace),
    so a crash loses nothing already paid for. prune() keeps only the entries this run used.
    It offers the same complete_json_resuming call as GroqClient.
    """

    def __init__(self, llm: GroqClient, path: Path, entries: dict[str, Any]) -> None:
        self.llm = llm
        self.path = path
        self._entries = entries
        self._used: set[str] = set()
        self.hits = self.misses = 0

    @classmethod
    async def open(cls, llm: GroqClient, folder: Path) -> CachedReplies:
        path = folder / CACHE_FILE
        entries: dict[str, Any] = {}
        try:
            raw = await anyio.to_thread.run_sync(path.read_text, "utf-8")
            data = json.loads(raw)
            if data.get("version") == CACHE_VERSION and isinstance(data.get("entries"), dict):
                entries = data["entries"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError):
            logger.warning("llm_cache_unreadable folder_exists=%s", folder.exists())
        return cls(llm, path, entries)

    def key(self, messages: Sequence[Message], schema: type[BaseModel], temperature: float, max_tokens: int) -> str:
        blob = json.dumps(
            [self.llm.model, schema.__name__, temperature, max_tokens, [list(m) for m in messages]],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    async def complete_json_resuming(
        self,
        messages: Sequence[Message],
        schema: type[M],
        *,
        temperature: float,
        max_tokens: int,
        checkpoint: Callable[[], Awaitable[None]],
    ) -> LLMResult[M]:
        key = self.key(messages, schema, temperature, max_tokens)
        self._used.add(key)
        if key in self._entries:
            try:
                value = schema.model_validate(self._entries[key])
            except ValidationError:
                del self._entries[key]  # a schema change since it was stored
            else:
                self.hits += 1
                return LLMResult(value, 0, 0, 0)
        result = await self.llm.complete_json_resuming(
            messages, schema, temperature=temperature, max_tokens=max_tokens, checkpoint=checkpoint
        )
        self.misses += 1
        self._entries[key] = result.value.model_dump(mode="json")
        await self._save()
        return result

    async def prune(self) -> None:
        """Drop entries this run did not use (old chunks or prompts)."""
        stale = set(self._entries) - self._used
        if stale:
            for key in stale:
                del self._entries[key]
            await self._save()

    async def _save(self) -> None:
        data = json.dumps({"version": CACHE_VERSION, "entries": self._entries}, ensure_ascii=False)
        tmp = self.path.with_suffix(".tmp")

        def write() -> None:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, self.path)

        await anyio.to_thread.run_sync(write)

EXTRACTION_BATCH = 6  # 4.2
EXTRACTION_TEMPERATURE = 0.0  # R-19
EXTRACTION_MAX_TOKENS = 4000  # R-19 as changed by SC-11

_DELIMITERS = (CASE_MATERIAL_START, CASE_MATERIAL_END, "<<<PROVISIONS>>>", "<<<END PROVISIONS>>>")


@dataclass(frozen=True)
class ChunkText:
    id: str
    text: str


@dataclass
class ExtractionOutput:
    batches: list[BatchEntities]
    warnings: list[str] = field(default_factory=list)
    failed_batches: int = 0
    llm_calls: int = 0


def neutralise(text: str) -> str:
    """Remove block delimiters from case text (R-06)."""
    for marker in _DELIMITERS:
        text = text.replace(marker, "[marker removed]")
    return text


def batch_labels(chunks: list[ChunkText]) -> dict[str, str]:
    """A-29: the label shown to the model (C1, C2, ...) -> chunk id."""
    return {f"C{i + 1}": c.id for i, c in enumerate(chunks)}


def case_material(chunks: list[ChunkText]) -> str:
    body = "\n\n".join(
        CHUNK_BLOCK.format(label=label, text=neutralise(c.text))
        for label, c in zip(batch_labels(chunks), chunks)
    )
    return f"{CASE_MATERIAL_START}\n{body}\n{CASE_MATERIAL_END}"


def batches_of(chunks: list[ChunkText], size: int = EXTRACTION_BATCH) -> list[list[ChunkText]]:
    return [chunks[i : i + size] for i in range(0, len(chunks), size)]


@dataclass
class _Counts:
    discarded: int = 0
    bad_dates: int = 0


def _valid_ids(labels: list[str], allowed: dict[str, str]) -> list[str]:
    """Map the batch labels the model cited to chunk ids; anything else is dropped (R-15)."""
    return list(dict.fromkeys(allowed[x.strip()] for x in labels if x.strip() in allowed))


def _party_chunks(names: list[str], chunks: list[ChunkText]) -> list[str]:
    wanted = [f" {n} " for n in (normalise_name(x) for x in names) if n]
    found = []
    for chunk in chunks:
        text = f" {normalise_name(chunk.text)} "
        if any(w in text for w in wanted):
            found.append(chunk.id)
    return found


def clean_batch(chunks: list[ChunkText], result: ExtractionResult, counts: _Counts) -> BatchEntities:
    """Apply R-15 and R-17 to one validated batch."""
    allowed = batch_labels(chunks)
    out = BatchEntities()
    for p in result.parties:
        names = [p.name, *p.aliases]
        found = _party_chunks(names, chunks)
        if not found:
            counts.discarded += 1
            continue
        aliases = [a for a in p.aliases if a.strip() and a.strip() != p.name.strip()]
        out.parties.append(PartyMention(p.name.strip(), p.kind, p.role, aliases, found))
    for e in result.events:
        ids = _valid_ids(e.chunk_ids, allowed)
        if not ids or not e.description.strip():
            counts.discarded += 1
            continue
        when, ok = iso_date(e.date)
        if not ok:
            counts.bad_dates += 1
        out.events.append(
            EventMention(e.description.strip(), when, e.location, list(e.participants), ids)
        )
    for c in result.claims:
        ids = _valid_ids(c.chunk_ids, allowed)
        if not ids or not c.text.strip():
            counts.discarded += 1
            out.claims.append(None)  # keep indexes for evidence.supports
            continue
        out.claims.append(ClaimMention(c.text.strip(), c.claim_type, c.asserted_by, ids))
    for v in result.evidence:
        ids = _valid_ids(v.chunk_ids, allowed)
        if not ids or not v.description.strip():
            counts.discarded += 1
            continue
        out.evidence.append(
            EvidenceMention(v.kind, v.description.strip(), list(v.supports), list(v.contradicts), ids)
        )
    return out


async def extract_document(
    llm: GroqClient | CachedReplies,
    chunks: list[ChunkText],
    checkpoint: Callable[[], Awaitable[None]],
) -> ExtractionOutput:
    """B6 for one document's chunks, in order."""
    output = ExtractionOutput(batches=[])
    counts = _Counts()
    groups = batches_of(chunks)
    for index, group in enumerate(groups):
        await checkpoint()  # E-10 at every batch boundary
        messages = [
            ("system", EXTRACTION_SYSTEM),
            ("user", EXTRACTION_USER.format(material=case_material(group))),
        ]
        try:
            reply = await llm.complete_json_resuming(
                messages,
                ExtractionResult,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=EXTRACTION_MAX_TOKENS,
                checkpoint=checkpoint,
            )
        except AppError as exc:
            if exc.details.get("reason") != "invalid_structured_output":
                raise
            output.failed_batches += 1  # E-02: the job continues
            output.llm_calls += 2
            logger.warning("extraction_batch_failed batch=%d chunks=%d", index, len(group))
            continue
        output.llm_calls += reply.llm_calls
        output.batches.append(clean_batch(group, reply.value, counts))

    if output.failed_batches:
        output.warnings.append(
            f"extraction: {output.failed_batches} of {len(groups)} batch(es) failed validation "
            "and were skipped (E-02)"
        )
    if counts.discarded:
        output.warnings.append(
            f"extraction: {counts.discarded} item(s) without a valid chunk id were discarded (R-15)"
        )
    if counts.bad_dates:
        output.warnings.append(
            f"extraction: {counts.bad_dates} date(s) were not ISO 8601 and were stored as null (R-17)"
        )
    return output
