"""Ingest a statute PDF into the Legal KB (1.4 A, Phase 2).

Run:
  python -m scripts.ingest_legal data/legal/CPA2019.pdf --act-id cpa-2019
  python -m scripts.ingest_legal "data/legal/Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National Commission) Rules, 2021.pdf" --act-id cp-jurisdiction-rules-2021
  python -m scripts.ingest_legal "data/legal/E commerce rules.pdf" --act-id cp-e-commerce-rules-2020
  python -m scripts.ingest_legal "data/legal/2021 May 17.pdf" --act-id cp-e-commerce-amendment-rules-2021

Known act ids have preset metadata: cpa-2019 from 05-roadmap.md Phase 2, and the
Jurisdiction Rules from the Gazette page itself (G.S.R. 912(E), 30 December 2021) with the
PRESCRIBES_FOR mapping of SC-6. Any flag overrides the preset. Unknown act ids need every
required flag. The run creates a Job node like the admin endpoint does (A2).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import uuid

from app.config import get_settings
from app.ingestion.legal.legal_processor import LegalProcessor, create_legal_job, run_legal_job
from app.schemas.api import LegalIngestMetadata
from app.services.media.ocr_service import OcrService
from app.services.retrieval import graph_queries
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.neo4j_service import Neo4jService

PRESETS: dict[str, dict] = {
    "cpa-2019": {
        "act_id": "cpa-2019",
        "title": "The Consumer Protection Act, 2019",
        "short_title": "Consumer Protection Act, 2019",
        "year": 2019,
        "number": "35 of 2019",
        "jurisdiction": "IN",
        "as_on_date": "2026-09-01",
        "kind": "act",
        "expected_sections": 107,
    },
    "cp-jurisdiction-rules-2021": {
        "act_id": "cp-jurisdiction-rules-2021",
        "title": "The Consumer Protection (Jurisdiction of the District Commission, the State "
                 "Commission and the National Commission) Rules, 2021",
        "short_title": "Consumer Protection (Jurisdiction of the District Commission, the State "
                       "Commission and the National Commission) Rules, 2021",
        "year": 2021,
        "number": "G.S.R. 912(E)",
        "jurisdiction": "IN",
        "as_on_date": "2021-12-30",
        "kind": "rule",
        "issued_under": {"act_id": "cpa-2019", "sections": ["34", "47", "58"]},
        "prescribes_for": [
            {"rule": "3", "section": "34", "note": "Pecuniary jurisdiction of the District Commission"},
            {"rule": "4", "section": "47", "note": "Pecuniary jurisdiction of the State Commission"},
            {"rule": "5", "section": "58", "note": "Pecuniary jurisdiction of the National Commission"},
        ],
        "expected_sections": 5,
    },
    # D-4: the Gazette original (G.S.R. 462(E), 23 July 2020), file data/legal/E commerce
    # rules.pdf. It predates the 17-5-2021 substitution of rule 4(1); as_on_date says so.
    "cp-e-commerce-rules-2020": {
        "act_id": "cp-e-commerce-rules-2020",
        "title": "The Consumer Protection (E-Commerce) Rules, 2020",
        "short_title": "Consumer Protection (E-Commerce) Rules, 2020",
        "year": 2020,
        "number": "G.S.R. 462(E)",
        "jurisdiction": "IN",
        "as_on_date": "2020-07-23",
        "kind": "rule",
        "issued_under": {"act_id": "cpa-2019", "sections": ["101"]},
        "expected_sections": 8,
    },
    # SC-9: G.S.R. 328(E), 17 May 2021, file data/legal/2021 May 17.pdf. In force on
    # publication (its rule 1(2)), so effective_on defaults to as_on_date.
    "cp-e-commerce-amendment-rules-2021": {
        "act_id": "cp-e-commerce-amendment-rules-2021",
        "title": "The Consumer Protection (E-Commerce) (Amendment) Rules, 2021",
        "short_title": "Consumer Protection (E-Commerce) (Amendment) Rules, 2021",
        "year": 2021,
        "number": "G.S.R. 328(E)",
        "jurisdiction": "IN",
        "as_on_date": "2021-05-17",
        "kind": "rule",
        "issued_under": {"act_id": "cpa-2019", "sections": ["101"]},
        "amends": {"act_id": "cp-e-commerce-rules-2020"},
        "expected_sections": 2,
    },
}


def build_metadata(args: argparse.Namespace) -> LegalIngestMetadata:
    data = dict(PRESETS.get(args.act_id, {"act_id": args.act_id}))
    overrides = {
        "title": args.title, "short_title": args.short_title, "year": args.year,
        "number": args.number, "jurisdiction": args.jurisdiction, "as_on_date": args.as_on_date,
        "kind": args.kind, "expected_sections": args.expected_sections,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    if args.metadata:
        # A path to a .json file (data/legal/*.metadata.json, SETUP.md) or an inline JSON object.
        path = Path(args.metadata)
        raw = path.read_text(encoding="utf-8") if path.suffix == ".json" and path.is_file() else args.metadata
        data.update(json.loads(raw))
    data["replace"] = args.replace
    return LegalIngestMetadata.model_validate(data)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    meta = build_metadata(args)
    neo4j = Neo4jService(
        settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD.get_secret_value(),
        args.database or settings.NEO4J_DATABASE,
    )
    embeddings = EmbeddingService(settings.OLLAMA_URL, settings.EMBED_MODEL, settings.EMBED_DIM)
    processor = LegalProcessor(neo4j, embeddings, OcrService(settings.OCR_LANGS))
    job_uuid = str(uuid.uuid4())
    try:
        await embeddings.verify_meta(neo4j)  # 2.2: never embed against a mismatched database
        await create_legal_job(neo4j, job_uuid, settings.DEFAULT_USER_ID)
        report = await run_legal_job(processor, job_uuid, args.pdf, meta)
        job = (await neo4j.read(graph_queries.JOB_STATUS, {"id": f"job:{job_uuid}"}))[0]
    finally:
        neo4j.close()
        await embeddings.aclose()

    print(f"job={job_uuid} act_id={meta.act_id} status={job['status']}")
    if report is None:
        print(f"error_code={job['code']}\n{job['msg']}")
        return 1
    print(f"parse: {report.parse_report}")
    print(f"units embedded: {report.units}  ocr pages: {report.ocr_pages}")
    print(f"graph counts: {report.counts}")
    print(f"timings ms: {report.timings_ms}")
    print(f"warnings: {len(report.warnings)}")
    for w in report.warnings[: args.show_warnings]:
        print(f"  - {w}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest a statute PDF into the Legal KB.")
    p.add_argument("pdf")
    p.add_argument("--act-id", required=True)
    p.add_argument("--title")
    p.add_argument("--short-title")
    p.add_argument("--year", type=int)
    p.add_argument("--number")
    p.add_argument("--jurisdiction")
    p.add_argument("--as-on-date")
    p.add_argument("--kind", choices=["act", "rule", "regulation"])
    p.add_argument("--expected-sections", type=int)
    p.add_argument("--metadata", help="JSON object, or a path to a .json file, merged over the preset")
    p.add_argument("--replace", action="store_true", help="Delete nodes no longer in the parse")
    p.add_argument("--database", help="Target database (default: NEO4J_DATABASE)")
    p.add_argument("--show-warnings", type=int, default=20)
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
