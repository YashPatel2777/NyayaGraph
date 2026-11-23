---
trigger: glob
globs: app/ingestion/legal/**/*.py, app/ingestion/chunking/legal_splitter.py, tests/golden/**/*.py
description: Rules for parsing IndiaCode statute PDFs into the Legal KB graph
---

# Statute parsing rules

- Implement parsing rules P-01 to P-12 from `docs/04-rules-and-edge-cases.md` section 4.3, in that order. Cite the rule id in a comment at each implementation point.
- Keep page boundaries until footnote handling (P-05) has run.
- Do not correct typos in the source text (for example the Chapter IV title). Key chapters by roman numeral.
- A reference followed by "of <Other Act>" must create `REFERS_TO_ACT` to an external stub Act and no `REFERS_TO` edge (P-10).
- The Statement of Objects and Reasons is stored as `Act.objects_text` and is never a LegalUnit or embedded (P-03).
- Any gap or duplicate in section numbering fails the job with `PARSE_VALIDATION_FAILED` and writes nothing (P-12, E-14).
- The golden tests G1 to G9 in `docs/05-roadmap.md` (Phase 2) are the acceptance test. Run them against a throwaway database and show the output.
