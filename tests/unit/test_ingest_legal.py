"""scripts/ingest_legal.py: --metadata accepts inline JSON or a data/legal/*.metadata.json path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.ingest_legal import build_metadata

LEGAL = Path(__file__).resolve().parents[2] / "data" / "legal"


def _args(**kw) -> argparse.Namespace:
    base = dict(act_id="cp-jurisdiction-rules-2021", title=None, short_title=None, year=None, number=None,
                jurisdiction=None, as_on_date=None, kind=None, expected_sections=None, metadata=None, replace=False)
    return argparse.Namespace(**{**base, **kw})


@pytest.mark.parametrize("path", sorted(LEGAL.glob("*.metadata.json")), ids=lambda p: p.stem[:30])
def test_every_saved_metadata_file_is_valid(path: Path) -> None:
    meta = build_metadata(_args(act_id=json.loads(path.read_text(encoding="utf-8"))["act_id"], metadata=str(path)))
    assert meta.act_id == json.loads(path.read_text(encoding="utf-8"))["act_id"]


def test_file_and_inline_json_give_the_same_metadata() -> None:
    path = LEGAL / ("Consumer Protection (Jurisdiction of the District Commission, the State Commission "
                    "and the National Commission) Rules, 2021.metadata.json")
    from_file = build_metadata(_args(metadata=str(path)))
    inline = build_metadata(_args(metadata=path.read_text(encoding="utf-8")))
    assert from_file == inline
    assert [(p.rule, p.section) for p in from_file.prescribes_for] == [("3", "34"), ("4", "47"), ("5", "58")]
