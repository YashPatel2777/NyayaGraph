"""Entity resolution (R-14, R-17, B8 BEFORE, A-24; 07 T3.8)."""

from __future__ import annotations

from app.ingestion.case.entity_resolver import (
    BatchEntities,
    ClaimMention,
    EntityResolver,
    EventMention,
    EvidenceMention,
    PartyMention,
    PartyRecord,
    before_edges,
    iso_date,
    normalise_name,
)
from app.schemas.graph_schema import entity_id

CASE = "3f2a1b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b"


def party(name: str, kind: str = "organisation", role: str = "opposite_party", chunks=("c1",), aliases=()):
    return PartyMention(name, kind, role, list(aliases), list(chunks))


class Judge:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, existing: PartyRecord, mention: PartyMention) -> bool:
        self.calls.append((existing.name, mention.name))
        return self.answer


def test_normalise_name() -> None:
    assert normalise_name("M/s. QuickKart Retail Pvt. Ltd.") == "quickkart retail pvt ltd"
    assert normalise_name("  Mr.  Ravi   MEHTA ") == "ravi mehta"
    assert normalise_name("Dr Asha Rao, Ms. Rao") == "asha rao rao"
    assert normalise_name("Mrsa Store") == "mrsa store"  # only whole honorifics go
    assert normalise_name("रवि मेहता") == "रवि मेहता"


def test_iso_dates_r17() -> None:
    assert iso_date("2026-01-12") == ("2026-01-12", True)
    assert iso_date("2026-01") == ("2026-01", True)
    assert iso_date("2026") == ("2026", True)
    assert iso_date(None) == (None, True)
    for bad in ("12/01/2026", "2026-02-30", "2026-13", "January 2026", "26-01-12"):
        assert iso_date(bad) == (None, False), bad


def test_before_edges_chain_and_partial_dates() -> None:
    edges = before_edges([
        ("buy", "2026-01-02"), ("deliver", "2026-01-04"), ("same-day", "2026-01-04"),
        ("month", "2026-01"), ("defect", "2026-01-06"), ("feb", "2026-02"),
        ("undated", None), ("bad", "06/01/2026"),
    ])
    assert ("buy", "deliver") in edges and ("buy", "same-day") in edges
    assert ("deliver", "defect") in edges and ("same-day", "defect") in edges
    assert ("defect", "feb") in edges
    assert ("buy", "defect") not in edges  # a chain, not every pair
    # 2026-01 overlaps every January date, so it is ordered only against February.
    assert [e for e in edges if "month" in e] == [("month", "feb")]
    assert not any("undated" in e or "bad" in e for e in edges)
    assert ("deliver", "same-day") not in edges


async def test_two_spellings_of_one_party_merge_automatically() -> None:
    judge = Judge(False)
    resolver = EntityResolver(CASE, [], judge)
    a = await resolver.add_party(party("QuickKart Retail Pvt. Ltd.", chunks=["c1"]))
    b = await resolver.add_party(party("M/s Quickkart Retail Pvt Ltd", chunks=["c2"]))
    c = await resolver.add_party(party("QuickKart Retails Pvt Ltd", chunks=["c3"]))  # 98
    assert a == b == c
    assert judge.calls == []
    merged = resolver.parties[a]
    assert merged.name == "QuickKart Retail Pvt. Ltd."
    assert merged.aliases == ["QuickKart Retails Pvt Ltd"]  # normalised duplicates not repeated
    assert merged.chunk_ids == {"c1", "c2", "c3"}  # every MENTIONS kept
    assert a == entity_id("pty", CASE, "organisation quickkart retail pvt ltd")


async def test_80_to_91_asks_once_and_follows_the_answer() -> None:
    yes = Judge(True)
    resolver = EntityResolver(CASE, [], yes)
    a = await resolver.add_party(party("QuickKart Retail Private Limited"))
    b = await resolver.add_party(party("QuickKart Retail Pvt Ltd", chunks=["c2"]))  # 85.7
    c = await resolver.add_party(party("QuickKart Retail Pvt Ltd", chunks=["c3"]))
    assert a == b == c
    assert yes.calls == [("QuickKart Retail Private Limited", "QuickKart Retail Pvt Ltd")]

    no = Judge(False)
    resolver = EntityResolver(CASE, [], no)
    ravi = await resolver.add_party(party("Ravi Mehta", kind="person", role="complainant"))
    rani = await resolver.add_party(party("Rani Mehta", kind="person", role="witness"))  # 90
    assert ravi != rani
    assert len(no.calls) == 1
    assert resolver.adjudications == 1


async def test_below_80_and_different_kinds_stay_separate() -> None:
    judge = Judge(True)
    resolver = EntityResolver(CASE, [], judge)
    a = await resolver.add_party(party("Ravi Mehta", kind="person"))
    b = await resolver.add_party(party("Ravi Kumar Mehta", kind="person"))  # 76.9
    c = await resolver.add_party(party("Ravi Mehta", kind="organisation"))
    assert len({a, b, c}) == 3
    assert judge.calls == []


async def test_existing_case_parties_take_part_and_role_is_upgraded() -> None:
    existing = PartyRecord("pty:x:1", "Zenith Mobiles", "organisation", "other", ["Zenith"])
    resolver = EntityResolver(CASE, [existing], Judge(False))
    pid = await resolver.add_party(party("ZENITH MOBILES", role="opposite_party", chunks=["c9"]))
    assert pid == "pty:x:1"
    assert existing.role == "opposite_party"
    assert existing.chunk_ids == {"c9"}


async def test_resolve_links_participants_claims_and_evidence() -> None:
    resolver = EntityResolver(CASE, [], Judge(False))
    batch = BatchEntities(
        parties=[party("Ravi Mehta", kind="person", role="complainant", chunks=["c1"]),
                 party("QuickKart Retail Pvt Ltd", chunks=["c1"])],
        events=[EventMention("Phone delivered", "2026-01-04", None, ["Mr. Ravi Mehta", "Unknown Courier"], ["c1"])],
        claims=[None, ClaimMention("Screen defective within 2 days", "defect", "Ravi Mehta", ["c2"])],
        evidence=[EvidenceMention("message", "Seller refuses refund", [1, 0, 7], [], ["c2"])],
    )
    batch2 = BatchEntities(
        events=[EventMention("phone   DELIVERED", "2026-01-04", None, [], ["c7"])],  # same event
    )
    out = await resolver.resolve([batch, batch2])
    ravi = entity_id("pty", CASE, "person ravi mehta")
    [event] = out.events.values()
    assert event.chunk_ids == {"c1", "c7"}  # A-24: identical normalised text and date merge
    assert out.participated == {(ravi, event.id)}
    [claim] = out.claims.values()
    assert claim.props["asserted_by"] == ravi
    [evd] = out.evidence.values()
    assert out.supports == {(evd.id, claim.id)}
    # 'Unknown Courier', the discarded claim 0 and index 7 are counted, not guessed.
    assert out.warnings == [
        "extraction: 3 participant, claimant or claim reference(s) did not match an "
        "extracted item and were dropped"
    ]


async def test_ids_are_stable_across_runs() -> None:
    async def run():
        resolver = EntityResolver(CASE, [], Judge(False))
        batch = BatchEntities(
            parties=[party("QuickKart Retail Pvt Ltd")],
            claims=[ClaimMention("No refund given", "deficiency", None, ["c1"])],
        )
        out = await resolver.resolve([batch])
        return sorted(out.parties), sorted(out.claims)

    assert await run() == await run()


# SC-13: organisation names compared again without company suffixes


def test_core_name() -> None:
    from app.ingestion.case.entity_resolver import core_name

    assert core_name("quickkart retail private limited") == "quickkart"
    assert core_name("zenith mobiles pvt ltd") == "zenith mobiles"
    assert core_name("private limited") == "private limited"  # nothing left: kept whole
    assert core_name("ravi mehta") == "ravi mehta"


async def test_short_and_full_company_names_go_to_the_check_not_auto_merge() -> None:
    yes = Judge(True)
    resolver = EntityResolver(CASE, [], yes)
    full = await resolver.add_party(party("QuickKart Retail Private Limited"))
    short = await resolver.add_party(party("QuickKart", chunks=["c2"]))  # 43.9 as written
    spelt = await resolver.add_party(party("Quickcart", chunks=["c3"]))  # 88.9 on the core
    assert full == short == spelt
    assert yes.calls == [("QuickKart Retail Private Limited", "QuickKart"),
                         ("QuickKart Retail Private Limited", "Quickcart")]
    assert resolver.parties[full].aliases == ["QuickKart", "Quickcart"]

    no = Judge(False)
    resolver = EntityResolver(CASE, [], no)
    a = await resolver.add_party(party("QuickKart Retail Private Limited"))
    b = await resolver.add_party(party("QuickKart"))
    assert a != b and len(no.calls) == 1  # the model said no: they stay separate


async def test_suffix_rule_is_only_for_organisations_and_unrelated_names() -> None:
    judge = Judge(True)
    resolver = EntityResolver(CASE, [], judge)
    await resolver.add_party(party("Ravi Mehta Ltd", kind="person"))
    await resolver.add_party(party("Ravi", kind="person"))
    await resolver.add_party(party("QuickKart Retail Pvt Ltd"))
    await resolver.add_party(party("Zenith Service Centre Pvt Ltd"))  # different core
    assert judge.calls == []
    assert len(resolver.parties) == 4
