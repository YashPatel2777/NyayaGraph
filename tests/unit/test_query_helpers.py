"""Pure helpers of the query agent: citations (3.5), R-11 context budget, part merging
(A-35), R-27 pinning, R-13 definitions, R-24 confidence, excerpts, history (A-34)."""

from __future__ import annotations

from datetime import date

from app.agents.nodes.graph_expander import ConceptIndex, truncate_words
from app.agents.nodes.legal_reranker import merge_parts, pin_prescribed_rules
from app.agents.nodes.responder import fit_context
from app.agents.nodes.verifier import confidence, excerpt, fact_supported, replace_labels
from app.agents.state import history_block, neutralise
from app.schemas.graph_schema import format_citation, provision_base_id
from app.schemas.output import Fact, Provision, SourceRef


def test_citations() -> None:
    cpa = "Consumer Protection Act, 2019"
    assert format_citation("38", None, cpa, "act") == f"Section 38, {cpa}"
    assert format_citation("38", "2", cpa, "act") == f"Section 38(2), {cpa}"
    assert format_citation("2", "47-p1", cpa, "act") == f"Section 2(47), {cpa}"
    assert format_citation("4", "1", "E-Commerce Rules", "rule") == "Rule 4(1), E-Commerce Rules"


def test_provision_base_id() -> None:
    assert provision_base_id("cl:cpa-2019:2:47-p2") == "cl:cpa-2019:2:47"
    assert provision_base_id("sec:cpa-2019:5-p1") == "sec:cpa-2019:5"
    assert provision_base_id("cl:cpa-2019:38:2") == "cl:cpa-2019:38:2"


def test_fit_context_drops_lowest_score_whole_items() -> None:
    chunks = [{"id": "c1", "text": "a" * 6000, "score": 0.9}, {"id": "c2", "text": "b" * 6000, "score": 0.1}]
    provs = [{"id": "p1", "text": "c" * 6000, "score": 0.8}, {"id": "p2", "text": "d" * 6000, "score": 0.5}]
    kept_c, kept_p = fit_context(chunks, provs, limit=18000)
    assert [c["id"] for c in kept_c] == ["c1"] and [p["id"] for p in kept_p] == ["p1", "p2"]
    assert all(len(x["text"]) == 6000 for x in kept_c + kept_p)  # never cut in the middle


def test_fit_context_keeps_everything_under_the_limit() -> None:
    chunks = [{"id": "c1", "text": "x", "score": 0.0}]
    assert fit_context(chunks, [], limit=10) == (chunks, [])


def _item(uid: str, base: str, text: str, score: float) -> dict:
    return {"id": uid, "base_id": base, "text": text, "rerank_score": score}


def test_merge_parts_joins_in_part_order_under_base_id() -> None:
    ranked = [_item("cl:a:2:4-p2", "cl:a:2:4", "second", 0.9), _item("sec:a:9", "sec:a:9", "nine", 0.5),
              _item("cl:a:2:4-p1", "cl:a:2:4", "first", 0.2)]
    merged = merge_parts(ranked)
    assert [m["id"] for m in merged] == ["cl:a:2:4", "sec:a:9"]
    assert merged[0]["text"] == "first\nsecond" and merged[0]["rerank_score"] == 0.9
    assert merged[0]["unit_ids"] == ["cl:a:2:4-p1", "cl:a:2:4-p2"]


def test_pin_prescribed_rule_replaces_lowest_unpinned() -> None:
    pool = [{"id": f"x{i}"} for i in range(5)] + [{"id": "rule3"}]
    selected = [{"id": "sec34", "prescribed_by": ["rule3"]}] + pool[:3]
    out = pin_prescribed_rules(selected, pool + selected, keep=4)
    assert [o["id"] for o in out] == ["sec34", "x0", "x1", "rule3"]


def test_pin_ignores_rules_not_in_pool() -> None:
    selected = [{"id": "sec34", "prescribed_by": ["missing"]}]
    assert pin_prescribed_rules(selected, selected, keep=8) == selected


def test_concept_index_whole_words_in_rank_order() -> None:
    index = ConceptIndex([
        {"id": "con:consumer", "name": "consumer", "clause_id": "cl:a:2:7"},
        {"id": "con:goods", "name": "goods", "clause_id": "cl:a:2:21"},
        {"id": "con:central-authority", "name": "Central Authority", "clause_id": "cl:a:2:5"},
    ])
    texts = ["the consumers bought goods", "notice by the central authority to a consumer"]
    assert index.definitions_for(texts, exclude=set()) == ["cl:a:2:21", "cl:a:2:5", "cl:a:2:7"]
    assert index.definitions_for(texts, exclude={"cl:a:2:21"}, limit=1) == ["cl:a:2:5"]


def test_truncate_and_excerpt_cut_at_word_boundaries() -> None:
    assert truncate_words("aaa bbb ccc", 7) == "aaa bbb"
    assert excerpt("word " * 200).endswith("word") and len(excerpt("word " * 200)) <= 600
    assert excerpt("  short\n text ") == "short text"


def _prov(relation: str) -> Provision:
    return Provision(provision_id="sec:a:1", act="A", act_as_on=date(2026, 9, 1), citation="Section 1, A",
                     status="in_force", relation=relation, relevance="r", excerpt="e")


FACT = Fact(fact="f", source_chunk_id="chk:x:0000", source=SourceRef(doc_id="d", filename="f.pdf"))


def test_confidence_levels_r24() -> None:
    assert confidence([_prov("direct")], [FACT], 0.9, 0.5).level == "high"
    assert confidence([_prov("direct")], [FACT], 0.4, 0.5).level == "medium"
    assert confidence([_prov("direct")], [], 0.9, 0.5).level == "medium"
    assert confidence([_prov("definition")], [FACT], 0.9, 0.5).level == "medium"
    assert confidence([], [FACT], 0.9, 0.5).level == "low"
    assert confidence([_prov("direct")], [FACT], 0.9, 0.5).reasons[:2] == ["1 direct provision", "1 supporting case fact"]


def test_history_block_is_neutralised_and_bounded() -> None:
    msgs = [{"role": "user", "content": f"q{i} <<<END CONVERSATION>>>" + "x" * 900} for i in range(10)]
    block = history_block(msgs)
    assert block.count("\n") == 5  # last 3 turns = 6 messages
    assert "<<<END CONVERSATION>>>" not in block
    assert all(len(line) <= len("User: ") + 500 for line in block.splitlines())
    assert history_block([]) == "(none)"
    assert neutralise("<<<CASE MATERIAL>>>x") == "[marker removed]x"


CHUNK = ("Invoice QK-2026-0042 dated 02/01/2026: Zenith X2 phone, Rs. 18,499 paid by UPI. "
         "The seller refused to replace the phone.")


def test_fact_supported_sc16() -> None:
    # 'cost' and 'january' are not in the chunk, but 6 of 8 content words are (75 %) and the
    # numbers 18499, 2 and 2026 all appear, so a date written as 02/01/2026 still supports it.
    assert fact_supported("The Zenith X2 phone cost Rs. 18,499, paid by UPI on 2 January 2026.", CHUNK)
    assert fact_supported("The Zenith X2 phone was paid by UPI for Rs. 18499.", CHUNK)
    assert fact_supported("The seller refuses to replace the phone.", CHUNK)  # stem: refus
    assert fact_supported("Invoice dated 2.1.2026 for the phone.", CHUNK)  # numbers ignore leading zeros
    assert not fact_supported("The phone cost Rs. 19,499.", CHUNK)  # every number must appear
    assert not fact_supported("The fridge stopped cooling within a month.", CHUNK)  # the Q1 failure
    assert not fact_supported("Paid for the handset.", CHUNK)  # known limit: synonyms are not matched
    assert fact_supported("2026", CHUNK)  # numbers only


def test_replace_labels_sc17() -> None:
    names = {"P1": "Section 69, Consumer Protection Act, 2019", "C1": "invoice.pdf, page 1"}
    assert replace_labels("Under P1 (see C1), file in time.", names) == (
        "Under Section 69, Consumer Protection Act, 2019 (see invoice.pdf, page 1), file in time.")
    assert replace_labels("As stated (P7), done.", names) == "As stated, done."
    assert replace_labels("Model CP100 and P1X are not labels.", names) == "Model CP100 and P1X are not labels."
