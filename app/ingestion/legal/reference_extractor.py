"""Cross-reference extraction (P-10, P-11, P-17).

Runs on every Section (title and text) and every Clause text. Produces REFERS_TO edges to
Sections, Clauses and Chapters, and REFERS_TO_ACT edges to external stub Acts. Unknown
targets are warnings only (P-12).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.schemas.graph_schema import act_id, chapter_id, clause_id, section_id, slug

NUM = r"[0-9]{1,3}[A-Z]?"
NUM_LIST = NUM + r"(?:[ ]*(?:,|and|to)[ ]*" + NUM + r")*"

# P-10 qualifiers. 'subsection' is accepted because the PDF drops the hyphen in places.
SUBSECTION_OF_RE = re.compile(
    r"(?i)(?<![A-Za-z-])sub-?section[ ]*[(](?P<sub>[0-9]+[A-Z]?)[)][ ]+of[ ]+section[ ]+(?P<sec>" + NUM + r")"
)
CLAUSE_OF_SECTION_2_RE = re.compile(
    r"(?i)(?<![A-Za-z-])clause[ ]*[(](?P<cl>[0-9]+)[)][ ]+of[ ]+section[ ]+2(?![0-9A-Z])"
)
# P-10 section references; not preceded by 'sub-' (sub-section 2 is not section 2).
SECTION_RE = re.compile(r"(?i)(?<![A-Za-z-])sections?[ ]+(?P<nums>" + NUM_LIST + r")")
CHAPTER_REF_RE = re.compile(r"(?<![A-Za-z])Chapter[ ]+(?P<roman>[IVXLC]+)(?![A-Za-z])")
# P-17 rule references inside rules documents.
SUBRULE_OF_RE = re.compile(
    r"(?i)(?<![A-Za-z-])sub-?rule[ ]*[(](?P<sub>[0-9]+)[)][ ]+of[ ]+rule[ ]+(?P<rule>" + NUM + r")"
)
RULE_RE = re.compile(r"(?i)(?<![A-Za-z-])rules?[ ]+(?P<nums>" + NUM_LIST + r")(?![0-9])")
# P-10 other-Act test, extended (A-14) so 'Code of Criminal Procedure, 1973' and
# 'Code of Civil Procedure, 1908' are recognised as P-11 expects.
OTHER_ACT_RE = re.compile(
    r"^[ ]*of[ ]+(?:the[ ]+)?(?P<name>(?:Code of [A-Z][A-Za-z]*(?:[ ][A-Z][A-Za-z]*)*|[A-Z][A-Za-z&' ]+?(?:Act|Code))"
    r"(?:,[ ]*[0-9]{4})?)(?![A-Za-z])"
)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    qualifier: str | None = None


@dataclass(frozen=True)
class ActEdge:
    source: str
    target: str  # act:{slug}
    name: str  # as written
    section_ref: str | None


@dataclass
class References:
    refers_to: set[Edge] = field(default_factory=set)
    refers_to_act: set[ActEdge] = field(default_factory=set)
    stub_acts: dict[str, str] = field(default_factory=dict)  # id -> name as written
    warnings: list[str] = field(default_factory=list)


def expand(nums: str) -> list[str]:
    """'20 and 21', '35, 36 and 39', '3 to 9' -> individual numbers (P-10)."""
    tokens = re.findall(NUM + r"|to", nums)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "to":
            i += 1
            continue
        if i + 2 < len(tokens) and tokens[i + 1] == "to" and tokens[i].isdigit() and tokens[i + 2].isdigit():
            lo, hi = int(tokens[i]), int(tokens[i + 2])
            out.extend(str(n) for n in range(lo, hi + 1)) if hi - lo <= 200 else out.append(tokens[i])
            i += 3
            continue
        out.append(tokens[i])
        i += 1
    return out


class ReferenceExtractor:
    def __init__(
        self,
        act_slug: str,
        act_titles: Iterable[str],
        known_ids: set[str],
        *,
        parent_slug: str | None = None,
        parent_titles: Iterable[str] = (),
    ) -> None:
        """act_titles: this document's title forms (P-10 'not this Act's title').
        known_ids: ids that exist (this document's, plus the parent Act's for rules).
        parent_slug, parent_titles: set for rules documents (P-17). A reference naming the
        parent Act resolves to the parent's sections, not to a stub Act."""
        self.act_slug = act_slug
        self.own_titles = {slug(t) for t in act_titles if t}
        self.parent_titles = {slug(t) for t in parent_titles if t}
        self.known = known_ids
        self.parent = parent_slug
        self.result = References()

    # 'section N' resolves to the parent Act in rules (P-17), to this Act otherwise.
    def _section_scope(self) -> str:
        return self.parent or self.act_slug

    def _other_act(self, after: str) -> str | None:
        m = OTHER_ACT_RE.match(after)
        if not m:
            return None
        name = m.group("name").strip()
        return None if slug(name) in self.own_titles | self.parent_titles else name

    def _add_act_edge(self, source: str, name: str, section_ref: str | None) -> None:
        target = act_id(name)  # P-11: the year stays as written
        self.result.stub_acts.setdefault(target, name)
        self.result.refers_to_act.add(ActEdge(source, target, name, section_ref))

    def _add(self, source: str, own_section: str, target: str, qualifier: str | None) -> None:
        if target in (source, own_section):
            return  # P-10: self-references are ignored
        if target not in self.known:
            self.result.warnings.append(f"reference from {source} to unknown {target}")
            return
        self.result.refers_to.add(Edge(source, target, qualifier))

    def extract(self, source: str, own_section: str, text: str) -> None:
        """Add the references found in text. own_section is the id of the Section the text
        belongs to (the source itself for a Section)."""
        text = re.sub(r"\s+", " ", text)
        consumed: list[tuple[int, int]] = []

        def free(a: int, b: int) -> bool:
            return all(b <= x or a >= y for x, y in consumed)

        scope = self._section_scope()

        for m in SUBSECTION_OF_RE.finditer(text):
            consumed.append(m.span())
            other = self._other_act(text[m.end() :])
            if other:
                self._add_act_edge(source, other, m.group("sec"))
                continue
            sec, sub = m.group("sec"), m.group("sub")
            cl = clause_id(scope, sec, sub)
            target = cl if cl in self.known else section_id(scope, sec)
            self._add(source, own_section, target, f"sub-section ({sub})")

        for m in CLAUSE_OF_SECTION_2_RE.finditer(text):
            if not free(*m.span()):
                continue
            consumed.append(m.span())
            after = text[m.end() :]
            other = self._other_act(after)
            if other:
                self._add_act_edge(source, other, "2")
                continue
            self._add(source, own_section, clause_id(scope, "2", m.group("cl")), f"clause ({m.group('cl')})")

        for m in SECTION_RE.finditer(text):
            if not free(*m.span()):
                continue
            after = text[m.end() :]
            other = self._other_act(after)
            if other:
                self._add_act_edge(source, other, m.group("nums"))
                continue
            for n in expand(m.group("nums")):
                self._add(source, own_section, section_id(scope, n), None)

        for m in CHAPTER_REF_RE.finditer(text):
            other = self._other_act(text[m.end() :])
            if other:
                self._add_act_edge(source, other, None)
                continue
            if self.parent is None:
                self._add(source, own_section, chapter_id(self.act_slug, m.group("roman")), None)

        if self.parent is not None:  # P-17: rules refer to this document by 'rule N'
            for m in SUBRULE_OF_RE.finditer(text):
                consumed.append(m.span())
                cl = clause_id(self.act_slug, m.group("rule"), m.group("sub"))
                target = cl if cl in self.known else section_id(self.act_slug, m.group("rule"))
                self._add(source, own_section, target, f"sub-rule ({m.group('sub')})")
            for m in RULE_RE.finditer(text):
                if not free(*m.span()):
                    continue
                for n in expand(m.group("nums")):
                    self._add(source, own_section, section_id(self.act_slug, n), None)
