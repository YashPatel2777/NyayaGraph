"""Statute parser: page text in, Chapters, Sections, Clauses, footnote data out (4.3).

Rules P-01 to P-09 and P-12 (Acts) and P-13 to P-16 and P-18 (rules documents, SC-5) are
applied in order; each is cited where it is implemented. References (P-10, P-11, P-17) are
extracted separately by reference_extractor.py.

The parser is pure: no I/O, no Neo4j. It raises ParseError on P-02/P-13 and P-12 failures,
so nothing is written (E-14).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Kind = Literal["act", "rule", "regulation"]

MONTHS = {
    m: i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"],
        start=1,
    )
}

# P-01, plus Gazette running heads inside the body (A-17)
WATERMARK_RE = re.compile(r"IndiaCode[0-9]*")
DIGITS_ONLY_RE = re.compile(r"^[0-9]+$")
GAZETTE_HEAD_RE = re.compile(r"THE GAZETTE OF INDIA[ ]*:[ ]*EXTRAORDINARY|भारत का राजपत्र")
# P-02, P-03
ACT_BODY_START_RE = re.compile(r"^An Act to")
OBJECTS_RE = re.compile(r"^STATEMENT OF OBJECTS AND REASONS", re.IGNORECASE)
# P-13, P-14 and the rules body end (A-13)
ENABLING_START_RE = re.compile(r"In exercise of the powers conferred by")
RULES_BODY_END_RE = re.compile(r"^[\[][ ]*F[.][ ]*No[.]")
# P-04
CHAPTER_RE = re.compile(r"^CHAPTER ([IVXLC]+)$")
# P-05
FOOTNOTE_START_RE = re.compile(
    r"^([*]|[0-9]{1,2})[.][ ]+(Ins[.]|Subs[.]|Omitted|Vide|"
    r"[0-9]{1,2}(st|nd|rd|th)[ ]+[A-Z][a-z]+,[ ]+[0-9]{4})"
)
# P-06. Rules documents (A-17) accept the separators Gazette typesetting uses between title
# and text: '.—', '. —', '. –', '. ---', ': -'.
HEADING_RE = re.compile(r"^([0-9]{1,3}[A-Z]?)[.][ ]+([A-Z].{2,300}?)[.]—(.*)$")
RULE_HEADING_RE = re.compile(r"^([0-9]{1,3}[A-Z]?)[.][ ]+([A-Z].{2,300}?)[ ]*[.:][ ]*(?:—|–|-{1,3})[ ]*(.*)$")
HEADING_SEP = {"act": re.compile(r"[.]—"), "rule": re.compile(r"[.:][ ]*(?:—|–|-{1,3})")}
HEADING_PREFIX_RE = re.compile(r"^([0-9]{1,3}[A-Z]?)[.][ ]+[A-Z]")
# P-07 (a footnote marker such as 1[ may precede the label, P-09)
SUBSECTION_RE = re.compile(r"^(?:[0-9]{1,2}[\[])?[(]([0-9]+[A-Z]?)[)][ ]")
# P-08 (Acts) and P-16 (rules)
ACT_DEFINITION_RE = re.compile(r"^(?:[0-9]{1,2}[\[])?[(]([0-9]+)[)][ ][“]")
# Rules: the Gazette text layer renders some quotes as ― or ‖ (A-17); they are recognised,
# not rewritten.
RULE_DEFINITION_RE = re.compile(r"^(?:[0-9]{1,2}[\[])?[(]([a-z]{1,3})[)][ ]+[“\"―‘]")
TERM_RE = re.compile(r"[“\"]([^”\"]+)[”\"]")
RULE_TERM_RE = re.compile(r"[“\"―‘]([^”\"‖’]+)[”\"‖’]")
# P-09
MARKER_RE = re.compile(r"(?<![0-9A-Za-z])([0-9]{1,2})[\[]")
AMENDING_ACT_RE = re.compile(r"Act[ ]+([0-9]{1,3})[ ]+of[ ]+([0-9]{4})")
WEF_RE = re.compile(r"w[.]e[.]f[.][ ]*([0-9]{2})-([0-9]{2})-([0-9]{4})")
DATE_PARA_RE = re.compile(r"^[ ]*([0-9]{1,2})(?:st|nd|rd|th)[ ]+([A-Z][a-z]+),[ ]+([0-9]{4})")
NUMBER_LIST = r"[0-9]{1,3}[A-Z]?(?:[ ]*(?:,|and|to)[ ]*[0-9]{1,3}[A-Z]?)*"
COMMENCEMENT_ITEM_RE = re.compile(
    r"(?<![A-Za-z])(?P<of>of[ ]+)?(?:ss|S|s)[.][ ]*(?P<nums>" + NUMBER_LIST + r")"
    r"(?:[ ]*[(]both inclusive[)])?(?:[ ]*[\[](?P<bracket>[^\]]*)[\]])?"
)
SPELLED_SECTION_RE = re.compile(r"of[ ]+section[ ]+([0-9]{1,3}[A-Z]?)")
ITEM_START_RE = re.compile(r"^(?:[(][^\s()]{1,6}[)]|Provided|Explanation)")
# A printed line ending in 'sub-section', 'clause' etc. continues on the next line with the
# label, as in 'sub-section' + '(1) of section 28': not a new item.
WRAPPED_REF_END_RE = re.compile(r"(?i)(?:sub-?)?(?:section|clause|rule)s?$")

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


class ParseError(Exception):
    """Fails the job; nothing is written (E-14)."""

    def __init__(self, code: str, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.problems = problems or []


@dataclass
class Line:
    text: str
    page: int


@dataclass
class ParsedClause:
    label: str
    kind: Literal["subsection", "definition"]
    text: str
    term: str | None = None


@dataclass
class ParsedSection:
    number: str
    title: str
    chapter_number: str | None
    text: str
    clauses: list[ParsedClause] = field(default_factory=list)
    amendment_notes: list[str] = field(default_factory=list)
    in_force_from: date | None = None
    pages: list[int] = field(default_factory=list)
    status: str = "in_force"


@dataclass
class ParsedChapter:
    number: str
    title: str


@dataclass
class Amendment:
    section_number: str
    note: str
    act_number: str | None  # for example '33 of 2021'
    effective_on: date | None


@dataclass
class ParseResult:
    kind: Kind
    chapters: list[ParsedChapter]
    sections: list[ParsedSection]
    objects_text: str | None
    enabling_text: str | None
    amendments: list[Amendment]
    warnings: list[str]

    @property
    def report(self) -> dict[str, int]:
        clauses = [c for s in self.sections for c in s.clauses]
        return {
            "chapters": len(self.chapters),
            "sections": len(self.sections),
            "subsection_clauses": sum(c.kind == "subsection" for c in clauses),
            "definition_clauses": sum(c.kind == "definition" for c in clauses),
            "amendments": len(self.amendments),
        }


# Helpers


def _successor_numbers(prev: str | None) -> set[str]:
    """P-06 number guard: prev + 1, or prev with the next letter suffix (55 -> 55A)."""
    if prev is None:
        return {"1"}
    m = re.fullmatch(r"([0-9]+)([A-Z]?)", prev)
    assert m
    n, letter = int(m.group(1)), m.group(2)
    nxt = {str(n + 1), f"{n}{chr(ord(letter) + 1) if letter else 'A'}"}
    return nxt


def _successor_labels(prev: str | None, numeric: bool) -> set[str]:
    if numeric:
        return _successor_numbers(prev)
    if prev is None:
        return {"a"}
    # a..z, then aa, bb ... as printed in Indian rules (z is followed by za, zb ...).
    if prev == "z":
        return {"za", "aa"}
    last = prev[-1]
    return {prev[:-1] + chr(ord(last) + 1)} if last != "z" else {prev + "a"}


def _reflow(lines: list[str]) -> str:
    """Join printed lines into text. A new line is kept before list items, provisos and
    explanations; other line breaks become spaces. A trailing hyphen after a letter joins
    without a space ("sub-" + "clause"). Wording is never changed (P-04 spirit)."""
    out = ""
    for raw in lines:
        line = re.sub(r"[ ]{2,}", " ", raw.strip())  # print spacing, not wording
        if not line:
            continue
        if not out:
            out = line
        elif ITEM_START_RE.match(line) and not WRAPPED_REF_END_RE.search(out):
            out += "\n" + line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            out += line
        else:
            out += " " + line
    return out


def _strip_markers(text: str) -> str:
    # P-09 markers (1[) are editorial; the bracket stays, the digit goes.
    return MARKER_RE.sub("[", text)


def _parse_date(day: str, month: str, year: str) -> date | None:
    if month not in MONTHS:
        return None
    try:
        return date(int(year), MONTHS[month], int(day))
    except ValueError:
        return None


def _expand_numbers(nums: str) -> list[str]:
    out: list[str] = []
    tokens = re.findall(r"[0-9]{1,3}[A-Z]?|to", nums)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "to":
            i += 1
            continue
        if i + 2 < len(tokens) and tokens[i + 1] == "to" and tok.isdigit() and tokens[i + 2].isdigit():
            out.extend(str(n) for n in range(int(tok), int(tokens[i + 2]) + 1))
            i += 3
            continue
        out.append(tok)
        i += 1
    return out


# Stages


def clean_pages(pages: list[str]) -> list[Line]:
    """P-01: drop the IndiaCode watermark, digit-only lines and Gazette running heads (A-17).
    Page boundaries are kept."""
    out: list[Line] = []
    for page_no, text in enumerate(pages, start=1):
        for raw in text.splitlines():
            line = WATERMARK_RE.sub("", raw).strip()
            if not line or DIGITS_ONLY_RE.fullmatch(line) or GAZETTE_HEAD_RE.search(line):
                continue
            out.append(Line(line, page_no))
    return out


def _try_heading(
    lines: list[Line], i: int, expected: set[str], kind: Kind = "act"
) -> tuple[re.Match, int] | None:
    """P-06: a heading line, joining up to 2 following lines until '.—' appears."""
    sep = HEADING_SEP["act" if kind == "act" else "rule"]
    heading_re = HEADING_RE if kind == "act" else RULE_HEADING_RE
    first = lines[i].text
    m = HEADING_PREFIX_RE.match(first)
    if not m or m.group(1) not in expected:
        return None
    joined = first
    for extra in range(3):
        if extra:
            if i + extra >= len(lines):
                return None
            joined += " " + lines[i + extra].text
        if sep.search(joined):
            hm = heading_re.match(joined)
            if hm and hm.group(1) == m.group(1):
                return hm, extra + 1
            return None
    return None


def split_body(lines: list[Line], kind: Kind) -> tuple[list[Line], str | None, str | None]:
    """P-02/P-03 for Acts, P-13/P-14 and the F. No. end for rules.

    Returns (body lines, objects_text, enabling_text).
    """
    if kind == "act":
        start = next((i for i, ln in enumerate(lines) if ACT_BODY_START_RE.match(ln.text)), None)
        if start is None:
            raise ParseError("PARSE_BODY_START_NOT_FOUND", "no line begins with 'An Act to' (P-02)")
        body = lines[start:]
        end = next((i for i, ln in enumerate(body) if OBJECTS_RE.match(ln.text)), None)
        objects = None
        if end is not None:
            objects = _reflow([ln.text for ln in body[end:]]) or None
            body = body[:end]
        return body, objects, None

    # Rules (SC-5)
    start = next((i for i in range(len(lines)) if _try_heading(lines, i, {"1"}, kind)), None)
    if start is None:
        raise ParseError(
            "PARSE_BODY_START_NOT_FOUND", "no heading numbered 1 was found (P-13)"
        )
    enabling = None
    en = next((i for i in range(start) if ENABLING_START_RE.search(lines[i].text)), None)
    if en is not None:
        para = lines[en].text
        para = para[ENABLING_START_RE.search(para).start():]  # drop a 'G.S.R. 912(E).—' prefix
        enabling = _reflow([para] + [ln.text for ln in lines[en + 1 : start]])
    body = lines[start:]
    end = next((i for i, ln in enumerate(body) if RULES_BODY_END_RE.match(ln.text)), None)
    if end is not None:
        body = body[:end]
    return body, None, enabling


def extract_footnotes(body: list[Line]) -> tuple[list[Line], dict[int, list[str]]]:
    """P-05: a footnote block runs from its first line to the end of that page."""
    kept: list[Line] = []
    blocks: dict[int, list[str]] = {}
    in_block_page: int | None = None
    for ln in body:
        if in_block_page == ln.page:
            blocks[ln.page].append(ln.text)
            continue
        in_block_page = None
        if FOOTNOTE_START_RE.match(ln.text):
            in_block_page = ln.page
            blocks.setdefault(ln.page, []).append(ln.text)
            continue
        kept.append(ln)
    return kept, blocks


@dataclass
class _OpenSection:
    number: str
    title: str
    chapter: str | None
    first_text: str
    lines: list[Line] = field(default_factory=list)


def _structure(
    body: list[Line], kind: Kind
) -> tuple[list[ParsedChapter], list[_OpenSection], list[str]]:
    """P-04 chapters and P-06 section headings with the number guard."""
    chapters: list[ParsedChapter] = []
    sections: list[_OpenSection] = []
    rejected: list[str] = []
    prev: str | None = None
    chapter: str | None = None
    i = 0
    while i < len(body):
        text = body[i].text
        cm = CHAPTER_RE.match(text)
        if cm:
            # P-04: the upper-case line(s) after the heading are the title, as written.
            roman, j, title_lines = cm.group(1), i + 1, []
            while j < len(body) and body[j].text.upper() == body[j].text and re.search(r"[A-Z]", body[j].text) \
                    and not HEADING_PREFIX_RE.match(body[j].text):
                title_lines.append(body[j].text)
                j += 1
            chapters.append(ParsedChapter(roman, " ".join(title_lines)))
            chapter = roman
            i = j
            continue
        found = _try_heading(body, i, _successor_numbers(prev), kind)
        if found:
            hm, used = found
            sections.append(_OpenSection(hm.group(1), hm.group(2).strip(), chapter, hm.group(3).strip()))
            sections[-1].lines.append(Line(hm.group(3).strip(), body[i].page))
            prev = hm.group(1)
            i += used
            continue
        pm = HEADING_PREFIX_RE.match(text)
        sep = HEADING_SEP["act" if kind == "act" else "rule"]
        if pm and sep.search(" ".join(b.text for b in body[i : i + 3])) and sections:
            rejected.append(pm.group(1))
        if sections:
            sections[-1].lines.append(body[i])
        i += 1
    return chapters, sections, rejected


def _clauses(sec: _OpenSection, kind: Kind) -> list[ParsedClause]:
    """P-07 subsections; P-08 (Acts) or P-16 (rules) definitions in the Definitions section."""
    is_definitions = sec.title.strip().rstrip(".") == "Definitions"
    clauses: list[tuple[ParsedClause, list[str]]] = []
    prev_sub: str | None = None
    prev_def: str | None = None
    for ln in sec.lines:
        text = ln.text
        if is_definitions:
            dm = (ACT_DEFINITION_RE if kind == "act" else RULE_DEFINITION_RE).match(text)
            if dm and dm.group(1) in _successor_labels(prev_def, numeric=(kind == "act")):
                body = text[dm.end(1) + 1 :].strip()
                tm = (TERM_RE if kind == "act" else RULE_TERM_RE).search(body)
                clauses.append((ParsedClause(dm.group(1), "definition", "", tm.group(1).strip() if tm else None), [body]))
                prev_def = dm.group(1)
                continue
        sm = SUBSECTION_RE.match(text)
        # Acts' Definitions sections have no numbered subsections (P-08); in rules both exist.
        if sm and not (is_definitions and kind == "act") and sm.group(1) in _successor_labels(prev_sub, True):
            clauses.append((ParsedClause(sm.group(1), "subsection", ""), [text[sm.end() :].strip()]))
            prev_sub = sm.group(1)
            continue
        if clauses:
            clauses[-1][1].append(text)
    out = []
    for clause, lines in clauses:
        clause.text = _strip_markers(_reflow(lines))
        out.append(clause)
    return out


def _apply_footnotes(
    blocks: dict[int, list[str]], sections: list[ParsedSection], warnings: list[str]
) -> list[Amendment]:
    """P-09 (and P-18 for notifications): amendment notes, AMENDS data, in_force_from."""
    by_number = {s.number: s for s in sections}
    amendments: list[Amendment] = []
    commencement: dict[str, date] = {}

    for page, block in sorted(blocks.items()):
        # Split the block into footnotes at each footnote-start line.
        notes: list[list[str]] = []
        for line in block:
            if FOOTNOTE_START_RE.match(line) or not notes:
                notes.append([line])
            else:
                notes[-1].append(line)
        for note_lines in notes:
            fm = FOOTNOTE_START_RE.match(note_lines[0])
            marker = fm.group(1) if fm else None
            keyword = fm.group(2) if fm else ""
            body = " ".join(note_lines)
            body = body[fm.end(1) + 1 :].strip() if fm else body
            if keyword.startswith(("Ins", "Subs", "Omitted")):
                target = _marker_section(marker, page, sections)
                if target is None:
                    warnings.append(f"amendment footnote {marker} on page {page} has no section")
                    continue
                target.amendment_notes.append(body)
                am = AMENDING_ACT_RE.search(body)
                wm = WEF_RE.search(body)
                eff = None
                if wm:
                    try:
                        eff = date(int(wm.group(3)), int(wm.group(2)), int(wm.group(1)))
                    except ValueError:
                        eff = None
                # P-18: notifications (no Act number) give a note only.
                amendments.append(
                    Amendment(target.number, body, f"{am.group(1)} of {am.group(2)}" if am else None, eff)
                )
                continue
            if DATE_PARA_RE.match(body):
                _commencement(body, by_number, commencement, warnings)
                continue
            warnings.append(f"footnote {marker} on page {page} is not an amendment or commencement note")

    for s in sections:
        s.in_force_from = commencement.get(s.number)
    return amendments


def _marker_section(marker: str | None, page: int, sections: list[ParsedSection]) -> ParsedSection | None:
    """P-09: the section where the marker (digit before '[') appears on that page, else the
    last section heading on that page."""
    on_page = [s for s in sections if page in s.pages]
    if marker and marker.isdigit():
        for s in on_page:
            if re.search(rf"(?<![0-9A-Za-z]){marker}[\[]", s.text):
                return s
    starts = [s for s in sections if s.pages and s.pages[0] == page]
    return starts[-1] if starts else (on_page[-1] if on_page else None)


def _commencement(
    text: str, by_number: dict[str, ParsedSection], out: dict[str, date], warnings: list[str]
) -> None:
    """P-09: each paragraph starting with a date lists the sections brought into force."""
    starts = [m.start() for m in re.finditer(r"[0-9]{1,2}(?:st|nd|rd|th)[ ]+[A-Z][a-z]+,[ ]+[0-9]{4}", text)]
    paras = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    for para in paras:
        dm = DATE_PARA_RE.match(para)
        if not dm:
            continue
        when = _parse_date(dm.group(1), dm.group(2), dm.group(3))
        if when is None:
            warnings.append(f"unparseable commencement date {dm.group(0)!r}")
            continue
        para = re.sub(r"\s+", " ", para[dm.end() :])
        assigned: set[str] = set()
        excluded: set[str] = set()
        for item in COMMENCEMENT_ITEM_RE.finditer(para):
            nums = _expand_numbers(item.group("nums"))
            bracket = (item.group("bracket") or "").strip()
            clause_level_first = bool(item.group("of"))  # 'of s. 101, 104': 101 is clause-level
            whole = nums[1:] if clause_level_first else nums
            if clause_level_first and nums and nums[0] in by_number:
                by_number[nums[0]].amendment_notes.append(
                    f"{dm.group(0).strip()}: clause-level commencement in section {nums[0]}"
                )
            if bracket and not bracket.lower().startswith("except"):
                # 'S. 2 [clauses (4), ...]': only those clauses commence.
                for n in whole:
                    if n in by_number:
                        by_number[n].amendment_notes.append(f"{dm.group(0).strip()}: {bracket} commence")
                continue
            if bracket.lower().startswith("except"):
                rest = bracket[6:].strip()
                if re.match(r"(?i)(sections?|ss?[.])[ ]", rest):
                    excluded.update(_expand_numbers(re.sub(r"(?i)^(sections?|ss?[.])[ ]*", "", rest)))
                else:
                    # Clause-level exception: recorded only in amendment_notes (P-09), on the
                    # section the bracket names, else on the last section of the item.
                    named = [m.group(1) for m in SPELLED_SECTION_RE.finditer(rest)] or whole[-1:]
                    for n in named:
                        if n in by_number:
                            by_number[n].amendment_notes.append(
                                f"{dm.group(0).strip()}: commenced except {rest}"
                            )
            assigned.update(whole)
        # Spelled-out 'of section N' items outside brackets are clause-level only.
        for sm in SPELLED_SECTION_RE.finditer(re.sub(r"[\[][^\]]*[\]]", "", para)):
            if sm.group(1) in by_number:
                by_number[sm.group(1)].amendment_notes.append(
                    f"{dm.group(0).strip()}: clause-level commencement in section {sm.group(1)}"
                )
        for n in assigned - excluded:
            if n not in by_number:
                warnings.append(f"commencement lists unknown section {n}")
                continue
            out.setdefault(n, when)  # the first date listed wins


def validate(sections: list[ParsedSection], rejected: list[str], expected: int | None) -> None:
    """P-12: continuous numbering and the expected count, else PARSE_VALIDATION_FAILED."""
    problems: list[str] = []
    prev: str | None = None
    seen: set[str] = set()
    for s in sections:
        if s.number in seen:
            problems.append(f"duplicate section {s.number}")
        if s.number not in _successor_numbers(prev):
            problems.append(f"gap before section {s.number} (after {prev})")
        seen.add(s.number)
        prev = s.number
    if rejected:
        problems.append(f"out-of-sequence headings treated as body text: {sorted(set(rejected), key=lambda x: (len(x), x))}")
    if expected is not None and len(sections) != expected:
        problems.append(f"expected {expected} sections, parsed {len(sections)} (last {prev})")
    if not sections:
        problems.append("no sections parsed")
    if problems:
        raise ParseError("PARSE_VALIDATION_FAILED", "section numbering failed validation (P-12)", problems)


# Amendment notifications (SC-9, P-19)

AMENDING_PARA_RE = re.compile(r"^([0-9]{1,3})[.][ ]+(.*)$")
SUBRULE_SUBST_RE = re.compile(
    r"(?is)^In the (?P<title>.+?), in rule (?P<rule>[0-9]{1,3}[A-Z]?), for sub-rule[ ]*[(](?P<sub>[0-9]+[A-Z]?)[)],"
    r"[ ]*the following sub-rule shall be substituted,[ ]*namely[ ]*[:—–-]*[ ]*[“\"](?P<text>.+)[”\"][ ]*$"
)
RULE_SUBST_RE = re.compile(
    r"(?is)^In the (?P<title>.+?), for rule (?P<rule>[0-9]{1,3}[A-Z]?),"
    r"[ ]*the following rule shall be substituted,[ ]*namely[ ]*[:—–-]*[ ]*[“\"](?P<text>.+)[”\"][ ]*$"
)


@dataclass
class Substitution:
    """One P-19 instruction: replace rule N (sub is None) or sub-rule (sub) of rule N."""

    principal_title: str
    rule: str
    sub: str | None
    text: str  # the new text as printed inside the quotes, label included


def parse_amending(pages: list[str], expected_sections: int | None) -> ParseResult:
    """P-19: an amendment notification's English part is a list of numbered paragraphs
    without titles. The body starts at the first '1.' after the English enabling text and
    ends at '[F. No.'."""
    lines = clean_pages(pages)  # P-01
    en = next((i for i, ln in enumerate(lines) if ENABLING_START_RE.search(ln.text)), None)
    if en is None:
        raise ParseError("PARSE_BODY_START_NOT_FOUND", "no English enabling text (P-19)")
    start = next((i for i in range(en + 1, len(lines)) if re.match(r"^1[.][ ]+", lines[i].text)), None)
    if start is None:
        raise ParseError("PARSE_BODY_START_NOT_FOUND", "no paragraph numbered 1 after the enabling text (P-19)")
    para = lines[en].text
    enabling = _reflow([para[ENABLING_START_RE.search(para).start():]] + [ln.text for ln in lines[en + 1 : start]])
    body = lines[start:]
    end = next((i for i, ln in enumerate(body) if RULES_BODY_END_RE.match(ln.text)), None)
    body = body[:end] if end is not None else body
    body, _blocks = extract_footnotes(body)  # P-05

    opened: list[_OpenSection] = []
    rejected: list[str] = []
    prev: str | None = None
    for ln in body:
        m = AMENDING_PARA_RE.match(ln.text)
        if m and m.group(1) in _successor_numbers(prev):
            opened.append(_OpenSection(m.group(1), "", None, m.group(2)))
            opened[-1].lines.append(Line(m.group(2), ln.page))
            prev = m.group(1)
        elif opened:
            opened[-1].lines.append(ln)
    sections = []
    for os_ in opened:
        sec = ParsedSection(os_.number, "", None, _reflow([l.text for l in os_.lines]),
                            pages=sorted({l.page for l in os_.lines}))
        sec.clauses = _clauses(os_, "rule")
        sections.append(sec)
    validate(sections, rejected, expected_sections)  # P-12
    return ParseResult("rule", [], sections, None, enabling, [], [])


def extract_substitutions(parsed: ParseResult) -> list[Substitution]:
    """P-19: every paragraph after the short title (paragraph 1) must be exactly one
    supported substitution; anything else fails the job so nothing is half-applied."""
    subs: list[Substitution] = []
    problems: list[str] = []
    for sec in parsed.sections:
        if sec.number == "1":
            continue
        text = re.sub(r"\s+", " ", sec.text).strip()
        if len(re.findall(r"(?i)shall be (substituted|inserted|omitted)", text)) != 1:
            problems.append(f"paragraph {sec.number}: not exactly one substitution instruction")
            continue
        m = SUBRULE_SUBST_RE.match(text) or RULE_SUBST_RE.match(text)
        if not m:
            problems.append(f"paragraph {sec.number}: unsupported amendment wording")
            continue
        sub = m.groupdict().get("sub")
        new = m.group("text").strip()
        label_ok = new.startswith(f"({sub})") if sub else new.startswith(f"{m.group('rule')}.")
        if not label_ok:
            problems.append(f"paragraph {sec.number}: quoted text does not start with its label")
            continue
        subs.append(Substitution(m.group("title").strip(), m.group("rule"), sub, new))
    if problems:
        raise ParseError("PARSE_VALIDATION_FAILED", "amendment notification not understood (P-19)", problems)
    if not subs:
        raise ParseError("PARSE_VALIDATION_FAILED", "no substitution found (P-19)")
    return subs


def apply_substitution(parsed: ParseResult, number: str, sub: str | None, new_text: str) -> str:
    """P-19: apply one substitution to a parsed principal in memory. Returns the previous
    text (for AMENDS.previous_text, P-20). Raises ParseError when the target is missing."""
    sec = next((s for s in parsed.sections if s.number == number), None)
    if sec is None:
        raise ParseError("PARSE_VALIDATION_FAILED", f"rule {number} is not in the principal rules (P-19)")
    if sub is None:
        # Whole rule: the quoted text is '{N}. Title.—text'.
        lines = clean_pages([new_text])
        prev = None if number == "1" else _predecessor(number)
        found = _try_heading(lines, 0, _successor_numbers(prev), parsed.kind)
        if not found:
            raise ParseError("PARSE_VALIDATION_FAILED", f"substituted rule {number} has no heading (P-19)")
        hm, used = found
        opened = _OpenSection(number, hm.group(2).strip(), sec.chapter_number, hm.group(3).strip())
        opened.lines = [Line(hm.group(3).strip(), 0)] + lines[used:]
        previous = sec.text
        sec.title, sec.text = opened.title, _reflow([l.text for l in opened.lines])
        sec.clauses = _clauses(opened, parsed.kind)
        return previous
    clause = next((c for c in sec.clauses if c.label == sub and c.kind == "subsection"), None)
    if clause is None:
        raise ParseError("PARSE_VALIDATION_FAILED", f"sub-rule ({sub}) of rule {number} is not in the principal rules (P-19)")
    body = new_text[len(f"({sub})"):].strip()
    old_full = f"({sub}) {clause.text}"
    if old_full not in sec.text:
        raise ParseError("PARSE_VALIDATION_FAILED", f"cannot locate sub-rule ({sub}) inside rule {number}'s text (P-19)")
    previous = clause.text
    sec.text = sec.text.replace(old_full, f"({sub}) {body}", 1)
    clause.text = body
    return previous


def _predecessor(number: str) -> str | None:
    m = re.fullmatch(r"([0-9]+)([A-Z]?)", number)
    assert m
    n, letter = int(m.group(1)), m.group(2)
    if letter:
        return str(n) if letter == "A" else f"{n}{chr(ord(letter) - 1)}"
    return str(n - 1) if n > 1 else None


def parse_statute(pages: list[str], kind: Kind, expected_sections: int | None) -> ParseResult:
    """Run P-01 to P-09 and P-12 (P-13 to P-16 and P-18 for rules) over per-page text."""
    warnings: list[str] = []
    lines = clean_pages(pages)  # P-01
    body, objects_text, enabling_text = split_body(lines, kind)  # P-02/P-03, P-13/P-14
    body, blocks = extract_footnotes(body)  # P-05, before headings (page boundaries kept)
    chapters, open_sections, rejected = _structure(body, kind)  # P-04, P-06, P-15

    sections: list[ParsedSection] = []
    for os_ in open_sections:
        pages_seen = sorted({ln.page for ln in os_.lines})
        raw_text = _reflow([ln.text for ln in os_.lines])
        sec = ParsedSection(
            number=os_.number,
            title=os_.title,
            chapter_number=os_.chapter,
            text=raw_text,  # markers kept until P-09 has located them
            pages=pages_seen,
        )
        sec.clauses = _clauses(os_, kind)
        sections.append(sec)

    validate(sections, rejected, expected_sections)  # P-12
    amendments = _apply_footnotes(blocks, sections, warnings)  # P-09, P-18
    for sec in sections:
        sec.text = _strip_markers(sec.text)
    return ParseResult(kind, chapters, sections, objects_text, enabling_text, amendments, warnings)
