"""Case document loaders (B3). Each loader turns a stored upload into text units that keep
their page (pdf, image) or start and end seconds (audio), so the chunker never guesses a
page from text (IN-05)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MediaType = Literal["pdf", "image", "audio", "text"]
# How the chunker packs units: prose (pages of text), chat (whole messages, SC-2) or audio
# (timestamped segments).
UnitKind = Literal["prose", "chat", "audio"]

# E-07: a page with fewer non-space characters than this goes to OCR, and a whole file under
# it after OCR fails with OCR_EMPTY.
MIN_TEXT_CHARS = 40


@dataclass(frozen=True)
class TextUnit:
    text: str
    page: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None


@dataclass
class LoadedDocument:
    media_type: MediaType
    kind: UnitKind
    units: list[TextUnit]
    page_count: int | None = None
    duration_sec: float | None = None
    language: str | None = None
    warnings: list[str] = field(default_factory=list)


class LoaderError(Exception):
    """A document that cannot be turned into text. code is a job error_code (4.4)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def non_space_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())
