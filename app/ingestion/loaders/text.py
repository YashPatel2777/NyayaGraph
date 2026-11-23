"""Text uploads (B3, IN-13, A-1, SC-1).

Decoded as UTF-8 (a BOM is dropped). Bytes that are not UTF-8 are replaced, never silently
dropped, and the job gets a warning (A-1). A WhatsApp or Telegram export becomes one unit
per message for the chunker (SC-2); other text is one prose unit with no page.
"""

from __future__ import annotations

from app.ingestion.loaders import LoadedDocument, LoaderError, TextUnit, non_space_len
from app.ingestion.loaders.chat import is_chat_export, parse_chat

NOT_UTF8_WARNING = "text: the file is not valid UTF-8; undecodable bytes were replaced"
EMPTY_CHAT_WARNING = "text: looked like a chat export but no messages were found; read as text"


def decode_text(data: bytes) -> tuple[str, list[str]]:
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        return data.decode("utf-8-sig", errors="replace"), [NOT_UTF8_WARNING]


def load_text(path: str) -> LoadedDocument:
    with open(path, "rb") as fh:
        content, warnings = decode_text(fh.read())
    if non_space_len(content) == 0:
        raise LoaderError("TEXT_EMPTY", "The text file has no content.")

    if is_chat_export(content):
        try:
            messages = parse_chat(content)
        except ValueError:  # malformed Telegram JSON
            messages = []
        if messages:
            return LoadedDocument(
                media_type="text",
                kind="chat",
                units=[TextUnit(m.line()) for m in messages],
                warnings=warnings,
            )
        warnings.append(EMPTY_CHAT_WARNING)

    return LoadedDocument(
        media_type="text", kind="prose", units=[TextUnit(content)], warnings=warnings
    )
