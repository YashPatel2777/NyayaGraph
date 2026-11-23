"""WhatsApp and Telegram exports inside text uploads (SC-1, SC-2, A-3).

Ported from the prototype's loaders/chat.py (07 section 5): the four line layouts, the
system-message filter, multi-line messages and Telegram JSON. Fixes:

- IN-02: timestamps are parsed field by field instead of through a format list, so the
  Android '12/01/2026, 14:33' layout normalises to '2026-01-12 14:33:00'.
- IN-09: day-first or month-first is inferred from every date in the file: any first number
  over 12 means day-first, else any second number over 12 means month-first, else day-first.
- Group-event patterns ('X left', 'X was added') are anchored to the whole message and apply
  only to messages the export marks as system lines, so a real message such as 'only 2 days
  left' is kept. An unmatched line is a continuation unless it starts with a timestamp.

A message is returned as one string '[YYYY-MM-DD HH:MM:SS] Sender: body' (SC-2); the chunker
packs whole messages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Invisible direction and zero-width marks that exports insert.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")
# Narrow and normal no-break spaces (newer exports put one before AM/PM).
_NBSP = re.compile(r"[  ]")

_DATE = r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP]\.?\s?[mM]\.?)?"

# WhatsApp Android: 12/01/2026, 14:32 - Sender: Message
WA_ANDROID = re.compile(rf"^({_DATE}),?\s+({_TIME})\s+-\s+([^:]+?):\s?(.*)$")
# WhatsApp iOS and Telegram desktop text: [12/01/2026, 14:32:05] Sender: Message
BRACKET = re.compile(rf"^\[({_DATE}),?\s+({_TIME})\]\s*([^:]+?):\s?(.*)$")
# Telegram two-line: 'Sender, [15.01.2026 14:32]' then the message on the next line.
TG_HEADER = re.compile(rf"^([^,\[\]]+),\s*\[({_DATE}),?\s+({_TIME})\]$")
# Any line that starts with a timestamp (used to recognise sender-less system lines).
_TIMESTAMPED = re.compile(rf"^\[?{_DATE},?\s+{_TIME}\]?\s")

_DATE_PARTS = re.compile(r"^(\d{1,4})[/.-](\d{1,2})[/.-](\d{1,4})$")
_TIME_PARTS = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:([aApP])\.?\s?[mM]\.?)?$")

# Whole-message placeholders and notices (searched).
_PLACEHOLDERS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^<media omitted>$",
        r"^<\s*attached:[^>]+>$",
        r"^(image|video|audio|document|sticker|gif|contact card) omitted$",
        r"^.{0,120}\bdocument omitted$",
        r"^messages and calls are end-to-end encrypted\b",
        r"^this chat is with a business account\b",
        r"^this message was deleted\.?$",
        r"^you deleted this message\.?$",
        r"^waiting for this message\b",
        r"^null$",
    )
]
# Group and call events. Anchored to the whole message, and applied only to messages that
# the export marks as system lines with U+200E (WhatsApp iOS), so ordinary sentences such as
# 'only 2 days left' survive. Android system lines carry no sender and are dropped by the
# timestamp rule in parse_chat_content.
_SYSTEM_MARK = "‎"
_EVENTS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^.{1,80} created (the )?group .{0,120}$",
        r"^.{1,80} changed the subject (from .{0,120} )?to .{0,120}$",
        r"^.{1,80} changed (this group's|the group) (icon|description|settings)\.?$",
        r"^.{1,80} (joined using this group's invite link|joined the group|joined)$",
        r"^.{1,80} (was added|was removed|left|left the group|left the channel)$",
        r"^.{1,80} (added|removed) .{1,80}$",
        r"^(your )?security code (with .{1,80} )?changed\.?.{0,80}$",
        r"^.{1,80}'s security code changed\.?.{0,80}$",
        r"^.{1,80} pinned a message$",
        r"^missed (voice|video) call.{0,40}$",
        r"^you're now an admin$",
        r"^channel created$",
    )
]
_EDITED_TAG = re.compile(r"<\s*this message was edited\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class ChatMessage:
    timestamp: str  # 'YYYY-MM-DD HH:MM:SS', or the raw text if it could not be read
    sender: str
    body: str

    def line(self) -> str:
        return f"[{self.timestamp}] {self.sender}: {self.body}"


def _clean(line: str) -> str:
    return _NBSP.sub(" ", _INVISIBLE.sub("", line)).rstrip()


def _is_system(body: str, marked: bool = False) -> bool:
    text = body.strip()
    if any(p.search(text) for p in _PLACEHOLDERS):
        return True
    return marked and any(p.match(text) for p in _EVENTS)


def infer_day_first(dates: list[str]) -> bool:
    """IN-09: decide the date order once for the whole file."""
    firsts: list[int] = []
    seconds: list[int] = []
    for raw in dates:
        m = _DATE_PARTS.match(raw)
        if not m or len(m.group(1)) == 4:  # year-first dates say nothing about the order
            continue
        firsts.append(int(m.group(1)))
        seconds.append(int(m.group(2)))
    if any(n > 12 for n in firsts):
        return True
    if any(n > 12 for n in seconds):
        return False
    return True


def normalise_timestamp(date_str: str, time_str: str, day_first: bool) -> str | None:
    """'12/01/2026', '14:33' -> '2026-01-12 14:33:00'. None if the fields are not a date."""
    d = _DATE_PARTS.match(date_str.strip())
    t = _TIME_PARTS.match(time_str.strip())
    if not d or not t:
        return None
    a, b, c = d.groups()
    if len(a) == 4:
        year, month, day = int(a), int(b), int(c)
    elif day_first:
        day, month, year = int(a), int(b), int(c)
    else:
        month, day, year = int(a), int(b), int(c)
    if len(c) <= 2 and len(a) != 4:
        year += 2000
    hour, minute, second = int(t.group(1)), int(t.group(2)), int(t.group(3) or 0)
    meridiem = (t.group(4) or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "p" else 0)
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60 and second < 60):
        return None
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _header(line: str) -> tuple[str, str, str, str | None] | None:
    """(date, time, sender, body) for a message's first line; body None for TG two-line."""
    for pattern in (WA_ANDROID, BRACKET):
        m = pattern.match(line)
        if m:
            return m.group(1), m.group(2), m.group(3).strip(), m.group(4)
    m = TG_HEADER.match(line)
    if m:
        return m.group(2), m.group(3), m.group(1).strip(), None
    return None


def is_chat_export(content: str) -> bool:
    """True for a Telegram JSON export, or text whose first 20 lines hold 2 message headers."""
    stripped = content.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            return False
        return isinstance(data, dict) and isinstance(data.get("messages"), list)
    lines = [_clean(line) for line in content.splitlines()[:20]]
    return sum(1 for line in lines if _header(line)) >= 2


def parse_chat_content(content: str) -> list[ChatMessage]:
    """Parse a WhatsApp or Telegram text export. System lines and placeholders are dropped."""
    raw_lines = content.splitlines()
    lines = [_clean(line) for line in raw_lines]
    headers = [_header(line) for line in lines]
    day_first = infer_day_first([h[0] for h in headers if h])

    messages: list[ChatMessage] = []
    sender: str | None = None
    stamp = ""
    marked = False
    body: list[str] = []

    def flush() -> None:
        nonlocal sender, body
        if sender is not None:
            text = _EDITED_TAG.sub("", "\n".join(body)).strip()
            if text and not _is_system(text, marked):
                messages.append(ChatMessage(stamp, sender, text))
        sender, body = None, []

    i = 0
    while i < len(lines):
        line, head = lines[i], headers[i]
        if head:
            flush()
            date_str, time_str, who, first = head
            marked = _SYSTEM_MARK in raw_lines[i]
            if first is None:  # Telegram two-line: the body is on the next line
                i += 1
                first = lines[i] if i < len(lines) else ""
            stamp = normalise_timestamp(date_str, time_str, day_first) or f"{date_str} {time_str}"
            sender, body = who, [first]
        elif _TIMESTAMPED.match(line):
            flush()  # a sender-less system line such as '12/01/2026, 14:32 - X joined'
        elif sender is not None:
            body.append(line)
        i += 1
    flush()
    return messages


def parse_chat_json(content: str) -> list[ChatMessage]:
    """Parse a Telegram desktop JSON export (result.json)."""
    data = json.loads(content)
    messages: list[ChatMessage] = []
    for m in data.get("messages", []):
        if not isinstance(m, dict) or m.get("type") != "message":
            continue
        raw = m.get("text", "")
        if isinstance(raw, list):
            raw = "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in raw)
        text = str(raw).strip()
        if not text or _is_system(text):
            continue
        date = str(m.get("date", ""))
        messages.append(ChatMessage(date.replace("T", " "), str(m.get("from") or "Unknown"), text))
    return messages


def parse_chat(content: str) -> list[ChatMessage]:
    if content.lstrip().startswith("{"):
        return parse_chat_json(content)
    return parse_chat_content(content)
