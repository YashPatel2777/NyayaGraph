import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import logfire
except ImportError:
    class _MockLogfire:
        def span(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
    logfire = _MockLogfire()


@dataclass
class ChatMessage:
    timestamp: str  # Normalized ISO string e.g., "2026-01-15 14:32:00"
    raw_timestamp: str
    sender: str
    body: str

    def format_thread_line(self) -> str:
        return f"[{self.timestamp}] {self.sender}: {self.body}"


# System patterns to strip
SYSTEM_PATTERNS = [
    re.compile(r"Messages and calls are end-to-end encrypted.*", re.IGNORECASE),
    re.compile(r"This chat is with a business account.*", re.IGNORECASE),
    re.compile(r"<Media omitted>", re.IGNORECASE),
    re.compile(r"<\s*attached:[^>]+>", re.IGNORECASE),
    re.compile(r"image omitted|video omitted|audio omitted|document omitted|sticker omitted|GIF omitted", re.IGNORECASE),
    re.compile(r"Waiting for this message\..*", re.IGNORECASE),
    re.compile(r".*created group.*", re.IGNORECASE),
    re.compile(r".*changed the subject to.*", re.IGNORECASE),
    re.compile(r".*changed the group (icon|description|settings).*", re.IGNORECASE),
    re.compile(r".*(joined using this group's invite link|joined the group|joined the channel).*", re.IGNORECASE),
    re.compile(r".*(was added|was removed|left the group|left the channel).*", re.IGNORECASE),
    re.compile(r".*security code changed.*", re.IGNORECASE),
    re.compile(r".*pinned a message.*", re.IGNORECASE),
    re.compile(r".*missed (voice|video) call.*", re.IGNORECASE),
    re.compile(r".*You're now an admin.*", re.IGNORECASE),
    re.compile(r".*channel created.*", re.IGNORECASE),
]

# WhatsApp Regexes
# 1. Android format: 12/01/2026, 14:32 - Sender: Message or 12/01/26, 2:32 pm - Sender: Message
WA_ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\s*-\s*([^:]+?):\s*(.*)$"
)

# 2. iOS format: [12/01/2026, 14:32:05] Sender: Message or [1/15/26, 2:32:05 PM] Sender: Message
WA_IOS_PATTERN = re.compile(
    r"^\[(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\]\s*([^:]+?):\s*(.*)$"
)

# Telegram Regexes
# 1. Desktop TXT export: [15.01.2026 14:32:05] Sender: Message
TG_BRACKET_PATTERN = re.compile(
    r"^\[(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+?):\s*(.*)$"
)

# 2. Telegram two-line format: Sender, [15.01.2026 14:32]
TG_HEADER_PATTERN = re.compile(
    r"^([^,\n]+),\s*\[(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?)\]$"
)


def _clean_unicode_artifacts(text: str) -> str:
    """Strip invisible zero-width and directional control characters added by WhatsApp/Telegram."""
    return re.sub(r"[\u200e\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", text).strip()


def _is_system_message(text: str) -> bool:
    """Check if message matches any known system notification pattern."""
    clean = text.strip()
    for pat in SYSTEM_PATTERNS:
        if pat.search(clean):
            return True
    return False


def _normalize_timestamp(date_str: str, time_str: Optional[str] = None) -> str:
    """Normalize diverse chat date/time strings into YYYY-MM-DD HH:MM:SS format."""
    combined = f"{date_str} {time_str}".strip() if time_str else date_str.strip()
    
    # Try common legal & messaging date formats
    formats = [
        "%d/%m/%Y, %H:%M:%S",
        "%d/%m/%Y, %H:%M",
        "%d/%m/%y, %H:%M:%S",
        "%d/%m/%y, %H:%M",
        "%m/%d/%Y, %I:%M:%S %p",
        "%m/%d/%Y, %I:%M %p",
        "%m/%d/%y, %I:%M:%S %p",
        "%m/%d/%y, %I:%M %p",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(combined, fmt)
            # Handle 2-digit years correctly if needed
            if dt.year < 100:
                dt = dt.replace(year=2000 + dt.year)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
            
    # Return raw combined if none match exactly
    return combined


def is_chat_export(content: str) -> bool:
    """Helper to detect if a plain text string looks like a WhatsApp or Telegram export."""
    first_lines = content.splitlines()[:20]
    matched = 0
    for line in first_lines:
        line = _clean_unicode_artifacts(line)
        if (
            WA_ANDROID_PATTERN.match(line)
            or WA_IOS_PATTERN.match(line)
            or TG_BRACKET_PATTERN.match(line)
            or TG_HEADER_PATTERN.match(line)
        ):
            matched += 1
    return matched >= 2


def parse_chat_content(raw_text: str) -> List[ChatMessage]:
    """
    Parse WhatsApp and Telegram raw text export content using regular expressions.
    Extracts timestamps, sender handles, and message body while stripping system logs.
    Handles multiline message continuations gracefully.
    """
    lines = raw_text.splitlines()
    messages: List[ChatMessage] = []
    
    current_sender: Optional[str] = None
    current_timestamp: Optional[str] = None
    current_raw_timestamp: Optional[str] = None
    current_body_lines: List[str] = []

    def flush_current():
        nonlocal current_sender, current_timestamp, current_raw_timestamp, current_body_lines
        if current_sender and current_body_lines:
            body = "\n".join(current_body_lines).strip()
            # Strip edited tags or empty body
            body = re.sub(r"<\s*This message was edited\s*>", "", body, flags=re.IGNORECASE).strip()
            if body and not _is_system_message(body):
                messages.append(
                    ChatMessage(
                        timestamp=current_timestamp or "UNKNOWN_TIME",
                        raw_timestamp=current_raw_timestamp or "",
                        sender=current_sender,
                        body=body,
                    )
                )
        current_sender = None
        current_timestamp = None
        current_raw_timestamp = None
        current_body_lines = []

    i = 0
    while i < len(lines):
        line = _clean_unicode_artifacts(lines[i])
        if not line:
            if current_body_lines:
                current_body_lines.append("")
            i += 1
            continue

        # Check WhatsApp iOS: [12/01/2026, 14:32:05] Sender: Body
        match_ios = WA_IOS_PATTERN.match(line)
        if match_ios:
            flush_current()
            d_part, t_part, sender, body = match_ios.groups()
            if not _is_system_message(body) and not _is_system_message(sender):
                current_sender = sender.strip()
                current_raw_timestamp = f"{d_part} {t_part}"
                current_timestamp = _normalize_timestamp(d_part, t_part)
                current_body_lines = [body]
            i += 1
            continue

        # Check WhatsApp Android: 12/01/2026, 14:32 - Sender: Body
        match_android = WA_ANDROID_PATTERN.match(line)
        if match_android:
            flush_current()
            d_part, t_part, sender, body = match_android.groups()
            if not _is_system_message(body) and not _is_system_message(sender):
                current_sender = sender.strip()
                current_raw_timestamp = f"{d_part} {t_part}"
                current_timestamp = _normalize_timestamp(d_part, t_part)
                current_body_lines = [body]
            i += 1
            continue

        # Check Telegram Bracket: [15.01.2026 14:32:05] Sender: Body
        match_tg_bracket = TG_BRACKET_PATTERN.match(line)
        if match_tg_bracket:
            flush_current()
            raw_ts, sender, body = match_tg_bracket.groups()
            if not _is_system_message(body) and not _is_system_message(sender):
                current_sender = sender.strip()
                current_raw_timestamp = raw_ts.strip()
                current_timestamp = _normalize_timestamp(raw_ts)
                current_body_lines = [body]
            i += 1
            continue

        # Check Telegram Header: Sender, [15.01.2026 14:32]
        match_tg_header = TG_HEADER_PATTERN.match(line)
        if match_tg_header and i + 1 < len(lines):
            flush_current()
            sender, raw_ts = match_tg_header.groups()
            next_line = _clean_unicode_artifacts(lines[i + 1])
            if not _is_system_message(next_line) and not _is_system_message(sender):
                current_sender = sender.strip()
                current_raw_timestamp = raw_ts.strip()
                current_timestamp = _normalize_timestamp(raw_ts)
                current_body_lines = [next_line]
            i += 2
            continue

        # System message without sender match (e.g. "12/01/2026, 14:32 - Messages and calls...")
        if _is_system_message(line):
            flush_current()
            i += 1
            continue

        # Multiline continuation
        if current_sender:
            current_body_lines.append(line)
        
        i += 1

    flush_current()
    return messages


def parse_chat_json(json_content: str) -> List[ChatMessage]:
    """Parse Telegram desktop JSON export format (result.json)."""
    data = json.loads(json_content)
    messages_data = data.get("messages", [])
    messages: List[ChatMessage] = []

    for m in messages_data:
        if m.get("type") != "message":
            continue
        sender = m.get("from") or m.get("actor") or "Unknown"
        date_str = m.get("date", "")
        text_val = m.get("text", "")

        # Text in telegram json can be str or list of formatting objects
        if isinstance(text_val, list):
            text_str = "".join(part if isinstance(part, str) else part.get("text", "") for part in text_val)
        else:
            text_str = str(text_val)

        text_str = text_str.strip()
        if not text_str or _is_system_message(text_str):
            continue

        messages.append(
            ChatMessage(
                timestamp=date_str.replace("T", " ") if "T" in date_str else date_str,
                raw_timestamp=date_str,
                sender=sender,
                body=text_str,
            )
        )
    return messages


def parse_chat(file_path: str) -> str:
    """
    Main loader function to convert WhatsApp or Telegram export files into clean conversation thread formats.
    Strips system logs and normalizes dialogue threads.
    
    Returns clean formatted conversation thread:
        [2026-01-15 14:32:00] Alice: Message text
        [2026-01-15 14:33:10] Bob: Response text
    """
    with logfire.span("💬 Chat Log Parsing", filename=file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Chat export file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Check if JSON
            if file_path.lower().endswith(".json"):
                try:
                    messages = parse_chat_json(content)
                except Exception:
                    messages = parse_chat_content(content)
            else:
                messages = parse_chat_content(content)

            if not messages:
                logfire.warning(f"No valid chat messages parsed from {file_path}. Possibly pure system log or unsupported format.")
                return ""

            thread_lines = [m.format_thread_line() for m in messages]
            clean_thread = "\n\n".join(thread_lines)

            logfire.info(f"✅ Successfully parsed {len(messages)} chat messages from {file_path}.")
            return clean_thread

        except Exception as e:
            logfire.error(f"❌ Chat Parse Failed for {file_path}: {e}")
            raise e


def parse_chat_structured(file_path: str) -> List[ChatMessage]:
    """Parse chat export and return list of structured ChatMessage objects."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if file_path.lower().endswith(".json"):
        try:
            return parse_chat_json(content)
        except Exception:
            return parse_chat_content(content)
    return parse_chat_content(content)
