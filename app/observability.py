"""Logfire setup with scrubbing on (R-04).

Logs, spans and traces carry ids, lengths and timings only. Scrubbing is a second line of
defence: the default Logfire patterns stay on and the extra patterns below redact any
attribute whose name suggests case text, transcripts or question text.
"""

from __future__ import annotations

import logging

import logfire

from app.config import Settings

SERVICE_NAME = "legal-graphrag"

# Attribute names that may carry case text, transcripts or question text (R-04).
SCRUB_PATTERNS = [
    "text",
    "question",
    "query",
    "transcript",
    "content",
    "prompt",
    "message",
    "excerpt",
    "summary",
    "fact",
]


def configure_observability(settings: Settings) -> None:
    token = settings.LOGFIRE_TOKEN.get_secret_value() if settings.LOGFIRE_TOKEN else None
    logfire.configure(
        token=token,
        send_to_logfire="if-token-present",
        service_name=SERVICE_NAME,
        environment=settings.APP_ENV,
        # R-04: scrubbing stays on. Never pass scrubbing=False.
        scrubbing=logfire.ScrubbingOptions(extra_patterns=SCRUB_PATTERNS),
        # Do not record the arguments of logged calls; they may carry text.
        inspect_arguments=False,
    )
    root = logging.getLogger()
    if not any(isinstance(h, logfire.LogfireLoggingHandler) for h in root.handlers):
        root.addHandler(logfire.LogfireLoggingHandler())
    root.setLevel(logging.INFO)
