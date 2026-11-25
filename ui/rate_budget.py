"""Groq per-minute token budget estimate for the UI's cooldown note (SC-24, O-6).

Pure functions, no Streamlit, so they can be unit-tested.
"""

from __future__ import annotations

import math

WINDOW_S = 60


def tokens_used(answer: dict) -> int:
    """Prompt plus completion tokens from an AnalysisResponse trace (3.5), or 0."""
    llm = (answer.get("trace") or {}).get("llm") or {}
    return int(llm.get("prompt_tokens", 0)) + int(llm.get("completion_tokens", 0))


def seconds_until_ready(usage: list[tuple[float, int]], now: float, tpm: int) -> int:
    """Seconds until one more question fits the per-minute token budget.

    `usage` holds (monotonic time, tokens) per answered question, oldest first. The next
    question is assumed to cost as much as the last one; usage older than the window no
    longer counts. An estimate only: other clients of the same Groq key are not seen.
    """
    recent = sorted((t, n) for t, n in usage if now - t < WINDOW_S)
    if not recent:
        return 0
    need = usage[-1][1]
    total = sum(n for _, n in recent)
    if total + need <= tpm:
        return 0
    for t, n in recent:
        total -= n
        if total + need <= tpm:
            return max(0, math.ceil(t + WINDOW_S - now))
    return max(0, math.ceil(recent[-1][0] + WINDOW_S - now))
