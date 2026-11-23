"""Guardrails hook (1.3). Guardrails are out of scope for v1: guard() is a no-op while
ENABLE_GUARDRAILS is false, and the agent only calls it when the flag is true (C2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class GuardResult(BaseModel):
    blocked: bool
    reply: Optional[str] = None


def guard(text: str, user_id: str) -> GuardResult:
    return GuardResult(blocked=False, reply=None)
