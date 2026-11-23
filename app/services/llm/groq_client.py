"""Groq chat client: JSON mode, Pydantic validation with one repair retry, tenacity backoff.

- 1.2: every call is retried with exponential backoff 2, 4, 8, 16, 32 s with full jitter, at
  most 5 attempts; on HTTP 429 the retry-after header is honoured.
- R-22, E-02: output is validated with Pydantic; on failure one retry with the validation
  error appended; a second failure raises 502 LLM_UPSTREAM_ERROR, reason
  'invalid_structured_output'.
- 4.2: per-call timeout 30 s. The LLM has no tools (R-06).
- R-04: logs carry the model, attempt counts, lengths and timings only.

The client is async (langchain-groq's ainvoke), so nothing blocks the event loop (1.2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

import groq
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.errors import AppError, ErrorCode
from app.prompts.repair import REPAIR_PROMPT

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

Message = tuple[str, str]  # (role, content); role is system, user or assistant

GROQ_TIMEOUT_S = 30.0  # 4.2
MAX_ATTEMPTS = 5  # 1.2
# E-01 in a job: after the 1.2 retries are exhausted the stage resumes (A-28).
RESUME_WAIT_S = 30.0
MAX_RESUMES = 20
# Full jitter over 2, 4, 8, 16, 32 s (1.2).
_backoff = wait_random_exponential(multiplier=2, max=32)

# D-1: gpt-oss models spend output tokens on reasoning, and those count against max_tokens.
# Low effort keeps reasoning inside the R-19 token caps (observed: 45 reasoning tokens for a
# planner-sized call at low effort, 138 at the default).
REASONING_EFFORT = "low"
REASONING_MODEL_PREFIXES = ("openai/gpt-oss",)

RETRYABLE = (
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,  # includes APITimeoutError
)


JSON_VALIDATE_FAILED = "json_validate_failed"  # Groq's 400 code for invalid JSON-mode output


class _InvalidJsonReply(Exception):
    """Groq refused the reply as invalid JSON (400 json_validate_failed, A-42)."""

    def __init__(self, failed_generation: str) -> None:
        super().__init__("Groq rejected the reply as invalid JSON")
        self.failed_generation = failed_generation


def _error_field(exc: BaseException, name: str) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get(name) is not None:
            return str(inner[name])
    return ""


def _error_code(exc: BaseException) -> str:
    return _error_field(exc, "code") or "unknown"


@dataclass
class LLMResult(Generic[M]):
    value: M
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int


def _retry_after_seconds(exc: BaseException) -> float | None:
    if not isinstance(exc, groq.RateLimitError):
        return None
    raw = exc.response.headers.get("retry-after") if exc.response is not None else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _wait(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = _retry_after_seconds(exc) if exc else None
    if retry_after is not None:
        return retry_after  # 1.2: honour retry-after on 429
    return _backoff(retry_state)


def _validation_error_text(exc: ValidationError | json.JSONDecodeError | ValueError) -> str:
    if isinstance(exc, ValidationError):
        # include_input=False: the model's rejected values are not repeated back.
        lines = [
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        ]
        return "\n".join(lines)
    return str(exc)


def _parse(content: str, schema: type[M]) -> M:
    data = json.loads(content)
    return schema.model_validate(data)


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        chat: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self._sleep = sleep
        if chat is None:
            from langchain_groq import ChatGroq

            extra: dict[str, Any] = {}
            if model.startswith(REASONING_MODEL_PREFIXES):
                extra["reasoning_effort"] = REASONING_EFFORT
            # max_retries=0: tenacity below owns the retry policy (1.2).
            chat = ChatGroq(
                model=model, api_key=api_key, timeout=GROQ_TIMEOUT_S, max_retries=0, **extra
            )
        self._chat = chat

    async def _invoke(
        self, messages: Sequence[Message], temperature: float, max_tokens: int
    ) -> AIMessage:
        bound = self._chat.bind(
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        retrying = AsyncRetrying(
            retry=retry_if_exception(lambda e: isinstance(e, RETRYABLE)),
            wait=_wait,
            stop=stop_after_attempt(MAX_ATTEMPTS),
            sleep=self._sleep,
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "groq_retry model=%s attempt=%d error=%s",
                self.model,
                rs.attempt_number,
                type(rs.outcome.exception()).__name__ if rs.outcome else "unknown",
            ),
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await bound.ainvoke(list(messages))
        except RETRYABLE as exc:
            # E-01: retries exhausted.
            raise AppError(
                ErrorCode.LLM_UPSTREAM_ERROR, details={"reason": "retries_exhausted"}
            ) from exc
        except groq.APIError as exc:
            code = _error_code(exc)
            if code == JSON_VALIDATE_FAILED:
                # E-02, A-42: Groq rejected the model's reply as invalid JSON; this is invalid
                # structured output, so complete_json runs the R-22 repair round.
                raise _InvalidJsonReply(_error_field(exc, "failed_generation")) from exc
            # R-04: the error code only, never the body text.
            logger.warning("groq_upstream_error model=%s status=%s code=%s",
                           self.model, getattr(exc, "status_code", None), code)
            raise AppError(
                ErrorCode.LLM_UPSTREAM_ERROR, details={"reason": "upstream_error"}
            ) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete_json(
        self,
        messages: Sequence[Message],
        schema: type[M],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult[M]:
        """Call Groq in JSON mode and validate the reply against schema (R-22).

        Temperatures and token caps come from the caller (R-19).
        """
        conversation = list(messages)
        prompt_tokens = completion_tokens = calls = 0
        start = time.perf_counter()
        last_error: Exception | None = None

        for repair_round in range(2):  # the first call plus one repair retry (R-22)
            try:
                reply = await self._invoke(conversation, temperature, max_tokens)
            except _InvalidJsonReply as rejected:  # E-02, A-42: same path as invalid output
                reply = AIMessage(content=rejected.failed_generation)
                reply_error: Exception | None = ValueError(str(rejected))
            else:
                reply_error = None
            calls += 1
            usage = reply.usage_metadata or {}
            prompt_tokens += int(usage.get("input_tokens", 0))
            completion_tokens += int(usage.get("output_tokens", 0))
            content = reply.content if isinstance(reply.content, str) else ""
            try:
                if reply_error is not None:
                    raise reply_error
                value = _parse(content, schema)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "groq_invalid_output model=%s schema=%s round=%d reply_chars=%d",
                    self.model,
                    schema.__name__,
                    repair_round,
                    len(content),
                )
                conversation += [
                    ("assistant", content or "(reply rejected as invalid JSON)"),
                    ("user", REPAIR_PROMPT.format(validation_error=_validation_error_text(exc))),
                ]
                continue
            logger.info(
                "groq_ok model=%s schema=%s calls=%d prompt_tokens=%d completion_tokens=%d ms=%d",
                self.model,
                schema.__name__,
                calls,
                prompt_tokens,
                completion_tokens,
                (time.perf_counter() - start) * 1000,
            )
            return LLMResult(value, prompt_tokens, completion_tokens, calls)

        raise AppError(
            ErrorCode.LLM_UPSTREAM_ERROR, details={"reason": "invalid_structured_output"}
        ) from last_error

    async def complete_json_resuming(
        self,
        messages: Sequence[Message],
        schema: type[M],
        *,
        temperature: float,
        max_tokens: int,
        checkpoint: Callable[[], Awaitable[None]],
        wait_s: float = RESUME_WAIT_S,
        max_resumes: int = MAX_RESUMES,
    ) -> LLMResult[M]:
        """complete_json for ingestion jobs (E-01, A-28).

        When Groq retries are exhausted the job stays running: wait wait_s, call checkpoint()
        (which raises to stop, for example when the case is being deleted, E-10), and resume.
        After max_resumes the AppError is raised. Invalid output (E-02) and other upstream
        errors are raised at once.
        """
        for resume in range(max_resumes + 1):
            try:
                return await self.complete_json(
                    messages, schema, temperature=temperature, max_tokens=max_tokens
                )
            except AppError as exc:
                if exc.details.get("reason") != "retries_exhausted" or resume == max_resumes:
                    raise
                logger.warning(
                    "groq_stage_resume model=%s schema=%s resume=%d wait_s=%.0f",
                    self.model, schema.__name__, resume + 1, wait_s,
                )
                await self._sleep(wait_s)
                await checkpoint()
        raise AssertionError("unreachable")  # pragma: no cover
