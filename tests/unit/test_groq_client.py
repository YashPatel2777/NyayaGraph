"""GroqClient: JSON mode, one repair retry (R-22), tenacity backoff and retry-after (1.2)."""

from __future__ import annotations

import httpx
import groq
import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.errors import AppError, ErrorCode
from app.services.llm.groq_client import MAX_ATTEMPTS, GroqClient


class Plan(BaseModel):
    intent: str
    search_query: str


GOOD = '{"intent": "LEGAL_ONLY", "search_query": "limitation period"}'


def _reply(content: str, inp: int = 10, out: int = 5) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
    )


def _rate_limit(retry_after: str | None) -> groq.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://groq.test"))
    return groq.RateLimitError("rate limited", response=resp, body=None)


def _server_error() -> groq.InternalServerError:
    resp = httpx.Response(503, request=httpx.Request("POST", "https://groq.test"))
    return groq.InternalServerError("unavailable", response=resp, body=None)


class FakeChat:
    """Stands in for ChatGroq: records bind kwargs and messages, replays scripted outcomes."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.bind_kwargs: list[dict] = []
        self.calls: list[list] = []

    def bind(self, **kwargs):
        self.bind_kwargs.append(kwargs)
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _client(chat: FakeChat, sleeps: Sleeps | None = None) -> GroqClient:
    return GroqClient("key", "openai/gpt-oss-120b", chat=chat, sleep=sleeps or Sleeps())


MESSAGES = [("system", "Return JSON."), ("user", "question")]


async def test_valid_reply_uses_json_mode_and_caller_parameters() -> None:
    chat = FakeChat([_reply(GOOD)])
    result = await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert result.value == Plan(intent="LEGAL_ONLY", search_query="limitation period")
    assert chat.bind_kwargs[0] == {
        "temperature": 0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    assert (result.prompt_tokens, result.completion_tokens, result.llm_calls) == (10, 5, 1)


async def test_invalid_json_gets_one_repair_retry_with_error_appended() -> None:
    chat = FakeChat([_reply("not json at all"), _reply(GOOD)])
    result = await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert result.value.intent == "LEGAL_ONLY"
    assert result.llm_calls == 2
    assert (result.prompt_tokens, result.completion_tokens) == (20, 10)
    second = chat.calls[1]
    assert second[:2] == MESSAGES
    assert second[2] == ("assistant", "not json at all")
    role, repair = second[3]
    assert role == "user" and "Validation error" in repair and "Expecting value" in repair


async def test_schema_violation_names_the_field_without_echoing_input() -> None:
    chat = FakeChat([_reply('{"intent": "SECRET-CASE-TEXT"}'), _reply(GOOD)])
    await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    repair = chat.calls[1][3][1]
    assert "search_query" in repair
    assert "SECRET-CASE-TEXT" not in repair


async def test_second_invalid_reply_is_502_invalid_structured_output() -> None:
    chat = FakeChat([_reply("{}"), _reply('{"intent": 1}')])
    with pytest.raises(AppError) as info:
        await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert info.value.code is ErrorCode.LLM_UPSTREAM_ERROR
    assert info.value.http_status == 502
    assert info.value.details == {"reason": "invalid_structured_output"}
    assert len(chat.calls) == 2  # exactly one repair retry


async def test_retry_after_header_is_honoured_on_429() -> None:
    chat = FakeChat([_rate_limit("7"), _reply(GOOD)])
    sleeps = Sleeps()
    await _client(chat, sleeps).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert sleeps.waits == [7.0]


async def test_429_without_retry_after_uses_jittered_backoff() -> None:
    chat = FakeChat([_rate_limit(None), _reply(GOOD)])
    sleeps = Sleeps()
    await _client(chat, sleeps).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert len(sleeps.waits) == 1 and 0 <= sleeps.waits[0] <= 2


async def test_backoff_is_full_jitter_under_2_4_8_16_and_stops_at_5_attempts() -> None:
    chat = FakeChat([_server_error() for _ in range(MAX_ATTEMPTS)])
    sleeps = Sleeps()
    with pytest.raises(AppError) as info:
        await _client(chat, sleeps).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert MAX_ATTEMPTS == 5
    assert len(chat.calls) == 5
    assert len(sleeps.waits) == 4
    for wait, ceiling in zip(sleeps.waits, [2, 4, 8, 16]):
        assert 0 <= wait <= ceiling
    assert info.value.code is ErrorCode.LLM_UPSTREAM_ERROR
    assert info.value.details == {"reason": "retries_exhausted"}


async def test_connection_error_is_retried() -> None:
    err = groq.APIConnectionError(request=httpx.Request("POST", "https://groq.test"))
    chat = FakeChat([err, _reply(GOOD)])
    result = await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert result.value.intent == "LEGAL_ONLY"


async def test_non_retryable_api_error_fails_fast_as_502() -> None:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://groq.test"))
    chat = FakeChat([groq.BadRequestError("bad", response=resp, body=None)])
    sleeps = Sleeps()
    with pytest.raises(AppError) as info:
        await _client(chat, sleeps).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert info.value.details == {"reason": "upstream_error"}
    assert sleeps.waits == [] and len(chat.calls) == 1


def test_real_chat_has_no_builtin_retries_and_30s_timeout() -> None:
    client = GroqClient("key", "openai/gpt-oss-120b")
    chat = client._chat
    assert chat.max_retries == 0
    assert chat.request_timeout == 30.0
    assert chat.reasoning_effort == "low"  # D-1


def test_non_reasoning_model_gets_no_reasoning_effort() -> None:
    client = GroqClient("key", "llama-3.3-70b-versatile")
    assert client._chat.reasoning_effort is None


# E-01 in a job: the stage resumes after the retries are exhausted (A-28).


async def test_resuming_waits_checks_the_case_and_resumes() -> None:
    chat = FakeChat([_server_error() for _ in range(MAX_ATTEMPTS)] + [_reply(GOOD)])
    sleeps = Sleeps()
    checks = []

    async def checkpoint() -> None:
        checks.append(1)

    result = await _client(chat, sleeps).complete_json_resuming(
        MESSAGES, Plan, temperature=0, max_tokens=200, checkpoint=checkpoint, wait_s=30.0
    )
    assert result.value.intent == "LEGAL_ONLY"
    assert checks == [1]
    assert sleeps.waits[-1] == 30.0


async def test_resuming_stops_when_the_checkpoint_raises() -> None:
    chat = FakeChat([_server_error() for _ in range(MAX_ATTEMPTS)])

    async def checkpoint() -> None:
        raise RuntimeError("case deleted")

    with pytest.raises(RuntimeError):
        await _client(chat).complete_json_resuming(
            MESSAGES, Plan, temperature=0, max_tokens=200, checkpoint=checkpoint
        )


async def test_resuming_gives_up_after_max_resumes_and_never_resumes_invalid_output() -> None:
    chat = FakeChat([_server_error() for _ in range(MAX_ATTEMPTS * 2)])

    async def checkpoint() -> None:
        return None

    with pytest.raises(AppError) as exc:
        await _client(chat).complete_json_resuming(
            MESSAGES, Plan, temperature=0, max_tokens=200, checkpoint=checkpoint, max_resumes=1
        )
    assert exc.value.details == {"reason": "retries_exhausted"}

    chat = FakeChat([_reply("not json"), _reply("still not json")])
    with pytest.raises(AppError) as exc:
        await _client(chat).complete_json_resuming(
            MESSAGES, Plan, temperature=0, max_tokens=200, checkpoint=checkpoint
        )
    assert exc.value.details == {"reason": "invalid_structured_output"}
    assert len(chat.calls) == 2  # the R-22 repair retry only


def _json_validate_failed(failed_generation: str | None) -> groq.BadRequestError:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://groq.test"))
    body = {"error": {"message": "Failed to generate JSON", "type": "invalid_request_error",
                      "code": "json_validate_failed", "failed_generation": failed_generation}}
    return groq.BadRequestError("bad", response=resp, body=body)


async def test_json_validate_failed_takes_the_repair_path() -> None:
    # A-42, E-02: Groq's 400 for invalid JSON-mode output is invalid structured output.
    chat = FakeChat([_json_validate_failed('{"intent": "LEGAL_ONLY", '),
                     _reply('{"intent": "LEGAL_ONLY", "search_query": "q"}')])
    result = await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert result.value.intent == "LEGAL_ONLY" and result.llm_calls == 2
    repair = chat.calls[1]
    assert repair[-2] == ("assistant", '{"intent": "LEGAL_ONLY", ')
    assert repair[-1][0] == "user"


async def test_json_validate_failed_twice_is_invalid_structured_output() -> None:
    chat = FakeChat([_json_validate_failed(None), _json_validate_failed(None)])
    with pytest.raises(AppError) as info:
        await _client(chat).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert info.value.details == {"reason": "invalid_structured_output"}
    assert chat.calls[1][-2] == ("assistant", "(reply rejected as invalid JSON)")


async def test_other_upstream_errors_log_their_code_only(caplog) -> None:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://groq.test"))
    err = groq.BadRequestError("bad", response=resp, body={"error": {"code": "context_length_exceeded",
                                                                        "message": "secret case text"}})
    with caplog.at_level("WARNING"), pytest.raises(AppError):
        await _client(FakeChat([err])).complete_json(MESSAGES, Plan, temperature=0, max_tokens=200)
    assert "code=context_length_exceeded" in caplog.text and "secret case text" not in caplog.text
