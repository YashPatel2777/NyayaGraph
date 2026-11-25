"""SC-24: cooldown estimate for Groq's free-tier tokens per minute (O-6)."""

from __future__ import annotations

from ui.rate_budget import seconds_until_ready, tokens_used

TPM = 8000


def test_no_usage_is_ready():
    assert seconds_until_ready([], now=100.0, tpm=TPM) == 0


def test_one_small_question_leaves_room_for_another():
    assert seconds_until_ready([(100.0, 3500)], now=101.0, tpm=TPM) == 0


def test_one_large_question_waits_until_it_leaves_the_window():
    # 5,800 + another 5,800 > 8,000: wait until the first is 60 s old.
    assert seconds_until_ready([(100.0, 5800)], now=121.0, tpm=TPM) == 39
    assert seconds_until_ready([(100.0, 5800)], now=160.0, tpm=TPM) == 0


def test_waits_only_for_enough_old_usage_to_expire():
    usage = [(100.0, 2000), (130.0, 3000)]  # next estimated at 3,000; 5,000 + 3,000 = 8,000 fits
    assert seconds_until_ready(usage, now=131.0, tpm=TPM) == 0
    usage = [(100.0, 3000), (130.0, 4000)]  # 7,000 + 4,000 > 8,000; after the first expires: 8,000 fits
    assert seconds_until_ready(usage, now=131.0, tpm=TPM) == 29


def test_question_larger_than_the_budget_waits_for_the_whole_window():
    assert seconds_until_ready([(100.0, 9000)], now=110.0, tpm=TPM) == 50


def test_tokens_used_reads_the_trace():
    assert tokens_used({"trace": {"llm": {"prompt_tokens": 5343, "completion_tokens": 489}}}) == 5832
    assert tokens_used({"trace": None}) == 0
    assert tokens_used({}) == 0
