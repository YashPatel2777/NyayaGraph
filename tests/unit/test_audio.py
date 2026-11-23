"""Audio loader (B3, E-08, T3.5). The transcriber is a fake here."""

from __future__ import annotations

import pytest

from app.ingestion.loaders import LoaderError
from app.ingestion.loaders.audio import NO_SPEECH_WARNING, load_audio, merge_segments
from app.services.media.transcription_service import AudioDecodeError, Segment, Transcript


def test_segments_with_short_pauses_merge() -> None:
    units = merge_segments(
        [
            Segment(0.0, 4.0, "I bought the phone"),
            Segment(5.5, 9.0, "on 2 January."),  # 1.5 s pause: merged
            Segment(12.0, 15.0, "It broke."),  # 3 s pause: new unit
        ]
    )
    assert [(u.text, u.start_sec, u.end_sec) for u in units] == [
        ("I bought the phone on 2 January.", 0.0, 9.0),
        ("It broke.", 12.0, 15.0),
    ]
    assert all(u.page is None for u in units)


def test_merge_stops_at_the_length_cap() -> None:
    long = "word " * 118  # 589 characters; with ' more words here' it passes 600
    units = merge_segments([Segment(0.0, 20.0, long.strip()), Segment(20.5, 25.0, "more words here")])
    assert len(units) == 2


class FakeTranscriber:
    device = "cuda"

    def __init__(self, result: Transcript | Exception) -> None:
        self.result = result
        self.calls: list[str | None] = []

    def transcribe(self, path: str, language: str | None = None) -> Transcript:
        self.calls.append(language)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def transcript(prob: float | None = 0.97, device: str = "cuda", segments=None) -> Transcript:
    segs = [Segment(0.0, 3.0, "The screen has green lines.")] if segments is None else segments
    return Transcript(segs, "en", prob, 3.21, device)


def test_document_fields_and_language_hint() -> None:
    fake = FakeTranscriber(transcript(prob=None))
    doc = load_audio("x.wav", fake, language="en")
    assert fake.calls == ["en"]
    assert (doc.media_type, doc.kind, doc.language, doc.duration_sec) == ("audio", "audio", "en", 3.21)
    assert doc.units[0].start_sec == 0.0 and doc.units[0].end_sec == 3.0
    assert doc.warnings == []


def test_low_detection_probability_warns() -> None:
    doc = load_audio("x.wav", FakeTranscriber(transcript(prob=0.41)))
    assert doc.warnings == ["audio: language detected as 'en' with probability 0.41"]


def test_cpu_fallback_is_reported() -> None:
    doc = load_audio("x.wav", FakeTranscriber(transcript(device="cpu")))
    assert doc.warnings == ["audio: transcribed on the CPU with the small model (E-08)"]


def test_no_speech_is_a_warning_not_a_failure() -> None:
    doc = load_audio("x.wav", FakeTranscriber(transcript(segments=[])))
    assert doc.units == []
    assert doc.warnings == [NO_SPEECH_WARNING]


def test_decode_failure_is_audio_decode_failed() -> None:
    with pytest.raises(LoaderError) as exc:
        load_audio("x.wav", FakeTranscriber(AudioDecodeError("InvalidDataError")))
    assert exc.value.code == "AUDIO_DECODE_FAILED"
