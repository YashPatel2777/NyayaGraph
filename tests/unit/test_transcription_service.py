"""TranscriptionService: E-08 out-of-memory fallback and decode failures. Models are fakes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.media import transcription_service as ts
from app.services.media.transcription_service import AudioDecodeError, TranscriptionService


class FakeModel:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.kwargs: dict = {}

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs

        def gen():
            if self.fail:  # inference errors surface while the generator is consumed
                raise self.fail
            yield SimpleNamespace(start=0.0, end=2.0, text=" Hello seller ")
            yield SimpleNamespace(start=2.5, end=3.0, text="  ")

        return gen(), SimpleNamespace(language="en", language_probability=0.88)


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch):
    svc = TranscriptionService("medium")
    models: dict = {}

    def fake_model(name, device, compute):
        return models.setdefault((name, device, compute), FakeModel())

    monkeypatch.setattr(svc, "_model", fake_model)
    monkeypatch.setattr(TranscriptionService, "decode", staticmethod(lambda path: np.zeros(48000, np.float32)))
    return svc, models


def test_gpu_path_uses_vad_and_returns_segments(service) -> None:
    svc, models = service
    out = svc.transcribe("x.wav")
    model = models[("medium", "cuda", "float16")]
    assert model.kwargs == {"language": None, "vad_filter": True}
    assert [(s.start, s.end, s.text) for s in out.segments] == [(0.0, 2.0, "Hello seller")]
    assert (out.language, out.language_probability, out.duration_sec, out.device) == ("en", 0.88, 3.0, "cuda")


def test_language_hint_means_no_detection_probability(service) -> None:
    svc, _ = service
    out = svc.transcribe("x.wav", language="hi")
    assert out.language_probability is None


def test_out_of_memory_retries_once_on_cpu_small_int8(service) -> None:
    svc, models = service
    models[("medium", "cuda", "float16")] = FakeModel(fail=RuntimeError("CUDA failed with error out of memory"))
    out = svc.transcribe("x.wav")
    assert out.device == "cpu"
    assert ("small", "cpu", "int8") in models


def test_other_runtime_errors_are_not_retried(service) -> None:
    svc, models = service
    models[("medium", "cuda", "float16")] = FakeModel(fail=RuntimeError("cuDNN failed"))
    with pytest.raises(RuntimeError):
        svc.transcribe("x.wav")
    assert ("small", "cpu", "int8") not in models


def test_real_decoder_rejects_garbage(tmp_path: Path) -> None:
    path = tmp_path / "original.ogg"
    path.write_bytes(b"OggS\x00" + b"\x13" * 400)
    with pytest.raises(AudioDecodeError):
        TranscriptionService.decode(str(path))


def test_real_decoder_reads_fixture_wav() -> None:
    wav = Path(__file__).resolve().parents[1] / "fixtures" / "case" / "clean" / "voice_note.wav"
    audio = TranscriptionService.decode(str(wav))
    assert audio.dtype == np.float32
    assert 15 < len(audio) / ts.SAMPLE_RATE < 60
