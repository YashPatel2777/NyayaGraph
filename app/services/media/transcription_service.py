"""faster-whisper transcription (section 0, B3, E-08).

- WHISPER_MODEL (default medium) on CUDA with float16, vad_filter=True, timestamped segments.
- E-08: a CUDA out-of-memory error retries once on the CPU with compute_type int8 and the
  small model. Audio that cannot be decoded fails with AUDIO_DECODE_FAILED. Without a
  language hint the language is detected and returned with its probability; the caller
  warns below 0.5.
- Audio is decoded once with PyAV (faster_whisper.decode_audio), so a decode failure is
  told apart from an inference failure.
- The FastAPI lifespan calls load() once (1.2); the CPU fallback model loads only when
  needed. Calls are blocking and use the GPU: the caller holds the GPU semaphore and runs
  them through anyio.to_thread.run_sync.
- R-04: logs carry durations, segment counts and timings, never the transcript.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
GPU_DEVICE, GPU_COMPUTE = "cuda", "float16"
CPU_DEVICE, CPU_COMPUTE, CPU_MODEL = "cpu", "int8", "small"  # E-08


class AudioDecodeError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment]
    language: str | None
    language_probability: float | None
    duration_sec: float
    device: str


def is_out_of_memory(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


class TranscriptionService:
    def __init__(self, model_name: str, *, device: str = GPU_DEVICE) -> None:
        self.model_name = model_name
        self.device = device
        self._models: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()

    def _model(self, name: str, device: str, compute: str) -> Any:
        key = (name, device, compute)
        with self._lock:
            if key not in self._models:
                from faster_whisper import WhisperModel

                start = time.perf_counter()
                self._models[key] = WhisperModel(name, device=device, compute_type=compute)
                logger.info(
                    "whisper_loaded model=%s device=%s compute=%s ms=%d",
                    name, device, compute, (time.perf_counter() - start) * 1000,
                )
            return self._models[key]

    def _primary(self) -> tuple[str, str, str]:
        if self.device == GPU_DEVICE:
            return self.model_name, GPU_DEVICE, GPU_COMPUTE
        return self.model_name, CPU_DEVICE, CPU_COMPUTE

    def load(self) -> None:
        self._model(*self._primary())

    @staticmethod
    def decode(path: str) -> Any:  # -> numpy float32 array at 16 kHz
        from faster_whisper import decode_audio

        try:
            audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # PyAV raises a family of FFmpeg errors
            raise AudioDecodeError(type(exc).__name__) from exc
        if audio is None or len(audio) == 0:
            raise AudioDecodeError("no audio samples")
        return audio

    def _run(self, model: Any, audio: Any, language: str | None) -> tuple[list[Segment], Any]:
        segments, info = model.transcribe(audio, language=language, vad_filter=True)
        # Inference runs while the generator is consumed, so OOM can surface here.
        return [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments], info

    def transcribe(self, path: str, language: str | None = None) -> Transcript:
        audio = self.decode(path)
        duration = len(audio) / SAMPLE_RATE
        name, device, compute = self._primary()
        start = time.perf_counter()
        try:
            segments, info = self._run(self._model(name, device, compute), audio, language)
        except RuntimeError as exc:
            if device != GPU_DEVICE or not is_out_of_memory(exc):
                raise
            # E-08: retry once on the CPU with int8 and the small model.
            logger.warning("whisper_oom_fallback model=%s duration_sec=%.1f", name, duration)
            device = CPU_DEVICE
            segments, info = self._run(
                self._model(CPU_MODEL, CPU_DEVICE, CPU_COMPUTE), audio, language
            )
        logger.info(
            "whisper_ok device=%s duration_sec=%.1f segments=%d ms=%d",
            device, duration, len(segments), (time.perf_counter() - start) * 1000,
        )
        return Transcript(
            segments=[s for s in segments if s.text],
            language=getattr(info, "language", None) or language,
            language_probability=None if language else getattr(info, "language_probability", None),
            duration_sec=duration,
            device=device,
        )
