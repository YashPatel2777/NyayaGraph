"""Audio uploads (B3, E-08, SC-4).

faster-whisper returns timestamped segments (services/media/transcription_service.py).
Neighbouring segments separated by a pause under 2 seconds are merged into one unit, as in
the prototype (07 T3.5), up to MERGE_MAX_CHARS so each chunk's start and end seconds stay
close to its text. Speaker diarization is deferred (SC-4): units carry no speaker.

Blocking, and uses the GPU: the caller holds the GPU semaphore and runs this through
anyio.to_thread.run_sync (1.2).
"""

from __future__ import annotations

from app.ingestion.loaders import LoadedDocument, LoaderError, TextUnit
from app.services.media.transcription_service import (
    AudioDecodeError,
    Segment,
    TranscriptionService,
)

MERGE_GAP_SEC = 2.0
MERGE_MAX_CHARS = 600
MIN_LANGUAGE_PROBABILITY = 0.5  # E-08
NO_SPEECH_WARNING = "audio: no speech was detected"


def merge_segments(segments: list[Segment]) -> list[TextUnit]:
    units: list[TextUnit] = []
    for seg in segments:
        if units:
            last = units[-1]
            gap = seg.start - (last.end_sec or 0.0)
            if gap < MERGE_GAP_SEC and len(last.text) + 1 + len(seg.text) <= MERGE_MAX_CHARS:
                units[-1] = TextUnit(
                    f"{last.text} {seg.text}", start_sec=last.start_sec, end_sec=seg.end
                )
                continue
        units.append(TextUnit(seg.text, start_sec=seg.start, end_sec=seg.end))
    return units


def load_audio(
    path: str, transcriber: TranscriptionService, language: str | None = None
) -> LoadedDocument:
    try:
        transcript = transcriber.transcribe(path, language)
    except AudioDecodeError as exc:
        raise LoaderError("AUDIO_DECODE_FAILED", "The audio could not be decoded.") from exc

    warnings: list[str] = []
    prob = transcript.language_probability
    if language is None and prob is not None and prob < MIN_LANGUAGE_PROBABILITY:
        warnings.append(
            f"audio: language detected as '{transcript.language}' with probability {prob:.2f}"
        )
    if transcript.device != transcriber.device:
        warnings.append("audio: transcribed on the CPU with the small model (E-08)")
    units = merge_segments(transcript.segments)
    if not units:
        warnings.append(NO_SPEECH_WARNING)
    return LoadedDocument(
        media_type="audio",
        kind="audio",
        units=units,
        duration_sec=round(transcript.duration_sec, 2),
        language=transcript.language,
        warnings=warnings,
    )
