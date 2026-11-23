import os
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

try:
    import logfire
except ImportError:
    class _MockLogfire:
        def span(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
    logfire = _MockLogfire()


@dataclass
class AudioWord:
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None


@dataclass
class AudioSegment:
    speaker: str
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)

    def format_timestamp(self) -> str:
        """Format start and end times as HH:MM:SS or MM:SS."""
        start_str = format_seconds(self.start)
        end_str = format_seconds(self.end)
        return f"[{start_str} - {end_str}]"


@dataclass
class AudioTranscriptionResult:
    full_text: str
    segments: List[AudioSegment]
    speakers: List[str]
    language: str
    duration: float

    def __str__(self) -> str:
        return self.full_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_text": self.full_text,
            "speakers": self.speakers,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "speaker": s.speaker,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "timestamp": s.format_timestamp(),
                }
                for s in self.segments
            ],
        }


def format_seconds(seconds: float) -> str:
    """Helper to convert float seconds into HH:MM:SS or MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def merge_aligned_segments_by_speaker(
    segments: List[Dict[str, Any]], 
    max_pause_sec: float = 2.0
) -> List[AudioSegment]:
    """
    Merge word/segment utterances that share the same speaker label into unified blocks.
    Respects pauses between utterances to keep natural dialogue boundaries.
    """
    merged: List[AudioSegment] = []
    current_speaker: Optional[str] = None
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_words: List[str] = []
    current_word_dicts: List[Dict[str, Any]] = []

    for seg in segments:
        seg_speaker = seg.get("speaker") or "UNKNOWN_SPEAKER"
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = seg.get("text", "").strip()
        words = seg.get("words", [])

        if not seg_text:
            continue

        # If continuation of same speaker and within pause threshold
        if (
            current_speaker == seg_speaker
            and current_end is not None
            and (seg_start - current_end) <= max_pause_sec
        ):
            current_words.append(seg_text)
            current_end = max(current_end, seg_end)
            if words:
                current_word_dicts.extend(words)
        else:
            # Flush previous speaker block
            if current_speaker is not None and current_words:
                merged.append(
                    AudioSegment(
                        speaker=current_speaker,
                        start=current_start if current_start is not None else 0.0,
                        end=current_end if current_end is not None else 0.0,
                        text=" ".join(current_words),
                        words=current_word_dicts,
                    )
                )
            # Start new block
            current_speaker = seg_speaker
            current_start = seg_start
            current_end = seg_end
            current_words = [seg_text]
            current_word_dicts = list(words) if words else []

    # Flush last remaining speaker block
    if current_speaker is not None and current_words:
        merged.append(
            AudioSegment(
                speaker=current_speaker,
                start=current_start if current_start is not None else 0.0,
                end=current_end if current_end is not None else 0.0,
                text=" ".join(current_words),
                words=current_word_dicts,
            )
        )

    return merged


def parse_audio(
    file_path: str,
    model_name: str = "base",
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
) -> str:
    """
    Extract speech transcriptions with speaker identification from recorded courtroom proceedings,
    client calls, and witness depositions.
    
    Implementation details:
    1. Initialize WhisperX for speech-to-text transcription with word-level alignment.
    2. Integrate Pyannote.audio for speaker diarization to separate distinct speakers.
    3. Merge transcript segments with speaker labels and timestamps.
    
    Returns clean formatted multi-speaker transcript text:
        [00:00:15 - 00:00:42] SPEAKER_01: Text...
    """
    result = parse_audio_detailed(
        file_path=file_path,
        model_name=model_name,
        device=device,
        hf_token=hf_token,
        batch_size=batch_size,
        compute_type=compute_type,
    )
    return result.full_text


def parse_audio_detailed(
    file_path: str,
    model_name: str = "base",
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
) -> AudioTranscriptionResult:
    """
    Full pipeline returning structured AudioTranscriptionResult with segments, speakers, and metadata.
    """
    with logfire.span("🎙️ Audio Loading & Diarization", filename=file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")

        # Resolve compute device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        try:
            import whisperx
        except ImportError as e:
            logfire.warning(
                f"whisperx not installed ({e}). "
                "Attempting fallback or raising error."
            )
            return _fallback_audio_parsing(file_path)

        logfire.info(f"Loading WhisperX model '{model_name}' on {device} ({compute_type})...")
        
        # 1. Initialize WhisperX for speech-to-text transcription
        whisper_model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        audio = whisperx.load_audio(file_path)
        raw_result = whisper_model.transcribe(audio, batch_size=batch_size)
        detected_language = raw_result.get("language", "en")
        logfire.info(f"Speech transcription completed. Detected language: {detected_language}")

        # 1.1 Word-level alignment
        try:
            logfire.info("Aligning whisper transcript with word-level alignment model...")
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )
            aligned_result = whisperx.align(
                raw_result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        except Exception as align_err:
            logfire.warning(f"Word alignment failed, falling back to raw segments: {align_err}")
            aligned_result = raw_result

        # 2. Integrate Pyannote.audio for speaker diarization
        diarized_result = aligned_result
        try:
            logfire.info("Running Pyannote.audio speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
            )
            diarize_segments = diarize_model(audio)
            diarized_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            logfire.info("Speaker diarization successfully assigned to words and segments.")
        except Exception as diarize_err:
            logfire.warning(
                f"Speaker diarization skipped or failed (token required or dependency missing): {diarize_err}"
            )
            # Default speaker if diarization not available
            for s in diarized_result.get("segments", []):
                if "speaker" not in s:
                    s["speaker"] = "SPEAKER_00"

        # 3. Merge transcript segments with speaker labels and timestamps
        raw_segments = diarized_result.get("segments", [])
        merged_segments = merge_aligned_segments_by_speaker(raw_segments)

        # Build clean formatted transcript
        formatted_lines: List[str] = []
        unique_speakers = sorted(list({seg.speaker for seg in merged_segments}))
        total_duration = float(merged_segments[-1].end) if merged_segments else 0.0

        for seg in merged_segments:
            line = f"{seg.format_timestamp()} {seg.speaker}: {seg.text}"
            formatted_lines.append(line)

        full_text = "\n\n".join(formatted_lines)
        logfire.info(
            f"✅ Processed audio transcript: {len(merged_segments)} merged segments across {len(unique_speakers)} speakers."
        )

        return AudioTranscriptionResult(
            full_text=full_text,
            segments=merged_segments,
            speakers=unique_speakers,
            language=detected_language,
            duration=total_duration,
        )


def _fallback_audio_parsing(file_path: str) -> AudioTranscriptionResult:
    """
    Fallback mechanism when WhisperX is not directly installed.
    Uses openai whisper API or standard whisper if available, or returns placeholder with warning.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            logfire.info("Using OpenAI Whisper API fallback for audio transcription...")
            with open(file_path, "rb") as audio_file:
                transcript_resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                )
            
            raw_segments = getattr(transcript_resp, "segments", []) or []
            merged: List[AudioSegment] = []
            
            for seg in raw_segments:
                start = float(seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0))
                end = float(seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0))
                text = (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip()
                speaker = "SPEAKER_00"
                if text:
                    merged.append(AudioSegment(speaker=speaker, start=start, end=end, text=text))

            formatted_lines = [f"{s.format_timestamp()} {s.speaker}: {s.text}" for s in merged]
            full_text = "\n\n".join(formatted_lines) if formatted_lines else getattr(transcript_resp, "text", "")
            
            return AudioTranscriptionResult(
                full_text=full_text,
                segments=merged,
                speakers=["SPEAKER_00"],
                language="en",
                duration=float(merged[-1].end) if merged else 0.0,
            )
        except Exception as e:
            logfire.warning(f"OpenAI audio transcription fallback failed: {e}")

    raise ImportError(
        "WhisperX and Pyannote.audio are required for audio transcription. "
        "Install via `pip install whisperx pyannote.audio` or configure OPENAI_API_KEY."
    )
