"""Helpers for moving audio between files, raw bytes, and the base64 form used
in OpenAI-style messages."""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Any

from rolebreak.types import (
    AudioFormat,
    InputAudio,
    InputAudioContent,
    Message,
    OutputAudio,
)

_EXT_TO_FORMAT: dict[str, AudioFormat] = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".ogg": "ogg",
    ".opus": "opus",
}


def format_from_path(path: str | Path) -> AudioFormat:
    suffix = Path(path).suffix.lower()
    if suffix not in _EXT_TO_FORMAT:
        raise ValueError(f"Unsupported audio extension: {suffix!r}")
    return _EXT_TO_FORMAT[suffix]


def encode_bytes(data: bytes) -> str:
    """Raw audio bytes -> base64 string."""
    return base64.b64encode(data).decode("ascii")


def decode_b64(data: str) -> bytes:
    """base64 string -> raw audio bytes."""
    return base64.b64decode(data)


def input_audio_from_file(path: str | Path) -> InputAudioContent:
    """Read an audio file into an OpenAI ``input_audio`` content part."""
    path = Path(path)
    data = encode_bytes(path.read_bytes())
    return InputAudioContent(input_audio=InputAudio(data=data, format=format_from_path(path)))


def input_audio_from_bytes(data: bytes, fmt: AudioFormat = "wav") -> InputAudioContent:
    """Wrap a clip's raw bytes in an OpenAI ``input_audio`` content part.

    The in-memory counterpart of :func:`input_audio_from_file`, for audio that has
    no file of its own — a ``.wav`` member read out of a WebDataset shard, say —
    so it can be sent without being unpacked to disk first. A path carries its
    container format in its suffix; bytes have to declare theirs with ``fmt``.
    """
    return InputAudioContent(input_audio=InputAudio(data=encode_bytes(data), format=fmt))


def user_audio_message(source: str | Path | bytes, text: str | None = None, fmt: AudioFormat = "wav") -> Message:
    """Build a ``user`` turn from an audio file — or a clip's raw bytes.

    Optionally carries a text part alongside the audio. ``fmt`` names the
    container format when ``source`` is bytes; a path takes it from its suffix.
    """
    parts: list = []
    if text is not None:
        from rolebreak.types import TextContent

        parts.append(TextContent(text=text))
    if isinstance(source, bytes | bytearray | memoryview):
        parts.append(input_audio_from_bytes(bytes(source), fmt))
    else:
        parts.append(input_audio_from_file(source))
    return Message(role="user", content=parts)


def save_output_audio(audio: OutputAudio, path: str | Path) -> Path:
    """Write an assistant ``OutputAudio`` payload to disk."""
    path = Path(path)
    path.write_bytes(decode_b64(audio.data))
    return path


def wav_frame_count(data: bytes) -> int | None:
    """Number of audio frames in WAV ``data``, or ``None`` if it isn't parseable WAV.

    Reads the header only, so it's cheap enough to call on every clip. Use it to
    tell a real recording from a header-with-no-samples file — 44 bytes of WAV
    that decodes to an empty waveform, which is what a speech backend leaves
    behind for a turn that produced no speech.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes()
    except (wave.Error, EOFError):
        return None


def has_audio_frames(audio: OutputAudio) -> bool:
    """Whether ``audio`` actually carries samples (not an empty, header-only WAV).

    Only WAV payloads can be checked without decoding, so anything
    :func:`wav_frame_count` can't parse (mp3, opus, …) is taken at its word and
    reported as carrying audio.
    """
    return wav_frame_count(decode_b64(audio.data)) != 0


def waveform_to_wav_bytes(samples: Any, sample_rate: int) -> bytes:
    """Encode a mono waveform to WAV bytes (16-bit PCM) using only the stdlib.

    ``samples`` is a 1-D array-like; float input is assumed to be in [-1, 1] and
    is scaled to int16, integer input is cast to int16 as-is.
    """
    import numpy as np

    if sample_rate <= 0:
        # wave masks this on close as the misleading "sampling rate not
        # specified"; raise a clear error at the source instead.
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

    arr = np.asarray(samples).reshape(-1)
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).round().astype("<i2")
    else:
        arr = arr.astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def output_audio_from_waveform(samples: Any, sample_rate: int, transcript: str | None = None) -> OutputAudio:
    """Wrap a generated waveform in an :class:`OutputAudio` (WAV/base64)."""
    return OutputAudio(
        data=encode_bytes(waveform_to_wav_bytes(samples, sample_rate)),
        format="wav",
        transcript=transcript,
    )
