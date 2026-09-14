"""Model abstractions for speech-to-speech and text-to-speech.

Two parallel, backend-agnostic interfaces share one set of OpenAI-compatible
types (:mod:`.types`) and audio helpers (:mod:`.audio`):

* :class:`~.base.SpeechToSpeechModel` — conversation in, spoken reply out
  (transformers / vLLM / custom repos), built via :func:`~.registry.load_model`.
* :class:`~.tts.TextToSpeechModel` — a compact text-in / speech-out interface
  (CosyVoice, with Japanese support), built via :func:`~.tts.load_tts`.

Backends import their heavy deps lazily, so the package imports cleanly without
torch / transformers / any TTS engine installed.
"""

from rolebreak.models.audio import (
    has_audio_frames,
    input_audio_from_bytes,
    input_audio_from_file,
    output_audio_from_waveform,
    save_output_audio,
    user_audio_message,
    waveform_to_wav_bytes,
)
from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.models.registry import get_backend, load_model, register_backend
from rolebreak.models.tts import (
    TextToSpeechModel,
    get_tts_backend,
    load_tts,
    register_tts_backend,
)
from rolebreak.types import (
    AudioConfig,
    ChatRequest,
    ChatResponse,
    InputAudio,
    InputAudioContent,
    Message,
    OutputAudio,
    TextContent,
)

__all__ = [
    # speech-to-speech
    "SpeechToSpeechModel",
    "load_model",
    "get_backend",
    "register_backend",
    # text-to-speech
    "TextToSpeechModel",
    "load_tts",
    "get_tts_backend",
    "register_tts_backend",
    # types
    "ChatRequest",
    "ChatResponse",
    "Message",
    "TextContent",
    "InputAudio",
    "InputAudioContent",
    "OutputAudio",
    "AudioConfig",
    # audio helpers
    "input_audio_from_file",
    "input_audio_from_bytes",
    "user_audio_message",
    "save_output_audio",
    "output_audio_from_waveform",
    "has_audio_frames",
    "waveform_to_wav_bytes",
]
