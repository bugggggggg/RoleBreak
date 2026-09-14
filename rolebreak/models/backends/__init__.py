"""Concrete model backends (speech-to-speech and text-to-speech).

Backend modules are imported lazily by the registries (so optional deps stay
optional); this package init avoids importing them eagerly. Import a class
directly from its module when you need the symbol, e.g.
``from rolebreak.models.backends.cosyvoice import CosyVoiceTTSModel``.
"""

# Speech-to-speech backends.
SPEECH_TO_SPEECH = [
    "CovoAudioAPIModel",
    "MiniCPMO4p5APIModel",
    "MoshiAPIModel",
    "Qwen2p5OmniAPIModel",
    "Qwen3OmniAPIModel",
]

# Text-to-speech backends.
TEXT_TO_SPEECH = [
    "CosyVoiceTTSModel",
]

__all__ = SPEECH_TO_SPEECH + TEXT_TO_SPEECH
