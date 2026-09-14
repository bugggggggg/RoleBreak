"""Build a model by backend name without importing every backend up front.

    from rolebreak.models import load_model

    model = load_model("qwen3-omni-api", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    model = load_model("minicpm-o-4.5-api", "openbmb/MiniCPM-o-4_5")
    model = load_model("covo-audio-api", "tencent/Covo-Audio-Chat")

Register a model-specific backend with the :func:`register_backend` decorator.
"""

from __future__ import annotations

from typing import Any, Callable

from rolebreak.models.base import SpeechToSpeechModel

# backend name -> zero-arg importer returning the SpeechToSpeechModel subclass.
_REGISTRY: dict[str, Callable[[], type[SpeechToSpeechModel]]] = {}


def register_backend(
    name: str,
) -> Callable[[type[SpeechToSpeechModel]], type[SpeechToSpeechModel]]:
    """Class decorator registering a backend under ``name``."""

    def deco(cls: type[SpeechToSpeechModel]) -> type[SpeechToSpeechModel]:
        _REGISTRY[name] = lambda: cls
        return cls

    return deco


def _builtin(name: str) -> type[SpeechToSpeechModel]:
    # Lazy imports keep optional deps (torch/transformers) out of the
    # import path until a backend is actually requested.
    if name == "qwen3-omni-api":
        from rolebreak.models.backends.qwen3_omni import Qwen3OmniAPIModel

        return Qwen3OmniAPIModel
    if name == "qwen2.5-omni-api":
        from rolebreak.models.backends.qwen2p5_omni import Qwen2p5OmniAPIModel

        return Qwen2p5OmniAPIModel
    if name == "minicpm-o-4.5-api":
        from rolebreak.models.backends.minicpm_o_4p5 import MiniCPMO4p5APIModel

        return MiniCPMO4p5APIModel
    if name == "covo-audio-api":
        from rolebreak.models.backends.covo_audio import CovoAudioAPIModel

        return CovoAudioAPIModel
    if name == "qwen2-audio-api":
        from rolebreak.models.backends.qwen2_audio import Qwen2AudioAPIModel

        return Qwen2AudioAPIModel
    if name == "moshi-api":
        from rolebreak.models.backends.moshi_api import MoshiAPIModel

        return MoshiAPIModel
    if name == "personaplex-api":
        from rolebreak.models.backends.personaplex_api import PersonaplexAPIModel

        return PersonaplexAPIModel
    if name == "speech-pipeline-api":
        from rolebreak.models.backends.speech_pipeline_api import SpeechPipelineAPIModel

        return SpeechPipelineAPIModel
    raise KeyError(name)


def get_backend(name: str) -> type[SpeechToSpeechModel]:
    """Resolve a backend class by name. ``base:foo`` resolves to ``base``
    unless ``base:foo`` was explicitly registered."""
    if name in _REGISTRY:
        return _REGISTRY[name]()
    base = name.split(":", 1)[0]
    if base in _REGISTRY:
        return _REGISTRY[base]()
    try:
        return _builtin(base)
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Built-ins: qwen2-audio-api, qwen2.5-omni-api, qwen3-omni-api, minicpm-o-4.5-api, covo-audio-api, moshi-api, personaplex-api, speech-pipeline-api. "
            f"Registered: {sorted(_REGISTRY)}."
        ) from None


def load_model(backend: str, model: str, **config: Any) -> SpeechToSpeechModel:
    """Instantiate a model for ``backend``. Does not call ``.load()`` — do that
    yourself (or use the model as a context manager) when ready to allocate."""
    return get_backend(backend)(model, **config)
