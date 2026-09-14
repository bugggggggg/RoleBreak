"""Qwen2-Audio audio-understanding backend (vLLM / OpenAI-compatible).

``Qwen/Qwen2-Audio-7B-Instruct`` perceives audio (plus text) and emits **text
only** — unlike the Omni checkpoints it has no talker (speech) head, so replies
never carry an ``audio`` field. That makes this a plain audio-in, text-out chat
model.

Like :class:`~rolebreak.models.backends.qwen3_omni.Qwen3OmniAPIModel` this
backend keeps no weights: it forwards our OpenAI-shaped :class:`ChatRequest` to a
running ``vllm serve`` at ``base_url`` and parses the chat-completion back. But
because there is no spoken output it is much simpler than the Qwen3-Omni API
adapter — there is no top-level ``speaker`` field and no base64-WAV streaming to
reassemble, just a single non-streaming completion whose text we return.

We deliberately keep this as its own class (duplicating the small transport
plumbing) rather than sharing with the Qwen3-Omni adapter: one model, one class.

Requires (deploy yourself): a vLLM build serving Qwen2-Audio, e.g.::

    vllm serve Qwen/Qwen2-Audio-7B-Instruct --port 8902

    hf download Qwen/Qwen2-Audio-7B-Instruct
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.types import ChatRequest, ChatResponse, Message, OutputAudio

# A locally deployed `vllm serve` OpenAI-compatible endpoint for Qwen2-Audio.
DEFAULT_API_BASE_URL = "http://localhost:8901/v1"


class Qwen2AudioAPIModel(SpeechToSpeechModel):
    """Qwen2-Audio via a locally deployed vLLM OpenAI-compatible endpoint.

    Audio-in, text-out: the model has no speech head, so we force text-only
    generation and always return an assistant message whose ``content`` is the
    reply text (``audio`` stays ``None``). Any ``audio`` modality the caller
    requested is dropped, since this model cannot honour it.
    """

    backend = "qwen2-audio-api"

    def __init__(
        self,
        model: str = "Qwen/Qwen2-Audio-7B-Instruct",
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        **config: Any,
    ) -> None:
        super().__init__(model, base_url=base_url, **config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, request: ChatRequest) -> ChatResponse:
        payload = request.to_dict()
        payload["model"] = self.model
        # Qwen2-Audio only speaks text; never ask the endpoint for audio output.
        payload["modalities"] = ["text"]
        payload.pop("audio", None)
        return self._parse_completion(self._post(payload))

    # -- transport --------------------------------------------------------- #
    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse_completion(self, data: dict[str, Any]) -> ChatResponse:
        choice = data["choices"][0]["message"]
        # Text-out model, but stay tolerant of an ``audio`` field just in case.
        audio = OutputAudio.from_dict(choice["audio"]) if choice.get("audio") else None
        return ChatResponse(
            message=Message(role="assistant", content=choice.get("content"), audio=audio),
            model=data.get("model", self.model),
            raw=data,
        )
