"""MiniCPM-o 4.5 speech-to-speech backend, served over an OpenAI-compatible API.

``openbmb/MiniCPM-o-4_5`` perceives audio (also image/video) and emits both text
and speech. As with the Qwen omni models we drive it out-of-process: run the
checkpoint behind ``vllm serve --omni`` and point this backend at that endpoint.
See ``docs/models.md`` for the launch command — it needs a derived image
(``rolebreak:vllm-omni``) and an explicit ``--deploy-config``.

Deliberately a sibling of :mod:`rolebreak.models.backends.qwen2p5_omni` rather
than a subclass — the response shape is the same but the voice and thinking
behaviour are not (see :class:`MiniCPMO4p5APIModel`), so sharing an implementation
would mean a base class that is mostly overridden.
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

from rolebreak.models.audio import decode_b64, encode_bytes
from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.types import (
    ChatRequest,
    ChatResponse,
    Message,
    OutputAudio,
)

#: The reference clip the checkpoint's Code2Wav stage speaks with by default,
#: ``assets/HT_ref_audio.wav``. Recorded for provenance only — unlike the Qwen
#: omni servers there is no per-request voice to set (see the class docstring).
DEFAULT_VOICE_REF = "HT_ref_audio"

# A locally deployed `vllm serve --omni` OpenAI-compatible endpoint for
# MiniCPM-o 4.5. Distinct from the Qwen omni ports so they can run side by side.
DEFAULT_API_BASE_URL = "http://localhost:8092/v1"

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
# A cold three-stage Omni server needs minutes to come back: it reloads the
# weights, recompiles, and recaptures CUDA graphs per stage, and stage 2 then
# rebuilds the Token2wav flow/HiFT assets. Measured cold start here is ~3
# minutes, so the budget stays generous.
DEFAULT_HEALTH_TIMEOUT = 600.0
HEALTH_POLL_SECONDS = 10.0
# Let the freshly-restarted engine settle before the next request: /health answers
# as soon as the API server is up, which is a little before the stages are warm.
RETRY_SETTLE_SECONDS = 5.0

# Transport failures that mean "the server went away mid-request", not "this
# request is bad" — the stream is cut when the engine takes the process down.
_TRANSPORT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    socket.timeout,
    TimeoutError,
)

# An in-band error is worth another attempt only when the *server* broke.
# vllm-omni's talker (stage 1) can die on a CUDA illegal-memory-access under
# load, taking the whole orchestrator with it; the container restarts and the
# same request then succeeds. See docs/models.md.
_ENGINE_DEATH_MARKERS = (
    "enginecore",
    "engine core",
    "engine dead",
    "enginedead",
    "illegal memory access",
    "cuda error",
)
# A request the server refuses on its merits fails identically every time, so
# retrying only burns the budget and hides the real cause.
_FATAL_MARKERS = (
    "context",
    "maximum",
    "too long",
    "too large",
    "invalid",
)

#: A complete reasoning block at the head of the reply.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str | None) -> str | None:
    """Drop MiniCPM's ``<think>…</think>`` preamble from a reply.

    The chat template reasons by default, and the block lands in
    ``message.content`` rather than in a separate ``reasoning_content`` field —
    ``"<think>\\n\\n</think>\\nHello!"`` for a turn it had nothing to think
    about. Left in, it would be scored as the persona's spoken words and read
    aloud in the transcript, so it is removed here rather than in every caller.

    ``chat_template_kwargs={"enable_thinking": False}`` suppresses it on the
    text path but is not reliable once a turn carries audio, so this runs
    regardless of that flag. A block left unterminated by the ``max_tokens`` cap
    means the whole reply was reasoning and nothing was said: that yields
    ``None``, not a truncated thought.
    """
    if not text:
        return text
    stripped = _THINK_BLOCK.sub("", text, count=1)
    if "</think>" in stripped:
        # Belt and braces: an unbalanced opener earlier in the reply.
        stripped = stripped.rsplit("</think>", 1)[1]
    elif stripped.lstrip().startswith("<think>"):
        return None  # truncated mid-thought; there is no spoken reply in here
    return stripped.strip() or None


class MiniCPMO4p5APIModel(SpeechToSpeechModel):
    """MiniCPM-o 4.5 via a locally deployed vLLM OpenAI-compatible endpoint.

    This backend keeps no weights: it forwards our OpenAI-shaped
    :class:`ChatRequest` to a running ``vllm serve --omni`` at ``base_url`` and
    parses the chat-completion back.

    Three quirks of this server shape the backend:

    * like Qwen2.5-Omni, the **non-streaming** call returns speech but splits
      the reply across **two ``choices`` that both carry ``index: 0``** — one
      holds ``message.content`` (text, with ``audio: null``), the other holds
      ``message.audio`` (base64 WAV, mono 24 kHz, with an empty
      ``transcript``). Reading ``choices[0]`` the usual way would silently drop
      the speech, so :meth:`_parse_completion` folds every choice into one reply
      and backfills the transcript from the text choice.
    * **there is no voice to pick.** Qwen's talker takes a top-level ``speaker``
      field; MiniCPM's Code2Wav stage clones a reference clip instead, and that
      clip is a *deploy-time* setting (the stage connector's ``extra.prompt_wav``,
      defaulting to ``assets/HT_ref_audio.wav`` inside the checkpoint). A
      ``speaker`` or ``audio.voice`` in the request is accepted and ignored, so
      this backend does not pretend to send one. Measured with the WavLM SV
      judge: replies sit at ~0.8 cosine against the checkpoint's default clip
      whether or not the turn carried audio input, so — despite an
      ``_extract_first_audio_ref`` path in vllm-omni — the user's own clip does
      **not** become the assistant's voice, and the persona's timbre is stable
      across a conversation.
    * the reply's ``content`` carries a ``<think>…</think>`` preamble. See
      :func:`strip_thinking`.

    Non-streaming is the default path. Pass ``stream=True`` to use the SSE path
    instead: it carries the same content and additionally yields a
    time-to-first-token latency.
    """

    backend = "minicpm-o-4.5-api"

    def __init__(
        self,
        model: str = "openbmb/MiniCPM-o-4_5",
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
        stream: bool = False,
        enable_thinking: bool = False,
        **config: Any,
    ) -> None:
        super().__init__(model, base_url=base_url, **config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.health_timeout = health_timeout
        self.stream = stream
        self.enable_thinking = enable_thinking
        # Held by whichever thread is nursing a dead engine back up. An eval that
        # replays several examples at once has every in-flight turn fail at the
        # same instant when the talker takes the orchestrator down, and a cold
        # server does not need one warmup turn per thread on top of the restart
        # it is already paying for -- see :meth:`_recover`.
        self._recovery = threading.Lock()

    def generate(self, request: ChatRequest) -> ChatResponse:
        payload = request.to_dict()
        payload["model"] = self.model
        # No voice field: the reference clip is fixed server-side, so anything we
        # sent here would be silently ignored (see the class docstring).
        if not self.enable_thinking:
            # Honoured on the text path; the reply is stripped either way.
            template_kwargs = dict(payload.get("chat_template_kwargs") or {})
            template_kwargs.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = template_kwargs
        if self.stream:
            payload["stream"] = True
        return self._generate_with_retry(payload)

    # -- resilience --------------------------------------------------------- #
    def _dispatch(self, payload: dict[str, Any]) -> ChatResponse:
        """One attempt at the turn, over whichever path the payload asks for."""
        if payload.get("stream"):
            return self._generate_stream(payload)
        return self._parse_completion(self._post(payload))

    def _generate_with_retry(self, payload: dict[str, Any]) -> ChatResponse:
        """Re-issue the turn when the *server* dies under it.

        The talker stage of vllm-omni can crash the whole engine on a CUDA fault
        under load, which would otherwise end a long eval partway through.
        Re-issuing is sound here because this backend keeps no server-side
        session: every turn already resends the full windowed history as audio,
        so an attempt against a restarted server is indistinguishable from a
        first attempt. A partial reply from a killed stream is discarded rather
        than stitched.

        The retry count rides back on :attr:`ChatResponse.raw` so a run can log
        that a turn needed one — a recovered turn should be visible in the
        transcript, not silently identical to a clean one.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self._dispatch(payload)
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                logger.warning(
                    "%s: attempt %d/%d failed (%s); waiting for the server to come back",
                    self.backend,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                self._recover()
                continue
            if attempt:
                response.raw = {**(response.raw or {}), "retries": attempt}
            return response
        raise AssertionError("unreachable: the loop either returns or raises")

    def _recover(self) -> None:
        """Wait for the engine to come back, warming it once for all callers.

        One thread does the polling and the throwaway warmup turn; any other
        thread that failed on the same crash blocks here until that finishes and
        then retries against an already-warm server. Without this, N concurrent
        workers would each poll ``/health`` and each fire a warmup turn at a
        server that has just restarted -- which is the load that killed it.
        """
        if self._recovery.acquire(blocking=False):
            try:
                if self._wait_for_health():
                    self._warmup()
            finally:
                self._recovery.release()
            return
        logger.info("%s: another thread is recovering the server; waiting for it", self.backend)
        with self._recovery:
            pass

    def _is_retryable(self, exc: BaseException) -> bool:
        """Whether ``exc`` looks like a dead server rather than a bad request."""
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code >= 500  # a 4xx is our fault and will not improve
        if isinstance(exc, _TRANSPORT_ERRORS):
            return True
        if isinstance(exc, RuntimeError):
            # An in-band failure: vLLM answers 200 before generation runs, so a
            # mid-generation crash arrives as a payload, not an HTTP status.
            message = str(exc).lower()
            if any(marker in message for marker in _FATAL_MARKERS):
                return False
            return any(marker in message for marker in _ENGINE_DEATH_MARKERS)
        return False

    def _warmup(self) -> bool:
        """Run one throwaway turn so the retried request does not pay cold-start costs.

        ``/health`` answers as soon as the API server binds its port, which is
        before the stages have JIT-compiled their Triton kernels — a restarted
        server stalls on that during its first real request. Spending it on a
        turn whose answer we discard keeps it out of the eval's latency numbers,
        and keeps a slow first reply from tripping the request timeout.

        Best-effort by design: a failed warmup says nothing the next real attempt
        would not say better, so it logs and returns instead of raising.
        """
        # Short prompt, both modalities, so all three stages (thinker, talker,
        # Code2Wav) are exercised.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Introduce yourself in one short sentence."}],
            "modalities": ["text", "audio"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        try:
            self._post(payload)
        except Exception as exc:  # noqa: BLE001 - warmup is advisory, never fatal
            logger.warning("%s: warmup turn failed (%s); trying the real request anyway", self.backend, exc)
            return False
        logger.info("%s: warmup turn done in %.1fs", self.backend, time.monotonic() - started)
        return True

    def _health_url(self) -> str:
        """``/health`` on the endpoint's origin — it sits beside ``/v1``, not under it."""
        parts = urllib.parse.urlsplit(self.base_url)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))

    def _wait_for_health(self) -> bool:
        """Poll ``/health`` until the restarted server answers, or give up.

        Giving up does not raise: the next attempt's own failure is a better
        error to surface than one from this helper.
        """
        url = self._health_url()
        deadline = time.monotonic() + self.health_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10.0) as resp:
                    if resp.status < 400:
                        logger.info("%s: server healthy again at %s", self.backend, url)
                        time.sleep(RETRY_SETTLE_SECONDS)
                        return True
            except Exception:  # noqa: BLE001 - still down; keep polling until the deadline
                pass
            time.sleep(HEALTH_POLL_SECONDS)
        logger.error(
            "%s: server did not answer %s within %.0fs",
            self.backend,
            url,
            self.health_timeout,
        )
        return False

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
        """Fold the reply's several ``choices`` into one message.

        This server answers a spoken turn with two choices that *both* say
        ``index: 0`` — one text-only, one audio-only — so we take the first
        non-empty of each rather than indexing. A text-only turn comes back as
        the single ordinary choice, which this reads unchanged.
        """
        if "choices" not in data:
            raise RuntimeError(self._response_error(data))
        text: str | None = None
        audio: OutputAudio | None = None
        for choice in data["choices"]:
            message = choice.get("message") or {}
            if text is None and message.get("content"):
                text = strip_thinking(message["content"])
            if audio is None and message.get("audio"):
                audio = OutputAudio.from_dict(message["audio"])
        if audio is not None and not audio.transcript:
            # The audio choice ships an empty transcript; the text choice is it.
            audio.transcript = text
        return ChatResponse(
            message=Message(role="assistant", content=text, audio=audio),
            model=data.get("model", self.model),
            raw=data,
        )

    def _generate_stream(self, payload: dict[str, Any]) -> ChatResponse:
        """Consume the SSE stream, splitting text deltas from base64 WAV segments.

        Streaming is also the only place this backend can time the reply: the
        turn-based contract hands the model a finished clip, so the wait starts
        when the request goes out and ends at the thinker's first text — or, for
        a turn that produced only speech, the Code2Wav stage's first WAV segment.

        The reasoning preamble is stripped only after the deltas are joined: a
        ``<think>`` block arrives split across chunks, so no single delta can be
        classified on its own.
        """
        text_parts: list[str] = []
        wav_segments: list[bytes] = []
        started = time.monotonic()
        first_text_s: float | None = None
        first_audio_s: float | None = None
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                if "choices" not in data:
                    raise RuntimeError(self._response_error(data))
                if not data["choices"]:
                    continue  # usage-only final chunk carries no delta
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue
                # Base64 WAV segments start with "UklGR" ("RIFF"); else it's text.
                if content.startswith("UklGR"):
                    wav_segments.append(decode_b64(content))
                    if first_audio_s is None:
                        first_audio_s = time.monotonic() - started
                else:
                    text_parts.append(content)
                    if first_text_s is None:
                        first_text_s = time.monotonic() - started

        transcript = strip_thinking("".join(text_parts))
        audio = None
        if wav_segments:
            audio = OutputAudio(
                data=encode_bytes(_merge_wav_segments(wav_segments)),
                format="wav",
                transcript=transcript,
            )
        return ChatResponse(
            message=Message(role="assistant", content=transcript, audio=audio),
            model=self.model,
            raw={"text": transcript, "audio_segments": len(wav_segments)},
            latency=first_text_s if first_text_s is not None else first_audio_s,
        )

    def _response_error(self, data: dict[str, Any]) -> str:
        """Describe a payload that carries a failure instead of a completion.

        vLLM answers 200 and opens the stream before generation runs, so a
        failure raised partway through (context overflow, a talker crash) can
        only be reported in-band: it yields one payload shaped
        ``{"object": "error", "message": ...}`` with no ``choices``.
        """
        err = data.get("error", data)
        detail = err.get("message") if isinstance(err, dict) else err
        return f"{self.backend} request failed at {self.base_url}: {detail or json.dumps(data)[:500]}"


def _merge_wav_segments(segments: list[bytes]) -> bytes:
    """Concatenate the PCM frames of several WAV byte-strings into one WAV.

    Each streamed segment is a complete standalone WAV; joining them at the raw
    bytes level would leave stray headers mid-stream, so we read each with
    :mod:`wave` and rewrite the merged frames under a single header.
    """
    params = None
    frames = bytearray()
    for seg in segments:
        with wave.open(io.BytesIO(seg), "rb") as w:
            if params is None:
                params = w.getparams()
            frames += w.readframes(w.getnframes())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return buf.getvalue()
