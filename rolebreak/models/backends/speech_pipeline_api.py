"""Speech-to-speech *pipeline* backend, talking to a running server.

Deploy the pipeline first, e.g.::

    docker run --rm -it --name speech-pipeline --gpus '"device=1"' \\
      --network host -v "$HOME/.cache:/root/.cache" \\
      -v "$PWD/external/speech-to-speech/src:/usr/src/app/src" \\
      speech-pipeline speech-to-speech --mode websocket --ws_port 8765 \\
      --min_silence_ms 640 --manual_turn_end True \\
      --llm_backend chat-completions \\
      --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
      --qwen3_tts_device cuda --qwen3_tts_backend torch \\
      --qwen3_tts_language auto --qwen3_tts_non_streaming_mode True \\
      --model_name /models/Qwen3.5-4B \\
      --responses_api_base_url http://localhost:8901/v1 \\
      --responses_api_api_key "EMPTY" \\
      --chat_size 64 --compact_history False

Requires (install into your env): ``aiohttp``, ``numpy``, and ``sphn`` (used only
to decode input audio files)::

    uv pip install aiohttp numpy sphn
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rolebreak.models.audio import decode_b64, output_audio_from_waveform
from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.types import ChatRequest, ChatResponse, InputAudioContent, Message

logger = logging.getLogger(__name__)

# Fixed by the pipeline (``PIPELINE_SR`` in the server): 16 kHz mono int16.
SAMPLE_RATE = 16_000
# Server re-buffers into 512-sample VAD chunks; sending frames of that size keeps
# our writes aligned with what the server consumes.
FRAME_SIZE = 512
# Real-time duration of one frame. We sleep this between frames so audio reaches
# the server at ~playback speed — the pace its streaming VAD is built for.
FRAME_DT = FRAME_SIZE / SAMPLE_RATE


@dataclass
class _Commit:
    """When this turn's ``end_of_turn`` went out, i.e. the real turn boundary.

    ``at`` stays ``None`` when the reply beat the clip and :meth:`_stream` was
    cancelled before it could commit — which is itself the symptom the guard
    reports.
    """

    at: float | None = None


class SpeechPipelineAPIModel(SpeechToSpeechModel):
    """Turn-based adapter over a running cascaded speech-to-speech pipeline.

    Construction is cheap; ``load()`` only validates config. The server must be
    reachable at ``url`` before :meth:`generate` runs.
    """

    backend = "speech-pipeline-api"

    def __init__(
        self,
        model: str = "speech-pipeline",
        *,
        url: str = "ws://localhost:8765",
        silence_stop_s: float = 1.5,
        min_reply_s: float = 0.5,
        max_reply_s: float = 30.0,
        response_timeout_s: float = 20.0,
        connect_timeout_s: float = 30.0,
        reset_timeout_s: float = 5.0,
        commit_turns: bool = True,
        pace_realtime: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(model, **config)
        self.url = url
        self.silence_stop_s = silence_stop_s
        self.min_reply_s = min_reply_s
        self.max_reply_s = max_reply_s
        self.response_timeout_s = response_timeout_s
        self.connect_timeout_s = connect_timeout_s
        # How long to wait for the server's ``session_reset`` ack before running
        # the conversation anyway (an older server has no ``reset`` handler).
        self.reset_timeout_s = reset_timeout_s
        # Send ``end_of_turn`` at the end of every clip instead of leaving the
        # boundary to the server's silence heuristic. We know exactly where the
        # turn ends, so there is nothing to detect — pair it with the server's
        # ``--manual_turn_end`` and the VAD cannot cut an utterance short. Safe
        # against a server that has neither: an unknown control message is
        # ignored there, leaving the old behaviour untouched.
        self.commit_turns = commit_turns
        # Play the clip at real time (what a live pipeline needs: its VAD has to
        # hear the pauses to segment, and a burst would arrive as one blob of
        # speech). A turn-based server has no VAD, so the clip can go over in one
        # message -- 0.8ms instead of the clip's own duration, which is otherwise
        # ~90% of a run's wall clock. Only turn this off against such a server:
        # to a live one the whole utterance would look like a single instant.
        self.pace_realtime = pace_realtime

    # -- inference --------------------------------------------------------- #
    def generate(self, request: ChatRequest) -> ChatResponse:
        user_pcm = self._extract_user_pcm(request)
        # The request's ``system`` message is the only source of persona here; with
        # none, the server keeps its own configured one.
        text_prompt = self._extract_persona(request) or ""
        pcm, transcript, latency, early_s = asyncio.run(self._converse([user_pcm], text_prompt))[0]
        return self._make_response(pcm, transcript, latency, early_s)

    def converse(
        self,
        user_audios: Sequence[np.ndarray],
        *,
        text_prompt: str,
    ) -> Iterator[ChatResponse]:
        """Stream a multi-turn conversation over a single pipeline session.

        ``user_audios`` is one 16 kHz mono float PCM clip per user turn (decode
        files with :meth:`load_audio`). Yields one :class:`ChatResponse` per turn
        as the pipeline finishes replying — the connection stays open across the
        whole sequence so the server's conversation state carries between turns.

        ``text_prompt`` is this conversation's persona / system prompt, sent as a
        control message on connect. It is required — the persona always comes from
        the caller, never from adapter state — but may be ``""`` to leave the
        server's own configured persona in place.

        This is a generator: iterate it to completion (or close it) so the
        underlying WebSocket is torn down. Turn boundaries use the wall-clock idle
        heuristic; tune it with the ``silence_stop_s`` / ``min_reply_s`` /
        ``max_reply_s`` / ``response_timeout_s`` constructor args.
        """
        if not self._loaded:
            self.load()
        # One turn per ``run_until_complete``, not one call for the whole
        # conversation: the caller sees each reply as it lands, so a run that dies
        # mid-example has already recorded the turns that did finish.
        loop = asyncio.new_event_loop()
        try:
            session, ws = loop.run_until_complete(self._open(text_prompt))
            try:
                for user_pcm in user_audios:
                    clip = np.asarray(user_pcm, dtype=np.float32).reshape(-1)
                    pcm, transcript, latency, early_s = loop.run_until_complete(self._run_turn(ws, clip))
                    yield self._make_response(pcm, transcript, latency, early_s)
            finally:
                loop.run_until_complete(self._close(session, ws))
        finally:
            loop.close()

    def _make_response(
        self,
        pcm: np.ndarray,
        transcript: str,
        latency: float | None = None,
        early_s: float | None = None,
    ) -> ChatResponse:
        audio = output_audio_from_waveform(pcm, SAMPLE_RATE, transcript=transcript or None)
        raw: dict[str, Any] = {"num_samples": int(pcm.shape[-1]), "text": transcript}
        if early_s is not None:
            # Set only when the turn tripped the guard, so a caller can test the
            # key's presence rather than compare a float against zero.
            raw["early_reply_s"] = early_s
        return ChatResponse(
            message=Message(role="assistant", content=transcript or None, audio=audio),
            model=self.model,
            raw=raw,
            latency=latency,
        )

    async def _converse(
        self,
        user_audios: list[np.ndarray],
        text_prompt: str,
    ) -> list[tuple[np.ndarray, str, float | None, float | None]]:
        """Open one session, stream every user turn, and collect each reply.

        If ``text_prompt`` is non-empty, a ``text_prompt`` control message is sent
        first so the server adopts that persona for the whole session.
        """
        session, ws = await self._open(text_prompt)
        try:
            replies: list[tuple[np.ndarray, str, float | None, float | None]] = []
            for user_pcm in user_audios:
                clip = np.asarray(user_pcm, dtype=np.float32).reshape(-1)
                replies.append(await self._run_turn(ws, clip))
            return replies
        finally:
            await self._close(session, ws)

    async def _open(self, text_prompt: str) -> tuple[Any, Any]:
        """Open the session's WebSocket, reset it, and set this conversation's persona."""
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.connect_timeout_s)
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            ws = await session.ws_connect(self.url, max_msg_size=0)
            await self._reset_session(ws)
            if text_prompt:
                # After the reset, never before: the reset restores the server's
                # default persona, so it would wipe this one.
                await ws.send_str(json.dumps({"type": "text_prompt", "text": text_prompt}))
        except BaseException:
            await session.close()
            raise
        return session, ws

    @staticmethod
    async def _close(session: Any, ws: Any) -> None:
        """Tear the session down, tolerating a connection that is already gone."""
        with contextlib.suppress(Exception):
            if not ws.closed:
                await ws.close()
        await session.close()

    async def _reset_session(self, ws: Any) -> bool:
        """Ask the server to clear the previous conversation; return whether it acked.

        The server also resets on a 0 -> 1 client transition, but that is
        best-effort: it is skipped if the previous socket has not been reaped
        yet, and then this conversation silently continues the last one — the
        exact failure a benchmark must not have. An explicit ``reset`` message is
        unconditional, and waiting for the ``session_reset`` ack makes it ordered:
        by the time we send audio, the chat buffer is empty and the ``SESSION_END``
        that clears the handlers' own state is already queued ahead of it.

        Frames arriving before the ack are the previous conversation's stragglers
        and are dropped. A server without the ``reset`` handler never acks, so
        this gives up after ``reset_timeout_s`` and lets the conversation run on
        the connect-time reset alone.
        """
        import aiohttp

        await ws.send_str(json.dumps({"type": "reset"}))
        deadline = time.monotonic() + self.reset_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                msg = await ws.receive(timeout=min(remaining, 0.4))
            except (asyncio.TimeoutError, TimeoutError):
                continue
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                return False
            if msg.type == aiohttp.WSMsgType.TEXT and self._is_reset_ack(msg.data):
                return True

    @staticmethod
    def _is_reset_ack(data: str) -> bool:
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(event, dict) and event.get("type") == "session_reset"

    async def _run_turn(self, ws: Any, user_pcm: np.ndarray) -> tuple[np.ndarray, str, float | None, float | None]:
        """Send one user clip, commit the turn, collect the reply."""
        started = time.monotonic()
        clip_s = user_pcm.shape[-1] / SAMPLE_RATE
        commit = _Commit()
        streamer: asyncio.Task[None] | None = None
        if self.pace_realtime:
            streamer = asyncio.create_task(self._stream(ws, user_pcm, commit))
            # The reply cannot start before the clip is over, so the server gets
            # ``response_timeout_s`` from *there* rather than from the turn's
            # start. Clips here run to 16s against a 20s budget, so charging the
            # playback to the server would time out the longest turns on a
            # pipeline that was answering perfectly well.
            deadline_from = started + clip_s
        else:
            # The whole clip in one message, then the commit: nothing is in
            # flight by the time the server starts, so the budget starts now.
            await self._send_turn(ws, user_pcm, commit)
            deadline_from = commit.at or started
        out_chunks: list[np.ndarray] = []
        text_parts: list[str] = []
        try:
            first_text_at, first_audio_at, last_activity = await self._recv_reply(
                ws, out_chunks, text_parts, deadline_from=deadline_from
            )
        finally:
            if streamer is not None:
                streamer.cancel()
                try:
                    await streamer
                except (asyncio.CancelledError, Exception):
                    pass

        # A turn can end while the server is still talking (either wall-clock cap
        # in _recv_reply). The server drops inbound audio until its reply is done,
        # so sending the next clip now would lose its head — and the leftover
        # audio would be collected as *that* turn's reply, misaligning every turn
        # after it. Wait for real quiet first. A turn that ended on the idle rule
        # has already seen it, so this returns without waiting.
        await self._settle(ws, last_activity)

        # The TTS is what the listener waits on; the LLM's text lands earlier and
        # is only a fallback for a turn that produced no audio at all.
        reply_at = first_audio_at if first_audio_at is not None else first_text_at
        # Latency is measured from the end of the user's turn, not from the start
        # of their clip: the clip is played at real time, so anything else bills
        # the server for the seconds the user spent talking. The commit is that
        # boundary; without one (an uncommitted or cancelled stream) fall back to
        # where real-time pacing puts the clip's last frame.
        turn_end_at = commit.at if commit.at is not None else started + clip_s
        latency = reply_at - turn_end_at if reply_at is not None else None
        early_s = self._warn_if_early(latency, clip_s)
        pcm = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
        return pcm.astype(np.float32), "".join(text_parts).strip(), latency, early_s

    @staticmethod
    def _warn_if_early(latency: float | None, clip_s: float) -> float | None:
        """Warn when the reply began before the user's turn was over.

        ``latency`` is measured from the end of the user's turn, so a negative one
        is the server answering a *prefix*: it endpointed on a pause inside the
        utterance instead of waiting for the clip (or for the ``end_of_turn`` that
        follows it). The tail is then either dropped while the server talks or
        picked up as its own pseudo-turn, which drags every later turn out of
        alignment. That scores as a bad model rather than a bad session, so it has
        to be loud: at ``min_silence_ms 64`` it fires on ~87% of turns, and with
        the server's ``--manual_turn_end`` it should fire on none.

        Returns how early the reply was, in seconds, or ``None`` when the turn is
        sound — which :meth:`_make_response` puts on the reply as
        ``raw["early_reply_s"]`` for the caller to record.
        """
        if latency is None or latency >= 0:
            return None
        early_s = -latency
        logger.warning(
            "reply began %.2fs before the end of a %.2fs user turn: the server endpointed "
            "mid-utterance, so this turn answers a prefix (run it with --manual_turn_end)",
            early_s,
            clip_s,
        )
        return early_s

    async def _send_turn(self, ws: Any, user_pcm: np.ndarray, commit: _Commit | None = None) -> None:
        """Hand the whole clip over in one message, then commit — no pacing, no padding.

        The bytes are the same bytes ``_stream`` would have sent frame by frame,
        so what the server buffers for this turn is identical either way; only
        the 512-sample framing and the real-time gaps between frames are gone,
        along with the trailing silence (there is no VAD here to keep fed, and a
        turn-based server closes the turn on the commit alone).
        """
        await self._send_pcm_frame(ws, user_pcm)
        if self.commit_turns:
            await ws.send_str(json.dumps({"type": "end_of_turn"}))
            if commit is not None:
                commit.at = time.monotonic()

    async def _stream(self, ws: Any, user_pcm: np.ndarray, commit: _Commit | None = None) -> None:
        """Stream the clip at real time, commit the turn, then silence until cancelled.

        The commit goes out on the same socket as the audio, so the server sees
        it strictly after every frame of the clip — it cannot cut the turn short
        and it need not wait out a silence to know the turn is over. The trailing
        silence still follows: the server keeps its VAD fed, and a server without
        ``--manual_turn_end`` still needs the silence to endpoint at all.
        """
        silence = np.zeros(FRAME_SIZE, dtype=np.float32)
        for frame in self._frames(user_pcm):
            await self._send_pcm_frame(ws, frame)
            await asyncio.sleep(FRAME_DT)
        if self.commit_turns:
            await ws.send_str(json.dumps({"type": "end_of_turn"}))
            if commit is not None:
                commit.at = time.monotonic()
        while True:
            await self._send_pcm_frame(ws, silence)
            await asyncio.sleep(FRAME_DT)

    async def _send_pcm_frame(self, ws: Any, frame: np.ndarray) -> None:
        await ws.send_bytes(self._to_int16_bytes(frame))

    async def _recv_reply(
        self,
        ws: Any,
        out_chunks: list[np.ndarray],
        text_parts: list[str],
        deadline_from: float | None = None,
    ) -> tuple[float | None, float | None, float]:
        """Collect the reply's audio/text until the server goes quiet (wall clock).

        The server sends no explicit "done" marker, so the turn ends once we have
        produced audio/text and then seen ``silence_stop_s`` of real-time quiet.
        Because ``_stream`` paces audio at playback speed, a wall-clock gap is a
        real conversational pause rather than an artifact of how fast we sent.

        ``min_reply_s`` floors that quiet gap in *wall clock* since the first
        output, not in samples produced: the server sends nothing while it is
        quiet, so a sample floor is a gate the idle rule can never clear — a turn
        answered in text alone, or in less than ``min_reply_s`` of speech, would
        wait for a reply that is already over. ``max_reply_s`` caps the same span,
        so a server that never stops talking still ends the turn.

        ``deadline_from`` is when ``response_timeout_s`` starts counting: the end
        of the user's turn, since the server has heard nothing to answer before
        that. It defaults to now, which is right for a caller with no clip to
        play out.

        Returns the monotonic timestamps of the first ``assistant_text`` event and
        the first audio frame (either may be ``None``) — which :meth:`_run_turn`
        turns into this turn's latency — plus the last moment the server was
        heard from, which is how much quiet it has already observed.
        """
        import aiohttp

        produced_at: float | None = None  # monotonic time of first reply audio/text
        first_text_at: float | None = None
        first_audio_at: float | None = None
        last_activity = time.monotonic()  # last audio/text (or the turn's start)
        started = last_activity if deadline_from is None else deadline_from
        total_samples = 0
        max_samples = int(self.max_reply_s * SAMPLE_RATE)
        # Poll often enough to notice a quiet gap between the server's sends.
        poll = min(self.silence_stop_s, 0.4)

        while True:
            try:
                msg = await ws.receive(timeout=poll)
            except (asyncio.TimeoutError, TimeoutError):
                msg = None  # a quiet tick; fall through to the idle checks
            if msg is not None:
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                    break
                if msg.type == aiohttp.WSMsgType.BINARY and msg.data:
                    pcm = self._from_int16_bytes(msg.data)
                    if pcm.shape[-1]:
                        out_chunks.append(pcm)
                        total_samples += pcm.shape[-1]
                        last_activity = time.monotonic()
                        produced_at = produced_at or last_activity
                        first_audio_at = first_audio_at or last_activity
                elif msg.type == aiohttp.WSMsgType.TEXT and msg.data:
                    kind, text = self._read_event(msg.data)
                    if kind == "response_failed":
                        # The server has given up on this turn; nothing more is
                        # coming, so end it now instead of waiting out the timeout
                        # and recording the stall as a silent reply.
                        logger.warning("pipeline reported response_failed: %s", text)
                        break
                    if kind == "assistant_text" and text:
                        text_parts.append(text)
                        last_activity = time.monotonic()
                        produced_at = produced_at or last_activity
                        first_text_at = first_text_at or last_activity

            now = time.monotonic()
            if total_samples >= max_samples:
                break
            if produced_at is None:
                # Nothing back yet — give the STT+LLM+TTS chain time to start.
                if now - started >= self.response_timeout_s:
                    break
                continue
            # Reply seen: end on a real quiet gap, but never below the floor.
            if now - produced_at >= self.max_reply_s:
                break
            if now - produced_at >= self.min_reply_s and now - last_activity >= self.silence_stop_s:
                break

        return first_text_at, first_audio_at, last_activity

    @staticmethod
    def _read_event(data: str) -> tuple[str, str]:
        """Split a pipeline event (JSON) into its type and its text payload.

        ``assistant_text`` carries a chunk of the reply, ``response_failed`` the
        reason the turn produced none. Everything else (transcriptions, VAD, token
        usage) is returned with its type for the caller to ignore.
        """
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return "", ""
        if not isinstance(event, dict):
            return "", ""
        kind = event.get("type")
        text = event.get("text") if "text" in event else event.get("message")
        return (kind if isinstance(kind, str) else ""), (text if isinstance(text, str) else "")

    async def _settle(self, ws: Any, last_activity: float) -> None:
        """Read and discard until the server has been quiet for ``silence_stop_s``.

        ``last_activity`` is when it was last heard from, so a turn that already
        ended on the idle rule returns immediately. Budgeted by
        ``response_timeout_s`` — how long this backend ever waits on the server —
        and deliberately not by ``max_reply_s``: that is the cap whose firing is
        the reason there is anything left to drain.
        """
        import aiohttp

        deadline = time.monotonic() + self.response_timeout_s
        while time.monotonic() - last_activity < self.silence_stop_s:
            if time.monotonic() >= deadline:
                return
            try:
                msg = await ws.receive(timeout=min(self.silence_stop_s, 0.4))
            except (asyncio.TimeoutError, TimeoutError):
                continue  # a quiet tick; the while-condition decides
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                return
            if msg.data:  # the tail of the reply we already cut, or a late event
                last_activity = time.monotonic()

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _to_int16_bytes(frame: np.ndarray) -> bytes:
        """Float PCM in [-1, 1] -> little-endian int16 bytes (the wire format)."""
        clipped = np.clip(np.asarray(frame, dtype=np.float32), -1.0, 1.0)
        return (clipped * 32767.0).round().astype("<i2").tobytes()

    @staticmethod
    def _from_int16_bytes(data: bytes) -> np.ndarray:
        """Little-endian int16 wire bytes -> mono float32 PCM in [-1, 1]."""
        # A partial trailing byte can't form an int16 sample; drop it.
        usable = len(data) - (len(data) % 2)
        arr = np.frombuffer(memoryview(data)[:usable], dtype="<i2")
        return (arr.astype(np.float32) / 32768.0).reshape(-1)

    @staticmethod
    def _frames(pcm: np.ndarray) -> Iterator[np.ndarray]:
        """Yield fixed-size mono frames, zero-padding the final partial frame."""
        n = pcm.shape[-1]
        for start in range(0, n, FRAME_SIZE):
            frame = pcm[start : start + FRAME_SIZE]
            if frame.shape[-1] < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - frame.shape[-1]))
            yield np.ascontiguousarray(frame, dtype=np.float32)

    @classmethod
    def load_audio(cls, source: str | Path | bytes, fmt: str = "wav") -> np.ndarray:
        """Decode an audio file — or a container's raw bytes — to mono float32 PCM at 16 kHz.

        Same contract as
        :meth:`~rolebreak.models.backends.moshi_api.MoshiAPIModel.load_audio`: a
        path takes its container format from its suffix, ``bytes`` declare theirs
        with ``fmt``, so an in-memory clip needs no file on disk.
        """
        if isinstance(source, bytes | bytearray | memoryview):
            return cls._to_mono_16k(bytes(source), fmt)
        p = Path(source)
        return cls._to_mono_16k(p.read_bytes(), p.suffix.lstrip(".").lower() or fmt)

    def _extract_user_pcm(self, request: ChatRequest) -> np.ndarray:
        """Decode the last user audio in the request to 16 kHz mono float PCM."""
        audio_part = self._last_user_audio(request.messages)
        if audio_part is None:
            raise ValueError(
                "SpeechPipelineAPIModel needs user audio: no input_audio part found in the "
                "request. The pipeline is audio-in/audio-out only — text-only turns are unsupported."
            )
        ia = audio_part.input_audio
        raw = decode_b64(ia.data)
        return self._to_mono_16k(raw, ia.format)

    @staticmethod
    def _extract_persona(request: ChatRequest) -> str | None:
        """Return the persona from the request's last ``system`` message, if any."""
        for msg in reversed(request.messages):
            if msg.role == "system" and isinstance(msg.content, str) and msg.content.strip():
                return msg.content.strip()
        return None

    @staticmethod
    def _last_user_audio(messages: list[Message]) -> InputAudioContent | None:
        for msg in reversed(messages):
            if msg.role != "user" or not isinstance(msg.content, list):
                continue
            for part in reversed(msg.content):
                if isinstance(part, InputAudioContent):
                    return part
        return None

    @staticmethod
    def _to_mono_16k(raw: bytes, fmt: str) -> np.ndarray:
        """Decode container bytes to mono float32 PCM at 16 kHz using sphn."""
        import sphn

        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(raw)
            tmp_path = Path(f.name)
        try:
            try:
                data, _ = sphn.read(str(tmp_path), sample_rate=SAMPLE_RATE)
            except TypeError:  # older sphn without resample kwarg
                data, src_sr = sphn.read(str(tmp_path))
                if src_sr != SAMPLE_RATE:
                    data = sphn.resample(data, src_sample_rate=src_sr, dst_sample_rate=SAMPLE_RATE)
        finally:
            tmp_path.unlink(missing_ok=True)

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:  # (channels, samples) -> mono
            arr = arr.mean(axis=0)
        return arr.reshape(-1)
