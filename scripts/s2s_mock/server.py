"""The websocket half: the real server's protocol, answered one whole turn at a time.

The wire format is the pipeline's own ``WebSocketStreamer`` protocol, so an
existing client needs no changes:

client -> server
    binary                              int16 LE mono @16 kHz, appended to the
                                        turn being buffered (sent at any rate --
                                        there is no VAD to keep fed and no
                                        real-time pacing to respect)
    ``{"type": "reset"}``               forget the conversation, restore the
                                        configured persona; answered with
                                        ``session_reset``
    ``{"type": "text_prompt", ...}``    set this conversation's persona
    ``{"type": "end_of_turn"}``         the user has stopped talking; answered
                                        with ``turn_committed``, then the turn
                                        runs

server -> client, once the turn is completely finished
    ``{"type": "transcription_completed", ...}``
    ``{"type": "assistant_text", "text": <the whole reply>}``
    binary                              the whole reply's audio, back to back
    ``{"type": "response_done", ...}``  explicit end of turn, with stage timings
    ``{"type": "response_failed", ...}``  instead, when the turn produced no audio

The one invariant worth stating plainly: **between the commit and the finished
reply the server sends nothing at all.** No partial text, no first audio block,
no keep-alive. A client therefore cannot mistake a slow turn for a finished one,
which is the failure the mock exists to remove -- so do not be tempted to send
progress here.

The cost of that invariant is that the client's response timeout has to cover
the entire turn (STT + LLM + TTS), not just the wait for a first token. See
``README.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .pipeline import SAMPLE_RATE, TurnPipeline, TurnResult, trim_padding

logger = logging.getLogger(__name__)

# The reply goes out in frames this big, back to back with no pause. One frame
# would be fine for a client with no message-size limit, but a few large ones
# cost nothing and keep the mock usable from a client that caps message size.
AUDIO_FRAME_BYTES = 256 * 1024


@dataclass
class Session:
    """One websocket connection: its own history, persona and turn counter."""

    runtime_config: Any
    buffer: bytearray = field(default_factory=bytearray)
    turns: int = 0


class MockServer:
    """Serves the pipeline protocol, turn-based, on one port."""

    def __init__(
        self,
        pipeline: TurnPipeline,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_user_audio_s: float = 60.0,
    ) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.max_user_samples = int(max_user_audio_s * SAMPLE_RATE)
        # Sessions are per-connection, but the handlers behind them are shared,
        # so the end-of-session reset is only safe once nobody is left to
        # disturb. Several clients on one port is a supported way to run this
        # (see the note in pipeline.TurnPipeline).
        self._live = 0

    async def serve(self) -> None:
        import websockets

        async with websockets.serve(self._handle_client, self.host, self.port, max_size=None):
            logger.info("s2s mock listening on ws://%s:%d (turn-based)", self.host, self.port)
            await asyncio.Future()  # serve until cancelled

    # -- connection -------------------------------------------------------- #
    async def _handle_client(self, ws: Any) -> None:
        client = id(ws)
        self._live += 1
        logger.info("client %s connected (%d live)", client, self._live)
        session = Session(runtime_config=self.pipeline.new_session())
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    session.buffer.extend(message)
                    continue
                await self._handle_text(ws, session, message, client)
        except Exception:
            logger.exception("client %s failed", client)
        finally:
            self._live -= 1
            # Resetting shared handler state while another session is mid-turn
            # would clobber it, so only the last one out does it.
            if self._live == 0:
                self.pipeline.end_session()
            logger.info("client %s disconnected after %d turn(s)", client, session.turns)

    async def _handle_text(self, ws: Any, session: Session, message: str, client: int) -> None:
        try:
            event = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            logger.warning("client %s: ignoring non-JSON text message", client)
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")

        if kind == "reset":
            session.runtime_config = self.pipeline.new_session()
            session.buffer.clear()
            session.turns = 0
            logger.info("client %s: session reset", client)
            await ws.send(json.dumps({"type": "session_reset"}))
            return

        if kind == "text_prompt":
            text = event.get("text")
            if not isinstance(text, str):
                logger.warning("client %s: text_prompt without a string 'text'", client)
                return
            session.runtime_config.session.instructions = text
            logger.info("client %s: persona set (%d chars)", client, len(text))
            return

        if kind == "end_of_turn":
            # Acknowledged before the work starts, exactly as the real server
            # does: the ack is what a client measures its latency from, so it
            # must not wait behind STT.
            committed_at = perf_counter()
            await ws.send(json.dumps({"type": "turn_committed"}))
            await self._run_turn(ws, session, committed_at)
            return

        logger.debug("client %s: ignoring text message of type %r", client, kind)

    # -- one turn ---------------------------------------------------------- #
    async def _run_turn(self, ws: Any, session: Session, committed_at: float) -> None:
        pcm = self._take_turn_audio(session)
        session.turns += 1
        turn_id = f"turn_{session.turns}"
        if pcm.size == 0:
            logger.warning("%s: committed with no audio buffered", turn_id)
            await ws.send(json.dumps({"type": "response_failed", "message": "no user audio", "turn_id": turn_id}))
            return

        # The models are synchronous and hold the GIL for seconds at a time,
        # so they run off-thread: the event loop stays live, and the websocket
        # reader keeps draining the socket into its own frame queue while this
        # turn runs. The client's trailing silence therefore never backs up
        # against a full socket buffer -- it is simply waiting to be appended
        # to the *next* turn's buffer, whose leading zeros get trimmed.
        result = await asyncio.to_thread(
            self.pipeline.run_turn,
            session.runtime_config,
            pcm,
            turn_id=turn_id,
            committed_at=committed_at,
        )
        await self._send_reply(ws, result, turn_id=turn_id, committed_at=committed_at)

    def _take_turn_audio(self, session: Session) -> np.ndarray:
        """Consume the buffer this turn accumulated, as trimmed int16 samples."""
        raw = bytes(session.buffer)
        session.buffer.clear()
        usable = len(raw) - (len(raw) % 2)
        pcm = trim_padding(np.frombuffer(raw[:usable], dtype=np.int16))
        if pcm.size > self.max_user_samples:
            logger.warning(
                "user turn is %.1fs, keeping the last %.1fs",
                pcm.size / SAMPLE_RATE,
                self.max_user_samples / SAMPLE_RATE,
            )
            pcm = pcm[-self.max_user_samples :]
        return pcm

    async def _send_reply(self, ws: Any, result: TurnResult, *, turn_id: str, committed_at: float) -> None:
        """Send the finished turn as one uninterrupted burst."""
        await ws.send(
            json.dumps(
                {
                    "type": "transcription_completed",
                    "transcript": result.transcript,
                    "turn_id": turn_id,
                }
            )
        )
        if result.reply_text:
            await ws.send(json.dumps({"type": "assistant_text", "text": result.reply_text, "turn_id": turn_id}))
        for start in range(0, len(result.audio), AUDIO_FRAME_BYTES):
            await ws.send(result.audio[start : start + AUDIO_FRAME_BYTES])

        if not result.audio:
            await ws.send(
                json.dumps(
                    {
                        "type": "response_failed",
                        "message": result.error or "no audio produced",
                        "turn_id": turn_id,
                    }
                )
            )
            logger.warning("%s: no audio (%s)", turn_id, result.error)
            return

        total_s = perf_counter() - committed_at
        await ws.send(
            json.dumps(
                {
                    "type": "response_done",
                    "turn_id": turn_id,
                    "stt_s": round(result.stt_s, 3),
                    "llm_s": round(result.llm_s, 3),
                    "tts_s": round(result.tts_s, 3),
                    "total_s": round(total_s, 3),
                    "audio_s": round(result.audio_s, 3),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            )
        )
        logger.info(
            "%s done in %.2fs (stt %.2f + llm %.2f + tts %.2f) -> %.2fs of audio",
            turn_id,
            total_s,
            result.stt_s,
            result.llm_s,
            result.tts_s,
            result.audio_s,
        )
