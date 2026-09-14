"""Launch the turn-based mock with the real pipeline's own command line.

Run it inside the ``speech-pipeline`` image, with the same flags you would give
``speech-to-speech``::

    python -m s2s_mock --ws_port 8765 --llm_backend chat-completions ...

Every argument the real server accepts is parsed by the real server's parser, so
the launch command needs no translation. VAD flags (``--min_silence_ms``,
``--manual_turn_end``, ...) are accepted and ignored: there is no VAD here, and
the turn boundary is whatever the client commits.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from speech_to_speech.s2s_pipeline import parse_arguments, prepare_all_args, setup_logger

from .pipeline import TurnPipeline
from .server import MockServer

logger = logging.getLogger("s2s_mock")


def _split_mock_args() -> argparse.Namespace:
    """Pull the mock's own flags out of ``sys.argv`` before the pipeline parses it.

    The pipeline's parser rejects arguments it does not know, so anything added
    here has to be removed from ``sys.argv`` first.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mock_max_user_audio_s",
        type=float,
        default=60.0,
        help="Cap on one user turn's buffered audio (seconds); the most recent audio wins. Default 60.",
    )
    mock_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    return mock_args


def main() -> None:
    mock_args = _split_mock_args()
    args = parse_arguments()
    setup_logger(args.module_kwargs.log_level)

    if args.module_kwargs.mode != "websocket":
        logger.info("--mode %s ignored: the mock only speaks the websocket protocol", args.module_kwargs.mode)

    prepare_all_args(
        args.module_kwargs,
        args.whisper_stt_handler_kwargs,
        args.paraformer_stt_handler_kwargs,
        args.faster_whisper_stt_handler_kwargs,
        args.mlx_audio_whisper_stt_handler_kwargs,
        args.parakeet_tdt_stt_handler_kwargs,
        args.language_model_handler_kwargs,
        args.responses_api_language_model_handler_kwargs,
        args.chat_tts_handler_kwargs,
        args.facebook_mms_tts_handler_kwargs,
        args.pocket_tts_handler_kwargs,
        args.kokoro_tts_handler_kwargs,
        args.qwen3_tts_handler_kwargs,
    )

    pipeline = TurnPipeline(args)
    server = MockServer(
        pipeline,
        host=args.websocket_streamer_kwargs.ws_host,
        port=args.websocket_streamer_kwargs.ws_port,
        max_user_audio_s=mock_args.mock_max_user_audio_s,
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
