"""Deploy the pipeline first (see
``rolebreak/models/backends/speech_pipeline_api.py`` and ``docs/models.md``), e.g.::

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
      --responses_api_api_key "EMPTY" --responses_api_disable_thinking True \\
      --responses_api_max_output_tokens 512 \\
      --chat_size 64 --compact_history False

Extra deps (install once): ``aiohttp numpy sphn webdataset`` (sphn only decodes
input WAVs, webdataset reads the user-turn shards).

Usage
-----
    # one example; text turns are spoken from data/audio by default
    python -m tools.eval.eval_speech_pipeline --example ip-conan

    # synthesize the spoken user turns first if they aren't in data/audio yet
    python -m tools.synthesize_user_turns ip-conan --out-dir data/audio

    # every example instead of one; either way the out-dir holds one <example>.tar
    # run shard per example, indexed in runs.jsonl, with a shared chat.log
    python -m tools.eval.eval_speech_pipeline --all

    # an interrupted run is picked back up by default — the examples runs.jsonl
    # already lists are skipped; replay every example instead
    python -m tools.eval.eval_speech_pipeline --all --no-resume

    # score what came out
    python -m tools.score_run --run-dir var/generation/speech-pipeline/sad

    # replay an older batch of synthesized clips instead of the newest one
    python -m tools.eval.eval_speech_pipeline --all --audio-version neutral

    # name the backend behind the pipeline: var/generation/speech-pipeline-qwen3omni/<version>
    python -m tools.eval.eval_speech_pipeline --all --tag qwen3omni

    # point at a remote server / tune the turn cutoff
    python -m tools.eval.eval_speech_pipeline --example ip-conan --url ws://10.0.0.5:8765 \\
        --silence-stop 1.5 --max-reply 20

    # several servers: the examples are handed out to whichever is free and
    # replayed in parallel, one conversation (one worker thread) per --url
    python -m tools.eval.eval_speech_pipeline --all \\
        --url ws://10.0.0.5:8765 --url ws://10.0.0.6:8765
"""

from __future__ import annotations

import contextlib
import fcntl
import queue
import re
import threading
from pathlib import Path

import click
from tqdm import tqdm

from rolebreak.examples import EXAMPLES
from rolebreak.models import has_audio_frames, load_model
from rolebreak.models.audio import decode_b64
from rolebreak.runlog import RUNS_FILENAME, append_run
from rolebreak.runshard import RunShardWriter, wav_member
from tools.audio_version import generation_dir, label, read_dir
from tools.eval.resume import recorded_examples
from tools.eval.user_turns import has_shard, load_synth_audio, report_drift, resolve_turn_audio


# Serializes the runs.jsonl appends of a parallel (multi --url) run. The workers
# are threads in one process now, so an in-process lock would cover them; this
# file lock stays because it also covers a *second* eval process pointed at the
# same out-dir (a different set of --url servers filling in the same run).
RUNS_LOCK = ".runs.lock"


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def run_example(
    example_name: str,
    *,
    model,
    model_id: str,
    root: Path,
    audio_dir: str,
    emit,
) -> dict:
    """Replay one example over a single pipeline session into its run shard.

    Returns the run's row for the consolidated ``runs.jsonl`` — only reached once
    the shard is published, so what the index lists and what is on disk agree.
    """
    example = EXAMPLES[example_name]
    synth_audio = load_synth_audio(audio_dir, example_name)

    emit(f"\n{'#' * 72}\n# example: {example.name}\n{'#' * 72}")
    emit(f"system> {example.system_prompt}")
    emit(
        "(sent to the server as a text_prompt control message on connect, so it replaces the server's default persona.)"
    )

    # Resolve + decode every turn's audio up front — straight from the shard bytes —
    # so a missing clip fails before we open the connection for nothing.
    clips = [resolve_turn_audio(turn, synth_audio.get(i)) for i, turn in enumerate(example.turns)]
    user_pcms = [model.load_audio(clip.source) for clip in clips]
    report_drift(example.turns, clips, emit)

    shard = RunShardWriter(
        root,
        example=example.name,
        persona=example.system_prompt,
        model=model_id,
        voice=None,  # the pipeline's TTS voice is fixed by the server config
    )

    # The persona goes over the wire per conversation: the server restores its
    # configured default on every new connection, so each example must send its
    # own or it answers as the stock assistant — and scoring would grade it
    # against a system prompt it never received.
    replies_iter = model.converse(user_pcms, text_prompt=example.system_prompt)
    with shard, contextlib.closing(replies_iter) as replies:
        for i, resp in enumerate(replies, start=1):
            turn = example.turns[i - 1]
            clip = clips[i - 1]
            reply = (resp.transcript or "").strip()

            emit(f"\n{'=' * 72}\n[{i:02d}/{len(example.turns)}]")
            emit(f"  user> {turn.label}")
            emit(f"  in  > {clip.ref}")
            emit(f"  bot > {reply or '(no assistant text received)'}")
            if resp.latency is not None:
                emit(f"  lat > {resp.latency:.2f}s")
            # The backend sets this when the reply started before the clip had
            # finished streaming — the turn answers a prefix, so the score is the
            # session's fault, not the model's. Grep the log for it before
            # trusting a run: `grep -c 'early>' chat*.log`.
            early_s = (resp.raw or {}).get("early_reply_s")
            if early_s is not None:
                emit(f"  early> reply began {early_s:.2f}s before the user clip ended (VAD cut the turn)")

            # An audio payload can come back carrying no samples when the turn
            # produced no speech (empty transcript, no latency). Storing it would
            # leave a header-only WAV in the shard that reads as a spoken turn but
            # has nothing in it to score, so record the turn as unspoken instead.
            spoke = resp.audio is not None and has_audio_frames(resp.audio)
            wav_bytes = decode_b64(resp.audio.data) if spoke else None
            if wav_bytes is not None:
                emit(f"  wav > {shard.path}#{wav_member(example.name, i - 1)}")

            shard.add_turn(
                i - 1,
                user_text=clip.text,
                assistant_text=reply,
                user_audio=clip.ref,
                wav=wav_bytes,
                accepted_emotions=turn.accepted_emotions,
                expected_rubric=turn.rubric,
                latency=resp.latency,
            )
    return shard.row()


@click.command()
@click.option(
    "--example",
    "example_name",
    default="ip-conan",
    show_default=True,
    type=click.Choice(sorted(EXAMPLES)),
    help="Which example conversation to run (ignored when --all is set).",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Replay every example rather than the one --example names; the out-dir is laid out the same either way.",
)
@click.option(
    "--model",
    "model_id",
    default="speech-pipeline",
    show_default=True,
    help="Label only; the server picks the STT/LLM/TTS weights.",
)
@click.option(
    "--url",
    "urls",
    multiple=True,
    default=("ws://localhost:8765",),
    show_default=True,
    help="Pipeline server WebSocket URL; repeat it to spread the examples over "
    "several servers and replay them in parallel. One conversation per URL — a "
    "--mode websocket server feeds every connected client through one shared "
    "pipeline, so a second session against the same URL would splice the two "
    "conversations together. The exception is the turn-based mock "
    "(scripts/s2s_mock), whose sessions are per-connection: repeat the same URL "
    "to run several conversations on one endpoint (two is its throughput sweet spot).",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(),
    help="Root for run shards/logs; the run lands in <out-dir>/<audio-version>[.<revision>]/. Defaults to var/{model} (path-safe).",
)
@click.option(
    "--tag",
    default=None,
    help="Suffix on the default out-dir name, i.e. var/generation/<model>-<tag>/ — the model label says "
    "'speech-pipeline', so this is where the backend weights it was actually serving go. "
    "Ignored when --out-dir is given.",
)
@click.option(
    "--audio-dir",
    default="data/audio",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Speak text turns from the tools/synthesize_user_turns.py shards under this dir (its --out-dir).",
)
@click.option(
    "--audio-version",
    default=None,
    help="Which audio version under --audio-dir to speak, e.g. 'angry' or 'neutral.1'; "
    "'none' reads clips straight from --audio-dir. Default: the newest neutral version present.",
)
@click.option(
    "--bump-version",
    "bump_version",
    is_flag=True,
    default=False,
    help="Generate into the next revision of this audio version -- neutral -> neutral.1 -- keeping the "
    "run already under it. Default writes the newest revision present, which is what --resume "
    "picks back up.",
)
@click.option(
    "--silence-stop",
    "silence_stop_s",
    default=12,
    show_default=True,
    type=float,
    help="End a turn after the server has been quiet (wall clock) this long, once it has replied.",
)
@click.option(
    "--min-reply",
    "min_reply_s",
    default=0.5,
    show_default=True,
    type=float,
    help="Never end a turn within this many seconds of the server's first output.",
)
@click.option(
    "--max-reply",
    "max_reply_s",
    default=120.0,
    show_default=True,
    type=float,
    help="Hard cap on a single reply (seconds).",
)
@click.option(
    "--direct-audio",
    "direct_audio",
    is_flag=True,
    default=False,
    help="Send each user clip in one message instead of playing it at real time. Only for a turn-based "
    "server (scripts/s2s_mock) -- it has no VAD to hear the pauses, so it buffers the clip whole and "
    "the reply is byte-identical, but a live pipeline would hear the utterance as a single instant. "
    "Saves the clip's own duration per turn, which is ~90% of a run's wall clock.",
)
@click.option(
    "--response-timeout",
    "response_timeout_s",
    default=45.0,
    show_default=True,
    type=float,
    help="Give up on a turn if the server sends nothing back within this many seconds.",
)
@click.option(
    "--connect-timeout",
    "connect_timeout_s",
    default=30.0,
    show_default=True,
    type=float,
    help="WebSocket connect timeout (seconds).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Pick an interrupted run back up: skip the examples already recorded in "
    "<out-dir>/runs.jsonl and append to chat.log. --no-resume replays every example, "
    "overwriting both.",
)
def main(
    example_name: str,
    run_all: bool,
    model_id: str,
    urls: tuple[str, ...],
    out_dir: str | None,
    tag: str | None,
    audio_dir: str,
    audio_version: str | None,
    bump_version: bool,
    silence_stop_s: float,
    min_reply_s: float,
    max_reply_s: float,
    response_timeout_s: float,
    direct_audio: bool,
    connect_timeout_s: float,
    resume: bool,
) -> None:
    if out_dir is None:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
        safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag or "").strip("_")
        name = f"{safe_model}-{safe_tag}" if safe_tag else safe_model
        out_dir = f"var/generation/{name}"
    root = Path(out_dir)

    examples = sorted(EXAMPLES) if run_all else [example_name]

    # Synthesized clips live in versioned subdirs of --audio-dir; pin one version for
    # the whole run so every example is spoken from the same batch.
    resolved = read_dir(audio_dir, audio_version)
    audio_dir = str(resolved)
    # Which clips were spoken is part of what a run is, so the version names the
    # output dir too: a replay against a newer batch lands beside the old run
    # rather than on top of it, and a scored run traces back to what it heard.
    spoken_version = label(resolved)
    if spoken_version is not None:
        # A second generation against the same clips is a run of its own, so it gets
        # its own revision of that version rather than being mixed into the first.
        root = generation_dir(root, spoken_version, bump=bump_version)
    elif bump_version:
        raise click.UsageError(
            f"--bump-version numbers a generation against its audio version, and {audio_dir} "
            "has none -- name a versioned batch with --audio-version, or pass --out-dir yourself."
        )

    click.echo(f"Run shards/logs will land under: {root}/")

    root.mkdir(parents=True, exist_ok=True)

    # --example is a --all run with one example in it — same out-dir, same shared
    # chat.log, same runs.jsonl index — so the two differ only in what gets run.
    log_path = root / "chat.log"
    # A resumed run continues the log: what the interrupted run recorded for the
    # examples it did finish is the log of those examples, and they aren't re-run.
    log_file = log_path.open("a" if resume else "w")

    def emit(line: str = "") -> None:
        # tqdm.write, not click.echo: plain writes to stdout tear the bar apart.
        # tqdm.write(line)
        log_file.write(line + "\n")
        log_file.flush()

    # An audio version only holds the examples that were synthesized into it, and
    # a text turn has nothing to send without its clip: drop the examples this
    # batch never spoke rather than failing the whole run on the first one.
    unspoken = [name for name in examples if not has_shard(audio_dir, name)]
    if unspoken:
        note = f"Skipping {len(unspoken)} example(s) with no shard under {audio_dir}: {', '.join(unspoken)}"
        emit(note)
        click.echo(note)
        examples = [name for name in examples if name not in set(unspoken)]
    if not examples:
        log_file.close()
        click.echo(f"Nothing to run — synthesize with tools/synthesize_user_turns.py --out-dir {audio_dir}")
        return

    # One model per URL, each the sole owner of its server's conversation. They
    # are built here rather than inside the workers so a bad --url or setting
    # raises before any example starts; construction opens no connection.
    model_kwargs = dict(
        silence_stop_s=silence_stop_s,
        min_reply_s=min_reply_s,
        max_reply_s=max_reply_s,
        response_timeout_s=response_timeout_s,
        connect_timeout_s=connect_timeout_s,
        pace_realtime=not direct_audio,
    )
    sessions = [(url, load_model("speech-pipeline-api", model_id, url=url, **model_kwargs)) for url in urls]
    emit(f"Using speech pipeline ({model_id}) at: {', '.join(urls)}")
    emit(f"User-turn audio: {audio_dir}")
    emit(f"Concurrent conversations: {len(sessions)} (one per server)")
    emit(f"Running {len(examples)} example(s).")

    # Every finished example is appended to runs.jsonl as it lands, so the file is
    # the record of what got through — of this run and of any earlier one against
    # this out-dir, including the runs written before the shards. That makes it the
    # one thing a resumed run has to read to know what not to replay.
    combined_path = root / RUNS_FILENAME
    if not resume:
        # Truncate once, then each example appends its own row again.
        combined_path.write_text("", encoding="utf-8")
    done = recorded_examples(combined_path) if resume else set()
    already = [name for name in examples if name in done]
    if already:
        note = f"Skipping {len(already)} example(s) already in {combined_path.name}: {', '.join(already[:6])}" + (
            " ..." if len(already) > 6 else ""
        )
        emit(note)
        click.echo(note)
    pending = [name for name in examples if name not in done]

    # Handed out on demand rather than dealt up front. What this buys is not
    # balance — examples run 10-35 turns, but over a few hundred of them a
    # round-robin deal already evens out to within ~1% of the queue's makespan —
    # it is that a *failure* now costs one example instead of a slice. A server
    # that drops out mid-run leaves the rest of the queue there for the others to
    # drain, where a static split stranded everything dealt to it and ended the
    # run short. It is also what lets a server take more than one conversation
    # later on, since the parallelism is no longer pinned one-slice-per-URL.
    todo: queue.Queue[str] = queue.Queue()
    for name in pending:
        todo.put(name)

    # Ctrl-C stops the hand-out; the examples already in flight still finish, so
    # their shards are published and indexed instead of abandoned mid-conversation.
    stop = threading.Event()
    # The workers share this process's log file and runs.jsonl, so both writes are
    # serialized here. (The file lock inside `record` is the cross-*process* half.)
    output_lock = threading.Lock()
    failures: list[tuple[str, str]] = []  # (url, example) of the ones that raised
    progress = tqdm(total=len(pending), desc="examples", unit="ex", disable=len(pending) <= 1)

    def example_log():
        """An ``emit`` scoped to one example, plus the ``flush`` that commits it.

        With one server the lines go straight out, so watching a single-URL run
        still shows the conversation as it happens. With several, the examples
        overlap in time, so each one's lines are held and written as a single
        block: a chat.log with two conversations spliced together line by line is
        not something a scored run can be read against.
        """
        if len(sessions) <= 1:
            return emit, lambda: None
        held: list[str] = []

        def emit_held(line: str = "") -> None:
            held.append(line)

        def flush() -> None:
            if held:
                log_file.write("\n".join(held) + "\n")
                log_file.flush()
                held.clear()

        return emit_held, flush

    def record(row: dict) -> None:
        """Index one finished example, and count it on the bar."""
        # Indexed only once its shard is on disk, so an interrupted run leaves
        # runs.jsonl listing exactly the shards that are there to be scored. A row
        # can be long, so writers take a lock rather than trusting append writes
        # not to interleave.
        with output_lock, (root / RUNS_LOCK).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            append_run(combined_path, row)
            progress.update(1)

    def replay_worker(url: str, model) -> None:
        """Replay whatever is left in the queue over the one server at ``url``.

        One of these per URL, so a server is never asked to hold two conversations
        at once. It stops at the first example that raises — the same server is
        unlikely to answer the next one — and leaves the queue to the others.
        """
        with model:
            while not stop.is_set():
                try:
                    name = todo.get_nowait()
                except queue.Empty:
                    return
                emit_ex, flush = example_log()
                try:
                    row = run_example(
                        name,
                        model=model,
                        model_id=model_id,
                        root=root,
                        audio_dir=audio_dir,
                        emit=emit_ex,
                    )
                except Exception:
                    failures.append((url, name))
                    raise
                finally:
                    # An example that raised still gets its lines written: its
                    # shard is left unpublished so the next run replays it, and
                    # the log is where the turn it died on is recorded.
                    with output_lock:
                        flush()
                record(row)

    threads = [
        threading.Thread(target=replay_worker, args=(url, model), name=f"replay-{i}", daemon=True)
        for i, (url, model) in enumerate(sessions)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except BaseException:
        # Ctrl-C: stop handing out examples, then wait out the conversations that
        # are already running. A second Ctrl-C interrupts that wait and the
        # workers, being daemons, go with the process — the shards they were
        # writing stay .partial and are replayed by the next --resume run.
        stop.set()
        for thread in threads:
            thread.join()
        raise
    finally:
        progress.close()
        log_file.close()

    if failures:
        # Whatever each worker did finish is already indexed, so a resumed run
        # picks up from there — this only says the run is incomplete.
        detail = ", ".join(f"{url} on {name}" for url, name in failures)
        click.echo(f"\n{len(failures)} worker(s) stopped on an error: {detail}", err=True)
        click.echo(f"Tracebacks are above; {log_path} has each conversation up to the turn it died on.", err=True)

    skipped = f" ({len(already)} skipped)" if already else ""
    click.echo(f"\nDone. {len(examples)} example(s){skipped}. Outputs under: {root}/")
    click.echo(f"Chat log: {log_path}")
    click.echo(f"Run index: {combined_path}")
    click.echo(f"Score with: python -m tools.score_run --run-dir {root}")


if __name__ == "__main__":
    main()
