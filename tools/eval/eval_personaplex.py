"""Usage
-----
    # one example; text turns are spoken from data/audio by default
    python -m tools.eval.eval_personaplex --example ip-conan

    # synthesize the spoken user turns first if they aren't in data/audio yet
    python -m tools.synthesize_user_turns ip-conan --out-dir data/audio

    # every example instead of one; either way the out-dir holds one <example>.tar
    # run shard per example, indexed in runs.jsonl, with a shared chat.log
    python -m tools.eval.eval_personaplex --all

    # an interrupted run is picked back up by default — the examples runs.jsonl
    # already lists are skipped; replay every example instead
    python -m tools.eval.eval_personaplex --all --no-resume

    # score what came out
    python -m tools.score_run --run-dir var/generation/nvidia_personaplex-7b-v1/sad

    # replay an older batch of synthesized clips instead of the newest one
    python -m tools.eval.eval_personaplex --all --audio-version neutral

    # pick the PersonaPlex voice, point at a remote server / tune the turn cutoff
    python -m tools.eval.eval_personaplex --example ip-conan --voice NATM1.pt \\
        --host 10.0.0.5 --port 8998 --silence-stop 1.5 --max-reply 20
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import click

from rolebreak.examples import EXAMPLES
from rolebreak.models import has_audio_frames, load_model
from rolebreak.models.audio import decode_b64
from rolebreak.models.backends.personaplex_api import PersonaplexAPIModel
from rolebreak.runlog import RUNS_FILENAME, append_run
from rolebreak.runshard import RunShardWriter, wav_member
from tools.audio_version import label, read_dir
from tools.eval.resume import recorded_examples
from tools.eval.user_turns import has_shard, load_synth_audio, report_drift, resolve_turn_audio


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def run_example(
    example_name: str,
    *,
    model: PersonaplexAPIModel,
    model_id: str,
    voice: str,
    root: Path,
    audio_dir: str,
    emit,
) -> dict:
    """Replay one example over a single duplex session into its run shard.

    Returns the run's row for the consolidated ``runs.jsonl`` — only reached once
    the shard is published, so what the index lists and what is on disk agree.
    """
    example = EXAMPLES[example_name]
    synth_audio = load_synth_audio(audio_dir, example_name)

    emit(f"\n{'#' * 72}\n# example: {example.name}\n{'#' * 72}")
    emit(f"system> {example.system_prompt}")
    emit(f"voice > {voice}")

    # Resolve + decode every turn's audio up front — straight from the shard bytes —
    # so a missing clip fails before we hold the model on the GPU for nothing.
    clips = [resolve_turn_audio(turn, synth_audio.get(i)) for i, turn in enumerate(example.turns)]
    user_pcms = [model.load_audio(clip.source) for clip in clips]
    report_drift(example.turns, clips, emit)

    shard = RunShardWriter(
        root,
        example=example.name,
        persona=example.system_prompt,
        model=model_id,
        voice=voice,  # PersonaPlex voice-conditioning embedding used this run
    )
    # PersonaPlex honors the persona: the example's system prompt is the session's
    # role prompt, fixed for this whole conversation.
    session = model.converse(user_pcms, text_prompt=example.system_prompt)
    with shard, contextlib.closing(session) as replies:
        for i, resp in enumerate(replies, start=1):
            turn = example.turns[i - 1]
            clip = clips[i - 1]
            reply = (resp.transcript or "").strip()

            emit(f"\n{'=' * 72}\n[{i:02d}/{len(example.turns)}]")
            emit(f"  user> {turn.label}")
            emit(f"  in  > {clip.ref}")
            emit(f"  bot > {reply or '(no text tokens received)'}")
            if resp.latency is not None:
                emit(f"  lat > {resp.latency:.2f}s")

            # An audio payload can come back carrying no samples when the turn
            # produced no speech. Storing it would leave a header-only WAV in the
            # shard that reads as a spoken turn but has nothing in it to score, so
            # record the turn as unspoken instead.
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
    default="nvidia/personaplex-7b-v1",
    show_default=True,
    help="Label only; the server picks the weights.",
)
@click.option(
    "--voice",
    default="NATF2.pt",
    show_default=True,
    help="PersonaPlex voice-conditioning embedding filename on the server (e.g. NATF2.pt, NATM1.pt, VARF3.pt).",
)
@click.option("--url", default=None, help="Full ws URL, e.g. ws://host:8998/api/chat (overrides host/port).")
@click.option("--host", default="localhost", show_default=True, help="PersonaPlex server host.")
@click.option("--port", default=8998, show_default=True, type=int, help="PersonaPlex server port.")
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(),
    help="Root for run shards/logs; the run lands in <out-dir>/<audio-version>/. Defaults to var/{model} (path-safe).",
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
    "--silence-stop",
    "silence_stop_s",
    default=1.0,
    show_default=True,
    type=float,
    help="End a turn after this much PersonaPlex silence (VAD).",
)
@click.option(
    "--warmup",
    "warmup_s",
    default=2.0,
    show_default=True,
    type=float,
    help="Grace period after the user clip before a turn may end (lets the model start talking).",
)
@click.option(
    "--min-reply",
    "min_reply_s",
    default=1.0,
    show_default=True,
    type=float,
    help="Never end a turn before this many seconds of output.",
)
@click.option(
    "--max-reply",
    "max_reply_s",
    default=30.0,
    show_default=True,
    type=float,
    help="Hard cap on a single reply (seconds).",
)
@click.option(
    "--vad-threshold", default=0.5, show_default=True, type=float, help="Silero speech probability threshold."
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
    "--prime/--no-prime",
    default=True,
    show_default=True,
    help="Consume PersonaPlex's spontaneous opening turn before sending the first user turn.",
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
    voice: str,
    url: str | None,
    host: str,
    port: int,
    out_dir: str | None,
    audio_dir: str,
    audio_version: str | None,
    silence_stop_s: float,
    warmup_s: float,
    min_reply_s: float,
    max_reply_s: float,
    vad_threshold: float,
    connect_timeout_s: float,
    prime: bool,
    resume: bool,
) -> None:
    if out_dir is None:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
        out_dir = f"var/generation/{safe_model}"
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
        root = root / spoken_version

    root.mkdir(parents=True, exist_ok=True)

    # --example is a --all run with one example in it — same out-dir, same shared
    # chat.log, same runs.jsonl index — so the two differ only in what gets run.
    log_path = root / "chat.log"
    # A resumed run continues the log: what the interrupted run recorded for the
    # examples it did finish is the log of those examples, and they aren't re-run.
    log_file = log_path.open("a" if resume else "w")

    def emit(line: str = "") -> None:
        click.echo(line)
        log_file.write(line + "\n")
        log_file.flush()

    # An audio version only holds the examples that were synthesized into it, and
    # a text turn has nothing to send without its clip: drop the examples this
    # batch never spoke rather than failing the whole run on the first one.
    unspoken = [name for name in examples if not has_shard(audio_dir, name)]
    if unspoken:
        emit(f"Skipping {len(unspoken)} example(s) with no shard under {audio_dir}: {', '.join(unspoken)}")
        examples = [name for name in examples if name not in set(unspoken)]
    if not examples:
        log_file.close()
        click.echo(f"Nothing to run — synthesize with tools/synthesize_user_turns.py --out-dir {audio_dir}")
        return

    model = load_model(
        "personaplex-api",
        model_id,
        voice_prompt=voice,
        url=url,
        host=host,
        port=port,
        silence_stop_s=silence_stop_s,
        warmup_s=warmup_s,
        min_reply_s=min_reply_s,
        max_reply_s=max_reply_s,
        vad_threshold=vad_threshold,
        connect_timeout_s=connect_timeout_s,
        prime=prime,
    )
    emit(f"Using duplex PersonaPlex ({model_id}, voice={voice}) at {host}:{port} ...")
    emit(f"User-turn audio: {audio_dir}")
    emit(f"Running {len(examples)} example(s): {', '.join(examples)}")

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
        emit(
            f"Skipping {len(already)} example(s) already in {combined_path.name}: {', '.join(already[:6])}"
            + (" ..." if len(already) > 6 else "")
        )
    pending = [name for name in examples if name not in done]

    with contextlib.ExitStack() as stack:
        if pending:  # nothing to say to the server if every example is already recorded
            stack.enter_context(model)
        for name in pending:
            row = run_example(
                name,
                model=model,
                model_id=model_id,
                voice=voice,
                root=root,
                audio_dir=audio_dir,
                emit=emit,
            )
            # Indexed only once its shard is on disk, so an interrupted run leaves
            # runs.jsonl listing exactly the shards that are there to be scored.
            append_run(combined_path, row)

    log_file.close()
    skipped = f" ({len(already)} skipped)" if already else ""
    click.echo(f"\nDone. {len(examples)} example(s){skipped}. Outputs under: {root}/")
    click.echo(f"Chat log: {log_path}")
    click.echo(f"Run index: {combined_path}")
    click.echo(f"Score with: python -m tools.score_run --run-dir {root}")


if __name__ == "__main__":
    main()
