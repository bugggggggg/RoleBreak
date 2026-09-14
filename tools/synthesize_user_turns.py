"""Render the user turns of :class:`~rolebreak.examples.Example` s to audio."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import webdataset

from rolebreak.examples import EXAMPLES, Example, Turn
from rolebreak.models import TextToSpeechModel, load_tts
from rolebreak.voices import VoiceClip, pool_specs
from tools.audio_version import write_dir
from tools.wds import open_writer, sample_key, shard_path, write_sample

# Per-backend default checkpoint (kept in sync with scripts/run_cosyvoice.py); ``None``
# lets the backend pick its own default in __init__.
DEFAULT_MODEL: dict[str, str] = {
    "cosyvoice3": "pretrained_models/Fun-CosyVoice3-0.5B",
}

# Output versions are ``<emotion>`` directories directly under --out-dir, named and
# resolved by tools/audio_version.py (shared with the eval runners that read them
# back). The emotions live there; ``--bump-version`` numbers a re-delivery of the
# default one (``neutral`` -> ``neutral.1``), and ``--version`` names one outright.


def _slot(name: str, size: int) -> int:
    """Where in a pool of ``size`` voices ``name`` starts looking.

    A digest of the name rather than its position, so a role's voice depends on the
    role alone: inserting an example shifts no other example's pick (barring a
    collision), and the spread samples the whole bank instead of taking its first N
    speakers, which in VCTK skews to one accent.
    """
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big") % size


def _deal_user_voices(examples: Mapping[str, Example]) -> dict[str, VoiceClip]:
    """Give every example a user voice: its authored one, or a distinct pool speaker."""
    authored = {name: ex.user_voice for name, ex in examples.items() if ex.user_voice is not None}
    _warn_shared("authored the same user_voice", _by_spec({n: str(c.spec) for n, c in authored.items()}))

    need = sorted(name for name, ex in examples.items() if ex.user_voice is None)
    if not need:
        return dict(authored)

    spoken_for = {str(clip.spec) for clip in authored.values()}
    pool = [spec for spec in pool_specs() if spec not in spoken_for]
    if len(need) > len(pool):
        raise click.ClickException(
            f"Only {len(pool)} free pool voice(s) for {len(need)} example(s) without a user_voice"
            f" — {len(need) - len(pool)} short. Collect more reference audio, or add a bank to"
            " rolebreak.voices.USER_VOICE_BANKS."
        )

    dealt: dict[str, str] = {}
    taken: set[str] = set()
    for name in need:
        start = _slot(name, len(pool))
        # The pool is at least as large as ``need``, so a free candidate always exists.
        spec = next(c for c in (pool[(start + step) % len(pool)] for step in range(len(pool))) if c not in taken)
        dealt[name] = spec
        taken.add(spec)

    return {**authored, **{name: VoiceClip(spec) for name, spec in dealt.items()}}


def _by_spec(specs: Mapping[str, str]) -> dict[str, list[str]]:
    """example -> spec, inverted to spec -> the examples sharing it (2+ only)."""
    shared: dict[str, list[str]] = defaultdict(list)
    for name, spec in specs.items():
        shared[spec].append(name)
    return {spec: names for spec, names in shared.items() if len(names) > 1}


def _warn_shared(what: str, shared: Mapping[str, list[str]]) -> None:
    for spec, names in sorted(shared.items()):
        click.secho(f"! {', '.join(sorted(names))} {what} ({spec})", fg="yellow")


def _shard_done(out_dir: Path, name: str, *, write_files: bool, write_wds: bool) -> bool:
    """Is ``name`` already synthesized in full under this version dir?

    A shard is written whole (and moved into place only when its example finished),
    so its presence means every turn is there and the example can be skipped without
    loading the model. Per-turn wavs carry no such promise on their own — an
    interrupted run leaves some of them — so the files format resumes turn by turn
    inside :func:`_synthesize_example` instead, and ``--format both`` additionally
    waits on the ``manifest.json`` that run writes last.
    """
    if not write_wds or not shard_path(out_dir, name).exists():
        return False
    return not write_files or (out_dir / name / "manifest.json").exists()


def _synthesize_example(
    model: TextToSpeechModel,
    name: str,
    ex: Example,
    *,
    voice: VoiceClip | None,
    out_dir: Path,
    synth_kwargs: dict[str, Any],
    skip_existing: bool,
    write_files: bool,
    write_wds: bool,
) -> list[dict[str, Any]]:
    """Synthesize one example's text turns; return its manifest list.

    ``voice`` is what :func:`_deal_user_voices` settled on for this example — its
    authored ``user_voice`` or a pool speaker — clip and transcript both, resolved
    here into a per-example copy of ``synth_kwargs`` so ``--all`` gives every example
    its own user voice. It is recorded on every manifest/metadata entry, so audio on
    disk always says which voice produced it.
    """
    # Keep the original turn index so file names line up with the example's turn
    # order even when audio-only turns are skipped.
    text_turns: list[tuple[int, Turn]] = [(i, t) for i, t in enumerate(ex.turns) if t.text]
    if not text_turns:
        click.echo(f"  {name}: no text turns, skipping")
        return []

    # The voice may be an hf:// voice-bank spec; reading the clip downloads it and
    # hands back the local file plus the transcript the bank ships with it.
    clip = voice
    synth_kwargs = {**synth_kwargs, "voice": str(clip.path) if clip is not None else None}
    if clip is not None and clip.transcript is not None:
        synth_kwargs["prompt_text"] = clip.transcript
    authored = "" if ex.user_voice is None else " (authored)"
    click.echo(f"  voice: {clip.label if clip is not None else '<backend default>'}{authored}")
    voice_spec = None if clip is None else str(clip.spec)
    # Run-wide, but recorded per entry so a wav on disk says how it was delivered as
    # well as by whom; omitted entirely when unset, leaving plain runs' output as-is.
    provenance: dict[str, Any] = {"user_voice": voice_spec}
    if synth_kwargs.get("instruct"):
        provenance["instruct"] = synth_kwargs["instruct"]

    dest = out_dir / name
    if write_files:
        dest.mkdir(parents=True, exist_ok=True)
    wds: webdataset.TarWriter | None = None
    shard = tmp = None
    if write_wds:
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = shard_path(out_dir, name)
        # Written beside the real shard and moved onto it only once the example is
        # done, so a shard at its final path is always a finished one — which is what
        # lets the next run skip that example outright (see _shard_done).
        tmp = shard.with_name(shard.name + ".partial")
        wds = open_writer(tmp)

    manifest: list[dict[str, Any]] = []
    try:
        total = len(text_turns)
        for n, (turn_index, turn) in enumerate(text_turns, start=1):
            wav_path = dest / f"turn_{turn_index:03d}.wav"
            audio_bytes: bytes | None
            # Reuse an existing wav from this version; only the files format can skip
            # (the wds shard is rewritten wholesale each run).
            if skip_existing and write_files and wav_path.exists():
                click.echo(f"  [{name} {n}/{total}] skip (exists)")
                audio_bytes = wav_path.read_bytes() if write_wds else None
            else:
                click.echo(f"  [{name} {n}/{total}] {turn.label[:55]!r}")
                audio = model.synthesize(turn.text, **synth_kwargs)
                audio_bytes = base64.b64decode(audio.data)
                if write_files:
                    wav_path.write_bytes(audio_bytes)

            if wds is not None and audio_bytes is not None:
                write_sample(
                    wds,
                    sample_key(name, turn_index),
                    wav=audio_bytes,
                    text=turn.text,
                    meta={
                        "example": name,
                        "index": turn_index,
                        "text": turn.text,
                        "system": ex.system_prompt,
                        **provenance,
                    },
                )

            entry: dict[str, Any] = {"index": turn_index, "text": turn.text, **provenance}
            if write_files:
                entry["audio"] = str(wav_path)
            manifest.append(entry)
    finally:
        if wds is not None:
            wds.close()

    # Past the try/finally, so only a run that got here in one piece publishes; a
    # failed one leaves its .partial behind for the next run to overwrite.
    if tmp is not None and shard is not None:
        tmp.replace(shard)

    if write_files:
        (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


@click.command()
@click.argument("example", required=False)
@click.option("--all", "run_all", is_flag=True, default=False, help="Synthesize every registered example.")
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["files", "wds", "both"]),
    default="wds",
    show_default=True,
    help="Output: per-turn wav files, a WebDataset .tar shard, or both.",
)
@click.option(
    "--backend",
    default="cosyvoice3",
    show_default=True,
    help="TTS engine: cosyvoice3.",
)
@click.option("--model", "model_id", default=None, help="Model id / checkpoint (defaults per backend).")
@click.option("--language", default=None, help="Language, e.g. 'en' / 'zh' (defaults per backend/voice).")
@click.option(
    "--instruct",
    default=None,
    help="(CosyVoice) natural-language style control applied to every turn, e.g. "
    "'speak slowly, sounding a little worried'. Uses the reference clip's voice but "
    "not its transcript; pair with --version so one emotion means one delivery.",
)
@click.option("--speed", default=1.0, show_default=True, type=float, help="Playback-rate multiplier.")
@click.option("--device", default="auto", show_default=True, help="Device to load on (auto / cpu / cuda).")
@click.option(
    "--out-dir",
    "out_dir",
    default="data/audio",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root dir; each run writes a <out-dir>/<emotion>/ version holding <example>/ and <example>.tar.",
)
@click.option(
    "--bump-version",
    "bump_version",
    is_flag=True,
    default=False,
    help="Start the next revision of the default emotion and re-synthesize everything; default tops up the newest one.",
)
@click.option(
    "--version",
    "version",
    default=None,
    metavar="EMOTION[.N]",
    help="Write this version instead of one picked for you, e.g. 'angry' or 'neutral.2' -- to "
    "synthesize another emotion or re-open an older delivery. Tops it up like the default does; not "
    "combinable with --bump-version.",
)
def main(
    example: str | None,
    run_all: bool,
    out_format: str,
    backend: str,
    model_id: str | None,
    language: str | None,
    instruct: str | None,
    speed: float,
    device: str,
    out_dir: Path,
    bump_version: bool,
    version: str | None,
) -> None:
    names: list[str]
    if run_all:
        if example:
            raise click.UsageError("Pass either an EXAMPLE name or --all, not both.")
        names = sorted(EXAMPLES)
    elif example:
        if example not in EXAMPLES:
            raise click.UsageError(f"Unknown example {example!r}. Available: {sorted(EXAMPLES)}")
        names = [example]
    else:
        raise click.UsageError("Provide an EXAMPLE name or --all.")

    if model_id is None:
        model_id = DEFAULT_MODEL.get(backend)
    write_files = out_format in ("files", "both")
    write_wds = out_format in ("wds", "both")

    # CosyVoice-specific knobs only matter to that backend; pass them through the
    # uniform synthesize(**kwargs) and let other engines ignore the unset ones.
    # The voice comes from each example's own user_voice, so it is injected in
    # _synthesize_example rather than baked in here.
    synth_kwargs: dict[str, Any] = {"language": language, "speed": speed}
    # Left out unless asked for: passing it at all switches CosyVoice to its instruct
    # mode (and other engines would just be handed a kwarg they don't take).
    if instruct:
        synth_kwargs["instruct"] = instruct

    version_dir = write_dir(out_dir, bump=bump_version, version=version)
    click.echo(f"Version: {version_dir}{'' if bump_version else ' (skipping what is already there)'}")
    if instruct:
        click.echo(f"Instruct: {instruct!r} (reference transcript unused in this mode)")

    # Whole examples already sitting in this version are dropped before the model is
    # loaded, so topping up a finished version (or resuming one) costs nothing.
    done_already = set()
    if not bump_version:
        done_already = {n for n in names if _shard_done(version_dir, n, write_files=write_files, write_wds=write_wds)}
    for name in sorted(done_already):
        click.echo(f"  {name}: already synthesized, skipping ({shard_path(version_dir, name)})")
    names = [n for n in names if n not in done_already]
    if not names:
        click.echo(f"\nNothing to do: {len(done_already)} example(s) already synthesized -> {version_dir}/")
        return

    # Dealt across the whole registry, not just `names`, so an example's voice is the
    # same whether it is synthesized alone or with --all.
    user_voices = _deal_user_voices(EXAMPLES)

    click.echo(f"Loading TTS (backend={backend!r}, model={model_id!r}) on {device!r} ...")
    model = load_tts(backend, model_id, device=device)

    total_turns = 0
    with model:
        for name in names:
            click.echo(f"\n== {name} ==")
            manifest = _synthesize_example(
                model,
                name,
                EXAMPLES[name],
                voice=user_voices.get(name),
                out_dir=version_dir,
                synth_kwargs=synth_kwargs,
                skip_existing=not bump_version,
                write_files=write_files,
                write_wds=write_wds,
            )
            total_turns += len(manifest)

    done = f"\nDone: {total_turns} turn(s) across {len(names)} example(s)"
    click.echo(f"{done}{f' ({len(done_already)} already synthesized)' if done_already else ''} -> {version_dir}/")


if __name__ == "__main__":
    main()
