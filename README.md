# RoleBreak

A harness for benchmarking **role consistency** in speech-to-speech models: it
replays a library of role-play conversations against a model, records the audio
and text of every turn, and scores the recorded run.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment from the lockfile:

```bash
uv sync
```

That is the whole install — `uv` reads `.python-version`, fetches CPython 3.13
if it is missing, and builds `.venv/` from `uv.lock`. It is one flat dependency
list, so the eval side and the GPU-side scoring judges both work out of the box
(at the cost of pulling torch — expect ~6.6 GB and a few minutes on a cold cache).

Run anything in it with `uv run` (no activation needed), or
`source .venv/bin/activate` if you prefer:

```bash
uv run python -m tools.eval.eval_qwen3_omni --help
```

Some judges additionally need a vendored checkout under `external/` and a
downloaded checkpoint — see [`rolebreak/metrics/README.md`](rolebreak/metrics/README.md).

## User-turn audio

The spoken user turns are released on the Hub, one webdataset shard per scenario.
Download them into `data/audio` — that is where every eval runner reads them from:

```bash
# the neutral delivery (~2.2 GB) — what the examples below replay
uv run hf download Greenbean/RoleBreak --repo-type dataset \
    --include 'data/audio/neutral/*' --local-dir .

# or all four emotions (~8.7 GB): neutral, angry, sad, happy
uv run hf download Greenbean/RoleBreak --repo-type dataset \
    --include 'data/audio/*' --local-dir .
```

Each emotion lands in its own directory, which is what `--audio-version` selects
between. The same dataset also carries the recorded runs behind the paper's
numbers under `data/generation/`. To re-synthesize the clips yourself instead
(GPU, CosyVoice), see `tools/synthesize_user_turns.py` and the command lines in
[`docs/launch.md`](docs/launch.md).

## Usage

Model servers are launched separately (vLLM, Moshi, PersonaPlex, the speech
pipeline) — see [`docs/models.md`](docs/models.md). With one running and the
user-turn audio in place, the two stages are:

```bash
# 1. Replay the benchmark against a model and record the run
uv run python -m tools.eval.eval_qwen3_omni --all \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --base-url http://localhost:8901/v1 \
    --audio-version neutral --bump-version

# 2. Score a recorded run (GPU for the audio metrics)
uv run python tools/score_run.py \
    --run-dir var/generation/Qwen_Qwen3-Omni-30B-A3B-Instruct/neutral \
    --metric naturalness
```

Full command lines for every model and metric are in
[`docs/launch.md`](docs/launch.md).

## Development

Install the pre-commit hooks once (configured in `.pre-commit-config.yaml`):

```bash
uv run pre-commit install
```

Hooks then run on `git commit`. To run them over the whole tree:

```bash
uv run pre-commit run --all-files
```

Lint, format and type-check directly:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```

## Layout

| Path         | What's in it                                                      |
| ------------ | ----------------------------------------------------------------- |
| `rolebreak/` | The library: model backends, the role library, voice banks, metrics |
| `tools/`     | CLI entry points — user-turn synthesis, per-model evals, scoring   |
| `scripts/`   | Helpers that run *inside* a model container (e.g. `s2s_mock/`)     |
| `external/`  | Vendored upstream checkouts (CosyVoice, Moshi, PersonaPlex, …)     |
| `docs/`      | Model launch commands and run recipes                              |
