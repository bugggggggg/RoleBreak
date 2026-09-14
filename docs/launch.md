
## Synthesize User Turn

Only needed to regenerate the clips — the released ones download straight
into `data/audio` (see the README).

First fetch the CosyVoice 3 checkpoint into the vendored checkout, under the
directory name the TTS backend looks for:

```bash
uv run hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir external/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
```

Then:

```bash
# neutral
CUDA_VISIBLE_DEVICES=0 \
    uv run python -m tools.synthesize_user_turns --all \
        --version neutral

# angry
CUDA_VISIBLE_DEVICES=7 \
    uv run python -m tools.synthesize_user_turns --all \
        --version angry \
        --instruct "请非常生气地说一句话。"

# sad
CUDA_VISIBLE_DEVICES=3 \
    uv run python -m tools.synthesize_user_turns --all  \
        --version sad \
        --instruct "请非常伤心地说一句话。"

# happy
CUDA_VISIBLE_DEVICES=2 \
    uv run python -m tools.synthesize_user_turns --all  \
        --version happy \
        --instruct "请非常开心地说一句话。"
```

## Eval

### Qwen
```bash
uv run python -m tools.eval.eval_qwen3_omni --all \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --base-url http://localhost:8901/v1 \
    --temperature 0.0 \
    --audio-version neutral \
    --bump-version
```

#### Qwen2.5-Omni

```bash
uv run python -m tools.eval.eval_qwen2p5_omni --all \
    --stream \
    --audio-version neutral \
    --bump-version
```

#### MiniCPM-o 4.5

```bash
uv run python -m tools.eval.eval_minicpm_o_4p5 --all \
    --stream \
    --audio-version neutral \
    --bump-version
```


#### Covo-Audio-Chat
```bash
uv run python -m tools.eval.eval_covo_audio --all \
    --audio-version neutral
```


### Personplex

```bash
uv run python -m tools.eval.eval_personaplex --all \
    --audio-version neutral
```

### Speech-Pipeline

```bash
uv run python -m tools.eval.eval_speech_pipeline --all \
    --audio-version neutral \
    --silence-stop 12 \
    --max-reply 120 \
    --response-timeout 45 \
    --url ws://localhost:8765 \
    --url ws://localhost:8766 \
    --url ws://localhost:8767 \
    --url ws://localhost:8768 \
    --url ws://localhost:8769 \
    --tag Qwen3.5-4B \
    --bump-version
```

```bash
uv run python -m tools.eval.eval_speech_pipeline --all \
    --audio-version neutral \
    --tag Qwen3.5-2B \
    --direct-audio \
    --silence-stop 2 --max-reply 120 --response-timeout 180 \
    --url ws://localhost:8766 --url ws://localhost:8766 \
    --url ws://localhost:8765 --url ws://localhost:8765 \
    --url ws://localhost:8767 --url ws://localhost:8767 \
    --bump-version
```

## Score

```bash
CUDA_VISIBLE_DEVICES=0 \
    uv run python tools/score_run.py \
        --run-dir var/generation/speech-pipeline-Qwen3.5-2B/neutral.1 \
        --metric naturalness

CUDA_VISIBLE_DEVICES=5 \
    uv run python tools/score_run.py \
        --run-dir var/generation/speech-pipeline-Qwen3.5-2B/neutral.1 \
        --metric emotion

OPENAI_MODEL="deepseek-v4-pro" \
OPENAI_BASE_URL="https://api.deepseek.com" \
OPENAI_API_KEY="sk-..." \
    uv run python tools/score_run.py \
        --run-dir var/generation/nvidia_personaplex-7b-v1/neutral \
        --metric text_quality \
        --parallel 32
```
