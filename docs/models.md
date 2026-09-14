## Download

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct
hf download Qwen/Qwen2.5-Omni-7B
hf download openbmb/MiniCPM-o-4_5
hf download tencent/Covo-Audio-Chat
HF_TOKEN=hf_xxx HF_HUB_DISABLE_XET=1 hf download nvidia/personaplex-7b-v1
```

## How to launch

> All launch configurations below are tested on **NVIDIA L20** GPUs (48 GB).
> GPU counts, memory-utilization ratios and stage overrides may need adjusting on other hardware.

### Qwen/Qwen3-Omni-30B-A3B-Instruct

```bash
docker run --name qwen3-omni --runtime nvidia --gpus '"device=3,4,5"' \
    --restart unless-stopped \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8901:8901 \
    --ipc=host \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90 \
    -e TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
    -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
    -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/root/.cache/nccl_trace \
    vllm/vllm-omni:v0.26.0 \
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --omni \
        --host 0.0.0.0 --port 8901 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --allowed-local-media-path / \
        --stage-overrides '{
          "0": {
            "tensor_parallel_size": 2,
            "devices": "0,1",
            "gpu_memory_utilization": 0.9,
            "async_scheduling": false
          },
          "1": {
            "tensor_parallel_size": 1,
            "devices": "2",
            "gpu_memory_utilization": 0.6,
            "async_scheduling": false,
            "max_num_seqs": 8
          },
          "2": {
            "tensor_parallel_size": 1,
            "devices": "2",
            "gpu_memory_utilization": 0.1
          }
        }'
```

```bash
curl http://localhost:8901/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
        {
            "role": "user",
            "content": "Introduce yourself in one short sentence."
        }
        ],
        "modalities": ["text", "audio"],
        "speaker": "chelsie"
    }'
```

### Qwen/Qwen2.5-Omni-7B

```bash
docker run -d \
  --name qwen2.5-omni-7b \
  --runtime nvidia \
  --gpus '"device=0,7"' \
  --restart unless-stopped \
  -v ~/.cache:/root/.cache \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  --ipc=host \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90 \
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
  -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/root/.cache/nccl_trace \
  vllm/vllm-omni:v0.26.0 \
  vllm serve Qwen/Qwen2.5-Omni-7B \
    --omni \
    --host 0.0.0.0 \
    --port 8091 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    --stage-overrides '{
      "0": {"devices": "0", "gpu_memory_utilization": 0.85},
      "1": {"devices": "1", "gpu_memory_utilization": 0.55},
      "2": {"devices": "1", "gpu_memory_utilization": 0.30}
    }'
```

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "modalities": ["text", "audio"],
        "speaker": "Chelsie"
    }'
```

### openbmb/MiniCPM-o-4_5

```bash
docker build -f dockerfile/Dockerfile.vllm-omni -t rolebreak:vllm-omni .

docker run -d \
  --name minicpm-o-4_5 \
  --runtime nvidia \
  --gpus '"device=0,2"' \
  --restart unless-stopped \
  -v ~/.cache:/root/.cache \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8092:8092 \
  --ipc=host \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90 \
  rolebreak:vllm-omni \
  vllm serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --host 0.0.0.0 \
    --port 8092 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    --deploy-config /usr/local/lib/python3.12/dist-packages/vllm_omni/deploy/minicpmo_4_5_2gpu.yaml \
    --stage-overrides '{
      "0": {"devices": "0", "gpu_memory_utilization": 0.85, "max_num_seqs": 16, "enable_prefix_caching": true},
      "1": {"devices": "1", "gpu_memory_utilization": 0.40},
      "2": {"devices": "1", "gpu_memory_utilization": 0.25}
    }'
```

```bash
curl http://localhost:8092/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

### tencent/Covo-Audio-Chat

```bash
docker build -f dockerfile/Dockerfile.vllm-omni -t rolebreak:vllm-omni .

docker run -d \
  --name covo-audio-chat \
  --runtime nvidia \
  --gpus '"device=5,7"' \
  --restart unless-stopped \
  -v ~/.cache:/root/.cache \
  -p 8093:8093 \
  --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90 \
  rolebreak:vllm-omni \
  vllm serve tencent/Covo-Audio-Chat \
    --omni \
    --host 0.0.0.0 \
    --port 8093 \
    --trust-remote-code \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    --stage-overrides '{
      "0": {"devices": "1", "gpu_memory_utilization": 0.80, "max_num_seqs": 16, "enable_prefix_caching": true},
      "1": {"devices": "0", "gpu_memory_utilization": 0.20, "max_num_seqs": 4}
    }'
```

```bash
curl http://localhost:8093/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tencent/Covo-Audio-Chat",
        "messages": [
            {"role": "system", "content": "你是\"小腾\"，英文名是\"Covo\"，由腾讯开发的AI助手。\n1、请使用简洁、口语化的语言和用户聊天，你的态度积极、耐心，像一位值得信赖的朋友。\n2、不要使用列表或编号，避免输出网址、表情符号和复杂的公式。\n3、不评价竞争对手，不发表主观政治观点，针对色情类、政治类、恐怖类、歧视类、暴力类的用户问题，你要妥善应对潜在的安全风险，并给出幽默，情绪安抚以及安全的劝导。\n请用文本和音频进行对话，交替生成5个文本token和15个音频token，音频部分使用发音人：default_female"},
            {"role": "user", "content": "Say hello in one short sentence."}
        ],
        "modalities": ["text", "audio"]
    }'
```

### Moshi

```bash
cd external/moshi/moshi && \
docker build \
  -f Dockerfile \
  -t rolebreak:moshi \
  .
```

```bash
docker run --rm -it \
  --name moshi \
  --gpus '"device=0"' \
  --network host \
  -v "$HOME/.cache:/root/.cache" \
  -v "$PWD/external/moshi/moshi/moshi:/app/moshi" \
  rolebreak:moshi \
  uv run -p 3.10 \
    -m moshi.server \
    --host 0.0.0.0 \
    --hf-repo kyutai/moshika-pytorch-bf16
```

### PersonaPlex

PersonaPlex (`nvidia/personaplex-7b-v1`) is a full-duplex model built on the
Moshi architecture, adding persona control (a text role prompt) and voice
conditioning. It is a **gated** model — accept the license at
<https://huggingface.co/nvidia/personaplex-7b-v1> and export `HF_TOKEN` before
launching.

```bash
cd external/personaplex && \
docker build \
  -f Dockerfile \
  -t rolebreak:personaplex \
  .

docker run --rm -d \
  --name personaplex \
  --gpus '"device=2"' \
  --network host \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env NO_TORCH_COMPILE=1 \
  -v "$HOME/.cache:/root/.cache" \
  -v "$PWD/external/personaplex/moshi/moshi:/app/moshi" \
  rolebreak:personaplex \
  uv run -p 3.12 \
    -m moshi.server \
    --host 0.0.0.0
```

### Speech to Speech

#### asr + llm + tts

##### Qwen3.5-2B
```
docker run -d \
    --name qwen3.5-2b-vllm \
    --runtime nvidia \
    --gpus '"device=0"' \
    -v ~/.cache:/root/.cache \
    -v <path to Qwen3.5-2B>:/models/Qwen3.5-2B \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8901:8901 \
    --ipc=host \
    rolebreak:vllm \
    --model /models/Qwen3.5-2B \
    --host 0.0.0.0 \
    --port 8901 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

##### Qwen3.5-4B
```
docker run -d \
    --name qwen3.5-4b-vllm \
    --runtime nvidia \
    --gpus '"device=0"' \
    -v ~/.cache:/root/.cache \
    -v <path to Qwen3.5-4B>:/models/Qwen3.5-4B \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8901:8901 \
    --ipc=host \
    rolebreak:vllm \
    --model /models/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 8901 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

##### Qwen3.5-9B 
```
docker run -d \
    --name qwen3.5-9b-vllm \
    --runtime nvidia \
    --gpus '"device=0"' \
    -v ~/.cache:/root/.cache \
    -v <path to Qwen3.5-9B>:/models/Qwen3.5-9B \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8901:8901 \
    --ipc=host \
    rolebreak:vllm \
    --model /models/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8901 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

##### Qwen3.5-27B

```bash
docker run -d \
    --name qwen3.5-27b-vllm \
    --runtime nvidia \
    --gpus '"device=0,2"' \
    -v ~/.cache:/root/.cache \
    -v <path to Qwen3.5-27B>:/models/Qwen3.5-27B \
    --env "HF_TOKEN=$HF_TOKEN" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -p 8901:8901 \
    --ipc=host \
    rolebreak:vllm \
    --model /models/Qwen3.5-27B \
    --served-model-name /models/Qwen3.5-27B \
    --host 0.0.0.0 \
    --port 8901 \
    --dtype auto \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.72
```

##### Test

```bash
curl http://localhost:8901/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/models/Qwen3.5-2B",
      "messages": [
        {
          "role": "user",
          "content": "Hello!"
        }
      ],
      "temperature": 0,
      "max_tokens": 256,
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    }'
```

##### s2s
```bash
docker run -d -it \
  --name speech-pipeline-5 \
  --gpus '"device=5"' \
  --network host \
  -v "$HOME/.cache:/root/.cache" \
  -v "$PWD/external/speech-to-speech/src:/usr/src/app/src" \
  speech-pipeline \
  speech-to-speech \
  --mode websocket \
  --ws_port 8766 \
  --min_silence_ms 640 \
  --manual_turn_end True \
  --llm_backend chat-completions \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_device cuda \
  --qwen3_tts_backend torch \
  --qwen3_tts_language auto \
  --qwen3_tts_non_streaming_mode True \
  --model_name /models/Qwen3.5-27B \
  --responses_api_base_url http://localhost:8901/v1 \
  --responses_api_api_key "EMPTY" \
  --responses_api_disable_thinking True \
  --responses_api_max_output_tokens 512 \
  --chat_size 64 \
  --compact_history False
```

###### Turn-based mock (no streaming)

Everything above is a *live* pipeline being driven by a benchmark, which puts two
guesses in the loop: where the user's turn ended (the VAD's job, fixed by
`--manual_turn_end`) and when the reply ended (nothing fixes this — the server
sends no end-of-response marker, so a client waits out a silence and a slow
non-streaming TTS looks exactly like a finished reply). The second one is not a
slow run but a corrupt one: the late audio is collected as the *next* turn's
reply, so one bad turn takes a good one with it.

[`scripts/s2s_mock/`](../scripts/s2s_mock/README.md) removes both guesses. Same
three models, built by the pipeline's own `get_*_handler` from the same parsed
arguments, but no VAD and nothing streamed: the turn ends where the client
commits it, and between the commit and the finished reply the server sends
nothing at all. Two changes to the launch command above — mount the directory,
run `python -m s2s_mock` instead of `speech-to-speech`:

```bash
docker run -d --rm --name s2s-mock-8765 --gpus '"device=7"' --network host \
    -v "$HOME/.cache:/root/.cache" \
    -v "$PWD/external/speech-to-speech/src:/usr/src/app/src" \
    -v "$PWD/scripts/s2s_mock:/usr/src/app/s2s_mock" \
    speech-pipeline \
    python -m s2s_mock --ws_port 8765 \
    --llm_backend chat-completions \
    --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --qwen3_tts_device cuda --qwen3_tts_backend torch \
    --qwen3_tts_language auto --qwen3_tts_non_streaming_mode True \
    --model_name /models/Qwen3.5-4B \
    --responses_api_base_url http://localhost:8901/v1 \
    --responses_api_api_key "EMPTY" --responses_api_disable_thinking True \
    --responses_api_max_output_tokens 512 \
    --chat_size 64 --compact_history False
```

```bash
docker ps -aq --filter name=speech-pipeline --filter name=s2s-mock | xargs -r docker rm -f

# for gp in 2:8765 5:8766 7:8767; do
for gp in 5:8766; do
    gpu=${gp%%:*}; port=${gp##*:}; \
    docker run -d --rm --name s2s-mock-$port --gpus "\"device=$gpu\"" --network host \
        -v "$HOME/.cache:/root/.cache" \
        -v "$PWD/external/speech-to-speech/src:/usr/src/app/src" \
        -v "$PWD/scripts/s2s_mock:/usr/src/app/s2s_mock" \
        speech-pipeline python -m s2s_mock --ws_port $port \
        --llm_backend chat-completions \
        --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --qwen3_tts_device cuda --qwen3_tts_backend torch \
        --qwen3_tts_language auto --qwen3_tts_non_streaming_mode True \
        --model_name /models/Qwen3.5-2B \
        --responses_api_base_url http://localhost:8901/v1 \
        --responses_api_api_key "EMPTY" --responses_api_disable_thinking True \
        --responses_api_max_output_tokens 512 \
        --chat_size 64 --compact_history False; \
done

for p in 8765 8766 8767; do until docker logs s2s-mock-$p 2>&1 | grep -q "listening on ws"; do sleep 3; done; echo "s2s-mock-$p ready"; done
```
