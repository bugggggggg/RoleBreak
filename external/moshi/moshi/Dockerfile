FROM ghcr.io/astral-sh/uv:debian

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# Resolve + install deps into the image at build time so run-time startup is fast
RUN uv sync -p 3.10

#ENTRYPOINT ["bash", "-c", "sleep infinity"]
#ENTRYPOINT ["uv", "run", "-p", "3.10", "-m", "moshi.server", "--host", "0.0.0.0", "--hf-repo", "kyutai/moshika-pytorch-bf16"]
