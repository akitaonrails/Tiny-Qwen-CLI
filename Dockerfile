# 1) Use the official PyTorch 2.3.0 CUDA 12.1 runtime image
#    (this image already bundles torch + cudnn)
#    Tag exists on Docker Hub: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime :contentReference[oaicite:0]{index=0}
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 2) Add git, venv support, pip
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential python3-venv python3-pip python-is-python3 git \
 && rm -rf /var/lib/apt/lists/*

# 3) Create an unprivileged user and /models directory
RUN useradd -ms /bin/bash ubuntu \
 && mkdir -p /models \
 && chown ubuntu:ubuntu /models

USER ubuntu
WORKDIR /app

# 4) Create a venv and install only the lightweight Python deps
RUN python -m venv /home/ubuntu/venv \
 && /home/ubuntu/venv/bin/pip install --no-cache-dir \
      transformers \
      accelerate \
      safetensors \
      bitsandbytes \
      xformers

# Ensure the venv’s python/pip are first on PATH
ENV PATH="/home/ubuntu/venv/bin:$PATH"

# 5) Copy in your CLI code and give it execute rights
COPY --chown=ubuntu:ubuntu . /app
RUN chmod +x /app/qwen_cli.py

# 6) Entrypoint uses the venv python (via PATH)
ENTRYPOINT ["python", "/app/qwen_cli.py"]
