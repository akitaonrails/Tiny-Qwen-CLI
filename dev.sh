#!/usr/bin/env bash
# Enhanced development script for Qwen CLI

set -euo pipefail

# Configuration directories
CONFIG_DIR="$HOME/.config/qwen_cli"
MODELS_DIR="$CONFIG_DIR/models"
CACHE_DIR="$CONFIG_DIR/.cache"

# Create directories if they don't exist
mkdir -p "$MODELS_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$CONFIG_DIR"

# Check if the model is already downloaded
if [ ! -d "$MODELS_DIR/Qwen2.5-Coder-14B-Instruct" ]; then
  echo "Model not found. Downloading from Hugging Face (this may take a while)..."
  git clone --depth 1 https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct.git \
    "$MODELS_DIR/Qwen2.5-Coder-14B-Instruct"
  echo "Model download complete!"
else
  echo "Model already downloaded."
fi

# Build the Docker image if it doesn't exist
if ! docker image inspect qwen-cli:latest >/dev/null 2>&1; then
  echo "Building Docker image..."
  docker build -t qwen-cli:latest -f Dockerfile.txt .
  echo "Docker image built successfully!"
else
  echo "Docker image already exists."
fi

# Check if GPU is available
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected!"
    GPU_FLAG="--gpus all"
  else
    echo "NVIDIA drivers installed but no GPU detected. Running in CPU mode."
    GPU_FLAG=""
  fi
else
  echo "No NVIDIA drivers detected. Running in CPU mode."
  GPU_FLAG=""
fi

# Parse command line arguments
COMMAND=${1:-"new"}
shift 2 2>/dev/null || true # Shift to get remaining args, ignore error if not enough args

# Run the container
echo "Starting Qwen CLI"
docker run --rm -it \
  $GPU_FLAG \
  -v .:/project \
  -v "$MODELS_DIR":/models:rw \
  -v "$CACHE_DIR":/home/ubuntu/.cache:rw \
  -v "$CONFIG_DIR":/home/ubuntu/.config/qwen_cli:rw \
  -e TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface \
  -e CONFIG_DIR=/home/ubuntu/.config/qwen_cli \
  -e MODELS_DIR=/models \
  qwen-cli:latest \
  $COMMAND "$@"
