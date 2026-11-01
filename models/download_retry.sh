#!/bin/bash
# Retry the official Qwen3-VL-4B-Instruct download until both weight shards are complete.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for i in $(seq 1 30); do
  echo "=== attempt $i $(date '+%H:%M:%S')"
  .venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/Qwen3-VL-4B-Instruct 2>&1 | grep -v "Still waiting" | tail -3
  if [ -f models/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors ] && [ -f models/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors ]; then
    echo "DOWNLOAD COMPLETE"; exit 0
  fi
  sleep 20
done
echo "DOWNLOAD FAILED after 30 attempts"; exit 1
