#!/bin/bash
# 向已运行的服务发送静音文件触发模型首次加载
# 用法: ./scripts/warmup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
set -a; source "$ROOT_DIR/.env" 2>/dev/null; set +a

PORT="${PORT:-10241}"
MODEL="${MODEL:-mlx-community/Qwen3-ASR-1.7B-8bit}"

curl -sf -X POST "http://localhost:$PORT/v1/audio/transcriptions" \
  -F "file=@$ROOT_DIR/assets/silence.wav" \
  -F "model=$MODEL" > /dev/null 2>&1 \
  && echo "模型预热完成" || echo "预热跳过（非致命）"
