#!/bin/bash
# launchd wrapper — 解决 launchd 极简 PATH 问题
# launchd 不读 shell profile，也找不到 conda/venv 的 python
# 此脚本 source .env 拿配置，用绝对路径的 python 前台 exec

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
set -a; source "$ROOT_DIR/.env" 2>/dev/null; set +a

# launchd 使用极简 PATH，需要显式添加 Homebrew（ffmpeg 依赖）
# EXTRA_PATH 在 .env 中配置（如 /opt/homebrew/bin:/usr/local/bin）
[[ -n "$EXTRA_PATH" ]] && export PATH="$EXTRA_PATH:$PATH"

# 相对路径展开为绝对路径
[[ "$HF_HUB_CACHE" == ./* ]] && export HF_HUB_CACHE="$ROOT_DIR/${HF_HUB_CACHE#./}"
[[ "$WESPEAKER_CACHE" == ./* ]] && export WESPEAKER_CACHE="$ROOT_DIR/${WESPEAKER_CACHE#./}"

# 通过 run_server.py 注入本地 patches（M4A 格式修复等）
# --workers 1: 单 GPU 推理无需多 worker，且保证 monkey-patch 在请求处理进程内生效
# PYTHON_BIN 在 .env 中配置（如 /Users/woodragon/anaconda3/bin/python）
PYTHON="${PYTHON_BIN:-python3}"
exec "$PYTHON" "$ROOT_DIR/run_server.py" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-10241}" \
  --workers 1
