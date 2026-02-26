#!/bin/bash
# launchd wrapper — 解决 launchd 极简 PATH 问题
# launchd 不读 shell profile，也找不到 conda/venv 的 python
# 此脚本 source .env 拿配置，用绝对路径的 python 前台 exec

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
set -a; source "$ROOT_DIR/.env" 2>/dev/null; set +a

# 相对路径展开为绝对路径
[[ "$HF_HUB_CACHE" == ./* ]] && export HF_HUB_CACHE="$ROOT_DIR/${HF_HUB_CACHE#./}"

exec /Users/woodragon/anaconda3/bin/python -m mlx_audio.server \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-10241}"
