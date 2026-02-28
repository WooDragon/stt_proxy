#!/bin/bash
# ASR 服务启动脚本（后台模式，手动管理用）
# 用法: ./start_service.sh
# 停止: ./stop_service.sh
# 注意: 与 launchd 互斥，不要同时使用

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
set -a; source "$SCRIPT_DIR/.env" 2>/dev/null; set +a

# 应用额外 PATH（Homebrew ffmpeg 等依赖）
[[ -n "$EXTRA_PATH" ]] && export PATH="$EXTRA_PATH:$PATH"

# 相对路径展开为绝对路径（HF_HUB_CACHE=./models → /abs/path/models）
[[ "$HF_HUB_CACHE" == ./* ]] && export HF_HUB_CACHE="$SCRIPT_DIR/${HF_HUB_CACHE#./}"
[[ "$WESPEAKER_CACHE" == ./* ]] && export WESPEAKER_CACHE="$SCRIPT_DIR/${WESPEAKER_CACHE#./}"

PORT="${PORT:-10241}"
HOST="${HOST:-0.0.0.0}"
LOG="$SCRIPT_DIR/logs/asr_service.log"

# 检查端口是否被占用
if lsof -i :"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
  echo "错误: 端口 $PORT 已被占用"
  lsof -i :"$PORT" -sTCP:LISTEN
  exit 1
fi

# 后台启动 mlx-audio server（通过 run_server.py 注入本地 patches）
# --workers 1: 单 GPU 推理无需多 worker，且保证 monkey-patch 在请求处理进程内生效
PYTHON="${PYTHON_BIN:-python3}"
nohup "$PYTHON" "$SCRIPT_DIR/run_server.py" \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  > "$LOG" 2>&1 &

PID=$!
echo $PID > "$SCRIPT_DIR/logs/asr_service.pid"
echo "服务已启动 | PID: $PID | 端口: $PORT | 日志: $LOG"

# 等待服务就绪（进程死了立刻报错，不傻等）
for i in $(seq 1 60); do
  if ! kill -0 $PID 2>/dev/null; then
    echo "错误: 服务进程已退出，启动失败。检查日志: $LOG"
    rm -f "$SCRIPT_DIR/logs/asr_service.pid"
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "服务就绪"
    "$SCRIPT_DIR/scripts/warmup.sh"
    exit 0
  fi
  sleep 1
done

echo "错误: 服务启动超时（60s），请检查日志 $LOG"
exit 1
