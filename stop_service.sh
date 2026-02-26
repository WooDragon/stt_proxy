#!/bin/bash
# 优雅停止 ASR 服务
# 用法: ./stop_service.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/asr_service.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "服务未运行（PID 文件不存在）"
  exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "进程已不存在，已清理 PID 文件"
  exit 0
fi

echo "停止服务 (PID: $PID)..."
kill "$PID"

# 等待进程退出，最多 15 秒
for i in $(seq 1 15); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "服务已停止"
    exit 0
  fi
  sleep 1
done

# 超时强制终止
kill -9 "$PID" 2>/dev/null
rm -f "$PID_FILE"
echo "已强制停止（SIGKILL）"
