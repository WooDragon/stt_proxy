#!/bin/bash

echo "STT代理服务 - 启动脚本"
echo "===================="

# 设置Python路径
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "启动STT代理服务..."
python src/stt_proxy.py --config config/stt_config.json