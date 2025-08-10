#!/bin/bash

echo "STT代理服务 - 测试脚本"
echo "===================="

# 设置Python路径
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "1. 运行配置覆盖测试..."
cd tests && python test_config_override.py && cd ..

echo -e "\n2. 运行完整功能测试..."
cd tests && python test_full_functionality.py && cd ..

echo -e "\n所有测试完成！"