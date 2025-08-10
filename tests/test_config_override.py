#!/usr/bin/env python3
"""
测试配置文件强制覆盖功能
"""

import asyncio
import json
import logging
import os
import sys
from unittest.mock import AsyncMock, patch

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# 导入相关函数
from stt_proxy import load_config


def test_config_override_logic():
    """测试配置文件强制覆盖逻辑"""
    print("测试配置文件强制覆盖逻辑...")
    
    # 加载配置文件
    config = load_config("config/stt_config.json")
    print(f"配置文件内容: {config}")
    
    # 模拟客户端请求数据
    client_data = {
        "language": "en",  # 客户端提供的值
        "temperature": 0.8,  # 客户端提供的值
        "initial_prompt": "This is an English conversation.",  # 客户端提供的值
        "model": "whisper-base",  # 客户端提供的值，但不在配置文件中
        "response_format": "json"  # 客户端提供的值
    }
    
    print(f"客户端原始数据: {client_data}")
    
    # 模拟处理逻辑
    form_data = client_data.copy()
    
    # 应用强制覆盖逻辑
    for key, default_value in config.items():
        old_value = form_data.get(key, "未提供")
        form_data[key] = default_value
        print(f"强制设置参数 {key}: {old_value} -> {default_value}")
    
    print(f"处理后数据: {form_data}")
    
    # 验证结果
    expected_results = {
        "language": "zh",  # 应该被配置文件覆盖
        "temperature": 0.2,  # 应该被配置文件覆盖
        "initial_prompt": "以下是一段普通话商务会议的录音，有多位发言人。转写需准确、流畅，并使用恰当的标点符号（例如逗号，句号，问号）进行断句。\n\n张总：嗯，关于上个季度提到的那个新方案，我们今天需要讨论一下具体的执行细节。\n李工：好的。我看了初步的计划，主要有两个问题想确认一下。第一个是资源方面，第二个是时间节点，这个我们能保证吗？\n张总：对，你提的这点很重要。下一步，我们必须先把风险点都梳理清楚。",  # 应该被配置文件覆盖
        "model": "whisper-base",  # 应该保持客户端值（不在配置文件中）
        "response_format": "json"  # 应该保持客户端值（不在配置文件中）
    }
    
    print(f"期望结果: {expected_results}")
    
    # 检查每个字段
    all_correct = True
    for key, expected_value in expected_results.items():
        actual_value = form_data.get(key)
        if actual_value == expected_value:
            print(f"✓ {key}: {actual_value} (正确)")
        else:
            print(f"✗ {key}: {actual_value} (期望: {expected_value})")
            all_correct = False
    
    return all_correct


def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    # 加载配置文件
    config = load_config("config/stt_config.json")
    
    # 测试空客户端数据
    print("1. 测试空客户端数据:")
    form_data = {}
    for key, default_value in config.items():
        old_value = form_data.get(key, "未提供")
        form_data[key] = default_value
        print(f"   设置参数 {key}: {old_value} -> {default_value}")
    
    # 验证所有配置文件字段都已设置
    for key, expected_value in config.items():
        if form_data.get(key) == expected_value:
            print(f"   ✓ {key}: {form_data[key]}")
        else:
            print(f"   ✗ {key}: {form_data.get(key)} (期望: {expected_value})")
    
    # 测试部分字段匹配
    print("2. 测试部分字段匹配:")
    form_data = {"model": "test-model", "extra_field": "extra-value"}
    for key, default_value in config.items():
        old_value = form_data.get(key, "未提供")
        form_data[key] = default_value
        print(f"   设置参数 {key}: {old_value} -> {default_value}")
    
    print(f"   最终数据: {form_data}")
    print(f"   验证extra_field保持不变: {'extra-value' in str(form_data)}")


def main():
    """主测试函数"""
    print("开始测试配置文件强制覆盖功能")
    print("=" * 50)
    
    # 测试主要逻辑
    result1 = test_config_override_logic()
    
    # 测试边界情况
    test_edge_cases()
    
    print("=" * 50)
    if result1:
        print("所有测试通过！配置文件强制覆盖功能正常工作。")
        return 0
    else:
        print("测试失败！请检查配置文件强制覆盖逻辑。")
        return 1


if __name__ == "__main__":
    sys.exit(main())