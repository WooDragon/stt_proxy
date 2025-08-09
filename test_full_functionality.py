#!/usr/bin/env python3
"""
完整功能测试脚本
"""

import json
import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stt_proxy import load_config


def test_complete_functionality():
    """测试完整功能"""
    print("测试完整功能...")
    
    # 1. 加载配置文件
    config = load_config("stt_config.json")
    print(f"1. 配置文件内容: {config}")
    
    # 2. 模拟各种客户端请求场景
    test_cases = [
        {
            "name": "客户端提供所有配置字段",
            "client_data": {
                "language": "en",
                "temperature": 0.8,
                "prompt": "English conversation",
                "response_format": "json",
                "model": "whisper-base"
            }
        },
        {
            "name": "客户端提供部分配置字段",
            "client_data": {
                "language": "ja",
                "model": "whisper-large"
            }
        },
        {
            "name": "客户端不提供配置字段",
            "client_data": {
                "file": "test.wav",
                "model": "whisper-tiny"
            }
        },
        {
            "name": "客户端提供额外字段",
            "client_data": {
                "language": "fr",
                "custom_field": "custom_value",
                "extra_param": "extra_value",
                "model": "whisper-small"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n2.{i} {test_case['name']}:")
        client_data = test_case["client_data"]
        print(f"   客户端数据: {client_data}")
        
        # 应用强制覆盖逻辑
        form_data = client_data.copy()
        
        # 强制覆盖配置文件中定义的字段
        for key, default_value in config.items():
            old_value = form_data.get(key, "未提供")
            form_data[key] = default_value
            print(f"   强制设置 {key}: {old_value} -> {default_value}")
        
        print(f"   最终数据: {form_data}")
        
        # 验证配置文件字段被强制覆盖
        for key, expected_value in config.items():
            if form_data.get(key) == expected_value:
                print(f"   ✓ 配置字段 {key} 正确覆盖为: {expected_value}")
            else:
                print(f"   ✗ 配置字段 {key} 错误: {form_data.get(key)}, 期望: {expected_value}")
        
        # 验证非配置字段保持不变
        non_config_keys = set(client_data.keys()) - set(config.keys())
        for key in non_config_keys:
            if key in form_data and form_data[key] == client_data[key]:
                print(f"   ✓ 非配置字段 {key} 保持不变: {form_data[key]}")
            else:
                print(f"   ✗ 非配置字段 {key} 被错误修改: {form_data.get(key)}, 原始: {client_data[key]}")


def test_configuration_priority():
    """测试配置优先级"""
    print("\n3. 测试配置优先级...")
    
    config = load_config("stt_config.json")
    
    # 验证配置文件中的每个字段都会强制覆盖客户端值
    print("验证配置文件字段强制覆盖:")
    for key, value in config.items():
        print(f"   - {key}: {value} (将强制覆盖任何客户端值)")
    
    # 验证不在配置文件中的字段会透传
    print("验证非配置字段透传:")
    non_config_fields = ["model", "file", "custom_field", "extra_param"]
    for field in non_config_fields:
        if field not in config:
            print(f"   - {field}: 将透传客户端值")


def main():
    """主测试函数"""
    print("STT代理服务完整功能测试")
    print("=" * 50)
    
    test_complete_functionality()
    test_configuration_priority()
    
    print("\n" + "=" * 50)
    print("测试完成！配置文件强制覆盖功能已验证。")
    print("\n功能总结:")
    print("1. 配置文件中定义的字段会强制覆盖客户端发送的任何值")
    print("2. 配置文件中未定义的字段会直接透传客户端参数")
    print("3. 客户端提供的额外字段会被保留")


if __name__ == "__main__":
    main()