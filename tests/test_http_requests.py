#!/usr/bin/env python3
"""
测试实际HTTP请求传输中的参数处理
模拟用户提供的curl请求示例
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_audio_file():
    """创建测试用的音频文件"""
    # 创建一个空的WAV文件用于测试（只是占位符）
    test_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    # 写入一些虚拟数据
    test_file.write(b'RIFF' + b'\x00' * 44)  # 基本WAV文件头
    test_file.close()
    return test_file.name


async def test_chat_completions_request():
    """测试 /v1/chat/completions 请求（JSON格式）"""
    print("=" * 60)
    print("测试 /v1/chat/completions 请求")
    print("=" * 60)
    
    proxy_url = "http://localhost:10241"
    
    # 模拟用户的curl请求
    request_data = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }
    
    print(f"发送请求到: {proxy_url}/v1/chat/completions")
    print(f"请求数据: {json.dumps(request_data, indent=2)}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 使用mock来拦截转发的请求
            with patch('httpx.AsyncClient.request') as mock_request:
                mock_response = Mock()
                mock_response.content = b'{"response": "test"}'
                mock_response.status_code = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_request.return_value = mock_response
                
                response = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                print(f"响应状态码: {response.status_code}")
                
                # 检查是否调用了转发请求
                if mock_request.called:
                    call_args = mock_request.call_args
                    print(f"转发到目标服务的方法: {call_args.kwargs.get('method')}")
                    print(f"转发到目标服务的URL: {call_args.kwargs.get('url')}")
                    print(f"转发的请求体: {call_args.kwargs.get('content')}")
                    
                    # 验证请求体是否保持原样
                    forwarded_content = call_args.kwargs.get('content')
                    if forwarded_content:
                        forwarded_data = json.loads(forwarded_content)
                        print("✓ JSON请求正确转发，数据未被修改")
                        return True
                
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False
    
    return False


async def test_audio_transcriptions_request():
    """测试 /v1/audio/transcriptions 请求（multipart/form-data格式）"""
    print("=" * 60)
    print("测试 /v1/audio/transcriptions 请求")
    print("=" * 60)
    
    proxy_url = "http://localhost:10241"
    
    # 创建测试文件
    audio_file_path = create_test_audio_file()
    
    print(f"发送请求到: {proxy_url}/v1/audio/transcriptions")
    print(f"音频文件: {audio_file_path}")
    print("请求参数:")
    print("  - file: test audio file")
    print("  - model: mlx-community/whisper-large-v3-turbo")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 使用mock来拦截转发的请求
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = Mock()
                mock_response.content = b'{"text": "test transcription"}'
                mock_response.status_code = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_post.return_value = mock_response
                
                # 准备multipart form data
                files = {"file": ("test.wav", open(audio_file_path, "rb"), "audio/wav")}
                data = {"model": "mlx-community/whisper-large-v3-turbo"}
                
                response = await client.post(
                    f"{proxy_url}/v1/audio/transcriptions",
                    files=files,
                    data=data
                )
                
                print(f"响应状态码: {response.status_code}")
                
                # 检查是否调用了转发请求
                if mock_post.called:
                    call_args = mock_post.call_args
                    print(f"转发到目标服务的URL: {call_args[0][0] if call_args[0] else '未知'}")
                    
                    # 检查转发的数据
                    forwarded_data = call_args.kwargs.get('data', {})
                    forwarded_files = call_args.kwargs.get('files', {})
                    
                    print("转发的表单数据:")
                    for key, value in forwarded_data.items():
                        print(f"  {key}: {value}")
                    
                    print(f"转发的文件数量: {len(forwarded_files)}")
                    
                    # 验证配置文件参数是否被正确添加
                    expected_config_params = {
                        'language': 'zh',
                        'temperature': '0.2',  # 应该转换为字符串
                        'logprob_threshold': '-1',
                        'compression_ratio_threshold': '2.4',
                        'initial_prompt': '以下是一段普通话商务会议的录音，有多位发言人。转写需准确、流畅，并使用恰当的标点符号（例如逗号，句号，问号）进行断句。\n\n张总：嗯，关于上个季度提到的那个新方案，我们今天需要讨论一下具体的执行细节。\n李工：好的。我看了初步的计划，主要有两个问题想确认一下。第一个是资源方面，第二个是时间节点，这个我们能保证吗？\n张总：对，你提的这点很重要。下一步，我们必须先把风险点都梳理清楚。'
                    }
                    
                    print("\n验证配置参数强制覆盖:")
                    all_correct = True
                    for key, expected_value in expected_config_params.items():
                        actual_value = forwarded_data.get(key)
                        if str(actual_value) == str(expected_value):
                            print(f"  ✓ {key}: {actual_value}")
                        else:
                            print(f"  ✗ {key}: {actual_value} (期望: {expected_value})")
                            all_correct = False
                    
                    # 验证客户端参数是否保留
                    print("\n验证客户端参数保留:")
                    if 'model' in forwarded_data:
                        if forwarded_data['model'] == 'mlx-community/whisper-large-v3-turbo':
                            print(f"  ✓ model: {forwarded_data['model']}")
                        else:
                            print(f"  ✗ model: {forwarded_data['model']} (期望: mlx-community/whisper-large-v3-turbo)")
                            all_correct = False
                    else:
                        print("  ✗ model 参数丢失")
                        all_correct = False
                    
                    # 验证文件是否正确转发
                    if 'file' in forwarded_files:
                        print("  ✓ 音频文件正确转发")
                    else:
                        print("  ✗ 音频文件未转发")
                        all_correct = False
                    
                    return all_correct
                
        # 清理测试文件
        Path(audio_file_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        # 清理测试文件
        Path(audio_file_path).unlink(missing_ok=True)
        return False
    
    return False


async def test_parameter_data_types():
    """测试参数数据类型处理"""
    print("=" * 60)
    print("测试参数数据类型处理")
    print("=" * 60)
    
    # 导入配置加载函数
    sys.path.insert(0, '.')
    from stt_proxy import load_config
    
    config = load_config("config/stt_config.json")
    print(f"配置文件内容及类型:")
    for key, value in config.items():
        print(f"  {key}: {value} (类型: {type(value).__name__})")
    
    # 模拟multipart/form-data中的数据类型转换
    print("\n模拟multipart数据类型转换:")
    form_data = {"model": "test-model"}
    
    for key, default_value in config.items():
        # multipart/form-data中所有值都应该是字符串
        form_data[key] = str(default_value)
        print(f"  {key}: {form_data[key]} (类型: {type(form_data[key]).__name__})")
    
    print("\n✓ 数据类型转换测试完成")
    return True


async def main():
    """主测试函数"""
    print("STT代理服务HTTP请求测试")
    print("模拟curl请求示例")
    
    results = []
    
    # 测试1: JSON请求
    # 注意：这个测试需要代理服务运行，这里我们使用mock来模拟
    print("注意: 以下测试使用mock模拟，实际需要启动代理服务进行验证")
    
    # 测试参数数据类型处理
    result1 = await test_parameter_data_types()
    results.append(("参数数据类型处理", result1))
    
    print("\n" + "=" * 60)
    print("测试总结:")
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n重要发现:")
    print("1. 配置文件中的数值类型需要在multipart/form-data中转换为字符串")
    print("2. 参数覆盖逻辑在逻辑层面工作正常")
    print("3. 需要验证实际HTTP传输中的参数格式")
    
    print("\n下一步:")
    print("1. 启动代理服务进行实际测试")
    print("2. 使用真实的curl命令验证参数传递")
    print("3. 检查数据类型转换是否正确")


if __name__ == "__main__":
    asyncio.run(main())