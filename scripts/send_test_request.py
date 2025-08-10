#!/usr/bin/env python3
"""
测试脚本，用于向STT代理服务发送请求
"""

import asyncio
import httpx

async def test_stt_request():
    """发送测试STT请求"""
    url = "http://localhost:10241/audio/transcriptions"
    
    # 准备测试数据
    data = {
        "language": "en",
        "temperature": "0.8",
        "initial_prompt": "This is a test English conversation.",
        "response_format": "json"
    }
    
    print("发送测试请求...")
    print(f"请求URL: {url}")
    print(f"请求数据: {data}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            print(f"响应状态: {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_stt_request())