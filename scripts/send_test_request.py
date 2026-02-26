#!/usr/bin/env python3
"""
测试脚本：向 ASR 服务发送转写请求

用法:
    python scripts/send_test_request.py <音频文件路径>
    python scripts/send_test_request.py test.wav
    python scripts/send_test_request.py test.wav --port 10241
"""

import argparse
import os
import sys
import httpx


def send_transcription_request(audio_path: str, port: int, model: str) -> None:
    """发送音频文件到 ASR 服务进行转写"""
    if not os.path.isfile(audio_path):
        print(f"错误: 文件不存在 {audio_path}", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{port}/v1/audio/transcriptions"

    print(f"文件: {audio_path}")
    print(f"服务: {url}")
    print(f"模型: {model}")
    print("---")

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"model": model}

        response = httpx.post(url, files=files, data=data, timeout=120.0)

    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"转写结果: {result['text']}")
        if "segments" in result:
            for seg in result["segments"]:
                print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    else:
        print(f"错误: {response.text}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ASR 转写测试")
    parser.add_argument("audio_file", help="音频文件路径")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "10241")),
                        help="服务端口 (默认: 10241)")
    parser.add_argument("--model", default=os.getenv("MODEL", "mlx-community/Qwen3-ASR-1.7B-8bit"),
                        help="模型名称")
    args = parser.parse_args()

    send_transcription_request(args.audio_file, args.port, args.model)


if __name__ == "__main__":
    main()
