#!/usr/bin/env python3
"""
测试STT配置优化效果的脚本
"""

import json
import requests
import os
from pathlib import Path

def load_config(config_path="config/stt_config.json"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_stt_request(audio_file_path, config_params, proxy_url="http://localhost:10241"):
    """发送STT转写请求"""
    if not os.path.exists(audio_file_path):
        print(f"错误: 音频文件 {audio_file_path} 不存在")
        return None
    
    # 准备请求数据
    files = {
        'file': ('audio.wav', open(audio_file_path, 'rb'), 'audio/wav')
    }
    
    # 合并配置参数（客户端参数会被配置文件强制覆盖）
    data = {
        'model': 'whisper-large-v3',
        **config_params
    }
    
    try:
        # 发送请求到代理服务器
        response = requests.post(f"{proxy_url}/audio/transcriptions", files=files, data=data)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    finally:
        files['file'][1].close()

def analyze_transcription_quality(transcription):
    """分析转写质量"""
    if not transcription:
        return {"error": "转写失败"}
    
    text = transcription.get('text', '')
    
    # 检测重复模式
    words = text.split()
    repetition_count = 0
    last_word = ""
    consecutive_repeats = 0
    max_consecutive = 0
    
    for word in words:
        if word == last_word:
            consecutive_repeats += 1
            max_consecutive = max(max_consecutive, consecutive_repeats)
        else:
            if consecutive_repeats > 2:  # 连续重复超过2次
                repetition_count += consecutive_repeats
            consecutive_repeats = 1
        last_word = word
    
    # 检查末尾重复 - 扩展到20个词
    last_20_words = words[-20:] if len(words) >= 20 else words
    unique_last_words = len(set(last_20_words))
    
    # 检测特定重复词汇（拜拜、嗯等）
    problematic_words = ["拜拜", "嗯", "啊", "呃", "那个"]
    problematic_count = sum(text.count(word) for word in problematic_words)
    
    # 检测末尾的具体重复模式
    ending_repeats = 0
    if len(words) >= 10:
        last_10 = words[-10:]
        for i in range(len(last_10) - 1):
            if last_10[i] == last_10[i + 1]:
                ending_repeats += 1
    
    analysis = {
        "total_words": len(words),
        "repetition_count": repetition_count,
        "max_consecutive_repeats": max_consecutive,
        "last_20_words_diversity": unique_last_words / len(last_20_words) if last_20_words else 0,
        "problematic_words_count": problematic_count,
        "ending_consecutive_repeats": ending_repeats,
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "text_ending": text[-300:] if len(text) > 300 else text
    }
    
    return analysis

def main():
    """主测试函数"""
    print("=== STT配置优化效果测试 ===\n")
    
    # 加载优化后的配置
    config = load_config()
    print("当前配置参数:")
    for key, value in config.items():
        if key != 'initial_prompt':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: [已设置提示词样例]")
    print()
    
    # 检查是否有测试音频文件
    test_audio_files = [
        "test_audio.wav",
        "test_audio.mp3", 
        "sample.wav",
        "sample.mp3"
    ]
    
    available_files = [f for f in test_audio_files if os.path.exists(f)]
    
    if not available_files:
        print("提示: 没有找到测试音频文件")
        print("请将测试音频文件放在当前目录，支持的文件名:")
        for filename in test_audio_files:
            print(f"  - {filename}")
        print()
        print("或者手动指定音频文件路径进行测试")
        return
    
    # 测试每个可用的音频文件
    for audio_file in available_files:
        print(f"测试音频文件: {audio_file}")
        print("-" * 50)
        
        # 发送转写请求
        result = test_stt_request(audio_file, config)
        
        if result:
            # 分析转写质量
            analysis = analyze_transcription_quality(result)
            
            print("转写结果分析:")
            print(f"  总词数: {analysis['total_words']}")
            print(f"  重复词数: {analysis['repetition_count']}")
            print(f"  最大连续重复: {analysis['max_consecutive_repeats']}")
            print(f"  末尾20词多样性: {analysis['last_20_words_diversity']:.2f}")
            print(f"  问题词汇总数: {analysis['problematic_words_count']}")
            print(f"  末尾连续重复: {analysis['ending_consecutive_repeats']}")
            
            print("\n转写文本开头:")
            print(f"  {analysis['text_preview']}")
            
            print("\n转写文本结尾:")
            print(f"  ...{analysis['text_ending']}")
            
        else:
            print("转写失败")
        
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()