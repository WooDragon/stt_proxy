#!/usr/bin/env python3
"""
后处理清理模块 - 用于清理转写结果中的重复和幻觉内容
"""

import re
from typing import Dict, Any

def detect_repetitive_ending(text: str, min_repeats: int = 3) -> tuple[bool, int]:
    """
    检测文本末尾是否有重复模式
    返回: (是否有重复, 重复开始位置)
    """
    if not text:
        return False, -1
    
    words = text.strip().split()
    if len(words) < min_repeats * 2:
        return False, -1
    
    # 从末尾开始检测重复模式
    for pattern_len in range(1, len(words) // min_repeats + 1):
        pattern = words[-pattern_len:]
        repeat_count = 1
        
        # 向前检查重复
        pos = len(words) - pattern_len
        while pos >= pattern_len:
            if words[pos - pattern_len:pos] == pattern:
                repeat_count += 1
                pos -= pattern_len
            else:
                break
        
        # 如果发现足够的重复
        if repeat_count >= min_repeats:
            # 计算重复开始的字符位置
            repeat_start_word_idx = pos
            char_pos = 0
            for i, word in enumerate(words):
                if i == repeat_start_word_idx:
                    return True, char_pos
                char_pos += len(word) + 1  # +1 for space
            
    return False, -1

def clean_repetitive_content(text: str, max_consecutive_repeats: int = 3) -> str:
    """
    清理重复内容
    """
    if not text:
        return text
    
    # 1. 检测并移除末尾的重复模式
    has_repeat, repeat_pos = detect_repetitive_ending(text, min_repeats=3)
    if has_repeat and repeat_pos > 0:
        # 保留重复开始前的内容，加上一个重复单元
        words_before = text[:repeat_pos].strip().split()
        if words_before:
            # 尝试找到一个自然的句子结束点
            clean_text = text[:repeat_pos].strip()
            
            # 如果最后不是句号、问号或感叹号，尝试添加句号
            if clean_text and clean_text[-1] not in '。？！.?!':
                clean_text += '。'
            
            return clean_text
    
    # 2. 清理过度的连续重复词汇
    words = text.split()
    cleaned_words = []
    last_word = ""
    consecutive_count = 0
    
    for word in words:
        if word == last_word:
            consecutive_count += 1
            # 允许适度重复，但限制过度重复
            if consecutive_count < max_consecutive_repeats:
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            consecutive_count = 1
            last_word = word
    
    return ' '.join(cleaned_words)

def detect_common_hallucinations(text: str) -> list[str]:
    """
    检测常见的幻觉内容
    """
    hallucinations = []
    
    # 检测末尾的重复模式
    has_repeat, _ = detect_repetitive_ending(text)
    if has_repeat:
        hallucinations.append("末尾重复模式")
    
    # 检测过多的语气词
    filler_words = ["嗯", "啊", "呃", "哦"]
    total_words = len(text.split())
    filler_count = sum(text.count(word) for word in filler_words)
    
    if total_words > 0 and filler_count / total_words > 0.1:  # 超过10%是语气词
        hallucinations.append(f"过多语气词 ({filler_count}/{total_words})")
    
    # 检测异常的重复短语
    words = text.split()
    if len(words) > 10:
        last_10 = words[-10:]
        unique_in_last_10 = len(set(last_10))
        if unique_in_last_10 <= 3:  # 末尾10个词中只有3个不同词汇
            hallucinations.append("末尾词汇多样性过低")
    
    return hallucinations

def postprocess_transcription(transcription_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    对转写结果进行后处理清理
    """
    if not transcription_result or 'text' not in transcription_result:
        return transcription_result
    
    original_text = transcription_result['text']
    
    # 执行清理
    cleaned_text = clean_repetitive_content(original_text)
    
    # 检测问题
    detected_issues = detect_common_hallucinations(original_text)
    
    # 创建新的结果
    result = transcription_result.copy()
    result['text'] = cleaned_text
    result['postprocessing'] = {
        'original_length': len(original_text),
        'cleaned_length': len(cleaned_text),
        'detected_issues': detected_issues,
        'was_cleaned': cleaned_text != original_text
    }
    
    return result

def main():
    """测试后处理功能"""
    # 测试用例
    test_text = """这是一段正常的对话内容，讨论了一些商务问题。然后对话结束了。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。嗯，拜拜。"""
    
    print("原文:")
    print(test_text)
    print("\n" + "="*50)
    
    # 测试检测功能
    has_repeat, pos = detect_repetitive_ending(test_text)
    print(f"检测到重复: {has_repeat}, 位置: {pos}")
    
    # 测试清理功能
    cleaned = clean_repetitive_content(test_text)
    print(f"\n清理后:")
    print(cleaned)
    
    # 测试完整后处理
    fake_result = {"text": test_text}
    processed = postprocess_transcription(fake_result)
    print(f"\n后处理结果:")
    print(f"文本: {processed['text']}")
    print(f"处理信息: {processed['postprocessing']}")

if __name__ == "__main__":
    main()