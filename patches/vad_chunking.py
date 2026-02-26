"""Monkey-patch: VAD-driven audio chunking for split_audio_into_chunks.

Bug:  Default energy-based splitter cuts at lowest-energy points regardless of
      speech boundaries — frequently splits mid-sentence.
Fix:  Use webrtcvad to detect speech/silence boundaries, cut at >= 500ms silence
      gaps. Falls back to original energy-based splitting when no silence found.

Hits: mlx-audio <= 0.3.1 (split_audio_into_chunks uses energy-only heuristic)
Drop: when upstream adopts VAD-aware chunking.
"""

import numpy as np
import webrtcvad

import mlx_audio.stt.models.qwen3_asr.qwen3_asr as _qwen3_module

_original_split = _qwen3_module.split_audio_into_chunks

# VAD constants
_VAD_SAMPLE_RATE = 16000
_VAD_FRAME_MS = 30
_VAD_FRAME_SAMPLES = _VAD_SAMPLE_RATE * _VAD_FRAME_MS // 1000  # 480
_SILENCE_THRESHOLD_MS = 500


def _resample_linear(wav, sr_orig, sr_target):
    """Resample audio via linear interpolation. Good enough for VAD analysis."""
    if sr_orig == sr_target:
        return wav
    ratio = sr_target / sr_orig
    n_target = int(len(wav) * ratio)
    indices = np.linspace(0, len(wav) - 1, n_target)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(wav) - 1)
    frac = indices - idx_floor
    return wav[idx_floor] * (1 - frac) + wav[idx_ceil] * frac


def _detect_silence_regions(wav_16k):
    """Detect silence regions (>= 500ms) using webrtcvad.

    Args:
        wav_16k: Float32 audio at 16kHz, normalized to [-1.0, 1.0].

    Returns:
        List of (start_sample, end_sample) tuples marking silence regions
        at 16kHz sample indices.
    """
    vad = webrtcvad.Vad(3)  # most aggressive mode

    # float32 → int16 with overflow protection
    wav_clipped = np.clip(wav_16k, -1.0, 1.0)
    wav_int16 = (wav_clipped * 32767).astype(np.int16)

    # Scan frames — skip tail shorter than one frame
    n_frames = (len(wav_int16) - _VAD_FRAME_SAMPLES + 1) // _VAD_FRAME_SAMPLES
    if n_frames <= 0:
        return []

    # Build per-frame speech flags
    is_speech = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * _VAD_FRAME_SAMPLES
        frame_bytes = wav_int16[start : start + _VAD_FRAME_SAMPLES].tobytes()
        is_speech[i] = vad.is_speech(frame_bytes, _VAD_SAMPLE_RATE)

    # Find contiguous silence runs >= threshold
    min_silence_frames = _SILENCE_THRESHOLD_MS // _VAD_FRAME_MS  # 500/30 ≈ 16 frames
    silence_regions = []
    run_start = None

    for i in range(n_frames):
        if not is_speech[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_silence_frames:
                    silence_regions.append((
                        run_start * _VAD_FRAME_SAMPLES,
                        i * _VAD_FRAME_SAMPLES,
                    ))
                run_start = None

    # Handle trailing silence
    if run_start is not None:
        run_len = n_frames - run_start
        if run_len >= min_silence_frames:
            silence_regions.append((
                run_start * _VAD_FRAME_SAMPLES,
                n_frames * _VAD_FRAME_SAMPLES,
            ))

    return silence_regions


def split_audio_vad(wav, sr, chunk_duration=1200.0, min_chunk_duration=1.0, **kwargs):
    """Split audio at silence boundaries detected by VAD.

    Greedy algorithm: accumulate audio up to chunk_duration, cut at nearest
    silence midpoint. Falls back to original energy-based splitter when no
    silence point is found within a chunk.

    Args:
        wav: Audio waveform (1D or 2D numpy array).
        sr: Sample rate.
        chunk_duration: Maximum chunk duration in seconds.
        min_chunk_duration: Minimum chunk duration in seconds.
        **kwargs: Absorb upstream params (search_expand_sec, min_window_ms, etc.)

    Returns:
        List of (chunk_waveform, offset_seconds) tuples.
    """
    # Ensure mono
    if wav.ndim > 1:
        wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)

    total_samples = len(wav)
    total_sec = total_samples / sr

    # Short audio: return as-is (pad if needed)
    if total_sec <= chunk_duration:
        if total_sec < min_chunk_duration:
            min_samples = int(min_chunk_duration * sr)
            wav = np.pad(wav, (0, min_samples - len(wav)))
        return [(wav, 0.0)]

    # Resample to 16kHz for VAD analysis
    wav_16k = _resample_linear(wav, sr, _VAD_SAMPLE_RATE)
    scale = sr / _VAD_SAMPLE_RATE  # map 16kHz indices back to original sr

    silence_regions = _detect_silence_regions(wav_16k)

    # No silence found at all → fall back to original energy-based splitter
    if not silence_regions:
        return _original_split(
            wav, sr,
            chunk_duration=chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

    # Convert silence regions to original sample rate (midpoints for cutting)
    silence_midpoints_orig = []
    for s_start, s_end in silence_regions:
        mid_16k = (s_start + s_end) // 2
        mid_orig = int(mid_16k * scale)
        mid_orig = min(mid_orig, total_samples)
        silence_midpoints_orig.append(mid_orig)

    # Greedy chunking: accumulate up to chunk_duration, cut at nearest silence
    max_chunk_samples = int(chunk_duration * sr)
    min_chunk_samples = int(min_chunk_duration * sr)
    chunks = []
    start_sample = 0
    sp_idx = 0  # pointer into silence_midpoints_orig

    while start_sample < total_samples:
        end_limit = min(start_sample + max_chunk_samples, total_samples)

        # Last segment: take everything
        if end_limit >= total_samples:
            chunk = wav[start_sample:total_samples]
            if len(chunk) < min_chunk_samples:
                chunk = np.pad(chunk, (0, min_chunk_samples - len(chunk)))
            chunks.append((chunk, start_sample / sr))
            break

        # Advance pointer past start_sample
        while sp_idx < len(silence_midpoints_orig) and silence_midpoints_orig[sp_idx] <= start_sample:
            sp_idx += 1

        # Find the last silence midpoint before end_limit
        best_cut = None
        for j in range(sp_idx, len(silence_midpoints_orig)):
            mp = silence_midpoints_orig[j]
            if mp > end_limit:
                break
            # Only consider cuts that produce at least min_chunk_duration
            if mp - start_sample >= min_chunk_samples:
                best_cut = mp

        if best_cut is not None:
            cut_sample = best_cut
        else:
            # No silence within this chunk window → fall back to energy split
            # for just this segment
            segment = wav[start_sample:end_limit]
            sub_chunks = _original_split(
                segment, sr,
                chunk_duration=chunk_duration,
                min_chunk_duration=min_chunk_duration,
            )
            for sub_wav, sub_offset in sub_chunks:
                chunks.append((sub_wav, (start_sample / sr) + sub_offset))
            start_sample = end_limit
            continue

        chunk = wav[start_sample:cut_sample]
        if len(chunk) < min_chunk_samples:
            chunk = np.pad(chunk, (0, min_chunk_samples - len(chunk)))
        chunks.append((chunk, start_sample / sr))
        start_sample = cut_sample

    return chunks


_qwen3_module.split_audio_into_chunks = split_audio_vad
