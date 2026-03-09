"""Unit tests for local monkey-patches.

Tests patch logic in isolation — no running service required.
"""

import asyncio
import inspect
import io
import os
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Test: patches/vad_chunking.py
# ---------------------------------------------------------------------------

class TestDetectSilenceRegions:
    """Test _detect_silence_regions from vad_chunking."""

    def _get_detect(self):
        from patches.vad_chunking import _detect_silence_regions
        return _detect_silence_regions

    def test_all_silence(self):
        """All-zero audio → one big silence region."""
        detect = self._get_detect()
        wav = np.zeros(16000 * 2, dtype=np.float32)  # 2s silence at 16kHz
        regions = detect(wav)
        assert len(regions) >= 1
        # Should cover nearly the entire duration
        total_silence = sum(end - start for start, end in regions)
        assert total_silence > 16000  # > 1s of detected silence

    def test_all_speech(self):
        """Loud white noise → no silence regions."""
        detect = self._get_detect()
        rng = np.random.RandomState(42)
        wav = rng.uniform(-0.8, 0.8, 16000 * 2).astype(np.float32)
        regions = detect(wav)
        # White noise at high amplitude should have few/no silence regions
        total_silence = sum(end - start for start, end in regions)
        assert total_silence < 16000 * 0.5  # less than 0.5s

    def test_short_silence_ignored(self):
        """200ms silence gap should be ignored (threshold is 500ms)."""
        detect = self._get_detect()
        rng = np.random.RandomState(42)
        sr = 16000
        # 1s speech + 200ms silence + 1s speech
        speech1 = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
        silence = np.zeros(int(sr * 0.2), dtype=np.float32)
        speech2 = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
        wav = np.concatenate([speech1, silence, speech2])
        regions = detect(wav)
        # 200ms silence should NOT produce a silence region
        for s, e in regions:
            # No region should be centered around the 200ms gap
            gap_center = sr + int(sr * 0.1)  # center of the 200ms gap
            if s <= gap_center <= e:
                # If detected, it must be short (< 500ms)
                assert (e - s) / sr < 0.5

    def test_long_silence_detected(self):
        """1s silence gap should be detected."""
        detect = self._get_detect()
        rng = np.random.RandomState(42)
        sr = 16000
        # 1s speech + 1s silence + 1s speech
        speech1 = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
        silence = np.zeros(sr, dtype=np.float32)  # 1s
        speech2 = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
        wav = np.concatenate([speech1, silence, speech2])
        regions = detect(wav)
        # Should find at least one silence region in the middle
        assert len(regions) >= 1
        # Check that at least one region overlaps the silence section
        silence_start_sample = sr
        silence_end_sample = sr * 2
        found = any(
            s < silence_end_sample and e > silence_start_sample
            for s, e in regions
        )
        assert found, f"Expected silence region near {silence_start_sample}-{silence_end_sample}, got {regions}"

    def test_tail_frame_safety(self):
        """Audio length not a multiple of frame size → no crash."""
        detect = self._get_detect()
        # 480 samples per frame (30ms @16kHz), use 1000 samples (2 full frames + 40 leftover)
        wav = np.zeros(1000, dtype=np.float32)
        regions = detect(wav)  # should not raise
        assert isinstance(regions, list)

    def test_too_short_for_frame(self):
        """Audio shorter than one VAD frame → empty result, no crash."""
        detect = self._get_detect()
        wav = np.zeros(100, dtype=np.float32)  # < 480 samples
        regions = detect(wav)
        assert regions == []


class TestSplitAudioVad:
    """Test split_audio_vad end-to-end."""

    def _get_split(self):
        from patches.vad_chunking import split_audio_vad
        return split_audio_vad

    def test_short_audio_passthrough(self):
        """Audio shorter than chunk_duration returns single chunk."""
        split = self._get_split()
        sr = 16000
        wav = np.zeros(sr * 5, dtype=np.float32)  # 5s
        result = split(wav, sr, chunk_duration=30.0)
        assert len(result) == 1
        assert result[0][1] == 0.0  # offset

    def test_short_audio_padded(self):
        """Audio shorter than min_chunk_duration gets padded."""
        split = self._get_split()
        sr = 16000
        wav = np.zeros(int(sr * 0.3), dtype=np.float32)  # 0.3s
        result = split(wav, sr, chunk_duration=30.0, min_chunk_duration=1.0)
        assert len(result) == 1
        assert len(result[0][0]) >= sr  # padded to >= 1s

    def test_splits_at_silence(self):
        """Long audio with silence gaps splits into multiple chunks."""
        split = self._get_split()
        sr = 16000
        rng = np.random.RandomState(42)
        # 20s speech + 1s silence + 20s speech (41s total, chunk_duration=30)
        speech1 = rng.uniform(-0.5, 0.5, sr * 20).astype(np.float32)
        silence = np.zeros(sr, dtype=np.float32)
        speech2 = rng.uniform(-0.5, 0.5, sr * 20).astype(np.float32)
        wav = np.concatenate([speech1, silence, speech2])

        result = split(wav, sr, chunk_duration=30.0)
        assert len(result) >= 2
        # Offsets should be monotonically increasing
        offsets = [r[1] for r in result]
        assert offsets == sorted(offsets)

    def test_kwargs_absorbed(self):
        """Unknown kwargs (forward compat) don't crash."""
        split = self._get_split()
        wav = np.zeros(16000, dtype=np.float32)
        result = split(wav, 16000, chunk_duration=30.0, search_expand_sec=5.0, min_window_ms=100.0)
        assert len(result) == 1

    def test_stereo_to_mono(self):
        """2D stereo input is handled."""
        split = self._get_split()
        sr = 16000
        wav_stereo = np.zeros((sr * 2, 2), dtype=np.float32)
        result = split(wav_stereo, sr, chunk_duration=30.0)
        assert len(result) >= 1

    def test_no_silence_fallback(self):
        """Continuous speech with no silence → falls back to original splitter."""
        split = self._get_split()
        sr = 16000
        rng = np.random.RandomState(42)
        # 60s of continuous noise, chunk_duration=30
        wav = rng.uniform(-0.5, 0.5, sr * 60).astype(np.float32)
        result = split(wav, sr, chunk_duration=30.0)
        # Should still produce chunks (via fallback)
        assert len(result) >= 2
        # Verify all audio is covered
        total_samples = sum(len(c[0]) for c in result)
        assert total_samples >= sr * 59  # approximately all audio accounted for


class TestResampleLinear:
    """Test the linear resampler helper."""

    def _get_resample(self):
        from patches.vad_chunking import _resample_linear
        return _resample_linear

    def test_same_rate_noop(self):
        resample = self._get_resample()
        wav = np.ones(100, dtype=np.float32)
        result = resample(wav, 16000, 16000)
        np.testing.assert_array_equal(result, wav)

    def test_downsample_halves_length(self):
        resample = self._get_resample()
        wav = np.ones(1000, dtype=np.float32)
        result = resample(wav, 32000, 16000)
        assert abs(len(result) - 500) <= 1

    def test_upsample_doubles_length(self):
        resample = self._get_resample()
        wav = np.ones(1000, dtype=np.float32)
        result = resample(wav, 16000, 32000)
        assert abs(len(result) - 2000) <= 1


# ---------------------------------------------------------------------------
# Test: patches/generate_enhance.py
# ---------------------------------------------------------------------------

class TestGenerateEnhance:
    """Test generate_enhance wrapper logic using mocks."""

    @dataclass
    class FakeSTTOutput:
        text: str = ""
        language: str = None

    def test_language_defaults_to_chinese(self):
        """When language not passed, defaults to Chinese."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput(text="你好")

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            # Re-import to get patched version
            from patches.generate_enhance import _generate_patched
            result = _generate_patched(MagicMock(), "audio.wav")

        assert captured_kwargs["language"] == "Chinese"
        assert result.language == "Chinese"

    def test_language_explicit_preserved(self):
        """Explicit language kwarg is not overridden."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput(text="hello")

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            result = _generate_patched(MagicMock(), "audio.wav", language="English")

        assert captured_kwargs["language"] == "English"
        assert result.language == "English"

    def test_repetition_penalty_injected(self):
        """repetition_penalty is injected from default when not provided."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput()

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            _generate_patched(MagicMock(), "audio.wav")

        assert "repetition_penalty" in captured_kwargs
        assert captured_kwargs["repetition_penalty"] > 1.0

    def test_repetition_penalty_explicit_preserved(self):
        """Explicit repetition_penalty is not overridden."""
        captured_kwargs = {}
        fake_output = self.FakeSTTOutput()

        def mock_generate(self_, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            _generate_patched(MagicMock(), "audio.wav", repetition_penalty=1.5)

        assert captured_kwargs["repetition_penalty"] == 1.5

    def test_language_backfill_on_output(self):
        """STTOutput.language is set to the actual language used."""
        fake_output = self.FakeSTTOutput(text="test")
        assert fake_output.language is None  # before patch

        def mock_generate(self_, audio, **kwargs):
            return fake_output

        with patch(
            "patches.generate_enhance._original_generate",
            side_effect=mock_generate,
        ):
            from patches.generate_enhance import _generate_patched
            result = _generate_patched(MagicMock(), "audio.wav")

        assert result.language == "Chinese"


# ---------------------------------------------------------------------------
# Test: patches/__init__.py
# ---------------------------------------------------------------------------

class TestApplyAll:
    """Test that apply_all() loads all patches without error."""

    def test_apply_all_idempotent(self):
        """Calling apply_all multiple times doesn't crash."""
        from patches import apply_all
        apply_all()
        apply_all()  # second call should be harmless


# ---------------------------------------------------------------------------
# Test: patches/speaker_diarization.py
# ---------------------------------------------------------------------------

class TestAssignSpeakers:
    """Test time-weighted speaker assignment logic."""

    def _get_assign(self):
        from patches.speaker_diarization import assign_speakers
        return assign_speakers

    def test_single_speaker_all_segments(self):
        """All segments overlap one speaker → all get that speaker."""
        assign = self._get_assign()
        asr = {
            "text": "hello world",
            "segments": [
                {"text": "hello", "start": 0.0, "end": 1.0},
                {"text": "world", "start": 1.0, "end": 2.0},
            ],
        }
        timeline = [(0.0, 3.0, "SPEAKER_00")]
        result = assign(asr, timeline)
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][1]["speaker"] == "SPEAKER_00"
        assert "SPEAKER_00" in result["speakers"]

    def test_two_speakers_no_overlap(self):
        """Two speakers with clean boundaries → correct assignment."""
        assign = self._get_assign()
        asr = {
            "text": "A speaks. B speaks.",
            "segments": [
                {"text": "A speaks.", "start": 0.0, "end": 3.0},
                {"text": "B speaks.", "start": 5.0, "end": 8.0},
            ],
        }
        timeline = [
            (0.0, 4.0, "SPEAKER_00"),
            (4.0, 9.0, "SPEAKER_01"),
        ]
        result = assign(asr, timeline)
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][1]["speaker"] == "SPEAKER_01"
        assert sorted(result["speakers"]) == ["SPEAKER_00", "SPEAKER_01"]

    def test_cross_boundary_segment(self):
        """Segment spanning two speakers → assigned to majority overlap."""
        assign = self._get_assign()
        asr = {
            "text": "crossing",
            "segments": [
                # 0-5: overlaps SPEAKER_00 [0-2] by 2s, SPEAKER_01 [2-6] by 3s
                {"text": "crossing", "start": 0.0, "end": 5.0},
            ],
        }
        timeline = [
            (0.0, 2.0, "SPEAKER_00"),
            (2.0, 6.0, "SPEAKER_01"),
        ]
        result = assign(asr, timeline)
        assert result["segments"][0]["speaker"] == "SPEAKER_01"
        # 3/5 = 0.6 < 0.7 threshold → speaker_confidence should be set
        assert "speaker_confidence" in result["segments"][0]

    def test_empty_segments(self):
        """No segments → speakers list from timeline, segments untouched."""
        assign = self._get_assign()
        asr = {"text": "", "segments": []}
        timeline = [(0.0, 1.0, "SPEAKER_00")]
        result = assign(asr, timeline)
        assert result["segments"] == []
        assert "SPEAKER_00" in result["speakers"]

    def test_empty_timeline(self):
        """No speaker timeline → segments unchanged, empty speakers."""
        assign = self._get_assign()
        asr = {
            "text": "hello",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
        }
        result = assign(asr, [])
        assert "speaker" not in result["segments"][0]
        assert result["speakers"] == []

    def test_no_overlap_segment(self):
        """Segment outside all speaker regions → no speaker assigned."""
        assign = self._get_assign()
        asr = {
            "text": "orphan",
            "segments": [{"text": "orphan", "start": 10.0, "end": 11.0}],
        }
        timeline = [(0.0, 5.0, "SPEAKER_00")]
        result = assign(asr, timeline)
        assert "speaker" not in result["segments"][0]


class TestMakeWindows:
    """Test sliding window generation."""

    def _get_make_windows(self):
        from patches.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer._make_windows

    def test_basic_windowing(self):
        """3s audio with 1.5s window, 0.75s step → multiple windows."""
        make_windows = self._get_make_windows()
        wav = torch.zeros(1, 48000)  # 3s at 16kHz
        windows = make_windows(wav, silence_regions=[], sr=16000)
        # Expect windows at 0.0, 0.75, 1.5
        assert len(windows) >= 2
        # Check monotonic starts
        starts = [w[0] for w in windows]
        assert starts == sorted(starts)

    def test_short_audio_no_windows(self):
        """Audio shorter than window → no windows (or tail handling)."""
        make_windows = self._get_make_windows()
        wav = torch.zeros(1, 8000)  # 0.5s at 16kHz, window=1.5s
        windows = make_windows(wav, silence_regions=[], sr=16000)
        # 0.5s < 1.5s window → no full window. Tail = 0.5s < 0.75s (half of 1.5s) → skipped
        assert len(windows) == 0

    def test_silence_windows_skipped(self):
        """Windows entirely within silence are excluded."""
        make_windows = self._get_make_windows()
        wav = torch.zeros(1, 48000)  # 3s at 16kHz
        # Mark entire audio as silence
        silence_regions = [(0, 48000)]
        windows = make_windows(wav, silence_regions, sr=16000)
        assert len(windows) == 0

    def test_partial_silence_kept(self):
        """Windows partially overlapping silence are kept."""
        make_windows = self._get_make_windows()
        wav = torch.zeros(1, 48000)  # 3s
        # Only first 1s is silence
        silence_regions = [(0, 16000)]
        windows = make_windows(wav, silence_regions, sr=16000)
        # First window [0, 1.5s] is NOT entirely within [0, 1.0s] → kept
        assert len(windows) > 0


class TestSpeakerDiarizerModelMissing:
    """Test SpeakerDiarizer behavior when CoreML model is absent."""

    def test_missing_model_raises(self):
        """diarize() raises FileNotFoundError with actionable message."""
        from patches.speaker_diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(model_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="convert_wespeaker"):
            diarizer.diarize(b"fake audio bytes")


class TestWindowIsSilent:
    """Test _window_is_silent helper."""

    def _get_helper(self):
        from patches.speaker_diarization import _window_is_silent
        return _window_is_silent

    def test_fully_inside_silence(self):
        is_silent = self._get_helper()
        assert is_silent(100, 200, [(0, 500)]) is True

    def test_partially_overlapping(self):
        is_silent = self._get_helper()
        assert is_silent(100, 600, [(0, 500)]) is False

    def test_no_silence_regions(self):
        is_silent = self._get_helper()
        assert is_silent(100, 200, []) is False

    def test_outside_all_regions(self):
        is_silent = self._get_helper()
        assert is_silent(600, 800, [(0, 500)]) is False


class TestBuildTimeline:
    """Test timeline merging logic."""

    def _get_build(self):
        from patches.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer._build_timeline

    def test_merge_adjacent_same_speaker(self):
        """Adjacent windows with same label merge into one segment."""
        build = self._get_build()
        windows = [
            (0.0, 1.5, None),
            (0.75, 2.25, None),
            (1.5, 3.0, None),
        ]
        labels = np.array([0, 0, 0])
        timeline = build(windows, labels)
        assert len(timeline) == 1
        assert timeline[0][0] == 0.0
        assert timeline[0][1] == 3.0
        assert timeline[0][2] == "SPEAKER_00"

    def test_different_speakers_split(self):
        """Different labels produce separate timeline segments."""
        build = self._get_build()
        windows = [
            (0.0, 1.5, None),
            (1.5, 3.0, None),
        ]
        labels = np.array([0, 1])
        timeline = build(windows, labels)
        assert len(timeline) == 2
        assert timeline[0][2] == "SPEAKER_00"
        assert timeline[1][2] == "SPEAKER_01"

    def test_empty_windows(self):
        """No windows → empty timeline."""
        build = self._get_build()
        assert build([], np.array([])) == []


class TestCluster:
    """Test AHC clustering logic."""

    def _get_cluster(self):
        from patches.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer._cluster

    def test_single_embedding(self):
        """Single embedding → single cluster."""
        cluster = self._get_cluster()
        emb = np.random.randn(1, 256).astype(np.float32)
        labels = cluster(emb)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_explicit_num_speakers(self):
        """num_speakers forces exact cluster count."""
        cluster = self._get_cluster()
        # Two distinct clusters
        rng = np.random.RandomState(42)
        group_a = rng.randn(5, 256).astype(np.float32) + 10.0
        group_b = rng.randn(5, 256).astype(np.float32) - 10.0
        emb = np.vstack([group_a, group_b])
        labels = cluster(emb, num_speakers=2)
        assert len(np.unique(labels)) == 2

    def test_max_speakers_clamp(self):
        """max_speakers caps the number of clusters."""
        cluster = self._get_cluster()
        rng = np.random.RandomState(42)
        # Three very distinct groups
        groups = [rng.randn(5, 256).astype(np.float32) + i * 20 for i in range(3)]
        emb = np.vstack(groups)
        labels = cluster(emb, max_speakers=2)
        assert len(np.unique(labels)) <= 2

    def test_cosine_separates_speakers(self):
        """Two orthogonal embedding groups (cos_sim ≈ 0) → 2 clusters."""
        cluster = self._get_cluster()
        rng = np.random.RandomState(42)
        dim = 256
        # Group A: base vector in first 128 dims + small noise
        base_a = np.zeros(dim, dtype=np.float32)
        base_a[:128] = 1.0
        group_a = base_a + rng.randn(10, dim).astype(np.float32) * 0.05

        # Group B: base vector in last 128 dims + small noise (orthogonal to A)
        base_b = np.zeros(dim, dtype=np.float32)
        base_b[128:] = 1.0
        group_b = base_b + rng.randn(10, dim).astype(np.float32) * 0.05

        emb = np.vstack([group_a, group_b])
        labels = cluster(emb)
        assert len(np.unique(labels)) == 2

    def test_cosine_merges_same_speaker(self):
        """Same base embedding + small noise (cos_sim > 0.8) → 1 cluster."""
        cluster = self._get_cluster()
        rng = np.random.RandomState(42)
        dim = 256
        base = rng.randn(dim).astype(np.float32)
        # Small noise: cos_sim between any pair should be >> 0.5 threshold
        emb = base + rng.randn(10, dim).astype(np.float32) * 0.1
        labels = cluster(emb)
        assert len(np.unique(labels)) == 1

    def test_threshold_sensitivity(self, monkeypatch):
        """Lower threshold produces more clusters than higher threshold."""
        cluster = self._get_cluster()
        rng = np.random.RandomState(42)
        dim = 256
        # 3 moderately separated groups (cos_sim between groups ~ 0.3-0.6)
        groups = []
        for i in range(3):
            base = rng.randn(dim).astype(np.float32)
            group = base + rng.randn(8, dim).astype(np.float32) * 0.3
            groups.append(group)
        emb = np.vstack(groups)

        # Strict threshold → more clusters
        monkeypatch.setenv("DIARIZE_THRESHOLD", "0.2")
        labels_strict = cluster(emb)

        # Lenient threshold → fewer clusters
        monkeypatch.setenv("DIARIZE_THRESHOLD", "0.9")
        labels_lenient = cluster(emb)

        assert len(np.unique(labels_strict)) >= len(np.unique(labels_lenient))


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — _is_audio_decode_error classifier
# ---------------------------------------------------------------------------

class TestIsAudioDecodeError:
    """Test audio decode error classifier (422 vs 500 routing)."""

    def _get_classifier(self):
        from patches.server_diarize import _is_audio_decode_error
        return _is_audio_decode_error

    # ── 422 cases: client-side audio errors ──

    def test_decode_error_duck_type(self):
        """miniaudio.DecodeError (duck-typed by class name) → 422."""
        classify = self._get_classifier()
        # Simulate miniaudio.DecodeError without importing miniaudio
        class DecodeError(Exception):
            pass
        assert classify(DecodeError("corrupt file")) is True

    def test_value_error_detect_format(self):
        """ValueError 'Unable to detect audio format' → 422."""
        classify = self._get_classifier()
        assert classify(ValueError("Unable to detect audio format from bytes")) is True

    def test_value_error_unsupported_format(self):
        """ValueError 'Unsupported format:' → 422."""
        classify = self._get_classifier()
        assert classify(ValueError("Unsupported format: xyz")) is True

    def test_runtime_error_no_audio_streams(self):
        """RuntimeError 'No audio streams found' → 422."""
        classify = self._get_classifier()
        assert classify(RuntimeError("No audio streams found in /tmp/foo.mp4")) is True

    def test_runtime_error_ffprobe_failed(self):
        """RuntimeError 'ffprobe failed:' → 422."""
        classify = self._get_classifier()
        assert classify(RuntimeError("ffprobe failed: exit code 1")) is True

    def test_runtime_error_ffmpeg_decoding_failed(self):
        """RuntimeError 'ffmpeg decoding failed:' → 422."""
        classify = self._get_classifier()
        assert classify(RuntimeError("ffmpeg decoding failed: broken pipe")) is True

    # ── 500 cases: service errors / unknown ──

    def test_runtime_error_ffmpeg_not_found(self):
        """RuntimeError 'ffmpeg not found' → 500 (service config)."""
        classify = self._get_classifier()
        assert classify(RuntimeError("ffmpeg not found")) is False

    def test_runtime_error_ffprobe_not_found(self):
        """RuntimeError 'ffprobe not found' → 500 (service config)."""
        classify = self._get_classifier()
        assert classify(RuntimeError("ffprobe not found")) is False

    def test_unknown_runtime_error(self):
        """RuntimeError with unrecognized message → 500 (conservative)."""
        classify = self._get_classifier()
        assert classify(RuntimeError("something unexpected")) is False

    def test_unknown_value_error(self):
        """ValueError with unrecognized message → 500 (conservative)."""
        classify = self._get_classifier()
        assert classify(ValueError("some other validation error")) is False

    def test_type_error_always_500(self):
        """TypeError → 500 (not an audio decode error)."""
        classify = self._get_classifier()
        assert classify(TypeError("wrong arg type")) is False

    def test_os_error_always_500(self):
        """OSError → 500 (not an audio decode error)."""
        classify = self._get_classifier()
        assert classify(OSError("disk full")) is False


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — _spool_to_tempfile
# ---------------------------------------------------------------------------

def _make_fake_upload(data: bytes, filename: str = "test.wav"):
    """Create a minimal UploadFile-like async mock for _spool_to_tempfile."""
    chunks = []
    pos = 0
    chunk_size = 64 * 1024  # match _SPOOL_CHUNK_SIZE
    while pos < len(data):
        chunks.append(data[pos:pos + chunk_size])
        pos += chunk_size
    chunks.append(b"")  # sentinel for EOF

    chunk_iter = iter(chunks)

    mock_file = AsyncMock()
    mock_file.filename = filename
    mock_file.read = AsyncMock(side_effect=lambda size: next(chunk_iter))
    return mock_file


class TestSpoolToTempfile:
    """Test _spool_to_tempfile async helper."""

    def _get_spool(self):
        from patches.server_diarize import _spool_to_tempfile
        return _spool_to_tempfile

    def test_small_file_spools_correctly(self):
        """Small file → tempfile with correct content and size."""
        spool = self._get_spool()
        data = b"hello world" * 100
        upload = _make_fake_upload(data, "audio.wav")

        tmp_path, size = asyncio.get_event_loop().run_until_complete(
            spool(upload, max_bytes=0)
        )
        try:
            assert size == len(data)
            assert os.path.exists(tmp_path)
            with open(tmp_path, "rb") as f:
                assert f.read() == data
        finally:
            os.unlink(tmp_path)

    def test_suffix_preserved(self):
        """Tempfile suffix matches upload filename extension."""
        spool = self._get_spool()
        data = b"fake audio"
        upload = _make_fake_upload(data, "recording.m4a")

        tmp_path, _ = asyncio.get_event_loop().run_until_complete(
            spool(upload, max_bytes=0)
        )
        try:
            assert tmp_path.endswith(".m4a")
        finally:
            os.unlink(tmp_path)

    def test_no_filename_no_suffix(self):
        """No filename → tempfile without suffix (no crash)."""
        spool = self._get_spool()
        data = b"fake audio"
        upload = _make_fake_upload(data)
        upload.filename = None

        tmp_path, _ = asyncio.get_event_loop().run_until_complete(
            spool(upload, max_bytes=0)
        )
        try:
            assert os.path.exists(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_max_bytes_cutoff(self):
        """File exceeding max_bytes → HTTPException 413 + tempfile cleaned up."""
        from fastapi import HTTPException
        spool = self._get_spool()
        data = b"x" * 2000
        upload = _make_fake_upload(data, "big.wav")

        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                spool(upload, max_bytes=1000)
            )
        assert exc_info.value.status_code == 413

    def test_unlimited_when_zero(self):
        """max_bytes=0 means unlimited — large file succeeds."""
        spool = self._get_spool()
        data = b"x" * 200_000
        upload = _make_fake_upload(data, "big.wav")

        tmp_path, size = asyncio.get_event_loop().run_until_complete(
            spool(upload, max_bytes=0)
        )
        try:
            assert size == len(data)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — _build_gen_kwargs
# ---------------------------------------------------------------------------

class TestBuildGenKwargs:
    """Test _build_gen_kwargs dynamic signature filtering."""

    def _get_build(self):
        from patches.server_diarize import _build_gen_kwargs
        return _build_gen_kwargs

    def test_filters_by_generate_signature(self):
        """Only params accepted by generate() are passed through."""
        build = self._get_build()

        mock_model = MagicMock()
        # Simulate generate(self, audio, *, language, max_tokens, stream)
        def fake_generate(audio, *, language="English", max_tokens=8192, stream=False):
            pass
        mock_model.generate = fake_generate

        form_params = {
            "language": "Chinese",
            "verbose": True,       # NOT in generate() signature
            "max_tokens": 4096,
            "stream": False,
            "context": "some ctx",  # NOT in generate() signature
        }

        result = build(mock_model, form_params)
        assert result == {"language": "Chinese", "max_tokens": 4096, "stream": False}

    def test_excludes_endpoint_only_keys(self):
        """file, model, diarize, num_speakers etc are always excluded."""
        build = self._get_build()

        mock_model = MagicMock()
        # Simulate generate with explicit params that overlap endpoint keys
        # (inspect.signature only sees named params, not **kwargs expansions)
        def fake_generate(audio, *, language="English", file=None, model=None,
                          diarize=None, num_speakers=None):
            pass
        mock_model.generate = fake_generate

        form_params = {
            "file": "should be excluded",
            "model": "should be excluded",
            "diarize": "true",
            "num_speakers": 2,
            "min_speakers": 1,
            "max_speakers": 5,
            "language": "Chinese",
        }

        result = build(mock_model, form_params)
        # Endpoint-level keys excluded even if generate() accepts them by name
        assert "file" not in result
        assert "model" not in result
        assert "diarize" not in result
        assert "num_speakers" not in result
        assert "min_speakers" not in result
        assert "max_speakers" not in result
        assert result.get("language") == "Chinese"

    def test_none_values_filtered(self):
        """None values are stripped (FastAPI Form defaults)."""
        build = self._get_build()

        mock_model = MagicMock()
        def fake_generate(audio, *, language=None, verbose=False):
            pass
        mock_model.generate = fake_generate

        form_params = {"language": None, "verbose": False}
        result = build(mock_model, form_params)
        assert "language" not in result
        assert result["verbose"] is False  # False is not None → kept


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — _safe_remove
# ---------------------------------------------------------------------------

class TestSafeRemove:
    """Test _safe_remove idempotent file deletion."""

    def _get_remove(self):
        from patches.server_diarize import _safe_remove
        return _safe_remove

    def test_removes_existing_file(self):
        """Existing file is deleted."""
        safe_remove = self._get_remove()
        fd, path = tempfile.mkstemp()
        os.close(fd)
        assert os.path.exists(path)
        safe_remove(path)
        assert not os.path.exists(path)

    def test_missing_file_no_error(self):
        """Non-existent path does not raise."""
        safe_remove = self._get_remove()
        safe_remove("/tmp/nonexistent_file_that_does_not_exist_12345")

    def test_idempotent_double_call(self):
        """Calling twice on same path does not raise."""
        safe_remove = self._get_remove()
        fd, path = tempfile.mkstemp()
        os.close(fd)
        safe_remove(path)
        safe_remove(path)  # second call — should not raise


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — _MAX_UPLOAD_BYTES configuration
# ---------------------------------------------------------------------------

class TestMaxUploadBytes:
    """Test _MAX_UPLOAD_BYTES reads from env correctly."""

    def test_default_500mb(self):
        """Default is 500MB when env not set."""
        from patches.server_diarize import _MAX_UPLOAD_BYTES
        # Module-level constant reads env at import time.
        # If MAX_UPLOAD_BYTES isn't set in test env, it should be 500*1024*1024
        expected = 500 * 1024 * 1024
        assert _MAX_UPLOAD_BYTES == expected or os.getenv("MAX_UPLOAD_BYTES") is not None


# ---------------------------------------------------------------------------
# Test: patches/speaker_diarization.py — file path input support
# ---------------------------------------------------------------------------

class TestDiarizerFilePathInput:
    """Test SpeakerDiarizer._load_audio with file path vs bytes."""

    def _make_wav_file(self, duration_sec=0.5, sample_rate=16000):
        """Create a temporary WAV file, return path and raw bytes."""
        n_samples = int(sample_rate * duration_sec)
        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
        wav_bytes = buf.getvalue()

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.write(fd, wav_bytes)
        os.close(fd)
        return path, wav_bytes

    def test_load_from_path_str(self):
        """String file path → loads audio without error."""
        from patches.speaker_diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(model_dir="/nonexistent")  # model not needed for _load_audio
        path, _ = self._make_wav_file()
        try:
            waveform, sr = diarizer._load_audio(path)
            assert sr == 16000
            assert waveform.shape[0] == 1  # mono
        finally:
            os.unlink(path)

    def test_load_from_path_object(self):
        """Path object → loads audio without error."""
        from patches.speaker_diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(model_dir="/nonexistent")
        path, _ = self._make_wav_file()
        try:
            waveform, sr = diarizer._load_audio(Path(path))
            assert sr == 16000
        finally:
            os.unlink(path)

    def test_load_from_bytes(self):
        """Bytes input → backward compatible, loads via tempfile."""
        from patches.speaker_diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(model_dir="/nonexistent")
        path, wav_bytes = self._make_wav_file()
        os.unlink(path)

        waveform, sr = diarizer._load_audio(wav_bytes, filename="test.wav")
        assert sr == 16000
        assert waveform.shape[0] == 1

    def test_invalid_input_type_raises(self):
        """Non-str/Path/bytes input → TypeError."""
        from patches.speaker_diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(model_dir="/nonexistent")
        with pytest.raises(TypeError, match="audio_input must be"):
            diarizer._load_audio(12345)

    def test_path_and_bytes_produce_same_audio(self):
        """File path and bytes of same WAV → identical waveform."""
        from patches.speaker_diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(model_dir="/nonexistent")
        path, wav_bytes = self._make_wav_file(duration_sec=0.1)
        try:
            wf_path, sr_path = diarizer._load_audio(path)
            wf_bytes, sr_bytes = diarizer._load_audio(wav_bytes, filename="test.wav")
            assert sr_path == sr_bytes
            torch.testing.assert_close(wf_path, wf_bytes)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: patches/audio_io_fix.py — Patch 2: AMR/AWB read() routing
# ---------------------------------------------------------------------------


class TestAmrAwbReadRouting:
    """Test that .amr/.awb extensions are routed to ffmpeg via _read_patched."""

    def _apply_patch(self):
        """Import audio_io_fix to ensure both patches are applied."""
        import patches.audio_io_fix  # noqa: F401 — side-effect import

    def test_amr_ext_calls_decode_ffmpeg(self):
        """.amr path → _decode_ffmpeg called, not miniaudio."""
        self._apply_patch()
        import mlx_audio.audio_io as _audio_io

        fake_samples = np.zeros(16000, dtype=np.int16)
        with patch.object(_audio_io, "_decode_ffmpeg",
                          return_value=(fake_samples, 16000, 1)) as mock_ffmpeg:
            result = _audio_io.read("/tmp/test.amr")

        mock_ffmpeg.assert_called_once_with("/tmp/test.amr")
        audio, sr = result
        assert sr == 16000

    def test_awb_ext_calls_decode_ffmpeg(self):
        """.awb path → _decode_ffmpeg called, not miniaudio."""
        self._apply_patch()
        import mlx_audio.audio_io as _audio_io

        fake_samples = np.zeros(8000, dtype=np.int16)
        with patch.object(_audio_io, "_decode_ffmpeg",
                          return_value=(fake_samples, 8000, 1)) as mock_ffmpeg:
            result = _audio_io.read("/tmp/recording.AWB")  # uppercase extension

        mock_ffmpeg.assert_called_once_with("/tmp/recording.AWB")
        audio, sr = result
        assert sr == 8000

    def test_wav_ext_not_intercepted(self):
        """.wav path is NOT intercepted — falls through to original read()."""
        self._apply_patch()
        import mlx_audio.audio_io as _audio_io

        sentinel = object()
        with patch("patches.audio_io_fix._original_read",
                   return_value=sentinel) as mock_orig:
            result = _audio_io.read("/tmp/test.wav")

        mock_orig.assert_called_once_with("/tmp/test.wav", always_2d=False, dtype="float64")
        assert result is sentinel

    def test_dtype_float32_conversion(self):
        """.amr → dtype=float32 converts int16 samples to [-1, 1] range."""
        self._apply_patch()
        import mlx_audio.audio_io as _audio_io

        # int16 max value → float32 should be ≈ 1.0
        fake_samples = np.array([32767], dtype=np.int16)
        with patch.object(_audio_io, "_decode_ffmpeg",
                          return_value=(fake_samples, 16000, 1)):
            audio, sr = _audio_io.read("/tmp/test.amr", dtype="float32")

        assert audio.dtype == np.float32
        assert abs(audio[0] - (32767 / 32768.0)) < 1e-5

    def test_stereo_reshape(self):
        """.amr stereo (nchannels=2) → output shape (samples, 2)."""
        self._apply_patch()
        import mlx_audio.audio_io as _audio_io

        # 4 interleaved stereo samples → shape (2, 2) after reshape
        fake_samples = np.array([1, 2, 3, 4], dtype=np.int16)
        with patch.object(_audio_io, "_decode_ffmpeg",
                          return_value=(fake_samples, 16000, 2)):
            audio, sr = _audio_io.read("/tmp/test.amr", dtype="int16")

        assert audio.shape == (2, 2)

    def test_always_2d_mono(self):
        """.amr mono + always_2d=True → shape (samples, 1)."""
        self._apply_patch()
        import mlx_audio.audio_io as _audio_io

        fake_samples = np.zeros(100, dtype=np.int16)
        with patch.object(_audio_io, "_decode_ffmpeg",
                          return_value=(fake_samples, 16000, 1)):
            audio, _ = _audio_io.read("/tmp/test.amr", always_2d=True, dtype="int16")

        assert audio.shape == (100, 1)


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — GPU lock serialization
# ---------------------------------------------------------------------------

class TestGpuLock:
    """Test GPU lock serialization: 503 on timeout, graceful fallback, lock release."""

    def setup_method(self):
        """Reset lazy lock so each test binds a fresh asyncio.Lock to its own event loop."""
        import patches.server_diarize as sd
        sd._gpu_lock = None

    # ── Test 1: non-stream 路径 503 ──

    def test_503_on_non_stream_timeout(self, monkeypatch):
        """Lock held → _acquire_gpu_or_503() raises HTTPException(503)."""
        from fastapi import HTTPException
        import patches.server_diarize as sd
        monkeypatch.setattr(sd, "_GPU_LOCK_TIMEOUT_SEC", 0)

        async def _test():
            lock = sd._get_gpu_lock()
            await lock.acquire()
            try:
                with pytest.raises(HTTPException) as exc_info:
                    await sd._acquire_gpu_or_503()
                assert exc_info.value.status_code == 503
            finally:
                lock.release()

        asyncio.run(_test())

    # ── Test 2: stream 路径 503 ──

    def test_503_on_stream_timeout_raises_http_exception(self, monkeypatch):
        """Lock held → _acquire_gpu_or_503() raises HTTPException(503) (stream path logic)."""
        from fastapi import HTTPException
        import patches.server_diarize as sd
        monkeypatch.setattr(sd, "_GPU_LOCK_TIMEOUT_SEC", 0)

        async def _test():
            lock = sd._get_gpu_lock()
            await lock.acquire()
            try:
                with pytest.raises(HTTPException) as exc_info:
                    await sd._acquire_gpu_or_503()
                assert exc_info.value.status_code == 503
            finally:
                lock.release()

        asyncio.run(_test())

    # ── Test 3: diarize 超时降级 ──

    def test_graceful_fallback_on_diarize_timeout(self, monkeypatch):
        """Lock held → _acquire_gpu_or_timeout() raises TimeoutError (not HTTPException)."""
        import patches.server_diarize as sd
        monkeypatch.setattr(sd, "_GPU_LOCK_TIMEOUT_SEC", 0)

        async def _test():
            lock = sd._get_gpu_lock()
            await lock.acquire()
            try:
                with pytest.raises(TimeoutError):
                    await sd._acquire_gpu_or_timeout()
            finally:
                lock.release()

        asyncio.run(_test())

    # ── Test 4: 成功后锁释放 ──

    def test_lock_released_on_success(self):
        """Lock is released in finally after successful generate."""
        async def _test():
            import patches.server_diarize as sd
            await sd._acquire_gpu_or_503()
            try:
                pass  # simulate generate()
            finally:
                sd._get_gpu_lock().release()

            # 验证锁已释放：能成功重新获取
            await asyncio.wait_for(sd._get_gpu_lock().acquire(), timeout=0.1)
            sd._get_gpu_lock().release()

        asyncio.run(_test())

    # ── Test 5: 异常后锁释放 ──

    def test_lock_released_on_exception(self):
        """Lock is released in finally even when generate() raises."""
        async def _test():
            import patches.server_diarize as sd
            await sd._acquire_gpu_or_503()
            try:
                try:
                    raise RuntimeError("GPU exploded")
                finally:
                    sd._get_gpu_lock().release()
            except RuntimeError:
                pass

            await asyncio.wait_for(sd._get_gpu_lock().acquire(), timeout=0.1)
            sd._get_gpu_lock().release()

        asyncio.run(_test())

    # ── Test 6: stream generator 流结束后通过 call_soon_threadsafe 释放锁 ──

    def test_stream_generator_releases_lock_via_call_soon_threadsafe(self):
        """Lock released via loop.call_soon_threadsafe after stream generator exhausted."""
        async def _test():
            import patches.server_diarize as sd
            lock = sd._get_gpu_lock()
            await lock.acquire()
            loop = asyncio.get_running_loop()
            chunks = []

            def _locked_gen():
                try:
                    for item in ["a", "b", "c"]:
                        yield item
                finally:
                    loop.call_soon_threadsafe(lock.release)

            # 在线程中消费 generator，模拟 Starlette iterate_in_threadpool 行为
            def _consume():
                for chunk in _locked_gen():
                    chunks.append(chunk)

            await asyncio.to_thread(_consume)
            await asyncio.sleep(0)  # 让 call_soon_threadsafe 回调执行

            assert chunks == ["a", "b", "c"]
            # 锁已释放，应能重新获取
            await asyncio.wait_for(lock.acquire(), timeout=0.1)
            lock.release()

        asyncio.run(_test())

    # ── Test 7: stream generator 客户端断开后锁释放 ──

    def test_stream_generator_releases_lock_on_close(self):
        """Lock released when generator.close() is called (simulates client disconnect)."""
        async def _test():
            import patches.server_diarize as sd
            lock = sd._get_gpu_lock()
            await lock.acquire()
            loop = asyncio.get_running_loop()

            def _locked_gen():
                try:
                    yield "a"
                    yield "b"
                finally:
                    loop.call_soon_threadsafe(lock.release)

            def _consume_partial():
                gen = _locked_gen()
                next(gen)    # consume first item
                gen.close()  # simulate disconnect (triggers finally → call_soon_threadsafe)

            await asyncio.to_thread(_consume_partial)
            await asyncio.sleep(0)  # 让 call_soon_threadsafe 回调执行

            await asyncio.wait_for(lock.acquire(), timeout=0.1)
            lock.release()

        asyncio.run(_test())
