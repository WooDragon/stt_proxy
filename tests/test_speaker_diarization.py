"""Unit tests for patches/speaker_diarization.py."""

import io
import os
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

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


