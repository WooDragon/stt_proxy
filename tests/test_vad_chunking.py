"""Unit tests for patches/vad_chunking.py."""


import numpy as np

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


