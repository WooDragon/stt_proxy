"""Unit tests for patches/audio_io_fix.py."""

from unittest.mock import patch

import numpy as np

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


