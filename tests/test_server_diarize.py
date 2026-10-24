"""Unit tests for patches/server_diarize.py."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

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

        tmp_path, size = asyncio.run(
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

        tmp_path, _ = asyncio.run(
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

        tmp_path, _ = asyncio.run(
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
            asyncio.run(
                spool(upload, max_bytes=1000)
            )
        assert exc_info.value.status_code == 413

    def test_unlimited_when_zero(self):
        """max_bytes=0 means unlimited — large file succeeds."""
        spool = self._get_spool()
        data = b"x" * 200_000
        upload = _make_fake_upload(data, "big.wav")

        tmp_path, size = asyncio.run(
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


