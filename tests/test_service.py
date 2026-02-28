"""
ASR 服务端到端黑盒测试

纯黑盒测试：不 mock 第三方库内部，直接向运行中的服务发 HTTP 请求。
前置条件：服务必须在 localhost:PORT 上运行（默认 10241）。

运行方式:
    pytest tests/test_service.py -v
    PORT=12345 pytest tests/test_service.py -v

标记:
    @pytest.mark.integration — 需要运行中的服务，CI 中可跳过
"""

import io
import os
import struct
import wave

import httpx
import pytest

PORT = int(os.getenv("PORT", "10241"))
MODEL = os.getenv("MODEL", "mlx-community/Qwen3-ASR-1.7B-8bit")
BASE_URL = f"http://localhost:{PORT}"


def _service_is_running() -> bool:
    """检查服务是否在运行"""
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=3.0)
        return resp.status_code == 200
    except httpx.ConnectError:
        return False


def _make_wav(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    """生成内存中的静音 WAV 文件"""
    n_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
    return buf.getvalue()


# 服务未运行时跳过全部测试
pytestmark = pytest.mark.skipif(
    not _service_is_running(),
    reason=f"ASR 服务未在 localhost:{PORT} 运行",
)


@pytest.mark.integration
class TestModelsEndpoint:
    """GET /v1/models 端点测试"""

    def test_models_returns_json(self):
        """模型列表返回合法 JSON"""
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)


@pytest.mark.integration
class TestTranscriptionEndpoint:
    """POST /v1/audio/transcriptions 端点测试"""

    def test_transcribe_silence_returns_text(self):
        """静音 WAV 转写返回合法响应"""
        wav_data = _make_wav(duration_sec=0.5)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("silence.wav", wav_data, "audio/wav")},
            data={"model": MODEL},
            timeout=120.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert isinstance(body["text"], str)

    def test_missing_model_returns_422(self):
        """不传 model 参数返回 422"""
        wav_data = _make_wav(duration_sec=0.3)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("test.wav", wav_data, "audio/wav")},
            timeout=30.0,
        )
        assert resp.status_code == 422

    def test_missing_file_returns_422(self):
        """不传音频文件返回 422"""
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            data={"model": MODEL},
            timeout=30.0,
        )
        assert resp.status_code == 422


@pytest.mark.integration
class TestDiarization:
    """Speaker diarization endpoint tests."""

    def test_diarize_false_unchanged(self):
        """diarize=false returns normal transcription (no speaker fields)."""
        wav_data = _make_wav(duration_sec=0.5)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("silence.wav", wav_data, "audio/wav")},
            data={"model": MODEL, "diarize": "false"},
            timeout=120.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body

    def test_diarize_true_returns_speakers(self):
        """diarize=true returns speaker labels and speakers list."""
        wav_data = _make_wav(duration_sec=1.0)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("silence.wav", wav_data, "audio/wav")},
            data={"model": MODEL, "diarize": "true"},
            timeout=120.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert "speakers" in body
        assert isinstance(body["speakers"], list)
        # If segments exist, they should have speaker labels or diarization_error
        if "segments" in body and body["segments"]:
            for seg in body["segments"]:
                assert "text" in seg
        # diarization_error may be present if model not converted (graceful degradation)
        if "diarization_error" in body:
            assert isinstance(body["diarization_error"], str)

    def test_stream_plus_diarize_returns_400(self):
        """stream=true + diarize=true → HTTP 400."""
        wav_data = _make_wav(duration_sec=0.5)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("silence.wav", wav_data, "audio/wav")},
            data={"model": MODEL, "diarize": "true", "stream": "true"},
            timeout=30.0,
        )
        assert resp.status_code == 400


@pytest.mark.integration
class TestErrorDifferentiation:
    """Verify 422 (bad audio) vs 500 (service error) differentiation."""

    def test_corrupt_audio_returns_422(self):
        """Random garbage bytes → 422 with 'Audio decode failed' detail."""
        garbage = os.urandom(256)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("garbage.wav", garbage, "audio/wav")},
            data={"model": MODEL},
            timeout=30.0,
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "Audio decode failed" in body["detail"]

    def test_corrupt_audio_diarize_returns_422(self):
        """diarize=true + garbage bytes → 422 (not graceful degradation)."""
        garbage = os.urandom(256)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("garbage.wav", garbage, "audio/wav")},
            data={"model": MODEL, "diarize": "true"},
            timeout=30.0,
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "Audio decode failed" in body["detail"]

    def test_valid_wav_still_200(self):
        """Regression: valid WAV still returns 200."""
        wav_data = _make_wav(duration_sec=0.5)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"model": MODEL},
            timeout=120.0,
        )
        assert resp.status_code == 200


@pytest.mark.integration
class TestUploadSizeLimit:
    """Upload size limit (MAX_UPLOAD_BYTES) enforcement."""

    def test_oversized_upload_returns_413(self):
        """Upload exceeding MAX_UPLOAD_BYTES → HTTP 413.

        Uses a minimal payload just above the configured limit.
        Reads MAX_UPLOAD_BYTES from env (default 500MB), then sends limit+1 bytes.
        Skip if limit is too large to construct in-memory (>10MB test ceiling).
        """
        max_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
        test_ceiling = 10 * 1024 * 1024  # 10MB — don't allocate more in a test
        if max_bytes > test_ceiling:
            pytest.skip(
                f"MAX_UPLOAD_BYTES={max_bytes} too large for in-memory test "
                f"(ceiling={test_ceiling}). Set MAX_UPLOAD_BYTES={test_ceiling} to test."
            )

        # Payload just above limit
        garbage = b"x" * (max_bytes + 1)
        resp = httpx.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": ("huge.wav", garbage, "audio/wav")},
            data={"model": MODEL},
            timeout=30.0,
        )
        assert resp.status_code == 413
        body = resp.json()
        assert "exceeds" in body["detail"].lower() or "limit" in body["detail"].lower()
