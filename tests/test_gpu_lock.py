"""Unit tests for GPU lock serialization in patches/server_diarize.py."""

import asyncio

import pytest

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


# ---------------------------------------------------------------------------
# Test: patches/server_diarize.py — stream 断连后 GPU 锁必须释放（真实 uvicorn）
# ---------------------------------------------------------------------------
# 旧测试只验证测试内自写的 _locked_gen 副本，绕过了真实断连路径（已删除）。
# uvicorn 的 ASGI spec 2.3 下，Starlette 断连时取消 stream_response，
# 生成器被丢弃而不是 close。这里走真实服务器 + 真实路由 + 假模型。

class _FakeStreamModel:
    """假 STT 模型：每个推理步耗时 delay 秒（在线程池内），记录并发度与已产出块数。"""

    def __init__(self, chunks: int = 4, delay: float = 0.3):
        import threading
        self.chunks = chunks
        self.delay = delay
        self.produced = 0
        self.fail_at = None  # 非 None 时，流式生成在第 fail_at 个推理步前抛 RuntimeError
        self.max_active = 0
        self._active = 0
        self._mu = threading.Lock()

    def _step(self):
        import time
        with self._mu:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delay)
        finally:
            with self._mu:
                self._active -= 1

    def generate(self, path, *, stream=False, language=None, verbose=False,
                 max_tokens=8192, chunk_duration=30.0, frame_threshold=25,
                 context=None, prefill_step_size=2048, text=None):
        if not stream:
            self._step()
            return {"text": "ok"}
        return self._stream()

    def _stream(self):
        for i in range(self.chunks):
            if i == self.fail_at:
                raise RuntimeError("fake generate failure")
            self._step()
            self.produced += 1
            yield f"chunk{i}"


class _LiveServer:
    """真实 uvicorn 跑在后台线程，端口临时分配。"""

    def __init__(self, app):
        import socket
        import threading
        import uvicorn
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._sock]}, daemon=True
        )

    def start(self):
        import time
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=10)
        self._sock.close()


def _post_transcription(port: int, stream: bool, timeout: float = 10.0):
    """发 multipart POST，返回已连接且请求已发出的 socket。"""
    import socket
    boundary = "stt-test-boundary"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    ).encode() + b"RIFF-fake-audio" + (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nfake\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\n'
        f'{str(stream).lower()}\r\n--{boundary}--\r\n'
    ).encode()
    head = (
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.sendall(head + body)
    return sock


def _read_until(sock, marker: bytes) -> bytes:
    buf = b""
    while marker not in buf:
        data = sock.recv(4096)
        if not data:
            break
        buf += data
    return buf


def _status_code(sock) -> int:
    return int(_read_until(sock, b"\r\n").split()[1])


def _wait_lock_free(sd, timeout: float) -> bool:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lock = sd._gpu_lock
        if lock is not None and not lock.locked():
            return True
        time.sleep(0.05)
    return False


class TestStreamDisconnectReleasesGpuLock:
    """stream=true 客户端中途断开 → GPU 锁必须释放，后续请求不被卡住；且不依赖 GC。"""

    @pytest.fixture
    def live(self, monkeypatch, tmp_path):
        import gc
        import tempfile
        from fastapi import FastAPI, File, Form, UploadFile
        from mlx_audio.server import model_provider
        import patches.server_diarize as sd

        fake = _FakeStreamModel()
        monkeypatch.setattr(model_provider, "load_model", lambda name: fake)
        # 若锁泄漏，后续请求 3s 内失败（503），而不是挂 300s
        monkeypatch.setattr(sd, "_GPU_LOCK_TIMEOUT_SEC", 3)
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # spool 文件落入 tmp_path，便于断言清理
        sd._gpu_lock = None

        app = FastAPI()

        @app.post("/v1/audio/transcriptions")
        async def _original(file: UploadFile = File(...), model: str = Form(...)):
            raise AssertionError("original handler must be replaced by setup()")

        sd.setup(app)
        server = _LiveServer(app)
        gc.disable()  # 锁释放不得依赖循环垃圾回收
        try:
            server.start()
            yield server, fake, sd
        finally:
            server.stop()
            gc.enable()
            sd._gpu_lock = None

    def test_abort_right_after_headers_releases_lock(self, live, tmp_path):
        """事故复现：客户端见到 NDJSON 响应头（200 已写入 access log）立刻断开。"""
        server, fake, sd = live
        sock = _post_transcription(server.port, stream=True)
        status_line = _read_until(sock, b"\r\n\r\n").split(b"\r\n", 1)[0]
        assert b" 200 " in status_line
        sock.close()

        assert _wait_lock_free(sd, timeout=5.0), "断连后 GPU 锁未释放（泄漏）"
        assert fake.produced < fake.chunks, "断连后应提前停止生成，而不是跑完整段音频"
        assert not list(tmp_path.iterdir()), "断连后 tempfile 未清理"

        follow_up = _post_transcription(server.port, stream=False)
        assert _status_code(follow_up) == 200
        assert fake.max_active == 1, "GPU 推理出现并发"

    def test_abort_after_first_chunk_releases_lock(self, live, tmp_path):
        """客户端读到首块后断开。"""
        server, fake, sd = live
        sock = _post_transcription(server.port, stream=True)
        _read_until(sock, b"chunk0")
        sock.close()

        assert _wait_lock_free(sd, timeout=5.0), "断连后 GPU 锁未释放（泄漏）"
        assert fake.produced < fake.chunks, "断连后应提前停止生成，而不是跑完整段音频"
        assert not list(tmp_path.iterdir()), "断连后 tempfile 未清理"

        follow_up = _post_transcription(server.port, stream=False)
        assert _status_code(follow_up) == 200
        assert fake.max_active == 1, "GPU 推理出现并发"

    def test_complete_stream_releases_lock(self, live, tmp_path):
        """对照组：客户端读完整个流，锁释放，输出完整。"""
        server, fake, sd = live
        sock = _post_transcription(server.port, stream=True)
        body = b""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            body += data

        assert b"chunk3" in body
        assert _wait_lock_free(sd, timeout=2.0)
        assert not list(tmp_path.iterdir()), "流结束后 tempfile 未清理"
        assert fake.produced == fake.chunks

    def test_generate_error_releases_lock_and_cleans_tempfile(self, live, tmp_path):
        """worker 内 generate 抛错：锁释放、tempfile 清理、后续请求正常。"""
        server, fake, sd = live
        fake.fail_at = 1
        sock = _post_transcription(server.port, stream=True)
        try:
            while sock.recv(4096):  # 响应头后异常 → 服务端直接断开连接
                pass
        except ConnectionError:
            pass

        assert _wait_lock_free(sd, timeout=5.0), "generate 抛错后 GPU 锁未释放"
        assert not list(tmp_path.iterdir()), "generate 抛错后 tempfile 未清理"

        fake.fail_at = None
        follow_up = _post_transcription(server.port, stream=False)
        assert _status_code(follow_up) == 200
