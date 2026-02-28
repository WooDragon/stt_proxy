"""Endpoint wrapper: add speaker diarization + streaming tempfile to /v1/audio/transcriptions.

Replaces the original FastAPI route with a direct-to-model wrapper that:
    - Spools upload to tempfile (async chunked, with size limit → 413).
    - Calls model.generate(tempfile_path) directly — single decode, no re-encode.
    - diarize=false (default): generate → NDJSON response.
    - diarize=true: generate → diarize(tempfile) → JSONResponse with speakers.

Previous architecture delegated to the original handler (server.py stt_transcriptions),
which did decode→reencode→decode (double-decode, ~13GB peak for 200MB M4A).
Now: spool→generate(path) — single decode, ~6.6GB peak → ~2.5GB steady state.

Startup safety: inspect.signature() compares original handler parameters.
Mismatch → WARNING log (non-fatal, detects upstream API drift).
"""

import asyncio
import inspect
import json
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from typing import Optional

from patches.speaker_diarization import SpeakerDiarizer, assign_speakers

logger = logging.getLogger("stt_proxy.diarize")

# ── Upload size limit ────────────────────────────────────────────────────
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))  # 500MB
_SPOOL_CHUNK_SIZE = 64 * 1024  # 64KB read chunks

# ── Audio decode error patterns (from mlx-audio audio_io.py) ──
# Service config errors: ffmpeg/ffprobe not installed (always 500)
_SERVICE_ERROR_MARKERS = ("ffmpeg not found", "ffprobe not found")
# Audio decode errors: bad/corrupt/unsupported file (422)
_AUDIO_RUNTIME_MARKERS = (
    "No audio streams found",
    "ffprobe failed:",
    "ffmpeg decoding failed:",
)
_AUDIO_VALUE_MARKERS = (
    "Unable to detect audio format",
    "Unsupported format:",
)


def _is_audio_decode_error(exc: Exception) -> bool:
    """Classify exception as client-side audio error (True → 422) vs service error (False → 500).

    Conservative: defaults to False (500) for any unrecognized pattern.
    Matches exact upstream message prefixes from mlx-audio audio_io.py.
    """
    # Duck-type miniaudio.DecodeError (avoid importing undeclared dependency)
    if type(exc).__name__ == "DecodeError":
        return True

    msg = str(exc)

    if isinstance(exc, RuntimeError):
        # Service config errors take priority
        if any(marker in msg for marker in _SERVICE_ERROR_MARKERS):
            return False
        return any(msg.startswith(marker) for marker in _AUDIO_RUNTIME_MARKERS)

    if isinstance(exc, ValueError):
        return any(msg.startswith(marker) for marker in _AUDIO_VALUE_MARKERS)

    return False


# Module-level singleton (lazy model load on first diarize=true request)
_diarizer = SpeakerDiarizer()

# Parameters that the original handler is expected to have.
# Used for startup signature validation.
_EXPECTED_ORIGINAL_PARAMS = {
    "file", "model", "language", "verbose", "max_tokens",
    "chunk_duration", "frame_threshold", "stream", "context",
    "prefill_step_size", "text",
}


# ── Helper functions ─────────────────────────────────────────────────────

async def _spool_to_tempfile(upload_file: UploadFile, max_bytes: int):
    """Async chunked spool: UploadFile → tempfile on disk.

    Reads upload in 64KB chunks via await (non-blocking event loop).
    Counts bytes in real time; exceeds max_bytes → delete remnant + HTTP 413.

    Args:
        upload_file: FastAPI UploadFile (async readable).
        max_bytes: Hard ceiling. 0 = unlimited.

    Returns:
        (tmp_path, file_size) — caller owns the tempfile, caller must clean up.
    """
    # Extract suffix from original filename for ffmpeg format hint
    suffix = ""
    if upload_file.filename:
        dot_idx = upload_file.filename.rfind(".")
        if dot_idx >= 0:
            suffix = upload_file.filename[dot_idx:]  # e.g. ".m4a"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    total = 0
    try:
        while True:
            chunk = await upload_file.read(_SPOOL_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes > 0 and total > max_bytes:
                os.close(fd)
                _safe_remove(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {max_bytes} bytes limit",
                )
            os.write(fd, chunk)
        os.close(fd)
    except HTTPException:
        raise
    except Exception:
        # Clean up on unexpected errors during spool
        try:
            os.close(fd)
        except OSError:
            pass
        _safe_remove(tmp_path)
        raise

    return tmp_path, total


def _build_gen_kwargs(stt_model, form_params: dict) -> dict:
    """Build kwargs for stt_model.generate() from Form parameters.

    Filters by inspecting generate() signature — no hardcoded param names.
    Removes None values and non-generate keys (file, model, diarize, etc.).

    Args:
        stt_model: Loaded STT model instance with generate() method.
        form_params: Dict of all Form parameters from the endpoint.

    Returns:
        Dict of kwargs safe to pass to stt_model.generate().
    """
    # Keys that are endpoint-level, not model-level
    exclude_keys = {"file", "model", "diarize", "num_speakers", "min_speakers", "max_speakers"}

    # Filter out None values and endpoint-only keys
    candidates = {k: v for k, v in form_params.items()
                  if k not in exclude_keys and v is not None}

    # Intersect with generate() actual signature
    sig = inspect.signature(stt_model.generate)
    accepted = set(sig.parameters.keys())

    return {k: v for k, v in candidates.items() if k in accepted}


def _safe_remove(path: str):
    """Delete file silently. Idempotent — missing file is not an error."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _generate_stream(stt_model, tmp_path: str, gen_kwargs: dict):
    """Sync generator: call generate(stream=True) and yield NDJSON lines.

    Pure business logic — does NOT manage tempfile cleanup.
    Caller attaches BackgroundTask(_safe_remove) to the response.

    Args:
        stt_model: Loaded STT model.
        tmp_path: Path to audio tempfile.
        gen_kwargs: Kwargs for generate() (must include stream=True).
    """
    from mlx_audio.server import sanitize_for_json

    result = stt_model.generate(tmp_path, **gen_kwargs)

    # stream=True → generator; iterate and yield NDJSON
    if hasattr(result, "__iter__") and hasattr(result, "__next__"):
        accumulated_text = ""
        for chunk in result:
            if isinstance(chunk, str):
                accumulated_text += chunk
                chunk_data = {"text": chunk, "accumulated": accumulated_text}
            else:
                chunk_data = {
                    "text": chunk.text,
                    "start": getattr(chunk, "start_time", None),
                    "end": getattr(chunk, "end_time", None),
                    "is_final": getattr(chunk, "is_final", None),
                    "language": getattr(chunk, "language", None),
                }
            yield json.dumps(sanitize_for_json(chunk_data)) + "\n"
    else:
        # Shouldn't happen with stream=True, but handle gracefully
        yield json.dumps(sanitize_for_json(result)) + "\n"


# ── Route setup ──────────────────────────────────────────────────────────

def setup(app: FastAPI):
    """Replace /v1/audio/transcriptions route with direct-to-model wrapper.

    Must be called AFTER patches.apply_all() and BEFORE uvicorn.run().
    """
    # 1. Find the original handler (for signature validation + route removal)
    original_handler = None
    route_index = None
    for i, route in enumerate(app.routes):
        if hasattr(route, "path") and route.path == "/v1/audio/transcriptions":
            if hasattr(route, "endpoint"):
                original_handler = route.endpoint
                route_index = i
                break

    if original_handler is None:
        logger.error("Cannot find /v1/audio/transcriptions route — diarization disabled")
        return

    # 2. Startup signature validation (detect upstream API drift)
    orig_sig = inspect.signature(original_handler)
    orig_params = set(orig_sig.parameters.keys())
    missing = _EXPECTED_ORIGINAL_PARAMS - orig_params
    extra = orig_params - _EXPECTED_ORIGINAL_PARAMS
    if missing:
        logger.warning(
            "Upstream handler REMOVED params %s — wrapper may be stale. "
            "Update server_diarize.py to match upstream.", missing
        )
    if extra:
        logger.warning(
            "Upstream handler ADDED params %s — these params won't be forwarded "
            "to generate(). Update server_diarize.py to match upstream.", extra
        )

    # 3. Remove old route (prevent it from receiving requests)
    app.routes.pop(route_index)

    # 4. Register new direct-to-model handler
    @app.post("/v1/audio/transcriptions")
    async def wrapped_transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        verbose: bool = Form(False),
        max_tokens: int = Form(8192),
        chunk_duration: float = Form(30.0),
        frame_threshold: int = Form(25),
        stream: bool = Form(False),
        context: Optional[str] = Form(None),
        prefill_step_size: int = Form(2048),
        text: Optional[str] = Form(None),
        # ── Diarization params ──
        diarize: str = Form("false"),
        num_speakers: Optional[int] = Form(None),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
    ):
        """Transcribe audio, optionally with speaker diarization.

        Spools upload to tempfile → single model.generate(path) decode.
        No re-encoding, no double-decode.
        """
        from mlx_audio.server import model_provider, sanitize_for_json

        want_diarize = diarize.lower() in ("true", "1")

        if want_diarize and stream:
            raise HTTPException(
                status_code=400,
                detail="diarize=true is incompatible with stream=true",
            )

        # Spool upload to tempfile (async chunked + size limit)
        tmp_path, file_size = await _spool_to_tempfile(file, _MAX_UPLOAD_BYTES)

        try:
            stt_model = model_provider.load_model(model)

            # Build generate() kwargs from Form params (dynamic signature filter)
            form_params = dict(
                language=language, verbose=verbose, max_tokens=max_tokens,
                chunk_duration=chunk_duration, frame_threshold=frame_threshold,
                stream=stream, context=context, prefill_step_size=prefill_step_size,
                text=text,
            )
            gen_kwargs = _build_gen_kwargs(stt_model, form_params)

            if not want_diarize:
                # ── Non-diarize path ──
                if stream:
                    # stream=true: sync generator via StreamingResponse
                    response = StreamingResponse(
                        _generate_stream(stt_model, tmp_path, gen_kwargs),
                        media_type="application/x-ndjson",
                    )
                    response.background = BackgroundTask(_safe_remove, tmp_path)
                    return response
                else:
                    # stream=false: generate() returns STTOutput dataclass
                    result = await asyncio.to_thread(
                        stt_model.generate, tmp_path, **gen_kwargs
                    )
                    content = json.dumps(sanitize_for_json(result)) + "\n"
                    response = Response(
                        content=content, media_type="application/x-ndjson"
                    )
                    response.background = BackgroundTask(_safe_remove, tmp_path)
                    return response

            else:
                # ── Diarize path (stream already rejected above) ──
                # ASR: single decode via generate()
                # Force stream=False for ASR phase
                asr_kwargs = {k: v for k, v in gen_kwargs.items() if k != "stream"}
                result = await asyncio.to_thread(
                    stt_model.generate, tmp_path, **asr_kwargs
                )
                asr_result = sanitize_for_json(result)

                # Diarization: pass tempfile path directly (no extra decode)
                try:
                    speaker_timeline = await asyncio.to_thread(
                        _diarizer.diarize,
                        tmp_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    enriched = assign_speakers(asr_result, speaker_timeline)
                    return JSONResponse(enriched)

                except Exception as exc:
                    # Graceful degradation: ASR result + diarization error
                    logger.exception("Diarization failed, returning ASR-only result")
                    asr_result["diarization_error"] = str(exc)
                    return JSONResponse(asr_result)

                finally:
                    _safe_remove(tmp_path)

        except HTTPException:
            _safe_remove(tmp_path)
            raise
        except Exception as exc:
            _safe_remove(tmp_path)
            if _is_audio_decode_error(exc):
                raise HTTPException(
                    status_code=422,
                    detail=f"Audio decode failed: {exc}",
                ) from exc
            raise

    logger.info("Diarization endpoint registered at /v1/audio/transcriptions")
