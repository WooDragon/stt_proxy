"""Monkey-patch: mlx-audio audio_io format-routing fixes.

Patch 1 — _decode_ffmpeg M4A/AAC pipe seek bug (mlx-audio <= 0.3.1):
    Bug:  _decode_ffmpeg pipes bytes via stdin (pipe:0) to ffmpeg.
          MP4/M4A containers need seekable input — moov atom sits at EOF,
          ffmpeg can't seek on a pipe, silently returns 0 samples.
    Fix:  bytes → tempfile → file path, then call original _decode_ffmpeg.
    Drop: when upstream _decode_ffmpeg handles bytes via tempfile.

Patch 2 — read() AMR/AWB format routing:
    Bug:  mlx-audio only routes .m4a/.aac extensions to ffmpeg; .amr/.awb
          fall through to miniaudio which cannot decode AMR → DecodeError.
    Fix:  read() wrapper routes _FFMPEG_EXTS to _decode_ffmpeg before
          falling through to the original implementation.
    Drop: when upstream read() adds native AMR/AWB support.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

import mlx_audio.audio_io as _audio_io

# ── Patch 1: _decode_ffmpeg bytes input via tempfile ─────────────────────────

_original_decode_ffmpeg = _audio_io._decode_ffmpeg


def _decode_ffmpeg_patched(input_data):
    """Wrap _decode_ffmpeg: bytes input writes to tempfile for seekable read."""
    if not isinstance(input_data, bytes):
        return _original_decode_ffmpeg(input_data)

    fd, tmp_path = tempfile.mkstemp(suffix=".audio")
    try:
        os.write(fd, input_data)
        os.close(fd)
        return _original_decode_ffmpeg(tmp_path)
    finally:
        os.unlink(tmp_path)


_audio_io._decode_ffmpeg = _decode_ffmpeg_patched

# ── Patch 2: read() AMR/AWB extension routing ────────────────────────────────

# Extensions that need ffmpeg but mlx-audio's read() doesn't route to it.
_FFMPEG_EXTS = frozenset(("amr", "awb"))

_original_read = _audio_io.read


def _read_patched(file, always_2d=False, dtype="float64"):
    """Wrap read(): route .amr/.awb extensions to ffmpeg.

    mlx-audio's read() only routes .m4a/.aac to ffmpeg; AMR/AWB would fall
    through to miniaudio which cannot decode them.  We intercept file-path
    inputs with known AMR extensions and call _decode_ffmpeg directly.

    BytesIO inputs for AMR are not currently used by this project and are
    left to the original implementation (will raise DecodeError → 422).

    Args:
        file: Path to the audio file or a BytesIO object.
        always_2d: If True, always return a 2D array (samples, channels).
        dtype: Data type for the output array ('float32', 'float64', 'int16').

    Returns:
        Tuple of (audio_data, sample_rate).
    """
    if isinstance(file, (str, Path)):
        ext = Path(file).suffix.lstrip(".").lower()
        if ext in _FFMPEG_EXTS:
            # _decode_ffmpeg is already patched above; file-path input is fine
            samples, sample_rate, nchannels = _audio_io._decode_ffmpeg(str(file))
            if nchannels > 1:
                samples = samples.reshape(-1, nchannels)
            if dtype in ("float32", "float64"):
                samples = samples.astype(dtype) / 32768.0
            elif dtype != "int16":
                samples = samples.astype(dtype)
            if always_2d and samples.ndim == 1:
                samples = samples[:, np.newaxis]
            return samples, sample_rate

    return _original_read(file, always_2d=always_2d, dtype=dtype)


_audio_io.read = _read_patched
