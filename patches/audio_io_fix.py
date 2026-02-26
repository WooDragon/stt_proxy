"""Monkey-patch: mlx-audio _decode_ffmpeg M4A/AAC pipe seek bug.

Bug:  _decode_ffmpeg pipes bytes via stdin (pipe:0) to ffmpeg.
      MP4/M4A containers need seekable input — moov atom sits at EOF,
      ffmpeg can't seek on a pipe, silently returns 0 samples.
Fix:  bytes → tempfile → file path, then call original _decode_ffmpeg.
Hits: mlx-audio <= 0.3.1
Drop: when upstream _decode_ffmpeg handles bytes via tempfile.
"""

import os
import tempfile

import mlx_audio.audio_io as _audio_io

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
