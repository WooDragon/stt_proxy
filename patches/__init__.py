"""Local monkey-patches for mlx-audio.

Each patch targets a different module/function — import order is irrelevant.
"""


def apply_all():
    """Apply all monkey-patches. Call before importing mlx_audio.server."""
    from patches import audio_io_fix  # noqa: F401 — _audio_io._decode_ffmpeg
    from patches import generate_enhance  # noqa: F401 — Qwen3ASRModel.generate
    from patches import vad_chunking  # noqa: F401 — split_audio_into_chunks
