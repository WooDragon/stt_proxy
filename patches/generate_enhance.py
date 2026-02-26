"""Monkey-patch: Qwen3ASRModel.generate defaults + STTOutput.language backfill.

Fixes:
  1. language defaults to "Chinese" instead of "English" (Chinese ASR service).
  2. repetition_penalty injected from .env (suppresses hallucinated repetitions).
  3. STTOutput.language backfilled from actual language used.

Hits: mlx-audio <= 0.3.1 (generate() hardcodes language="English", never sets output.language)
Drop: when upstream defaults to Chinese and populates STTOutput.language.
"""

import functools
import os

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

_DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.2"))

_original_generate = Qwen3ASRModel.generate


@functools.wraps(_original_generate)
def _generate_patched(self, audio, **kwargs):
    """Wrap generate(): fix language default, inject repetition_penalty, backfill output.language."""
    # P3: default language to Chinese (setdefault preserves explicit caller value)
    kwargs.setdefault("language", "Chinese")

    # P1: inject repetition_penalty if not explicitly provided
    kwargs.setdefault("repetition_penalty", _DEFAULT_REPETITION_PENALTY)

    language_used = kwargs["language"]

    result = _original_generate(self, audio, **kwargs)

    # P3: backfill STTOutput.language (upstream never sets it)
    if hasattr(result, "language"):
        result.language = language_used

    return result


Qwen3ASRModel.generate = _generate_patched
