#!/usr/bin/env python
"""Patched mlx-audio server launcher.

Applies local monkey-patches before starting the server,
then delegates to mlx_audio.server via runpy (preserving __main__ semantics
so uvicorn workers can re-import mlx_audio.server:app cleanly).
Remove this wrapper when upstream fixes are merged.
"""

import patches.audio_io_fix  # noqa: F401 — M4A pipe seek fix

import runpy

runpy.run_module("mlx_audio.server", run_name="__main__", alter_sys=True)
