#!/usr/bin/env python
"""Patched mlx-audio server launcher.

Applies local monkey-patches before starting the server,
then delegates to mlx_audio.server via runpy (preserving __main__ semantics
so uvicorn workers can re-import mlx_audio.server:app cleanly).
Remove this wrapper when upstream fixes are merged.
"""

import patches

patches.apply_all()

import runpy  # noqa: E402

runpy.run_module("mlx_audio.server", run_name="__main__", alter_sys=True)
