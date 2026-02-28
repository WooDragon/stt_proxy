#!/usr/bin/env python
"""Patched mlx-audio server launcher.

Applies local monkey-patches, sets up diarization endpoint, then starts uvicorn.

Why not runpy: runpy.run_module("mlx_audio.server", run_name="__main__") immediately
enters __main__ → argparse → start_server() → uvicorn.run(), blocking forever.
We need a control window between "app created" and "uvicorn started" to call
server_diarize.setup(app). Direct import + manual uvicorn.run() gives us that window.

Remove this wrapper when upstream fixes are merged AND diarization is no longer needed.
"""

import argparse
import sys

# 1. Apply monkey-patches BEFORE importing mlx_audio.server
#    (patches replace module-level references at import time)
import patches
patches.apply_all()

# 2. Import server module — this creates the FastAPI `app` at module level
#    and registers all default routes. The __main__ block is NOT executed
#    because we import as a module, not run as __main__.
import mlx_audio.server  # noqa: E402

# 3. Inject diarization endpoint into the app's route table
from patches.server_diarize import setup as setup_diarize  # noqa: E402
setup_diarize(mlx_audio.server.app)


def main():
    parser = argparse.ArgumentParser(description="Patched mlx-audio STT server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10241)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Hard guard: workers > 1 is broken on macOS
    # Reason 1: macOS Python multiprocessing uses spawn — child processes
    #           don't inherit monkey-patches.
    # Reason 2: uvicorn.run(app_instance) with workers>1 can't re-import the app.
    # Reason 3: Single GPU — no throughput benefit from multiple workers.
    if args.workers > 1:
        print("FATAL: --workers must be 1. Multi-worker breaks monkey-patches on macOS.",
              file=sys.stderr)
        sys.exit(1)

    import uvicorn  # noqa: E402
    uvicorn.run(
        mlx_audio.server.app,
        host=args.host,
        port=args.port,
        workers=1,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
