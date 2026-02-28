#!/usr/bin/env python
"""Offline conversion: WeSpeaker ResNet34-LM ONNX → CoreML FP16.

Conversion pipeline:
    ONNX (HuggingFace, ~26.5MB)
      → onnx2torch → PyTorch
      → @register_torch_op(prod) — fix TSTP pooling ReduceProd
      → torch.jit.trace
      → coremltools.convert(FP16, macOS14)
      → models/wespeaker/wespeaker_resnet34_LM_fp16.mlpackage (~12.7MB)

Build-time dependencies (NOT in requirements.txt):
    pip install onnx onnx2torch

Run once:
    python scripts/convert_wespeaker.py

The resulting .mlpackage is loaded at runtime by SpeakerDiarizer.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

# ── Register custom torch op BEFORE coremltools conversion ──────────────
# onnx2torch maps ONNX ReduceProd to torch.prod, which coremltools
# doesn't natively support. Register a converter: torch.prod → MIL reduce_prod.
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs


@register_torch_op()
def prod(context, node):
    """Map torch.prod → MIL reduce_prod for TSTP pooling."""
    inputs = _get_inputs(context, node)
    x = inputs[0]
    if len(inputs) > 1 and inputs[1] is not None:
        axes = [inputs[1].val] if not isinstance(inputs[1].val, list) else inputs[1].val
        keepdim = inputs[2].val if len(inputs) > 2 else False
        result = mb.reduce_prod(x=x, axes=axes, keep_dims=keepdim)
    else:
        result = mb.reduce_prod(x=x, keep_dims=False)
    context.add(result, torch_name=node.name)


# ── Paths ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_DIR = _ROOT / "models" / "wespeaker"

ONNX_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/voxceleb_resnet34_LM.onnx"
)


def _download_onnx(model_dir: Path) -> Path:
    """Download ONNX model if not cached."""
    onnx_path = model_dir / "voxceleb_resnet34_LM.onnx"
    if onnx_path.exists():
        print(f"[skip] ONNX already at {onnx_path}")
        return onnx_path

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {ONNX_URL}")
    urllib.request.urlretrieve(ONNX_URL, onnx_path)
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"[done] {size_mb:.1f} MB → {onnx_path}")
    return onnx_path


def _onnx_to_pytorch(onnx_path: Path):
    """ONNX → PyTorch via onnx2torch."""
    from onnx2torch import convert as onnx2torch_convert

    print("[convert] ONNX → PyTorch")
    pt_model = onnx2torch_convert(str(onnx_path))
    pt_model.eval()
    n_params = sum(p.numel() for p in pt_model.parameters())
    print(f"  params: {n_params:,}")
    return pt_model


def _trace(pt_model, num_frames: int = 200):
    """JIT-trace PyTorch model with dummy input."""
    print("[trace] JIT tracing")
    dummy = torch.randn(1, num_frames, 80)
    with torch.no_grad():
        traced = torch.jit.trace(pt_model, dummy)
    print("  OK")
    return traced


def _convert_coreml(traced_model, output_path: Path):
    """Traced PyTorch → CoreML FP16 .mlpackage."""
    import coremltools as ct

    if output_path.exists():
        print(f"[skip] CoreML already at {output_path}")
        return

    print("[convert] PyTorch → CoreML FP16")
    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="feats",
                shape=ct.Shape(
                    shape=(1, ct.RangeDim(lower_bound=10, upper_bound=3000), 80)
                ),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="embs")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    model.save(str(output_path))
    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  saved {size_mb:.1f} MB → {output_path}")


def main():
    """Run the full conversion pipeline."""
    model_dir = Path(os.getenv("WESPEAKER_CACHE", str(_DEFAULT_MODEL_DIR)))
    output_path = model_dir / "wespeaker_resnet34_LM_fp16.mlpackage"

    if output_path.exists():
        print(f"CoreML model already exists at {output_path}")
        print("Delete it first if you want to reconvert.")
        return

    # 1. Download ONNX
    onnx_path = _download_onnx(model_dir)

    # 2. ONNX → PyTorch
    pt_model = _onnx_to_pytorch(onnx_path)

    # 3. JIT trace
    traced = _trace(pt_model)

    # 4. CoreML FP16 export
    _convert_coreml(traced, output_path)

    # 5. Quick sanity check
    import coremltools as ct
    cm = ct.models.MLModel(str(output_path))
    dummy = np.random.randn(1, 200, 80).astype(np.float32)
    pred = cm.predict({"feats": dummy})
    emb_key = "embs" if "embs" in pred else list(pred.keys())[0]
    emb = np.array(pred[emb_key])
    print(f"\n[verify] output shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    print("Conversion complete.")


if __name__ == "__main__":
    main()
