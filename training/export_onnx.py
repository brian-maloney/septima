#!/usr/bin/env python3
"""Export a trained YOLO stage to ONNX and install it into models/.

Loads the best.pt from training/runs/<stage>/weights, exports to ONNX, copies it
to models/<stage>.onnx, and verifies the model's class order matches
models/classes.json (panel_classes / digit_classes). The Go inference engine
relies on that order being authoritative.

Usage:
  python training/export_onnx.py --stage digits
  python training/export_onnx.py --stage panel
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODELS = HERE.parent / "models"


def expected_classes(stage: str) -> list[str]:
    cfg = json.loads((MODELS / "classes.json").read_text())
    return cfg["digit_classes"] if stage == "digits" else cfg["panel_classes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["digits", "panel"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default=None, help="override best.pt path")
    args = ap.parse_args()

    best = Path(args.weights) if args.weights else RUNS / args.stage / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"missing trained weights {best}; run train.py --stage {args.stage} first")

    model = YOLO(str(best))

    # Verify class order before exporting — a mismatch would silently corrupt
    # decoding on the Go side.
    names = [model.names[i] for i in range(len(model.names))]
    want = expected_classes(args.stage)
    if names != want:
        raise SystemExit(
            f"class order mismatch for {args.stage}:\n  model: {names}\n  classes.json: {want}\n"
            "Fix models/classes.json (or the dataset names) so they agree."
        )

    onnx_path = Path(model.export(format="onnx", opset=12, imgsz=args.imgsz, simplify=True))
    MODELS.mkdir(exist_ok=True)
    dest = MODELS / f"{args.stage}.onnx"
    shutil.copy2(onnx_path, dest)

    # Record the input size so the Go side and classes.json stay in sync. The
    # panel and digit stages can be exported at different sizes, so write the
    # stage-specific key (panel_input_size / digit_input_size) and leave the
    # other stage untouched. The legacy shared "input_size" is kept only as a
    # fallback for readers that predate the per-stage keys.
    size_key = f"{args.stage}_input_size"
    cfg = json.loads((MODELS / "classes.json").read_text())
    cfg[size_key] = args.imgsz
    (MODELS / "classes.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"exported {args.stage}: {dest}  (classes verified, {size_key}={args.imgsz})")


if __name__ == "__main__":
    main()
