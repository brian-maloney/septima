#!/usr/bin/env python3
"""Train a YOLO detection stage (digits or panel) with Ultralytics.

Transfer-learns from COCO weights on the merged dataset built by prepare.py +
render.py.  Defaults to Apple Silicon (device=mps); use --device 0 on CUDA.

Usage:
  python training/train.py --stage digits --epochs 100
  python training/train.py --stage panel  --epochs 60
  # AWS / L4 (24 GB):
  python training/train.py --stage digits --model yolo11m.pt --device 0 --batch 32

Results land in training/runs/<stage>/; export the best weights with
training/export_onnx.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = HERE / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["digits", "panel"], required=True)
    ap.add_argument("--data", default=None,
                    help="explicit data yaml (overrides the stage default; use for fine-tuning)")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="base weights to transfer from (yolo11n/s/m.pt or runs/…/best.pt to fine-tune)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps", help="mps | cpu | 0 (cuda)")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--cache", default="ram",
                    help="ultralytics cache mode: ram (fast, needs ~16 GB) | disk | False")
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader worker processes; set to nCPU on the training host (4 on g6.xlarge)")
    ap.add_argument("--name", default=None, help="run name (default: stage)")
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze the first N layers (e.g. 11 = the YOLO11 backbone) so a "
                         "fine-tune adapts only the neck/head — preserves digit features when "
                         "adding a new glyph style like colons")
    args = ap.parse_args()

    data_yaml = Path(args.data) if args.data else DATA / args.stage / f"data_{args.stage}.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"missing {data_yaml}; run prepare.py and render.py first")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(RUNS),
        name=args.name or args.stage,
        exist_ok=True,
        cache=args.cache,
        workers=args.workers,
        freeze=args.freeze,
        # Light geometric aug — synthetic data already varies perspective;
        # displays are axis-aligned so no rotation/mosaic (mosaic=0 also avoids
        # the heavy CPU cost of 4-image stitching per sample).
        degrees=3.0,
        perspective=0.0005,
        mosaic=0.0,
        fliplr=0.0,   # never mirror: digits are not symmetric
        flipud=0.0,
    )
    print(f"done. best weights: {RUNS / (args.name or args.stage) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
