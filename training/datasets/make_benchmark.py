#!/usr/bin/env python3
"""Build an end-to-end benchmark from the held-out digit test split.

Each test image's YOLO digit boxes define the true reading: cluster boxes into
rows (single-linkage on y, mirroring the Go assembler), sort left-to-right, map
classes to characters, join rows with newlines. The result is written as a
ground_truth.json next to the images so cmd/septima-bench can score the full
pipeline (panel crop + digit detect + assemble) on real images it never trained on.

Usage:
  python training/datasets/make_benchmark.py
  go run ./cmd/septima-bench training/data/digits/test
"""
from __future__ import annotations

import json
from pathlib import Path

DIGIT_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"]
HERE = Path(__file__).resolve().parent
TEST = HERE.parent / "data" / "digits" / "test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_boxes(lbl: Path):
    boxes = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, w, h = map(float, p[1:5])
        except ValueError:
            continue
        if 0 <= c < len(DIGIT_LABELS):
            boxes.append((c, cx, cy, w, h))
    return boxes


def derive_string(boxes) -> str:
    if not boxes:
        return ""
    heights = sorted(b[4] for b in boxes)
    med = heights[len(heights) // 2] or 0.05
    tol = med * 0.7

    # Single-linkage row clustering on center-y.
    rows, cur, prev = [], [], None
    for b in sorted(boxes, key=lambda b: b[2]):
        if prev is None or b[2] - prev <= tol:
            cur.append(b)
        else:
            rows.append(cur)
            cur = [b]
        prev = b[2]
    if cur:
        rows.append(cur)

    lines = []
    for row in rows:
        row.sort(key=lambda b: b[1])  # left-to-right
        lines.append("".join(DIGIT_LABELS[b[0]] for b in row))
    return "\n".join(lines)


def main():
    img_dir, lbl_dir = TEST / "images", TEST / "labels"
    if not lbl_dir.exists():
        raise SystemExit(f"no held-out test split at {TEST}; run prepare.py first")

    images = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = next((img_dir / (lbl.stem + e) for e in IMG_EXTS if (img_dir / (lbl.stem + e)).exists()), None)
        if img is None:
            continue
        value = derive_string(read_boxes(lbl))
        if not value:
            continue
        images.append({"file": f"images/{img.name}", "value": value})

    out = TEST / "ground_truth.json"
    out.write_text(json.dumps({
        "description": "Held-out real-image benchmark (GT strings derived from test-split boxes)",
        "images": images,
    }, indent=2))
    print(f"wrote {out} with {len(images)} benchmark images")


if __name__ == "__main__":
    main()
