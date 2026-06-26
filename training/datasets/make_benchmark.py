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

import argparse
import json
from pathlib import Path

from PIL import Image

DIGIT_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"]
HERE = Path(__file__).resolve().parent
TEST = HERE.parent / "data" / "digits" / "test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Benchmark quality gates (see analyze findings, 2026-06-26). The held-out test
# split is auto-derived from public-dataset boxes, so its GT inherits each
# source's annotation quality. Two model-INDEPENDENT gates keep the benchmark an
# honest measure of recognition rather than of annotation noise:
#
#  1. Resolution floor: drop images whose median digit box is below
#     MIN_DIGIT_PX tall — a glyph that small is under the detector's effective
#     resolution, so a miss reflects the image, not the model. (At 14px this
#     removes only a handful of degenerate sub-100px crops.)
#
#  2. Source exclusion: drop sources whose box-derived GT is not a coherent
#     single reading. loclaurote_price is grocery-scale photos with TWO separate
#     numeric fields (price + weight) plus printed product codes/barcodes; the
#     union of its boxes mashes those fields into one nonsensical string
#     (e.g. price "4.5" + weight "18.74" -> GT "1804451"), so it cannot score a
#     reader either way. Excluding the whole source (not just its failures) keeps
#     this a data-quality decision, not cherry-picking.
DEFAULT_MIN_DIGIT_PX = 14.0
DEFAULT_EXCLUDE_SOURCES = ("loclaurote_price",)


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
        lines.append(trim_edge_punct("".join(DIGIT_LABELS[b[0]] for b in row)))
    return "\n".join(line for line in lines if line)


def trim_edge_punct(s: str) -> str:
    """Mirror the Go assembler: a reading never starts with '.'/':' nor ends with
    '.'/':'/'-'. Keeps the GT consistent with the pipeline's normalization."""
    s = s.lstrip(".:")
    s = s.rstrip(".:-")
    return s


def source_of(stem: str, sources: tuple[str, ...]) -> str | None:
    """The configured source prefix this filename belongs to (prepare.py names
    every merged file '<source>_<stem>'), or None."""
    return next((s for s in sources if stem.startswith(s + "_")), None)


def median_digit_px(boxes, img_h: int) -> float:
    """Median digit-box height in pixels (the readability measure)."""
    hs = sorted(b[4] * img_h for b in boxes)
    return hs[len(hs) // 2] if hs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-digit-px", type=float, default=DEFAULT_MIN_DIGIT_PX,
                    help="drop images whose median digit box is shorter than this (readability floor)")
    ap.add_argument("--exclude-sources", default=",".join(DEFAULT_EXCLUDE_SOURCES),
                    help="comma-separated source prefixes to exclude (incoherent GT); '' to keep all")
    args = ap.parse_args()
    exclude = tuple(s for s in args.exclude_sources.split(",") if s)

    img_dir, lbl_dir = TEST / "images", TEST / "labels"
    if not lbl_dir.exists():
        raise SystemExit(f"no held-out test split at {TEST}; run prepare.py first")

    images = []
    dropped = {"no_image": 0, "empty_gt": 0, "excluded_source": 0, "low_res": 0}
    drop_src = {}  # excluded-source -> count, for transparency
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = next((img_dir / (lbl.stem + e) for e in IMG_EXTS if (img_dir / (lbl.stem + e)).exists()), None)
        if img is None:
            dropped["no_image"] += 1
            continue
        src = source_of(lbl.stem, exclude)
        if src is not None:
            dropped["excluded_source"] += 1
            drop_src[src] = drop_src.get(src, 0) + 1
            continue
        boxes = read_boxes(lbl)
        value = derive_string(boxes)
        if not value:
            dropped["empty_gt"] += 1
            continue
        with Image.open(img) as im:
            _, img_h = im.size
        if median_digit_px(boxes, img_h) < args.min_digit_px:
            dropped["low_res"] += 1
            continue
        images.append({"file": f"images/{img.name}", "value": value})

    out = TEST / "ground_truth.json"
    out.write_text(json.dumps({
        "description": "Held-out real-image benchmark (GT strings derived from test-split boxes; "
                       f"quality-gated: median digit >= {args.min_digit_px:g}px, excluding {list(exclude)})",
        "images": images,
    }, indent=2))
    print(f"wrote {out} with {len(images)} benchmark images")
    print(f"dropped: {dropped['excluded_source']} excluded-source {drop_src or ''}, "
          f"{dropped['low_res']} low-res (<{args.min_digit_px:g}px digits), "
          f"{dropped['empty_gt']} empty-GT, {dropped['no_image']} missing-image")


if __name__ == "__main__":
    main()
