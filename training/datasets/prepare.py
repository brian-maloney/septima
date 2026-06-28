#!/usr/bin/env python3
"""Download and merge public seven-segment / digital-display datasets into
septima's unified label space, feeding BOTH detection stages.

Per source (configured in sources.yaml):
  1. download (Kaggle or Roboflow) into training/datasets/_downloads/<name>/
  2. read its class names; classify each class as a digit/symbol or a panel class
  3. for every image, route boxes per class:
       - digit/symbol boxes -> digits dataset (classes 0-9 . : -)
       - explicit panel/display boxes -> panel dataset (class: display)
       - if an image has digit boxes but NO explicit panel box, synthesize one
         from the union of its digit boxes (the display panel) -> panel dataset

So a digit-only dataset trains both the digit detector and (via synthesized
panel boxes) the panel detector; a mixed dataset contributes to both directly.

Outputs (under --out, default training/data):
  digits/{train,val}/{images,labels}/  + digits/data_digits.yaml
  panel/{train,val}/{images,labels}/   + panel/data_panel.yaml

Usage:
  python training/datasets/prepare.py --inspect     # download + print class names
  python training/datasets/prepare.py               # full merge
  python training/datasets/prepare.py --no-download  # re-merge existing downloads

Requires Kaggle creds (~/.kaggle/kaggle.json) for kaggle sources and
ROBOFLOW_API_KEY for roboflow sources. See training/README.md.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

DIGIT_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"]
DIGIT_INDEX = {lbl: i for i, lbl in enumerate(DIGIT_LABELS)}
PANEL_LABELS = ["display"]

NAME_ALIASES = {
    "dot": ".", "decimal": ".", "point": ".", "period": ".", "decimalpoint": ".",
    "colon": ":", "double_dot": ":", "doubledot": ":",
    "minus": "-", "dash": "-", "neg": "-", "negative": "-", "hyphen": "-",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
PANEL_NAME_HINTS = {"display", "panel", "lcd", "led", "screen", "7-segment-display",
                    "seven-segment-display", "sevensegment", "7segment", "ssd",
                    "meter", "counter", "digitalmeter", "screendisplay"}

HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE / "_downloads"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify_name(raw: str, class_map: dict) -> tuple[str, object]:
    """Return ('digit', unified_idx) | ('panel', 0) | ('drop', None) for a class."""
    key = str(raw).strip()
    if key in class_map:
        v = class_map[key]
        if not v:
            return "drop", None
        if v in DIGIT_INDEX:
            return "digit", DIGIT_INDEX[v]
        return "panel", 0
    low = key.lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in DIGIT_INDEX:
        return "digit", DIGIT_INDEX[key]
    if low in NAME_ALIASES:
        return "digit", DIGIT_INDEX[NAME_ALIASES[low]]
    # "digit_0".."digit_9" (and "digit0"..) -> the bare digit.
    if low.startswith("digit") and low[5:] in DIGIT_INDEX:
        return "digit", DIGIT_INDEX[low[5:]]
    if low in PANEL_NAME_HINTS:
        return "panel", 0
    return "drop", None


# ----------------------------------------------------------------------------- download

def download_kaggle(src: dict) -> Path:
    dest = DOWNLOADS / src["name"]
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{src['name']}] downloading kaggle dataset {src['id']} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", src["id"],
                    "-p", str(dest), "--unzip"], check=True)
    return dest


def download_roboflow(src: dict) -> Path:
    dest = DOWNLOADS / src["name"]
    if dest.exists() and any(dest.iterdir()):
        return dest
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(f"[{src['name']}] skipped: ROBOFLOW_API_KEY not set", file=sys.stderr)
        return dest
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(src["workspace"]).project(src["project"])
        version = src.get("version")
        if not version:  # default to the latest version
            version = max(int(str(v.version).split("/")[-1]) for v in proj.versions())
            print(f"[{src['name']}] using latest version {version}")
        proj.version(int(version)).download("yolov8", location=str(dest))
    except Exception as e:
        print(f"[{src['name']}] roboflow download failed: {e}\n"
              f"  (set an explicit 'version:' in sources.yaml, or check workspace/project)",
              file=sys.stderr)
    return dest


def download(src: dict, do_download: bool) -> Path:
    dest = DOWNLOADS / src["name"]
    if not do_download:
        return dest
    if src["kind"] == "kaggle":
        return download_kaggle(src)
    if src["kind"] == "roboflow":
        return download_roboflow(src)
    raise ValueError(f"unknown source kind: {src['kind']}")


# ----------------------------------------------------------------------------- discovery

def read_class_names(root: Path) -> list[str]:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        hits = sorted(root.rglob(name))
        if hits:
            data = yaml.safe_load(hits[0].read_text()) or {}
            names = data.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            return [str(n) for n in names]
    return []


def iter_image_label_pairs(root: Path):
    for lbl in root.rglob("*.txt"):
        if lbl.name.lower() in {"readme.txt", "classes.txt"}:
            continue
        img = find_sibling_image(lbl)
        if img is not None:
            yield img, lbl


def find_sibling_image(lbl: Path):
    candidates = []
    parts = list(lbl.parts)
    if "labels" in parts:
        candidates.append(Path(*[("images" if s == "labels" else s) for s in parts]).parent)
    candidates.append(lbl.parent)
    for d in candidates:
        for ext in IMG_EXTS:
            cand = d / (lbl.stem + ext)
            if cand.exists():
                return cand
    return None


# ----------------------------------------------------------------------------- merge

def route_boxes(lbl: Path, routing: dict) -> tuple[list[str], list[str]]:
    """Split a YOLO label file into (digit_lines, panel_lines) using the per-class
    routing {src_idx: ('digit', uidx) | ('panel', 0) | ('drop', None)}.
    When the image has digit boxes but no explicit panel box, synthesize a panel
    box from the union of the digit boxes."""
    digit_lines, panel_lines = [], []
    digit_boxes = []  # (cx, cy, w, h) for union synthesis
    has_explicit_panel = False
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cls = int(float(p[0]))
            cx, cy, w, h = map(float, p[1:5])
        except ValueError:
            continue
        kind, target = routing.get(cls, ("drop", None))
        if kind == "digit":
            digit_lines.append(f"{target} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            digit_boxes.append((cx, cy, w, h))
        elif kind == "panel":
            panel_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            has_explicit_panel = True

    if not has_explicit_panel and digit_boxes:
        x0 = min(cx - w / 2 for cx, _, w, _ in digit_boxes)
        x1 = max(cx + w / 2 for cx, _, w, _ in digit_boxes)
        y0 = min(cy - h / 2 for _, cy, _, h in digit_boxes)
        y1 = max(cy + h / 2 for _, cy, _, h in digit_boxes)
        # Pad ~8% horizontally / 18% vertically to mimic a bezel, clamp to frame.
        pw, ph = (x1 - x0) * 0.08, (y1 - y0) * 0.18
        x0, x1 = max(0.0, x0 - pw), min(1.0, x1 + pw)
        y0, y1 = max(0.0, y0 - ph), min(1.0, y1 + ph)
        panel_lines.append(f"0 {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}")
    return digit_lines, panel_lines


def detect_split(lbl: Path):
    """Detect train/val/test from a dataset's own directory layout, so a
    dataset's test split stays HELD OUT (never trained on) for benchmarking."""
    parts = {p.lower() for p in lbl.parts}
    if "test" in parts:
        return "test"
    if "valid" in parts or "val" in parts:
        return "val"
    if "train" in parts:
        return "train"
    return None


def clean_source(roots: list[Path], prefix: str):
    """Remove a source's previously-merged files (so a re-run's reshuffled split
    can't leave stale duplicates in the wrong split). Only touches <prefix>_*,
    leaving synthetic (synth_*) and other sources intact."""
    for root in roots:
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                d = root / split / kind
                if d.exists():
                    for f in d.glob(f"{prefix}_*"):
                        f.unlink()


def copy_pair(img: Path, lines: list[str], root: Path, split: str, prefix: str, stem: str):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{stem}"
    shutil.copy2(img, root / split / "images" / (base + img.suffix.lower()))
    (root / split / "labels" / (base + ".txt")).write_text("\n".join(lines) + "\n")


def write_data_yaml(path: Path, root: Path, names: list[str]):
    path.write_text(yaml.safe_dump({
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))


def write_finetune_yaml(data_root: Path, names: list[str]):
    """Digit training config combining the open-source/synthetic digits with any
    real_* hard-negative datasets present (produced by cmd/septima-annotate).
    Discovers all real_*/train/images directories automatically so new datasets
    (real_tank, real_hard, …) are picked up without editing this function."""
    train = ["digits/train/images"]
    for real_dir in sorted(data_root.glob("real_*/train/images")):
        train.append(str(real_dir.relative_to(data_root)))
    (data_root / "data_finetune.yaml").write_text(yaml.safe_dump({
        "path": str(data_root.resolve()),
        "train": train,
        "val": "digits/val/images",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data")
    ap.add_argument("--sources", type=Path, default=HERE / "sources.yaml")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = yaml.safe_load(args.sources.read_text())
    sources = [s for s in cfg["sources"] if s.get("enabled", True)]

    if args.inspect:
        for src in sources:
            root = download(src, not args.no_download)
            names = read_class_names(root)
            routing = {i: classify_name(n, src.get("class_map", {})) for i, n in enumerate(names)}
            print(f"\n{src['name']} ({src['kind']}): {len(names)} classes")
            print(f"  names:   {names}")
            print(f"  routing: {routing}")
        return

    digits_root, panel_root = args.out / "digits", args.out / "panel"
    counts = {"digit_img": 0, "panel_img": 0, "digit_test": 0, "panel_test": 0}

    for src in sources:
        root = download(src, not args.no_download)
        if not root.exists() or not any(root.iterdir()):
            print(f"[{src['name']}] no data; skipping", file=sys.stderr)
            continue
        names = read_class_names(root)
        if not names:
            print(f"[{src['name']}] no data.yaml class names; skipping", file=sys.stderr)
            continue
        routing = {i: classify_name(n, src.get("class_map", {})) for i, n in enumerate(names)}
        kinds = {k for k, _ in routing.values()}
        print(f"[{src['name']}] {len(names)} classes -> "
              f"{'digits ' if 'digit' in kinds else ''}{'panel' if kinds & {'digit', 'panel'} else ''}")

        clean_source([digits_root, panel_root], src["name"])
        pairs = list(iter_image_label_pairs(root))
        random.shuffle(pairs)
        n_val = int(len(pairs) * args.val_frac)
        panel_only = src.get("panel_only", False)
        for i, (img, lbl) in enumerate(pairs):
            digit_lines, panel_lines = route_boxes(lbl, routing)
            if panel_only:
                digit_lines = []  # use boxes for panel synthesis only
            split = detect_split(lbl)  # respect the dataset's own split
            if split is None:          # flat dataset (e.g. Kaggle): carve a val set
                split = "val" if i < n_val else "train"
            if digit_lines:
                copy_pair(img, digit_lines, digits_root, split, src["name"], lbl.stem)
                counts["digit_img"] += 1
                if split == "test":
                    counts["digit_test"] += 1
            if panel_lines:
                copy_pair(img, panel_lines, panel_root, split, src["name"], lbl.stem)
                counts["panel_img"] += 1
                if split == "test":
                    counts["panel_test"] += 1

    if counts["digit_img"]:
        write_data_yaml(digits_root / "data_digits.yaml", digits_root, DIGIT_LABELS)
        write_finetune_yaml(args.out, DIGIT_LABELS)
    if counts["panel_img"]:
        write_data_yaml(panel_root / "data_panel.yaml", panel_root, PANEL_LABELS)

    print(f"\nmerged: {counts['digit_img']} digit images ({counts['digit_test']} held-out test) -> {digits_root}")
    print(f"        {counts['panel_img']} panel images ({counts['panel_test']} held-out test) -> {panel_root}")
    print("Held-out test splits are NOT in data_*.yaml. Build the end-to-end")
    print("benchmark with: python training/datasets/make_benchmark.py")
    print("Run training/synth/render.py to add synthetic samples before training.")


if __name__ == "__main__":
    main()
