#!/usr/bin/env python3
"""Download and merge public seven-segment YOLO datasets into septima's unified
label space.

Pipeline per source (configured in sources.yaml):
  1. download (Kaggle or Roboflow) into training/datasets/_downloads/<name>/
  2. locate its data.yaml + image/label pairs (any common YOLO layout)
  3. read the source's own class names, normalize each into septima's unified
     labels, and decide whether the source is digit-level or panel-level
  4. copy images + rewrite label class indices into the merged dataset

Outputs (under --out, default training/data):
  digits/{train,val}/{images,labels}/  + data_digits.yaml   (classes 0-9 . : -)
  panel/{train,val}/{images,labels}/   + data_panel.yaml    (class: display)

Usage:
  python training/datasets/prepare.py --inspect          # download + print class names only
  python training/datasets/prepare.py                    # full merge
  python training/datasets/prepare.py --no-download       # re-merge already-downloaded sources

Requires Kaggle credentials (~/.kaggle/kaggle.json) for kaggle sources and
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

# Unified label space — index order must match models/classes.json digit_classes.
DIGIT_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"]
DIGIT_INDEX = {lbl: i for i, lbl in enumerate(DIGIT_LABELS)}
PANEL_LABELS = ["display"]

# Common alias normalization for symbol / digit class names found in the wild.
NAME_ALIASES = {
    "dot": ".", "decimal": ".", "point": ".", "period": ".", "decimalpoint": ".",
    "colon": ":", "double_dot": ":", "doubledot": ":",
    "minus": "-", "dash": "-", "neg": "-", "negative": "-", "hyphen": "-",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
# Names that indicate a panel/display-level (single-object) dataset.
PANEL_NAME_HINTS = {"display", "panel", "lcd", "led", "screen", "7-segment-display",
                    "seven-segment-display", "sevensegment", "7segment", "ssd"}

HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE / "_downloads"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_name(raw: str, class_map: dict) -> str | None:
    """Map a source class name to a unified label, or None to drop it."""
    key = str(raw).strip()
    if key in class_map:
        v = class_map[key]
        return v or None
    low = key.lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in DIGIT_INDEX:           # already "0".."9" or ".", ":", "-"
        return key
    if low in NAME_ALIASES:
        return NAME_ALIASES[low]
    if len(key) == 1 and key in DIGIT_INDEX:
        return key
    return None


# ----------------------------------------------------------------------------- download

def download_kaggle(src: dict) -> Path:
    dest = DOWNLOADS / src["name"]
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{src['name']}] downloading kaggle dataset {src['id']} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", src["id"], "-p", str(dest), "--unzip"],
        check=True,
    )
    return dest


def download_roboflow(src: dict) -> Path:
    dest = DOWNLOADS / src["name"]
    if dest.exists() and any(dest.iterdir()):
        return dest
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(f"[{src['name']}] skipped: ROBOFLOW_API_KEY not set", file=sys.stderr)
        return dest
    from roboflow import Roboflow  # imported lazily; optional dependency
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(src["workspace"]).project(src["project"])
    proj.version(src["version"]).download("yolov8", location=str(dest))
    return dest


def download(src: dict, do_download: bool) -> Path:
    dest = DOWNLOADS / src["name"]
    if not do_download:
        return dest
    kind = src["kind"]
    if kind == "kaggle":
        return download_kaggle(src)
    if kind == "roboflow":
        return download_roboflow(src)
    raise ValueError(f"unknown source kind: {kind}")


# ----------------------------------------------------------------------------- discovery

def find_data_yaml(root: Path) -> Path | None:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def read_class_names(root: Path) -> list[str]:
    dy = find_data_yaml(root)
    if dy is None:
        return []
    with open(dy) as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", [])
    if isinstance(names, dict):  # {0: '0', 1: '1', ...}
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def iter_image_label_pairs(root: Path):
    """Yield (image_path, label_path) for every YOLO pair under root.

    Handles the standard layouts: <split>/images + <split>/labels, or a flat
    images/ + labels/. Pairs an image to a .txt with the same stem.
    """
    label_files = [p for p in root.rglob("*.txt")
                   if p.name.lower() not in {"readme.txt", "classes.txt"}]
    for lbl in label_files:
        img = find_sibling_image(lbl)
        if img is not None:
            yield img, lbl


def find_sibling_image(lbl: Path) -> Path | None:
    # Try the parallel "labels" -> "images" directory first, then same dir.
    candidates = []
    parts = list(lbl.parts)
    if "labels" in parts:
        img_dir = Path(*[("images" if seg == "labels" else seg) for seg in parts]).parent
        candidates.append(img_dir)
    candidates.append(lbl.parent)
    for d in candidates:
        for ext in IMG_EXTS:
            cand = d / (lbl.stem + ext)
            if cand.exists():
                return cand
    return None


# ----------------------------------------------------------------------------- merge

def build_remap(names: list[str], class_map: dict):
    """Return (target_space, remap) where target_space is 'digit' or 'panel' and
    remap maps source class index -> unified index (or None to drop)."""
    normalized = [normalize_name(n, class_map) for n in names]
    digit_hits = [n for n in normalized if n in DIGIT_INDEX]

    # Panel-level if it has no digit/symbol classes, or a single generic class.
    looks_panel = (not digit_hits) or all(
        (n is None and names[i].lower().replace(" ", "") in PANEL_NAME_HINTS)
        for i, n in enumerate(normalized)
    )
    if looks_panel or len(names) == 1 and not digit_hits:
        remap = {i: 0 for i in range(len(names))}  # everything -> the single panel class
        return "panel", remap

    remap = {}
    for i, n in enumerate(normalized):
        remap[i] = DIGIT_INDEX[n] if n in DIGIT_INDEX else None
    return "digit", remap


def rewrite_label(lbl: Path, remap: dict) -> list[str]:
    out = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        new = remap.get(cls)
        if new is None:
            continue
        out.append(" ".join([str(new), *parts[1:5]]))
    return out


def copy_pair(img: Path, lines: list[str], split_dir: Path, prefix: str, stem: str):
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{stem}"
    shutil.copy2(img, split_dir / "images" / (base + img.suffix.lower()))
    (split_dir / "labels" / (base + ".txt")).write_text("\n".join(lines) + "\n")


def write_data_yaml(path: Path, root: Path, names: list[str]):
    path.write_text(yaml.safe_dump({
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data")
    ap.add_argument("--sources", type=Path, default=HERE / "sources.yaml")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-download", action="store_true", help="reuse existing _downloads/")
    ap.add_argument("--inspect", action="store_true",
                    help="download and print each source's class names, then exit")
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = yaml.safe_load(args.sources.read_text())
    sources = [s for s in cfg["sources"] if s.get("enabled", True)]

    if args.inspect:
        for src in sources:
            root = download(src, not args.no_download)
            names = read_class_names(root)
            target, remap = (build_remap(names, src.get("class_map", {}))
                             if names else ("?", {}))
            print(f"\n{src['name']} ({src['kind']}): {len(names)} classes -> {target}")
            print(f"  names: {names}")
            print(f"  remap: {remap}")
        return

    digits_root = args.out / "digits"
    panel_root = args.out / "panel"
    counts = {"digit": 0, "panel": 0}

    for src in sources:
        root = download(src, not args.no_download)
        if not root.exists() or not any(root.iterdir()):
            print(f"[{src['name']}] no data; skipping", file=sys.stderr)
            continue
        names = read_class_names(root)
        if not names:
            print(f"[{src['name']}] no data.yaml class names; skipping", file=sys.stderr)
            continue
        target, remap = build_remap(names, src.get("class_map", {}))
        dest_root = digits_root if target == "digit" else panel_root
        print(f"[{src['name']}] {len(names)} classes -> {target} detector")

        pairs = list(iter_image_label_pairs(root))
        random.shuffle(pairs)
        n_val = int(len(pairs) * args.val_frac)
        for i, (img, lbl) in enumerate(pairs):
            lines = rewrite_label(lbl, remap)
            if not lines:
                continue  # image has no boxes that survive remapping
            split = "val" if i < n_val else "train"
            copy_pair(img, lines, dest_root / split, src["name"], lbl.stem)
            counts[target] += 1

    if counts["digit"]:
        write_data_yaml(digits_root / "data_digits.yaml", digits_root, DIGIT_LABELS)
    if counts["panel"]:
        write_data_yaml(panel_root / "data_panel.yaml", panel_root, PANEL_LABELS)

    print(f"\nmerged: {counts['digit']} digit images -> {digits_root}")
    print(f"        {counts['panel']} panel images -> {panel_root}")
    print("Note: synthetic + annotated-real data are added to these dirs by the "
          "render/annotate steps before training.")


if __name__ == "__main__":
    main()
