#!/usr/bin/env python3
"""Preflight-validate the digit fine-tune, then launch it ONLY if every check passes.

Background: the previous decimal-synth run regressed every metric. The two
suspected causes were (1) training fresh from stock COCO weights instead of
fine-tuning the existing 13-class digit model (which discards all prior real-data
adaptation and explains the broad, non-decimal regression), and (2) a synth set
that did not actually contain the new decimal/colon data the run is meant to add.

This script refuses to train unless:
  * the base weights LOAD as our 13-class digit model (a stock yolo11n/m.pt has
    80 COCO classes -> hard fail; this is the key guard),
  * data_finetune.yaml fine-tunes on digits + real_tank, with class names in the
    exact classes.json order, and never references the held-out test split,
  * real_tank hard-negatives are present (they hold tank at 100%),
  * the digit train set actually contains decimal and colon boxes (the new synth),
  * the requested device is available.

Only when all critical checks pass does it invoke training/train.py with the
correct fine-tune command.

Usage (AWS L4/A10G):
  python scripts/train_digits_decimal.py --device 0 --batch 32
Local MPS:
  python scripts/train_digits_decimal.py
Regenerate synth first, then validate + train:
  python scripts/train_digits_decimal.py --regen-synth --device 0 --batch 32
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # exclude .npy disk-cache files
CLASSES_JSON = ROOT / "models" / "classes.json"
FINETUNE_YAML = DATA / "data_finetune.yaml"
RENDER_PY = ROOT / "training" / "synth" / "render.py"
DEFAULT_BASE = ROOT / "training" / "runs" / "digits" / "weights" / "best.pt"
STOCK_NAMES = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
               "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"}

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Preflight:
    def __init__(self):
        self.fails: list[str] = []
        self.warns: list[str] = []

    def ok(self, msg: str):
        print(f"  {GREEN}[OK]{RESET}   {msg}")

    def fail(self, msg: str):
        print(f"  {RED}[FAIL]{RESET} {msg}")
        self.fails.append(msg)

    def warn(self, msg: str):
        print(f"  {YELLOW}[WARN]{RESET} {msg}")
        self.warns.append(msg)


def load_digit_classes() -> list[str]:
    return json.loads(CLASSES_JSON.read_text())["digit_classes"]


def refresh_finetune_yaml():
    """Rewrite data_finetune.yaml from the CURRENT data tree (reusing prepare.py's
    own generator) so it always reflects whether real_tank is present. This closes
    the stale-yaml footgun that silently dropped the tank hard-negatives last run:
    once real_tank is scp'd in, the yaml picks it up without re-running prepare.py."""
    sys.path.insert(0, str(ROOT / "training" / "datasets"))
    try:
        import prepare
    except Exception as e:
        print(f"{YELLOW}could not import prepare to refresh data_finetune.yaml: {e}{RESET}")
        return
    prepare.write_finetune_yaml(DATA, prepare.DIGIT_LABELS)
    has_rt = (DATA / "real_tank" / "train" / "images").exists()
    print(f"refreshed {FINETUNE_YAML.relative_to(ROOT)} from data tree "
          f"(real_tank {'included' if has_rt else 'ABSENT'})")


def check_classes(pf: Preflight) -> list[str]:
    if not CLASSES_JSON.exists():
        pf.fail(f"missing {CLASSES_JSON}")
        return []
    cls = load_digit_classes()
    if cls == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"]:
        pf.ok(f"classes.json: 13 digit classes in expected order")
    else:
        pf.fail(f"classes.json digit order unexpected: {cls}")
    return cls


def check_base_weights(pf: Preflight, base: Path, digit_classes: list[str]):
    """THE key guard: the base must load as our 13-class digit model, not stock COCO."""
    if base.name in STOCK_NAMES:
        pf.fail(f"base weights '{base.name}' are stock COCO weights — fine-tune from "
                f"training/runs/digits/weights/best.pt instead (this caused the last regression)")
        return
    if not base.exists():
        pf.fail(f"base weights not found: {base}")
        return
    try:
        from ultralytics import YOLO
        names = YOLO(str(base)).names  # {idx: name}
    except Exception as e:
        pf.fail(f"could not load base weights {base}: {e}")
        return
    loaded = [names[i] for i in sorted(names)]
    if len(loaded) != len(digit_classes):
        pf.fail(f"base weights have {len(loaded)} classes, expected {len(digit_classes)} "
                f"-> NOT our digit model (stock COCO has 80). Refusing to fine-tune.")
        return
    if loaded != digit_classes:
        pf.fail(f"base weights class names {loaded} != classes.json order {digit_classes}")
        return
    pf.ok(f"base weights are the 13-class digit model: {base.relative_to(ROOT)}")


def check_finetune_yaml(pf: Preflight, digit_classes: list[str]):
    if not FINETUNE_YAML.exists():
        pf.fail(f"missing {FINETUNE_YAML} — run training/datasets/prepare.py")
        return
    cfg = yaml.safe_load(FINETUNE_YAML.read_text())
    train = cfg.get("train", [])
    train = [train] if isinstance(train, str) else train
    val = cfg.get("val", "")
    paths = [str(p) for p in train] + [str(val)]

    if any("digits/train/images" in p for p in train):
        pf.ok("data_finetune.yaml trains on digits/train")
    else:
        pf.fail(f"data_finetune.yaml train does not include digits/train/images: {train}")
    if any("real_tank/train/images" in p for p in train):
        pf.ok("data_finetune.yaml trains on real_tank (tank hard-negatives present)")
    else:
        pf.fail("data_finetune.yaml does NOT include real_tank/train/images — tank will "
                "regress (phantom decimals). scp real_tank and re-run prepare.py.")
    if any("real_hard/train/images" in p for p in train):
        pf.ok("data_finetune.yaml trains on real_hard (tilted/thin-1 hard-negatives present)")
    else:
        pf.warn("data_finetune.yaml does not include real_hard/train/images — "
                "run: go run ./cmd/septima-annotate -in tests -out training/data/real_hard "
                "-panel-model models/panel.onnx -repeat 30")

    # Leakage guard: the held-out test split must never be trained/validated on.
    leak = [p for p in paths if "/test/" in p or p.endswith("/test") or "test/images" in p]
    if leak:
        pf.fail(f"data_finetune.yaml references the held-out TEST split (leakage): {leak}")
    else:
        pf.ok("no held-out test split referenced (no benchmark leakage)")

    names = cfg.get("names", {})
    loaded = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if loaded == digit_classes:
        pf.ok("data_finetune.yaml names match classes.json order")
    else:
        pf.fail(f"data_finetune.yaml names {loaded} != classes.json {digit_classes}")


def count_class_files(label_dir: Path, glob: str, class_id: int) -> int:
    """Number of label files containing at least one box of class_id."""
    n = 0
    tok = f"{class_id} "
    for lbl in label_dir.glob(glob):
        for line in lbl.read_text().splitlines():
            if line.startswith(tok):
                n += 1
                break
    return n


def check_data_present(pf: Preflight, min_decimal: int, min_colon: int):
    img_dir = DATA / "digits" / "train" / "images"
    lbl_dir = DATA / "digits" / "train" / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        pf.fail(f"missing digit train data under {DATA/'digits'/'train'}")
        return
    n_img = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if n_img >= 1000:
        pf.ok(f"digit train images: {n_img}")
    else:
        pf.fail(f"only {n_img} digit train images — did prepare.py + render.py run?")

    rt = DATA / "real_tank" / "train" / "images"
    n_rt = sum(1 for p in rt.iterdir() if p.suffix.lower() in IMG_EXTS) if rt.exists() else 0
    if n_rt > 0:
        pf.ok(f"real_tank train images: {n_rt}")
    else:
        pf.fail("real_tank/train/images is empty/missing")

    rh = DATA / "real_hard" / "train" / "images"
    n_rh = sum(1 for p in rh.iterdir() if p.suffix.lower() in IMG_EXTS) if rh.exists() else 0
    if n_rh > 0:
        pf.ok(f"real_hard train images: {n_rh} (tilted/thin-1 hard-negatives)")
    else:
        pf.warn("real_hard/train/images is empty/missing — run septima-annotate -in tests "
                "-out training/data/real_hard -panel-model models/panel.onnx -repeat 30")

    # The point of this run: decimal + colon synth must be present.
    n_dec = count_class_files(lbl_dir, "synth_*.txt", 10)
    n_col = count_class_files(lbl_dir, "synth_*.txt", 11)
    if n_dec >= min_decimal:
        pf.ok(f"synth decimal-'.' label files: {n_dec} (>= {min_decimal})")
    else:
        pf.fail(f"only {n_dec} synth files with a decimal box (< {min_decimal}) — "
                f"regenerate synth: python training/synth/render.py --digits 4000 --panels 1500")
    if n_col >= min_colon:
        pf.ok(f"synth colon-':' label files: {n_col} (>= {min_colon})")
    else:
        pf.warn(f"only {n_col} synth files with a colon box (< {min_colon})")


def check_render_code(pf: Preflight):
    """Soft check that render.py carries the diverse-colon generator (the targeted
    colon-synthesis change), so the run actually adds the new colon variety."""
    if not RENDER_PY.exists():
        pf.warn("render.py not found (cannot confirm generator version)")
        return
    src = RENDER_PY.read_text()
    if "Diversify colon appearance" in src and "square" in src:
        pf.ok("render.py is the diverse-colon generator (varied dot size/position/shape)")
    else:
        pf.warn("render.py lacks the diverse-colon code — synth may be the OLD generator")


def check_device(pf: Preflight, device: str, allow_cpu: bool):
    try:
        import torch
    except Exception as e:
        pf.fail(f"torch not importable: {e}")
        return
    if device == "cpu":
        (pf.ok if allow_cpu else pf.fail)("device=cpu (training will be very slow; pass --allow-cpu to permit)")
    elif device == "mps":
        if torch.backends.mps.is_available():
            pf.ok("device=mps available")
        else:
            pf.fail("device=mps requested but MPS unavailable")
    else:  # cuda index
        if torch.cuda.is_available():
            pf.ok(f"device={device}: CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            pf.fail(f"device={device} requested but CUDA unavailable")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help="base weights to fine-tune (must be the 13-class digit model)")
    ap.add_argument("--device", default=None,
                    help="mps | cpu | 0 (cuda); auto-detected when omitted")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache", default="disk", help="ram | disk | False")
    ap.add_argument("--name", default="digits_decimal")
    ap.add_argument("--min-decimal", type=int, default=300, help="min synth files with a '.' box")
    ap.add_argument("--min-colon", type=int, default=120, help="min synth files with a ':' box")
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze the first N layers (e.g. 11 = YOLO11 backbone) so the "
                         "fine-tune adapts only the neck/head (preserves digit features)")
    ap.add_argument("--regen-synth", action="store_true",
                    help="regenerate synth (4000/1500) before validating")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="validate only; do not train")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "0"
            elif torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"
        print(f"auto-detected device: {args.device}")

    if args.regen_synth:
        print("Regenerating synthetic data (4000 digit / 1500 panel) ...")
        rc = subprocess.run([sys.executable, str(RENDER_PY),
                             "--digits", "4000", "--panels", "1500"]).returncode
        if rc != 0:
            print(f"{RED}render.py failed (rc={rc}); aborting.{RESET}")
            return 2

    refresh_finetune_yaml()

    print("\nPreflight checks for digit fine-tune:")
    pf = Preflight()
    digit_classes = check_classes(pf)
    check_base_weights(pf, args.base, digit_classes)
    check_finetune_yaml(pf, digit_classes)
    check_data_present(pf, args.min_decimal, args.min_colon)
    check_render_code(pf)
    check_device(pf, args.device, args.allow_cpu)

    print()
    if pf.fails:
        print(f"{RED}{len(pf.fails)} check(s) FAILED — NOT training.{RESET}")
        for f in pf.fails:
            print(f"  - {f}")
        return 1
    if pf.warns:
        print(f"{YELLOW}{len(pf.warns)} warning(s) (non-blocking).{RESET}")

    cmd = [sys.executable, str(ROOT / "training" / "train.py"),
           "--stage", "digits",
           "--data", str(FINETUNE_YAML),
           "--model", str(args.base),
           "--epochs", str(args.epochs),
           "--batch", str(args.batch),
           "--device", args.device,
           "--cache", args.cache,
           "--name", args.name]
    if args.freeze is not None:
        cmd += ["--freeze", str(args.freeze)]
    print(f"{GREEN}All critical checks passed.{RESET}")
    print("Training command:\n  " + " ".join(cmd))
    if args.dry_run:
        print(f"{YELLOW}--dry-run: not launching.{RESET}")
        return 0

    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc == 0:
        print(f"\n{GREEN}Training done.{RESET} Next:")
        print(f"  python training/export_onnx.py --stage digits "
              f"--weights training/runs/{args.name}/weights/best.pt")
        print("  # then bench: tanktests (must hold 32/32), tests, and "
              "go run ./cmd/septima-analyze training/data/digits/test")
    else:
        print(f"\n{RED}Training exited rc={rc}.{RESET}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
