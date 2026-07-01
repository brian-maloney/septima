#!/usr/bin/env python3
"""Stage the real_hard digit hard-negative set for the combined colon fine-tune.

real_hard exists to anchor DIGIT precision (thin '1', tilted multi-row glyphs)
while the diverse-colon synth adapts the neck/head to ':'.  The two prior colon
runs drifted digit confidence on tilted/thin-'1' displays; the combined run
showed that oversampling the exact hard cases as hard-negatives protects them.

This script is the reproducible producer of training/data/real_hard/train.  It
reads ONE canonical labelled crop per case from review/ and oversamples each
REPEAT times into train/{images,labels}.  It is deterministic and idempotent:
train/ is wiped and regenerated every run, so it can be re-run after editing a
review label (e.g. the shell-pump decimal fix) and rsynced to the GPU box.

Why an explicit include list (not "everything in review/"):
  * The 4 passing tests/ crops (calculator, RSA-token, multimeter, gas-pump) +
    the shell pump are DIGIT cases — safe, label-complete hard-negatives.
  * The alarm-clock and microwave crops are EXCLUDED: their auto-labels carry NO
    colon box (and microwave's is missing a digit), so training on them would
    teach the model to SUPPRESS the colon on those exact images — fighting the
    synth, or, if we hand-labelled the colon, memorising the benchmark.  Colon
    recall must generalise from synth, not from the test image itself.
  * 0502 (propane gauge) is EXCLUDED: it already reads robustly (bright-on-dark,
    tank-like) and needs no protection.

The shell pump (jai5qyznvjky) is special: its review crop is the FULL 3024x4032
photo, not a panel crop.  Fed whole, its glyphs vanish when letterboxed to 640.
So it is cropped here to the digit-box union + margin (mode "crop_union") and its
labels are re-normalised to the crop, matching the framing of the other crops.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "training" / "data" / "real_hard"
REVIEW = ROOT / "review"
TRAIN = ROOT / "train"

REPEAT = 60  # oversample factor per case (was 30 in the combined run; stronger
#              hard-negative gradient vs the 8000-image colon synth pull)

# (stem, mode).  stem is the review/ basename without extension.
# mode "asis"       -> the review file is already a tight panel crop.
# mode "crop_union" -> the review file is a full photo; crop to digit-box union.
HARD_NEGS = [
    ("tank_2013meax1g981", "asis"),                       # calculator, 12 digits
    ("tank_68f79706-7dd3-4d7d-8247-5fda7366da14."
     "c3a5d25cda7dbc2795f37d0bff316e8d", "asis"),         # RSA token 156311
    ("tank_dVv50", "asis"),                               # multimeter 0.68
    ("tank_getting-weird-messages-from-my-gas-pump-"
     "v0-wwjamtn0tzyg1", "asis"),                         # gas pump (webp)
    ("tank_jai5qyznvjky", "crop_union"),                  # Shell pump 29.29 / 13.318
]

# fractional margin added around the digit-box union when crop_union-ing.
UNION_MARGIN = 0.20


def find_image(stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = REVIEW / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no review image for {stem!r}")


def read_label(stem: str) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in (REVIEW / f"{stem}.txt").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        c = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        rows.append((c, x, y, w, h))
    return rows


def crop_to_union(img: Image.Image, boxes):
    """Crop the image to the digit-box union + UNION_MARGIN and re-normalise boxes."""
    xs0 = min(x - w / 2 for _, x, _, w, _ in boxes)
    xs1 = max(x + w / 2 for _, x, _, w, _ in boxes)
    ys0 = min(y - h / 2 for _, _, y, _, h in boxes)
    ys1 = max(y + h / 2 for _, _, y, _, h in boxes)
    mx = (xs1 - xs0) * UNION_MARGIN
    my = (ys1 - ys0) * UNION_MARGIN
    cx0 = max(0.0, xs0 - mx)
    cy0 = max(0.0, ys0 - my)
    cx1 = min(1.0, xs1 + mx)
    cy1 = min(1.0, ys1 + my)
    cw, ch = cx1 - cx0, cy1 - cy0

    W, H = img.size
    crop = img.crop((int(cx0 * W), int(cy0 * H), int(cx1 * W), int(cy1 * H)))
    out = []
    for c, x, y, w, h in boxes:
        out.append((c, (x - cx0) / cw, (y - cy0) / ch, w / cw, h / ch))
    return crop, out


def label_text(boxes) -> str:
    return "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes)


def main() -> None:
    img_dir = TRAIN / "images"
    lbl_dir = TRAIN / "labels"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    total = 0
    for stem, mode in HARD_NEGS:
        src = find_image(stem)
        boxes = read_label(stem)
        img = Image.open(src).convert("RGB")
        if mode == "crop_union":
            img, boxes = crop_to_union(img, boxes)
        ext = ".jpg"
        lbl = label_text(boxes)
        for r in range(REPEAT):
            img.save(img_dir / f"{stem}_r{r}{ext}", quality=95)
            (lbl_dir / f"{stem}_r{r}.txt").write_text(lbl)
        total += REPEAT
        print(f"  {stem[:48]:48s} {mode:11s} {img.size}  x{REPEAT}")

    print(f"\nstaged {total} images / labels into {TRAIN.relative_to(REPO)}")
    print(f"  cases: {len(HARD_NEGS)}  repeat: {REPEAT}")


if __name__ == "__main__":
    main()
