#!/usr/bin/env python3
"""Synthetic seven-segment display generator.

Renders seven-segment strings with exact YOLO boxes for both training stages,
covering what the public LED price-display datasets miss: the reflective-LCD
dark-on-light look, decimals/colons/minus, multi-digit fields, and the tank
panel's split 4-digit framing. Boxes stay exact through geometric augmentation
because every glyph is drawn into a parallel integer label map that undergoes
the identical geometric transform; boxes are read back from the label map.

Two output kinds, appended to the same dataset dirs prepare.py uses:
  digits/ : tight crops of the display, one box per glyph (classes 0-9 . : -)
  panel/  : the display composited into a larger scene, one box (class display)

Usage:
  python training/synth/render.py --digits 2000 --panels 800
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Unified digit label order — must match models/classes.json digit_classes.
DIGIT_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"]
LABEL_INDEX = {c: i for i, c in enumerate(DIGIT_LABELS)}

# Active segments per character: a,b,c,d,e,f,g.
SEG = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def seg_polys(x0, y0, x1, y1):
    """Return {segment: polygon} for a digit cell, as chamfered hexagons."""
    w, h = x1 - x0, y1 - y0
    t = 0.16 * w
    midy = y0 + h / 2

    def hbar(cy):
        xa, xb = x0 + t * 0.6, x1 - t * 0.6
        return [(xa, cy), (xa + t / 2, cy - t / 2), (xb - t / 2, cy - t / 2),
                (xb, cy), (xb - t / 2, cy + t / 2), (xa + t / 2, cy + t / 2)]

    def vbar(cx, ya, yb):
        return [(cx, ya), (cx + t / 2, ya + t / 2), (cx + t / 2, yb - t / 2),
                (cx, yb), (cx - t / 2, yb - t / 2), (cx - t / 2, ya + t / 2)]

    return {
        "a": hbar(y0 + t / 2),
        "g": hbar(midy),
        "d": hbar(y1 - t / 2),
        "f": vbar(x0 + t / 2, y0 + t * 0.6, midy),
        "b": vbar(x1 - t / 2, y0 + t * 0.6, midy),
        "e": vbar(x0 + t / 2, midy, y1 - t * 0.6),
        "c": vbar(x1 - t / 2, midy, y1 - t * 0.6),
    }


def _dot(rgb, lab, cx, cy, r, gid, color, square):
    """Draw one punctuation dot (round or square) into image + label map; return
    its bbox. Real LCDs vary dot shape/size, so the renderer does too — a single
    clean circle is what made '.' the weakest, most-missed class."""
    box = [cx - r, cy - r, cx + r, cy + r]
    if square:
        rgb.rectangle(box, fill=color); lab.rectangle(box, fill=gid)
    else:
        rgb.ellipse(box, fill=color); lab.ellipse(box, fill=gid)
    return tuple(box)


def draw_glyph(rgb: ImageDraw.ImageDraw, lab: ImageDraw.ImageDraw, char, cell, gid, pal, rng):
    """Draw one glyph into the RGB image and the label map; return its YOLO box
    region as drawn-pixel extent (xa,ya,xb,yb), or None to skip."""
    x0, y0, x1, y1 = cell
    w, h = x1 - x0, y1 - y0
    on, off = pal["on"], pal["off"]

    if char in SEG:
        polys = seg_polys(x0, y0, x1, y1)
        active = SEG[char]
        for name, poly in polys.items():
            if name in active:
                rgb.polygon(poly, fill=on)
                lab.polygon(poly, fill=gid)
            elif off is not None:  # faint LCD ghost segment (appearance only)
                rgb.polygon(poly, fill=off)
        # Box the active extent so e.g. '1' boxes just the right bar.
        xs = [p[0] for n in active for p in polys[n]]
        ys = [p[1] for n in active for p in polys[n]]
        return (min(xs), min(ys), max(xs), max(ys))

    if char == "-":
        poly = seg_polys(x0, y0, x1, y1)["g"]
        rgb.polygon(poly, fill=on); lab.polygon(poly, fill=gid)
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))

    # Punctuation dots: vary size (dot_scale), shape (dot_square), contrast
    # (dot_color, sometimes fainter than the segments) and position, so the
    # detector learns '.'/':' robustly instead of memorizing one clean circle.
    square = pal["dot_square"]
    dotcol = pal["dot_color"]
    if char == ".":
        r = 0.16 * w * pal["dot_scale"]
        cx = x0 + r + rng.uniform(-0.05, 0.10) * w
        cy = (y1 - r) - rng.uniform(0.0, 0.10) * h   # near the baseline, slight jitter
        return _dot(rgb, lab, cx, cy, r, gid, dotcol, square)

    if char == ":":
        r = 0.14 * w * pal["dot_scale"]
        cx = x0 + w / 2 + rng.uniform(-0.05, 0.05) * w
        jit = rng.uniform(-0.03, 0.03) * h
        out = None
        for cy in (y0 + h * 0.33 + jit, y0 + h * 0.66 + jit):
            box = _dot(rgb, lab, cx, cy, r, gid, dotcol, square)
            out = box if out is None else (min(box[0], out[0]), min(box[1], out[1]),
                                           max(box[2], out[2]), max(box[3], out[3]))
        return out
    return None


def scatter_specks(rgb, boxes, pal, rng, W, H, dh):
    """Sprinkle a few unlabeled dark specks onto an LCD display as hard negatives.
    Real reflective LCDs carry dust/grime that the detector otherwise reads as
    phantom decimals (the dominant precision error). Drawn into the image only —
    never the label map — so they teach 'a small blob is not necessarily a dot'.
    Specks avoid glyph boxes (with margin) so they don't corrupt a real digit."""
    if pal["off"] is None or rng.random() > 0.45:   # LCD displays only, ~45%
        return
    glyphs = [b[1:] for b in boxes]
    for _ in range(rng.randint(1, 3)):
        r = rng.uniform(0.04, 0.10) * dh
        for _try in range(8):
            cx = rng.uniform(r, W - r)
            cy = rng.uniform(r, H - r)
            m = 0.5 * dh   # keep clear of real glyphs
            if all(cx < gx0 - m or cx > gx1 + m or cy < gy0 - m or cy > gy1 + m
                   for gx0, gy0, gx1, gy1 in glyphs):
                break
        else:
            continue
        col = pal["dot_color"] if rng.random() < 0.6 else pal["on"]
        box = [cx - r, cy - r, cx + r, cy + r]
        if pal["dot_square"]:
            rgb.rectangle(box, fill=col)
        else:
            rgb.ellipse(box, fill=col)


def cell_width(char, dh):
    """Width of a glyph cell given digit height dh (narrow for ./:/-)."""
    if char in ".:":
        return dh * 0.30
    if char == "1":
        return dh * 0.42
    return dh * 0.60


def random_text(rng) -> str:
    kind = rng.random()
    n = rng.randint(1, 6)
    digits = "".join(rng.choice("0123456789") for _ in range(n))
    if kind < 0.40:                     # plain integer (tank-like)
        return digits
    if kind < 0.65 and n >= 2:          # decimal (boosted: '.' was the weakest class)
        k = rng.randint(1, max(1, n - 1))
        return digits[:k] + "." + digits[k:]
    if kind < 0.82 and n >= 3:          # clock-like colon
        return digits[:2] + ":" + digits[2:]
    if kind < 0.90:                     # negative
        return "-" + digits
    return digits


def render_display(text, dh, pal, rng):
    """Render text onto a tight panel image; return (PIL RGB, label np array, boxes).
    boxes: list of (label_index, x0,y0,x1,y1) in image pixels."""
    pad = dh * 0.25
    widths = [cell_width(c, dh) for c in text]
    gap = dh * 0.10
    total_w = sum(widths) + gap * (len(text) - 1) + 2 * pad
    W, H = int(total_w), int(dh + 2 * pad)

    img = Image.new("RGB", (W, H), pal["bg"])
    lab = Image.new("L", (W, H), 0)
    drgb, dlab = ImageDraw.Draw(img), ImageDraw.Draw(lab)

    boxes = []
    x = pad
    for i, c in enumerate(text):
        cw = widths[i]
        cell = (x, pad, x + cw, pad + dh)
        region = draw_glyph(drgb, dlab, c, cell, gid=i + 1, pal=pal, rng=rng)
        if region is not None and c in LABEL_INDEX:
            boxes.append((LABEL_INDEX[c], *region))
        x += cw + gap
    scatter_specks(drgb, boxes, pal, rng, W, H, dh)
    return img, np.array(lab), boxes


# ----------------------------------------------------------------------------- palettes & aug

def dot_style(pal, rng):
    """Add per-display punctuation-dot appearance: shape (round/square), size
    scale, and contrast. Decided once per display so all dots stay consistent
    (as on a real device), but varied across displays so '.'/':' generalize."""
    pal["dot_square"] = rng.random() < 0.4               # ~40% square (LCD-style)
    pal["dot_scale"] = rng.uniform(0.85, 1.7)            # size variety
    # Usually the segment colour; sometimes a lower-contrast dot (blend toward bg)
    # so recall is robust to faint decimals as well as bold ones.
    on, bg = pal["on"], pal["bg"]
    if rng.random() < 0.35:
        f = rng.uniform(0.25, 0.6)
        pal["dot_color"] = tuple(int(o + f * (b - o)) for o, b in zip(on, bg))
    else:
        pal["dot_color"] = on
    return pal


def random_palette(rng):
    if rng.random() < 0.5:
        # LCD: dark segments on light gray, faint ghost off-segments.
        base = rng.randint(150, 215)
        bg = (base + rng.randint(-10, 10), base + rng.randint(-10, 10), base + rng.randint(-5, 15))
        d = rng.randint(20, 70)
        on = (d, d, d + rng.randint(0, 20))
        ghost = rng.random() < 0.7
        off = (base - rng.randint(12, 28),) * 3 if ghost else None
        return dot_style({"bg": bg, "on": on, "off": off}, rng)
    # LED: bright segments on dark background, no ghost.
    bg = tuple(rng.randint(0, 35) for _ in range(3))
    hue = rng.choice([(255, 40, 30), (40, 255, 80), (60, 160, 255), (255, 200, 40), (240, 240, 240)])
    on = tuple(min(255, c + rng.randint(-25, 25)) for c in hue)
    return dot_style({"bg": bg, "on": on, "off": None}, rng)


def find_coeffs(src, dst):
    """Perspective coefficients mapping dst->src for PIL Image.transform."""
    a = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        a.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        a.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
    A = np.array(a, dtype=float)
    b = np.array([c for pt in src for c in pt], dtype=float)
    res = np.linalg.solve(A, b)
    return res.tolist()


def warp_pair(img, lab, rng, out_size, place):
    """Apply identical rotation+perspective+placement to img (RGB) and lab (L)
    onto an out_size×out_size canvas. place='fill' centers/scales to most of the
    frame (digit crops); place='scene' drops it small into a scene (panel)."""
    W, H = img.size
    # Slight rotation.
    ang = rng.uniform(-7, 7)
    img = img.rotate(ang, expand=True, resample=Image.BILINEAR, fillcolor=None)
    lab = lab.rotate(ang, expand=True, resample=Image.NEAREST, fillcolor=0)

    # Mild perspective.
    W, H = img.size
    m = 0.12
    dst = [(0, 0), (W, 0), (W, H), (0, H)]
    src = [(rng.uniform(0, W * m), rng.uniform(0, H * m)),
           (W - rng.uniform(0, W * m), rng.uniform(0, H * m)),
           (W - rng.uniform(0, W * m), H - rng.uniform(0, H * m)),
           (rng.uniform(0, W * m), H - rng.uniform(0, H * m))]
    coeffs = find_coeffs(src, dst)
    img = img.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    lab = lab.transform((W, H), Image.PERSPECTIVE, coeffs, Image.NEAREST)

    # Scale + paste onto canvas.
    if place == "fill":
        scale = out_size * rng.uniform(0.6, 0.92) / max(W, H)
    else:
        scale = out_size * rng.uniform(0.18, 0.45) / max(W, H)
    nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    lab = lab.resize((nw, nh), Image.NEAREST)

    bgcol = tuple(rng.randint(0, 60) for _ in range(3)) if place == "scene" else \
        tuple(rng.randint(80, 200) for _ in range(3))
    canvas = Image.new("RGB", (out_size, out_size), bgcol)
    if place == "scene":
        canvas = scene_background(canvas, rng)
    labcanvas = Image.new("L", (out_size, out_size), 0)
    ox = rng.randint(0, max(0, out_size - nw))
    oy = rng.randint(0, max(0, out_size - nh))
    canvas.paste(img, (ox, oy))
    labcanvas.paste(lab, (ox, oy))
    return canvas, np.array(labcanvas)


def scene_background(canvas, rng):
    d = ImageDraw.Draw(canvas)
    for _ in range(rng.randint(0, 6)):
        x0, y0 = rng.randint(0, canvas.width), rng.randint(0, canvas.height)
        x1, y1 = x0 + rng.randint(10, 200), y0 + rng.randint(10, 200)
        c = tuple(rng.randint(0, 120) for _ in range(3))
        d.rectangle([x0, y0, x1, y1], fill=c)
    return canvas


def appearance(img, rng):
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    arr = np.asarray(img).astype(np.float32)
    arr += rng.uniform(0, 12) * np.random.randn(*arr.shape)        # sensor noise
    arr *= rng.uniform(0.8, 1.2)                                    # brightness
    if rng.random() < 0.35:                                         # glare blob
        gy, gx = np.ogrid[:arr.shape[0], :arr.shape[1]]
        cy, cx = rng.randint(0, arr.shape[0]), rng.randint(0, arr.shape[1])
        r = rng.randint(20, 90)
        mask = ((gx - cx) ** 2 + (gy - cy) ** 2) < r * r
        arr[mask] += rng.uniform(40, 120)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def boxes_from_label(lab_arr, out_size):
    """YOLO boxes for each glyph id present in the label map."""
    out = []
    ids = np.unique(lab_arr)
    for gid in ids:
        if gid == 0:
            continue
        ys, xs = np.where(lab_arr == gid)
        if xs.size == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        out.append((int(gid), (x0 + x1) / 2 / out_size, (y0 + y1) / 2 / out_size,
                    (x1 - x0) / out_size, (y1 - y0) / out_size))
    return out


# ----------------------------------------------------------------------------- generation

def write_sample(stem, img, yolo_lines, root, split):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(root / split / "images" / f"{stem}.jpg", quality=random.randint(60, 92))
    (root / split / "labels" / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n")


def warp_appearance(text, dh, pal, rng, out_size, place):
    """Render text, then apply identical geometric warp to image + label map and
    appearance-only noise to the image. Returns (RGB canvas, label np array)."""
    img, lab_arr, _ = render_display(text, dh, pal, rng)
    canvas, lab_canvas = warp_pair(img, Image.fromarray(lab_arr), rng, out_size, place)
    return appearance(canvas, rng), lab_canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=2000, help="digit-crop samples")
    ap.add_argument("--panels", type=int, default=800, help="panel-scene samples")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    S = args.img_size
    digits_root, panel_root = DATA / "digits", DATA / "panel"

    for i in range(args.digits):
        text = random_text(rng)
        dh = rng.uniform(70, 150)
        pal = random_palette(rng)
        canvas, lab = warp_appearance(text, dh, pal, rng, S, "fill")
        id_to_class = {gid + 1: LABEL_INDEX[c] for gid, c in enumerate(text) if c in LABEL_INDEX}
        lines = []
        for gid, cx, cy, w, h in boxes_from_label(lab, S):
            if gid in id_to_class and w > 0 and h > 0:
                lines.append(f"{id_to_class[gid]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            continue
        split = "val" if rng.random() < args.val_frac else "train"
        write_sample(f"synth_d{i:05d}", canvas, lines, digits_root, split)

    for i in range(args.panels):
        text = random_text(rng)
        dh = rng.uniform(50, 120)
        pal = random_palette(rng)
        canvas, lab = warp_appearance(text, dh, pal, rng, S, "scene")
        ys, xs = np.where(lab > 0)
        if xs.size == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2 / S, (y0 + y1) / 2 / S
        bw, bh = (x1 - x0) / S, (y1 - y0) / S
        # Pad the panel box a little beyond the digits to mimic a bezel.
        bw, bh = min(1.0, bw * 1.3), min(1.0, bh * 1.6)
        line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        split = "val" if rng.random() < args.val_frac else "train"
        write_sample(f"synth_p{i:05d}", canvas, [line], panel_root, split)

    print(f"synthetic: {args.digits} digit + {args.panels} panel samples -> {DATA}")


if __name__ == "__main__":
    main()
