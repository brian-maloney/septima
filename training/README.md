# septima training (offline)

Python pipeline that produces the two ONNX models the Go inference engine uses.
Nothing here is needed at inference time — only to (re)train and export models.

## Setup

```sh
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
```

### Kaggle credentials (for the public datasets)
The merge step downloads public 7-seg datasets via the Kaggle CLI. Create an API
token at https://www.kaggle.com/settings → "Create New Token", then:

```sh
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

(Optional) Roboflow source: `export ROBOFLOW_API_KEY=...` and set `enabled: true`
for `roboflow_7seg` in `datasets/sources.yaml`.

## Step 1 — public datasets → unified YOLO data

```sh
# First, inspect what classes each source actually ships (downloads, prints, exits):
python training/datasets/prepare.py --inspect
```

Use the printed class names to fill in `class_map` overrides in
`datasets/sources.yaml` if any source uses odd names (e.g. `dot`, `colon`).
Digit/symbol names normalize automatically; a single generic "display" class is
auto-routed to the panel detector. Then merge:

```sh
python training/datasets/prepare.py        # writes training/data/{digits,panel}/...
```

Output: `training/data/digits/` (classes `0-9 . : -`, `data_digits.yaml`) and
`training/data/panel/` (class `display`, `data_panel.yaml`), each split
train/val. Filenames are prefixed by source so sets never collide.

## Step 2 — synthetic data (no credentials needed)

```sh
python training/synth/render.py --digits 4000 --panels 1500
```

Appends synthetic samples to `training/data/{digits,panel}`. This is the highest-
leverage source for the reflective-LCD tank look, plus the only source of `.`/`:`/`-`
glyphs and (currently) of panel-detector data. Boxes stay exact through the
geometric augmentation via a parallel label map.

## Step 3 — train (Ultralytics YOLO-nano, Apple Silicon)

```sh
python training/train.py --stage digits --epochs 100   # -> training/runs/digits/
python training/train.py --stage panel  --epochs 60    # -> training/runs/panel/
```

`--device mps` by default (override with `--device cpu` or `0` for CUDA).

## Step 4 — export to ONNX (installs into models/)

```sh
python training/export_onnx.py --stage digits   # -> models/digits.onnx
python training/export_onnx.py --stage panel    # -> models/panel.onnx
```

Export verifies the model's class order against `models/classes.json` and records
the input size. The Go engine then loads these directly.

## Step 5 — fine-tune on real crops (closes the synthetic→real gap)

The bootstrap annotator (Go) runs the live pipeline on the real images and writes
trustworthy labels wherever the decode already matches ground truth, oversampling
them into a fine-tune set; disagreements go to a `review/` folder.

```sh
go run ./cmd/septima-annotate -in tanktests        # -> training/data/real_tank + data_finetune.yaml
```

Fine-tune from the base weights, then export and bench:

```sh
python training/train.py --stage digits \
    --data training/data/data_finetune.yaml \
    --model training/runs/digits/weights/best.pt \
    --epochs 40 --name digits_ft
python training/export_onnx.py --stage digits \
    --weights training/runs/digits_ft/weights/best.pt
go run ./cmd/septima-bench tanktests               # the real metric
```

The fine-tune set mixes the synthetic/public digits with oversampled real crops;
validation is on the synthetic val split, but the Go bench on `tanktests` is the
metric that matters. Any images the annotator routes to `review/` (the tank's
separated leading `1` is the usual culprit) can be hand-corrected and added to
`real_tank/train` for a second pass.

## Unified label space
`0 1 2 3 4 5 6 7 8 9 . : -` (digits) and `display` (panel). The digit order is
authoritative and must match `models/classes.json` `digit_classes`.
