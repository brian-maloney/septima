# septima training (offline)

Python pipeline that produces the two ONNX models the Go inference engine uses.
Nothing here is needed at inference time — only to (re)train and export models.

## Setup

```sh
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
```

### Dataset credentials
Kaggle (CLI): create a token at https://www.kaggle.com/settings → "Create New
Token", then:

```sh
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Roboflow (real meter/display photos for generality): get a key at
https://app.roboflow.com → Settings → API, then `export ROBOFLOW_API_KEY=...`.
The roboflow sources in `datasets/sources.yaml` download the latest version of
each project automatically.

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

## Step 2.5 — held-out benchmark (measure real-world accuracy)

`prepare.py` keeps each dataset's own `test/` split HELD OUT (never in
`data_*.yaml`). Turn it into an end-to-end benchmark — ground-truth strings are
derived from the test boxes:

```sh
python training/datasets/make_benchmark.py          # -> data/digits/test/ground_truth.json
go run ./cmd/septima-bench training/data/digits/test # exact + char accuracy on real held-out images
```

This is the number that reflects "world can use" generality — score it after
every training run alongside `tanktests` and `tests`.

## Step 3 — train both stages (Ultralytics YOLO, Apple Silicon or GPU)

Model size scales with `--model`: `yolo11n.pt` (nano, fast) → `yolo11s.pt` /
`yolo11m.pt` (more accurate, want a GPU). On a CUDA box use `--device 0`; an
A10G/L4 (AWS g5/g6) turns the multi-hour MPS runs into ~30–60 min, making a
larger model + more epochs practical.

```sh
# example: larger model on a GPU box
python training/train.py --stage panel  --model yolo11s.pt --device 0 --epochs 80
python training/train.py --stage digits --model yolo11s.pt --device 0 --epochs 120 \
    --data training/data/data_finetune.yaml
```


The digit detector trains on the combined digit data (public + synthetic + real
tank crops) via `data_finetune.yaml`; the panel detector trains on the panel data
(synthetic scenes + panel boxes synthesized from every digit image + any explicit
display/meter boxes from real datasets).

```sh
# digit detector (general): from base weights on the full digit corpus
python training/train.py --stage digits --data training/data/data_finetune.yaml --epochs 100
# panel detector
python training/train.py --stage panel  --epochs 60
```

`--device mps` by default (override with `--device cpu` or `0` for CUDA). To
fine-tune from existing weights instead of training fresh, add
`--model training/runs/<run>/weights/best.pt`.

## Step 4 — export to ONNX (installs into models/)

```sh
python training/export_onnx.py --stage digits --weights training/runs/digits/weights/best.pt
python training/export_onnx.py --stage panel  --weights training/runs/panel/weights/best.pt
go run ./cmd/septima-bench tanktests   # regression
go run ./cmd/septima-bench tests       # generalization
```

Export verifies the model's class order against `models/classes.json` and records
the input size. Once `models/panel.onnx` exists, the Go pipeline uses it for
stage-1 automatically (falling back to the bright-panel heuristic only if it
finds nothing).

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
