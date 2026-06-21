#!/usr/bin/env bash
#
# AWS GPU-box setup + data prep for septima training.
#
# Regenerates the entire training corpus from pinned open-source datasets on the
# box itself (so nothing large is uploaded from your machine), builds the
# held-out benchmark, and optionally runs training + export.
#
# Prereqs on the box:
#   - NVIDIA GPU + driver (nvidia-smi)            for fast training
#   - ~/.kaggle/kaggle.json                       for the Kaggle datasets
#   - export ROBOFLOW_API_KEY=...                 for the Roboflow datasets
#   - (optional) training/data/real_tank/ synced  protects the tank 32/32
#
# Usage:
#   scripts/aws_setup.sh           # setup + data prep, then print train commands
#   scripts/aws_setup.sh --train   # setup + data prep + train + export
#
# Tunables (env vars): SEPTIMA_MODEL, SEPTIMA_DEVICE, SEPTIMA_EPOCHS_PANEL,
#   SEPTIMA_EPOCHS_DIGITS, SEPTIMA_BATCH, SEPTIMA_WORKERS, SEPTIMA_CACHE,
#   SEPTIMA_CUDA (pytorch wheel tag, e.g. cu124).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${SEPTIMA_MODEL:-yolo11m.pt}"
DEVICE="${SEPTIMA_DEVICE:-0}"
EPOCHS_PANEL="${SEPTIMA_EPOCHS_PANEL:-80}"
EPOCHS_DIGITS="${SEPTIMA_EPOCHS_DIGITS:-120}"
BATCH="${SEPTIMA_BATCH:-32}"          # L4 (24 GB) fits yolo11m@640 at batch=32; drop to 16 if OOM
WORKERS="${SEPTIMA_WORKERS:-4}"      # match vCPU count of the instance (g6.xlarge = 4)
CACHE="${SEPTIMA_CACHE:-ram}"         # ram = fast after epoch 1; disk = safe fallback; False = off
CUDA_TAG="${SEPTIMA_CUDA:-cu124}"
DO_TRAIN=0
[ "${1:-}" = "--train" ] && DO_TRAIN=1

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

# --- 1. Python environment ---------------------------------------------------
command -v python3 >/dev/null || die "python3 not found"

# Ubuntu/Debian ship python3 without venv/ensurepip by default — install it.
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  say "Installing python venv support (python${PYVER}-venv) — needs sudo"
  sudo apt-get update -qq
  sudo apt-get install -y "python${PYVER}-venv" python3-pip \
    || die "could not install python${PYVER}-venv; run: sudo apt install python${PYVER}-venv"
fi

say "Setting up venv at training/.venv"
# Recreate when bin/activate is missing — a venv that failed at the ensurepip
# step leaves bin/python3 behind but never writes the activate scripts, so check
# for activate specifically rather than the interpreter.
[ -f training/.venv/bin/activate ] || { rm -rf training/.venv; python3 -m venv training/.venv; }
[ -f training/.venv/bin/activate ] || die "venv creation failed; run: sudo apt install python3-venv"
# shellcheck disable=SC1091
source training/.venv/bin/activate
pip install --quiet --upgrade pip

# Install a CUDA build of torch first (so ultralytics doesn't pull the CPU wheel).
if command -v nvidia-smi >/dev/null 2>&1; then
  say "GPU detected; installing CUDA torch ($CUDA_TAG)"
  pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    || die "CUDA torch install failed — try a different SEPTIMA_CUDA tag (cu121/cu124) for your driver"
else
  printf '\n!! nvidia-smi not found — training will run on CPU (slow).\n'
fi
pip install --quiet -r training/requirements.txt

python - <<'PY'
import torch
print(f"==> torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"==> device: {torch.cuda.get_device_name(0)}")
PY

# --- 2. Credentials ----------------------------------------------------------
[ -f "$HOME/.kaggle/kaggle.json" ] || die "missing ~/.kaggle/kaggle.json (Kaggle datasets)"
chmod 600 "$HOME/.kaggle/kaggle.json"
[ -n "${ROBOFLOW_API_KEY:-}" ] || die "ROBOFLOW_API_KEY not set (Roboflow datasets)"

# --- 3. Data prep (regenerated from pinned sources) --------------------------
say "Downloading + merging open datasets (pinned versions)"
python training/datasets/prepare.py
say "Generating synthetic samples"
python training/synth/render.py
say "Building held-out benchmark"
python training/datasets/make_benchmark.py

# --- 4. real_tank sanity (protects the tank's 32/32) -------------------------
if [ -z "$(ls -A training/data/real_tank/train/images 2>/dev/null || true)" ]; then
  printf '\n!! training/data/real_tank is empty. data_finetune.yaml expects it.\n'
  printf '   rsync it from your local machine, or regenerate with cmd/septima-annotate,\n'
  printf '   else the digit model loses the tank-specific fine-tuning data.\n'
fi

say "Data ready."

# --- 5. Train (optional) -----------------------------------------------------
train_cmds() {
  cat <<EOF
  python training/train.py --stage panel  --model $MODEL --device $DEVICE --epochs $EPOCHS_PANEL --batch $BATCH --workers $WORKERS --cache $CACHE
  python training/train.py --stage digits --model $MODEL --device $DEVICE --epochs $EPOCHS_DIGITS --batch $BATCH --workers $WORKERS --cache $CACHE \\
      --data training/data/data_finetune.yaml
  python training/export_onnx.py --stage panel
  python training/export_onnx.py --stage digits
EOF
}

if [ "$DO_TRAIN" -eq 1 ]; then
  say "Training panel ($MODEL, device $DEVICE, $EPOCHS_PANEL epochs, batch $BATCH, workers $WORKERS, cache=$CACHE)"
  python training/train.py --stage panel  --model "$MODEL" --device "$DEVICE" --epochs "$EPOCHS_PANEL" --batch "$BATCH" --workers "$WORKERS" --cache "$CACHE"
  say "Training digits ($MODEL, device $DEVICE, $EPOCHS_DIGITS epochs, batch $BATCH, workers $WORKERS, cache=$CACHE)"
  python training/train.py --stage digits --model "$MODEL" --device "$DEVICE" --epochs "$EPOCHS_DIGITS" --batch "$BATCH" --workers "$WORKERS" --cache "$CACHE" \
      --data training/data/data_finetune.yaml
  python training/export_onnx.py --stage panel
  python training/export_onnx.py --stage digits
  say "Done. Pull models/panel.onnx + models/digits.onnx back to your local machine, then:"
  echo "  go run ./cmd/septima-bench training/data/digits/test   # held-out benchmark"
  echo "  go run ./cmd/septima-bench tanktests && go run ./cmd/septima-bench tests"
else
  say "Next — train + export (or re-run with --train):"
  train_cmds
fi
