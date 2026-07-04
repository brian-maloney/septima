#!/usr/bin/env bash
#
# Fresh yolo11l digit-model retrain (bigger backbone, untried lever).
#
# Every prior digit retrain in this repo FINE-TUNED from the existing 13-class
# yolo11m checkpoint (training/runs/digits/weights/best.pt). That checkpoint
# CANNOT be reused here: yolo11l has different per-layer channel widths than
# yolo11m, so its state dict does not load into an m-sized backbone. A model-
# size change means training FRESH from stock yolo11l.pt on the full corpus
# (data_finetune.yaml: public+synthetic digits + real_tank + real_hard hard-
# negatives), exactly like the original yolo11s/yolo11m baselines were produced
# (see scripts/aws_setup.sh) — NOT via scripts/train_digits_decimal.py, whose
# preflight explicitly REJECTS stock COCO weights (that guard is correct for the
# fine-tune workflow; it does not apply to standing up a new model size).
#
# Hypothesis (memory: "yolo11l retrain (untried)"): the remaining held-out
# failures are dominated by length-mismatch on small/degenerate crops and
# genuinely hard glyphs, not a concentrated confusion pattern — more model
# capacity may recover some of those where synth/threshold/selection tuning
# already proved tapped out. Panel detector is NOT touched (not a bottleneck
# per the diagnostics; stays on the current yolo11m panel.onnx).
#
# GPU is auto-detected (memory + core count) rather than hardcoded, since box
# sizes vary run to run: ultralytics batch=-1 (AutoBatch) picks a batch that
# fits the box's free VRAM unless SEPTIMA_BATCH overrides it.
#
# Prerequisites (on the training box):
#   1. Code synced (no git remote; scp/rsync from local, or git clone if remote add).
#   2. training/data/{digits,real_tank,real_hard} present + data_finetune.yaml
#      (rsync training/data from local, or regenerate: prepare.py + render.py +
#      stage_real_hard.py — real_tank/real_hard are hand-curated, prefer rsync).
#   3. training/.venv built (scripts/aws_setup.sh sets this up).
#   4. yolo11l.pt present at repo root, or let ultralytics auto-download it
#      (needs outbound network to github.com/ultralytics/assets).
#
# Usage:
#   bash scripts/aws_train_yolo11l.sh                 # full run
#   bash scripts/aws_train_yolo11l.sh --check-only     # validate only, no train
#
# Tunables (env vars):
#   SEPTIMA_EPOCHS   (120)   epochs — matches the original fresh-baseline default
#                            (aws_setup.sh EPOCHS_DIGITS) since this is the same
#                            kind of run (train fresh on the full corpus).
#   SEPTIMA_IMGSZ    (640)   training/export image size. imgsz=1280 was tried
#                            twice on the yolo11m digit model and regressed the
#                            held-out bench ~5pp each time (real data tops out
#                            ~640px) — stay at 640 unless that changes.
#   SEPTIMA_BATCH    (-1)    -1 = ultralytics AutoBatch (fits free VRAM on this
#                            box automatically). Set an explicit int to pin it.
#   SEPTIMA_WORKERS  (auto)  defaults to nproc.
#   SEPTIMA_DEVICE   (0)
#   SEPTIMA_NAME     (digits_yolo11l)
#   SEPTIMA_CACHE    (ram)   ram=fast (needs real RAM headroom for ~40k imgs at
#                            640px); disk=safe fallback; False=off.
#   SEPTIMA_MODEL    (yolo11l.pt)  override to try yolo11x.pt with the same script.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${SEPTIMA_MODEL:-yolo11l.pt}"
IMGSZ="${SEPTIMA_IMGSZ:-640}"
EPOCHS="${SEPTIMA_EPOCHS:-120}"
BATCH="${SEPTIMA_BATCH:--1}"
DEVICE="${SEPTIMA_DEVICE:-0}"
NAME="${SEPTIMA_NAME:-digits_yolo11l}"
CACHE="${SEPTIMA_CACHE:-ram}"
WORKERS="${SEPTIMA_WORKERS:-$(command -v nproc >/dev/null && nproc || echo 8)}"
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    *) printf 'unknown arg: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\n  \033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
say()  { printf '\n==> %s\n' "$*"; }

say "Environment validation (fresh yolo11l digit retrain)"

# 0. repo root sanity -------------------------------------------------------
[ -f go.mod ] && [ -d training ] || die "run from the septima repo root"

# 1. python / venv ------------------------------------------------------------
if [ -f training/.venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source training/.venv/bin/activate
  ok "venv activated (training/.venv)"
else
  command -v python3 >/dev/null \
    || die "no python3 and no training/.venv — run scripts/aws_setup.sh first"
  warn "no training/.venv; falling back to system python3"
fi

# 2. torch + CUDA + GPU info --------------------------------------------------
python - <<'PY' || die "python/torch/CUDA check failed — fix env (scripts/aws_setup.sh installs CUDA torch)"
import sys
try:
    import torch, ultralytics
except Exception as e:
    print(f"  MISSING package: {e}"); sys.exit(1)
print(f"  torch {torch.__version__}  ultralytics {ultralytics.__version__}")
if not torch.cuda.is_available():
    print("  CUDA NOT available — refusing to train on CPU"); sys.exit(2)
name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  CUDA device: {name}  ({mem_gb:.1f} GB)")
PY
ok "torch + ultralytics importable, CUDA available"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
fi
ok "workers=${WORKERS} (nproc)"

# 3. classes.json 13-class -----------------------------------------------------
python - <<'PY' || die "models/classes.json digit order is wrong/missing"
import json, sys
c = json.load(open("models/classes.json"))["digit_classes"]
sys.exit(0 if c == ["0","1","2","3","4","5","6","7","8","9",".",":","-"] else 1)
PY
ok "classes.json: 13 digit classes in expected order"

# 4. data_finetune.yaml present, no test-split leakage -------------------------
python - <<'PY' || die "data_finetune.yaml missing/invalid — run training/datasets/prepare.py"
import yaml, sys
cfg = yaml.safe_load(open("training/data/data_finetune.yaml"))
train = cfg.get("train", [])
train = [train] if isinstance(train, str) else train
paths = train + [str(cfg.get("val", ""))]
leak = [p for p in paths if "/test/" in p or str(p).endswith("/test") or "test/images" in str(p)]
if leak:
    print(f"  LEAKAGE: {leak}"); sys.exit(1)
if not any("real_tank/train/images" in p for p in train):
    print("  missing real_tank/train/images — tank will regress"); sys.exit(1)
if not any("real_hard/train/images" in p for p in train):
    print("  missing real_hard/train/images — hard-negatives absent"); sys.exit(1)
print(f"  train sources: {train}")
PY
ok "data_finetune.yaml: full corpus + real_tank + real_hard, no held-out leakage"

# 5. real data volume sanity ---------------------------------------------------
N_DIGITS="$(find training/data/digits/train/images -type f 2>/dev/null | wc -l | tr -d ' ')" || true
N_TANK="$(find training/data/real_tank/train/images -type f 2>/dev/null | wc -l | tr -d ' ')" || true
N_HARD="$(find training/data/real_hard/train/images -type f 2>/dev/null | wc -l | tr -d ' ')" || true
[ "${N_DIGITS:-0}" -ge 1000 ] || die "only ${N_DIGITS:-0} digit train images — run prepare.py + render.py"
[ "${N_TANK:-0}" -gt 0 ]      || die "real_tank/train/images empty — rsync it from local"
[ "${N_HARD:-0}" -gt 0 ]      || die "real_hard/train/images empty — rsync it, or: python scripts/stage_real_hard.py"
ok "digit train images: ${N_DIGITS}  real_tank: ${N_TANK}  real_hard: ${N_HARD}"

# 6. base weights: stock yolo11l.pt (fresh train, not a fine-tune) ------------
if [ -f "$MODEL" ]; then
  ok "base weights present: ${MODEL}"
else
  warn "${MODEL} not found locally — ultralytics will auto-download it (needs network)"
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "Validated (--check-only). Not training."
  exit 0
fi

# 7. train ----------------------------------------------------------------------
say "Training digits FRESH from ${MODEL} (device ${DEVICE}, ${EPOCHS} epochs, imgsz ${IMGSZ}, batch ${BATCH}, cache ${CACHE}, workers ${WORKERS}, name ${NAME})"
RUN_DIR="training/runs/${NAME}"
if [ -d "$RUN_DIR" ]; then
  warn "removing prior run dir ${RUN_DIR} for a clean run"
  rm -rf "$RUN_DIR"
fi
python training/train.py \
  --stage digits \
  --data training/data/data_finetune.yaml \
  --model "$MODEL" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --cache "$CACHE" \
  --workers "$WORKERS" \
  --name "$NAME"

# 8. export to models/digits.onnx ----------------------------------------------
say "Exporting ONNX -> models/digits.onnx (imgsz ${IMGSZ})"
python training/export_onnx.py \
  --stage digits \
  --imgsz "$IMGSZ" \
  --weights "training/runs/${NAME}/weights/best.pt"

# 9. verify (best-effort; needs Go + onnxruntime on box) -----------------------
say "Verification against the pre-run baseline gates — see scripts/gate_yolo11l.sh"
if command -v go >/dev/null 2>&1; then
  bash scripts/gate_yolo11l.sh || warn "gate script reported a failure — see above"
else
  warn "Go not found on this box — pull models/digits.onnx to local and run scripts/gate_yolo11l.sh there"
fi

say "Done. New model at models/digits.onnx (run: ${NAME})."
printf '\nIf trained remotely, scp models/digits.onnx back then run:\n'
printf '  bash scripts/gate_yolo11l.sh\n'
