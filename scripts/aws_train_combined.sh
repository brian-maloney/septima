#!/usr/bin/env bash
#
# Combined colon-synth + hard-negative digit fine-tune for AWS GPU.
#
# Strategy: add colon appearance diversity (diverse render.py, ~18% colon share)
# AND oversample hard real-world cases (real_hard: calculator 12-digit, shell pump
# tilted rows, etc.) as hard-negatives alongside real_tank.  The two prior runs
# showed full-backbone fine-tuning on colon-synth drifts thin-'1'/tilted-digit
# confidence (freeze=11 prevented colon learning and was WORSE).  The hypothesis:
# the hard-negative crops keep the backbone anchored while the colon synth adapts
# the neck/head to ':' — counter the drift with DATA not freezing.
#
# Prerequisites (on the AWS box):
#   1. Code synced (no git remote; use scp/rsync from local machine).
#   2. training/data/real_tank present (rsync from local).
#   3. training/data/real_hard present (rsync from local after running:
#        go run ./cmd/septima-annotate -in tests \
#          -out training/data/real_hard \
#          -panel-model models/panel.onnx \
#          -repeat 30
#   4. training/runs/digits/weights/best.pt present (the 13-class yolo11m baseline).
#   5. Python venv built (scripts/aws_setup.sh).
#
# Usage:
#   bash scripts/aws_train_combined.sh                  # full run
#   bash scripts/aws_train_combined.sh --check-only     # validate + regen synth, no train
#   bash scripts/aws_train_combined.sh --no-regen-synth # skip synth regen (use box's copy)
#
# Tunables (env vars):
#   SEPTIMA_EPOCHS   (20)    epochs — same as the "gentler" run; full backbone, less is safer
#   SEPTIMA_BATCH    (32)
#   SEPTIMA_DEVICE   (0)
#   SEPTIMA_NAME     (digits_combined)
#   SEPTIMA_CACHE    (ram)   ram=fast (needs ~25 GB RAM); disk=safe; False=off
#   SEPTIMA_DIGITS_SYNTH  (8000)
#   SEPTIMA_PANELS_SYNTH  (2500)
#
# No SEPTIMA_FREEZE: this run intentionally trains ALL layers.  Freezing the
# backbone (=11) prevented colon learning in run #3 and made digits WORSE.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${SEPTIMA_EPOCHS:-20}"
BATCH="${SEPTIMA_BATCH:-32}"
DEVICE="${SEPTIMA_DEVICE:-0}"
NAME="${SEPTIMA_NAME:-digits_combined}"
CACHE="${SEPTIMA_CACHE:-ram}"
DIGITS_SYNTH="${SEPTIMA_DIGITS_SYNTH:-8000}"
PANELS_SYNTH="${SEPTIMA_PANELS_SYNTH:-2500}"
REGEN_SYNTH=1
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check-only)     CHECK_ONLY=1 ;;
    --no-regen-synth) REGEN_SYNTH=0 ;;
    *) printf 'unknown arg: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\n  \033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
say()  { printf '\n==> %s\n' "$*"; }

say "Environment validation (combined colon + hard-negative fine-tune)"

# 0. repo root sanity -----------------------------------------------------------
[ -f go.mod ] && [ -d training ] || die "run from the septima repo root"

# 1. python / venv --------------------------------------------------------------
if [ -f training/.venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source training/.venv/bin/activate
  ok "venv activated (training/.venv)"
else
  command -v python3 >/dev/null \
    || die "no python3 and no training/.venv — run scripts/aws_setup.sh first"
  warn "no training/.venv; falling back to system python3"
fi

# 2. torch + CUDA ---------------------------------------------------------------
python - <<'PY' || die "python/torch/CUDA check failed — fix env (scripts/aws_setup.sh installs CUDA torch)"
import sys
try:
    import torch, ultralytics
except Exception as e:
    print(f"  MISSING package: {e}"); sys.exit(1)
print(f"  torch {torch.__version__}  ultralytics {ultralytics.__version__}")
if not torch.cuda.is_available():
    print("  CUDA NOT available — refusing to train on CPU"); sys.exit(2)
print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
PY
ok "torch + ultralytics importable, CUDA available"

# 3. GPU info -------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
else
  warn "nvidia-smi not found (torch sees CUDA, continuing)"
fi

# 4. disk space -----------------------------------------------------------------
FREE_GB="$(df -P . | awk 'NR==2{print int($4/1024/1024)}')" || true
if [ "${FREE_GB:-0}" -ge 1000000 ]; then ok "disk free: ample (network FS)"; \
elif [ "${FREE_GB:-0}" -ge 25 ];    then ok "disk free: ${FREE_GB} GB"; \
else warn "only ${FREE_GB:-?} GB free — consider SEPTIMA_CACHE=disk or False"; fi

# 5. render.py is the diverse-colon generator -----------------------------------
grep -q "Diversify colon appearance" training/synth/render.py \
  || die "render.py is NOT the diverse-colon version — sync the latest code"
ok "render.py is the diverse-colon generator"

# 6. classes.json 13-class ------------------------------------------------------
python - <<'PY' || die "models/classes.json digit order is wrong/missing"
import json, sys
c = json.load(open("models/classes.json"))["digit_classes"]
sys.exit(0 if c == ["0","1","2","3","4","5","6","7","8","9",".",":","-"] else 1)
PY
ok "classes.json: 13 digit classes in expected order"

# 7. base weights ---------------------------------------------------------------
BASE="training/runs/digits/weights/best.pt"
[ -f "$BASE" ] || die "missing $BASE — scp the 13-class best.pt from your local machine (gitignored *.pt)"
ok "base weights present: $BASE"

# 8. real digit data prepared ---------------------------------------------------
N_REAL="$(find training/data/digits/train/images -type f \
          \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) \
          ! -name 'synth_*' 2>/dev/null | wc -l | tr -d ' ')" || true
[ "${N_REAL:-0}" -ge 100 ] \
  || die "only ${N_REAL:-0} real digit train images — run training/datasets/prepare.py first"
ok "real digit data present (${N_REAL} real train images)"

# 9. real_tank (must hold tank 32/32) ------------------------------------------
[ -n "$(ls -A training/data/real_tank/train/images 2>/dev/null || true)" ] \
  || die "training/data/real_tank is empty — rsync from local; tank will regress without it"
N_TANK="$(find training/data/real_tank/train/images -type f \
          \( -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' ')" || true
ok "real_tank present (${N_TANK} train images)"

# 10. real_hard (tilted/thin-1 hard-negatives) ----------------------------------
N_HARD="$(find training/data/real_hard/train/images -type f \
          \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) 2>/dev/null \
          | wc -l | tr -d ' ')" || true
if [ "${N_HARD:-0}" -ge 10 ]; then
  ok "real_hard present (${N_HARD} train images)"
else
  die "training/data/real_hard is empty or missing — generate it locally first:
    go run ./cmd/septima-annotate \\
      -in tests \\
      -out training/data/real_hard \\
      -panel-model models/panel.onnx \\
      -repeat 30
  then rsync to this box:
    rsync -av training/data/real_hard <aws-host>:~/septima/training/data/"
fi

# 11. regenerate colon-diverse synth ON this box --------------------------------
if [ "$REGEN_SYNTH" -eq 1 ]; then
  say "Regenerating colon-diverse synth (${DIGITS_SYNTH} digit / ${PANELS_SYNTH} panel)"
  find training/data/digits training/data/panel \
       \( -name 'synth_d*' -o -name 'synth_p*' \) -delete 2>/dev/null || true
  python training/synth/render.py --digits "$DIGITS_SYNTH" --panels "$PANELS_SYNTH"
  find training/data -name 'labels.cache' -delete 2>/dev/null || true
  N_COL="$(grep -l '^11 ' training/data/digits/train/labels/synth_*.txt 2>/dev/null \
           | wc -l | tr -d ' ')" || true
  ok "synth regenerated; train files with a colon ':' box: ${N_COL:-0}"
else
  warn "--no-regen-synth: using the synth already on the box"
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "Validated (--check-only). Not training."
  exit 0
fi

# 12. fine-tune — no freeze (freeze=11 prevented colon learning in run #3) ------
say "Fine-tuning digits from best.pt (device ${DEVICE}, ${EPOCHS} epochs, batch ${BATCH}, cache ${CACHE}, NO backbone freeze, name ${NAME})"
python scripts/train_digits_decimal.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch  "$BATCH"  \
  --cache  "$CACHE"  \
  --name   "$NAME"
# No --freeze: train_digits_decimal.py defaults to no freeze when omitted.

# 13. export to models/digits.onnx ---------------------------------------------
say "Exporting ONNX -> models/digits.onnx"
python training/export_onnx.py \
  --stage digits \
  --weights "training/runs/${NAME}/weights/best.pt"

# 14. verify (best-effort; needs Go + onnxruntime on box) ----------------------
say "Verification"
if command -v go >/dev/null 2>&1; then
  printf '  TANK (must hold 32/32):\n'
  go run ./cmd/septima-bench tanktests \
    || warn "tank bench did not run (onnxruntime dylib?)"
  printf '  HELD-OUT benchmark (baseline: 329/408 = 80.6%%):\n'
  go run ./cmd/septima-bench training/data/digits/test \
    || warn "held-out bench did not run here"
else
  warn "Go not found — skipping bench. Pull models/digits.onnx to local and run:"
  printf '    go run ./cmd/septima-bench tanktests\n'
  printf '    go run ./cmd/septima-bench training/data/digits/test\n'
  printf '    go test ./tests/\n'
fi

say "Done. New model at models/digits.onnx (run: ${NAME})."
printf '\nIf trained on AWS, scp models/digits.onnx back then verify locally:\n'
printf '  go run ./cmd/septima-bench tanktests\n'
printf '  go run ./cmd/septima-bench training/data/digits/test\n'
printf '  go test ./tests/   # alarm clock 2:47 is the colon target\n'
printf '\nBaseline rollback: /tmp/digits_baseline_806.onnx\n'
