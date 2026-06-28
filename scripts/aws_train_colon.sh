#!/usr/bin/env bash
#
# One-shot AWS GPU runner for the diverse-colon digit fine-tune.
#
# VALIDATES the environment + code + data BEFORE doing anything expensive, then
# regenerates the colon-diverse synthetic data ON THIS BOX (so you never have to
# upload gigabytes of synth images), fine-tunes the digit model FROM the existing
# 13-class best.pt (not from stock COCO), exports models/digits.onnx, and — if Go
# is available here — runs the benchmarks.
#
# Run from the repo root on the AWS box:
#   bash scripts/aws_train_colon.sh                 # validate + regen synth + train + export
#   bash scripts/aws_train_colon.sh --check-only    # validate + regen synth, then STOP
#   bash scripts/aws_train_colon.sh --no-regen-synth # use the synth already on the box
#
# Tunables (env vars):
#   SEPTIMA_EPOCHS (20)  SEPTIMA_BATCH (32)  SEPTIMA_DEVICE (0)  SEPTIMA_NAME (digits_colon)
#   SEPTIMA_CACHE (ram)  SEPTIMA_DIGITS_SYNTH (8000)  SEPTIMA_PANELS_SYNTH (2500)
#
# Footguns this guards against (all have bitten us before):
#   - training fresh from stock yolo11m.pt instead of best.pt  -> broad regression
#     (delegated to train_digits_decimal.py's base-weights guard);
#   - real_tank absent -> data_finetune.yaml silently drops the tank hard-negatives
#     -> tank phantom-decimal regression;
#   - STALE render.py on the box (old fixed colon) -> the run adds no colon variety;
#   - a stale ultralytics label cache hiding the freshly regenerated synth;
#   - a CPU-only / mis-driver'd box silently training for days.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${SEPTIMA_EPOCHS:-20}"   # gentler than 40: a long colon-heavy run drifted hard-glyph confidence down
BATCH="${SEPTIMA_BATCH:-32}"
DEVICE="${SEPTIMA_DEVICE:-0}"
NAME="${SEPTIMA_NAME:-digits_colon}"
CACHE="${SEPTIMA_CACHE:-ram}"   # ram = fast after epoch 1 (needs ~25 GB system RAM); disk = safe; False = off
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

say "Environment validation"

# 0. repo root sanity ---------------------------------------------------------
[ -f go.mod ] && [ -d training ] || die "run from the septima repo root"

# 1. python / venv ------------------------------------------------------------
if [ -f training/.venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source training/.venv/bin/activate
  ok "venv activated (training/.venv)"
else
  command -v python3 >/dev/null || die "no python3 and no training/.venv — run scripts/aws_setup.sh first to build the env"
  warn "no training/.venv; falling back to system python3"
fi

# 2. core packages + CUDA -----------------------------------------------------
python - <<'PY' || die "python/torch/CUDA check failed — fix the env (scripts/aws_setup.sh installs CUDA torch) before training"
import sys
try:
    import torch, ultralytics
except Exception as e:
    print(f"  MISSING package: {e}"); sys.exit(1)
print(f"  torch {torch.__version__}  ultralytics {ultralytics.__version__}")
if not torch.cuda.is_available():
    print("  CUDA NOT available — refusing to train on CPU (would take days)"); sys.exit(2)
print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
PY
ok "torch + ultralytics importable, CUDA available"

# 3. nvidia-smi ---------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
else
  warn "nvidia-smi not found (torch sees CUDA, so continuing)"
fi

# 4. disk space (synth + ultralytics cache + run artifacts) -------------------
# A network FS (EFS) reports an effectively-unbounded size; show "ample" instead
# of the overflowed number.
FREE_GB="$(df -P . | awk 'NR==2{print int($4/1024/1024)}')" || true
if [ "${FREE_GB:-0}" -ge 1000000 ]; then ok "disk free: ample (network FS)"; \
  elif [ "${FREE_GB:-0}" -ge 25 ]; then ok "disk free: ${FREE_GB} GB"; \
  else warn "only ${FREE_GB:-?} GB free — the image cache may not fit (consider SEPTIMA_CACHE=disk or False)"; fi

# 5. code freshness: render.py MUST be the diverse-colon generator ------------
grep -q "Diversify colon appearance" training/synth/render.py \
  || die "training/synth/render.py is NOT the diverse-colon version — sync the latest code to this box (no git remote = scp/rsync the repo)"
ok "render.py is the diverse-colon generator"

# 6. classes.json 13-class in order -------------------------------------------
python - <<'PY' || die "models/classes.json digit order is wrong/missing"
import json, sys
c = json.load(open("models/classes.json"))["digit_classes"]
sys.exit(0 if c == ["0","1","2","3","4","5","6","7","8","9",".",":","-"] else 1)
PY
ok "classes.json: 13 digit classes in expected order"

# 7. base weights present (launcher asserts they ARE the 13-class model) -------
BASE="training/runs/digits/weights/best.pt"
[ -f "$BASE" ] || die "missing base weights $BASE — scp it from your local machine (the 13-class fine-tune base; it is gitignored as *.pt so it is NOT in the repo)"
ok "base weights present: $BASE"

# 8. real (non-synth) digit data prepared -------------------------------------
# NB: no `find | head` here — under `set -o pipefail`, head closing the pipe early
# sends find SIGPIPE (exit 141) and silently kills the script. Count everything.
N_REAL="$(find training/data/digits/train/images -type f \
          \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) \
          ! -name 'synth_*' 2>/dev/null | wc -l | tr -d ' ')" || true
[ "${N_REAL:-0}" -ge 100 ] \
  || die "training/data/digits/train has only ${N_REAL:-0} real images — run training/datasets/prepare.py first (needs ROBOFLOW_API_KEY + ~/.kaggle/kaggle.json)"
ok "real digit data present (${N_REAL} real train images)"

# 9. real_tank present (holds tank 32/32) -------------------------------------
[ -n "$(ls -A training/data/real_tank/train/images 2>/dev/null || true)" ] \
  || die "training/data/real_tank is empty — rsync it from local; data_finetune.yaml needs it or the tank regresses (phantom decimals)"
ok "real_tank present"

# 10. regenerate the colon-diverse synth ON this box --------------------------
if [ "$REGEN_SYNTH" -eq 1 ]; then
  say "Regenerating colon-diverse synth (${DIGITS_SYNTH} digit / ${PANELS_SYNTH} panel)"
  # render.py does NOT clear old synth; remove it first so stale fixed-colon synth
  # cannot dilute the new variety.
  find training/data/digits training/data/panel \( -name 'synth_d*' -o -name 'synth_p*' \) -delete 2>/dev/null || true
  python training/synth/render.py --digits "$DIGITS_SYNTH" --panels "$PANELS_SYNTH"
  # Invalidate ultralytics' cached file lists so the new synth is actually used.
  find training/data -name 'labels.cache' -delete 2>/dev/null || true
  # `|| true`: grep -l exits 1 when it matches nothing, which pipefail would turn
  # into a script-killing failure.
  N_COL="$(grep -l '^11 ' training/data/digits/train/labels/synth_*.txt 2>/dev/null | wc -l | tr -d ' ')" || true
  ok "synth regenerated; train files with a colon ':' box: ${N_COL:-0}"
else
  warn "--no-regen-synth: using the synth already on the box (ensure it is the diverse-colon synth)"
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "Environment + data validated (--check-only). Not training."
  exit 0
fi

# 11. fine-tune from best.pt --------------------------------------------------
# train_digits_decimal.py re-runs the FULL data preflight (refreshes
# data_finetune.yaml from the tree, asserts the 13-class base, real_tank present,
# no test-split leakage, colon/decimal synth present, device available).
say "Fine-tuning digits from best.pt (device ${DEVICE}, ${EPOCHS} epochs, batch ${BATCH}, cache ${CACHE}, name ${NAME})"
python scripts/train_digits_decimal.py --device "$DEVICE" --epochs "$EPOCHS" --batch "$BATCH" --cache "$CACHE" --name "$NAME"

# 12. export to models/digits.onnx --------------------------------------------
say "Exporting ONNX -> models/digits.onnx"
python training/export_onnx.py --stage digits --weights "training/runs/${NAME}/weights/best.pt"

# 13. verify (best-effort; needs Go + onnxruntime on this box) ----------------
say "Verification"
if command -v go >/dev/null 2>&1; then
  echo "  TANK (must hold 32/32):"
  go run ./cmd/septima-bench tanktests || warn "tank bench did not run here (onnxruntime dylib?)"
  echo "  HELD-OUT benchmark (baseline was 329/408 = 80.6%):"
  go run ./cmd/septima-bench training/data/digits/test || warn "held-out bench did not run here"
else
  warn "Go not found here — skipping bench. Pull models/digits.onnx to your local machine and run:"
  echo "      go run ./cmd/septima-bench tanktests              # MUST hold 32/32"
  echo "      go run ./cmd/septima-bench training/data/digits/test   # baseline 80.6%"
  echo "      go test ./tests/                                  # alarm clock 2:47 is the target"
fi

say "Done. New model at models/digits.onnx (run name: ${NAME})."
echo "  If you trained on AWS, scp models/digits.onnx back, then re-verify locally:"
echo "    go run ./cmd/septima-bench tanktests && go run ./cmd/septima-bench training/data/digits/test && go test ./tests/"
echo "  Keep /tmp/digits_baseline_806.onnx as the rollback if anything regresses."
