#!/usr/bin/env bash
#
# No-regression gate for the run #6 colon-geometry model.
#
# Run #5 showed colon synth can REGRESS things that already work: it read the
# standard clock-radio colon "12:17" as "12-17", and cost -1.2pp held-out strict
# (punctuation churn), while the microwave (its target) fails on DIGITS anyway.
# So run #6 is ADOPTED ONLY IF it holds everything that works AND ideally helps a
# real colon case.  This script checks the model currently at models/digits.onnx
# (put the downloaded run #6 there first) against fixed thresholds decided BEFORE
# the run, so the decision is not cherry-picked after seeing results.
#
# Baseline reference (committed model): tank 32/32; held-out 326/403 (80.9%)
# strict / 347/403 (86.1%) digits-only; standard clock colon reads "12:17".
#
# ADOPT run #6 iff ALL hard gates pass:
#   G1 tank            == 32/32                         (never regress the primary use case)
#   G2 digits-only     >= 86.1%   (>= 347/403)          (hold the run #5 digit-preservation win)
#   G3 strict          >= 80.4%   (>= 324/403)          (no worse than run #5; ideally >= baseline)
#   G4 clock colon     == "12:17" on the digits-only crop (standard colon NOT regressed)
#   G5 tmnsi colon     >= 57/59                          (memorised colon set not broken)
# GOAL (not required, this is the point of the run):
#   Gx microwave       == "21:24"  (or at least the colon ':' appears)
#
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=training/.venv/bin/python
CLOCK=tests/Digital-clock-radio-basic_hf.jpg
CROP=/tmp/gate_clock_crop.png
BENCH=/tmp/gate_bench.txt

say(){ printf '\n== %s\n' "$*"; }
pass(){ printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail(){ printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0

say "Model under test: $(ls -la models/digits.onnx | awk '{print $5, $NF}')"

# G1 tank ----------------------------------------------------------------------
say "G1  tank (want 32/32)"
TANK=$(go run ./cmd/septima-bench tanktests 2>/dev/null | awk -F'[ /]' '/^exact:/{print $2"/"$3}')
[ "$TANK" = "32/32" ] && pass "tank $TANK" || fail "tank $TANK (want 32/32)"

# G2/G3 held-out benchmark -----------------------------------------------------
say "G2/G3  held-out benchmark"
go run ./cmd/septima-bench training/data/digits/test >"$BENCH" 2>/dev/null
STRICT_N=$(awk -F'[ /]' '/^exact:/{print $2}' "$BENCH")
DIGO_N=$(awk -F'[ /]'  '/^digits-only exact:/{print $3}' "$BENCH")
grep -E '^exact:|^digits-only' "$BENCH" | sed 's/^/  /'
[ "${DIGO_N:-0}"   -ge 347 ] && pass "G2 digits-only ${DIGO_N}/403 (>=347)" || fail "G2 digits-only ${DIGO_N}/403 (<347 = digit regression)"
[ "${STRICT_N:-0}" -ge 324 ] && pass "G3 strict ${STRICT_N}/403 (>=324)"    || fail "G3 strict ${STRICT_N}/403 (<324)"

# G4 clock colon (digits-only crop, indicator excluded) ------------------------
say "G4  standard clock colon not regressed (want 12:17)"
$PY -c "from PIL import Image; Image.open('$CLOCK').convert('RGB').crop((560,520,1260,820)).save('$CROP')"
CLK=$(go run ./cmd/septima "$CROP" 2>/dev/null | tr -d '[:space:]')
[ "$CLK" = "12:17" ] && pass "clock colon '$CLK'" || fail "clock colon '$CLK' (want 12:17 -> standard colon REGRESSED)"

# G5 tmnsi colon subset --------------------------------------------------------
say "G5  tmnsi colon set (want >=57/59)"
TM=$(grep -E 'tmnsi_colon' "$BENCH" | grep -oE 'strict +[0-9]+/[0-9]+' | awk '{print $2}')
TM_N=${TM%%/*}
[ "${TM_N:-0}" -ge 57 ] && pass "tmnsi strict $TM" || fail "tmnsi strict $TM (want >=57/59)"

# Gx microwave GOAL ------------------------------------------------------------
say "Gx  microwave 21:24 (GOAL, not required)"
MW=$(go run ./cmd/septima tests/images.jpeg 2>/dev/null | tr -d '[:space:]')
if [ "$MW" = "21:24" ]; then pass "microwave '$MW' <- run #6 achieved its target"
elif printf '%s' "$MW" | grep -q ':'; then printf '  \033[33mPARTIAL\033[0m microwave %s (colon present)\n' "$MW"
else printf '  \033[33mMISS\033[0m microwave %s (no colon; may be resolution-limited)\n' "$MW"; fi

# verdict ----------------------------------------------------------------------
say "VERDICT"
if [ "$FAILED" -eq 0 ]; then
  printf '  \033[32mADOPT run #6\033[0m — all hard gates passed. Commit models/digits.onnx.\n'
else
  printf '  \033[31mREJECT run #6\033[0m — a hard gate failed. Restore baseline: git checkout models/digits.onnx\n'
fi
exit "$FAILED"
