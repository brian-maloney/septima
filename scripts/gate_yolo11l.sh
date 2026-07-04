#!/usr/bin/env bash
#
# No-regression gate for the yolo11l digit-model retrain.
#
# Thresholds are the LIVE numbers measured on main immediately before this run
# (commit 6fb5432, 2026-07-03), so the ADOPT/REJECT decision is not cherry-picked
# after seeing yolo11l's results:
#   tank              32/32   (100.0%)
#   held-out strict    333/403 (82.6%)
#   held-out digits-only 347/403 (86.1%)
#   tests/ exact       7/9    (77.8%)  (knownHard: microwave colon, alarm-bell-icon clock)
#
# Run this with the candidate model at models/digits.onnx (put the downloaded
# yolo11l weights there first, or let aws_train_yolo11l.sh call it directly).
#
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BENCH=/tmp/gate_yolo11l_bench.txt
TESTS=/tmp/gate_yolo11l_tests.txt

say(){ printf '\n== %s\n' "$*"; }
pass(){ printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail(){ printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0

say "Model under test: $(ls -la models/digits.onnx | awk '{print $5, $NF}')"

# G1 tank ----------------------------------------------------------------------
say "G1  tank (want 32/32)"
TANK=$(go run ./cmd/septima-bench tanktests 2>/dev/null | awk -F'[ /]' '/^exact:/{print $2"/"$3}')
[ "$TANK" = "32/32" ] && pass "tank $TANK" || fail "tank $TANK (want 32/32)"

# G2/G3 held-out benchmark ------------------------------------------------------
say "G2/G3  held-out benchmark (want strict >=333/403, digits-only >=347/403)"
go run ./cmd/septima-bench training/data/digits/test >"$BENCH" 2>/dev/null
STRICT_N=$(awk -F'[ /]' '/^exact:/{print $2}' "$BENCH")
DIGO_N=$(awk -F'[ /]'  '/^digits-only exact:/{print $3}' "$BENCH")
grep -E '^exact:|^digits-only' "$BENCH" | sed 's/^/  /'
[ "${STRICT_N:-0}" -ge 333 ] && pass "G2 strict ${STRICT_N}/403 (>=333)" || fail "G2 strict ${STRICT_N}/403 (<333 = regression)"
[ "${DIGO_N:-0}"   -ge 347 ] && pass "G3 digits-only ${DIGO_N}/403 (>=347)" || fail "G3 digits-only ${DIGO_N}/403 (<347 = digit regression)"

# G4 tests/ ----------------------------------------------------------------------
say "G4  tests/ (want exact >=7/9)"
go run ./cmd/septima-bench tests >"$TESTS" 2>/dev/null
cat "$TESTS" | sed 's/^/  /'
TESTS_N=$(awk -F'[ /]' '/^exact:/{print $2}' "$TESTS")
[ "${TESTS_N:-0}" -ge 7 ] && pass "G4 tests ${TESTS_N}/9 (>=7)" || fail "G4 tests ${TESTS_N}/9 (<7 = regression)"

# Gx GOAL: any of the two knownHard colon cases newly pass ---------------------
say "Gx  knownHard colon cases (GOAL, not required)"
if grep -q 'PASS.*images.jpeg' "$TESTS"; then
  pass "microwave colon now passes"
else
  printf '  \033[33mno change\033[0m microwave colon still fails (may be resolution-limited, not a model question)\n'
fi

# verdict ------------------------------------------------------------------------
say "VERDICT"
if [ "$FAILED" -eq 0 ]; then
  printf '  \033[32mADOPT yolo11l\033[0m — all hard gates passed. Commit models/digits.onnx.\n'
else
  printf '  \033[31mREJECT yolo11l\033[0m — a hard gate failed. Restore baseline: git checkout models/digits.onnx\n'
fi
exit "$FAILED"
