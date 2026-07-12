# Septima — Agent Notes

Seven-segment display OCR: two-stage local YOLO (panel detector → crop →
digit detector), trained in Python/Ultralytics, run in Go via ONNX Runtime.
See `README.md` for the pipeline diagram, layout table, and build/run commands
— don't duplicate that here. This file is process/pitfalls, not architecture.

## Workflow

- Verification loop for anything touching the models or postprocessing:
  1. `go build ./... && go test ./...`
  2. `go run ./cmd/septima-bench training/data/digits/test` — the held-out
     benchmark (strict exact-match + digits-only exact-match, ignores `.`/`:`
     placement — see "Two benchmark numbers" below).
  3. `go run ./cmd/septima-bench tanktests` — must stay 32/32 (100%). This is
     the primary real use case; treat any regression here as a blocker.
  4. `go test ./tests/...` — the 8 diverse fixtures (clocks, calculator, gas
     pumps, multimeter, RSA token). This is the sensitive canary for model
     drift that the held-out benchmark can mask (see below) — always run it
     after any digit-model retrain, not just when it looks relevant.
- Regenerate benchmark ground truth after any dataset/gate change:
  `training/.venv/bin/python training/datasets/make_benchmark.py` (needs the
  venv's PIL+yaml). `ground_truth.json` is gitignored/regenerated, not
  hand-edited.

## Two benchmark numbers, and why both matter

`septima-bench` reports strict exact-match and a digits-only exact-match
(strips `.`/`:` before comparing). The ~5pp gap between them is mostly
**ground-truth label noise**, not model error — several open datasets
inconsistently include/omit a decimal point for the same physical device
type. Confirmed by eye on multiple cases (model reads the decimal correctly,
GT omits it, and vice versa). Do not hand-edit GT to close this gap — that's
model-favorable cherry-picking. Digits-only is the honest recognition floor;
strict is the honest end-to-end number. Report both when comparing runs.

## Retraining the digit/panel model

- Training runs go to a GPU box (AWS) — local Apple Silicon MPS is too slow
  for a full run (~30h+ for 25 epochs was measured). Use
  `scripts/train_digits_decimal.py` (has preflight checks: correct base
  weights, class count, no test-split leakage, hard-negative data present) or
  `scripts/aws_train_combined.sh`.
- The AWS box has no git remote; changes are propagated by scp/heredoc. This
  has caused real bugs (a script fix present locally but stale on the AWS
  box, silently dropping training data). Before trusting an AWS run's result,
  confirm the AWS box actually has the code change you think it has.
- Always round-trip-verify: after syncing a new `.onnx` back, a local `.pt`
  in `training/runs/` can be stale relative to what's actually live in
  `models/`. Re-export and re-benchmark rather than assuming.
- Change ONE variable per retrain (synth recipe OR hard-negative data OR
  freeze schedule, not several at once) — coupled changes are unablatable
  when a run regresses, and regressions are common enough that you'll want
  to isolate the cause.
- Back up the pre-retrain `models/digits.onnx` (e.g. to `/tmp/`) before
  swapping in a new one. `models/*.onnx` is no longer git-tracked (fetched by
  `scripts/fetch-models.sh`, pinned in `models/MODELS_VERSION`), so
  `git checkout` won't revert it — either restore your `/tmp/` backup, or
  delete the local file and re-run `scripts/fetch-models.sh` to pull the
  last-published version back down.
- Once a retrain passes the full verification loop above, publish it with
  `scripts/publish-models.sh` (creates a new `models-vN` release and
  rewrites `models/MODELS_VERSION`) and commit the updated pin file — see
  "Model weights" in `README.md`.

## Dead ends (don't re-try without new information)

- **Raising inference resolution** (e.g. 960/1280 imgsz) without retraining
  at that resolution — regresses sharply; the detector is scale-sensitive to
  its training imgsz (640).
- **A global punctuation confidence threshold** (raising or lowering `.`/`:`
  conf uniformly) — monotonic tradeoff, no interior win beyond the current
  per-class floor (`Options.PunctThreshold` vs `ConfThreshold`).
- **Decimal-point synthetic data tuning** (varying dot shape/size/frequency
  in `training/synth/render.py`) — tried twice, regressed real accuracy both
  times even when synthetic val metrics improved. The domain gap between
  synthetic and real decimals didn't close this way.
- **Full fine-tuning for colon support** — works (recovers colon reads) but
  reliably costs ~1-1.5pp digits-only accuracy and breaks at least one hard
  tilted-digit test case, across multiple tried recipes (aggressive, gentler,
  frozen-backbone). The colon needs backbone adaptation; that adaptation is
  what drifts thin/tilted digit precision. No configuration tried has avoided
  this tradeoff — if you find one, it'd be worth revisiting.
- **Same-glyph-confusion targeting** (e.g. tuning specifically for 8↔0,
  6↔4 mixups) — the substitution tail is flat (no pair dominates), so
  this has no concentrated benefit.

## Useful dev tools (all in `cmd/`, not just `septima`/`septima-bench`)

- `septima-diag` — runs one stage on an image, dumps raw detections + an
  annotated PNG. The first thing to reach for when a specific image reads
  wrong; `-panel` flag uses the classical bright-panel finder instead of the
  ML panel model.
- `septima-analyze` — reproduces both localization candidates (full-frame vs
  panel-crop) per benchmark image faithfully against the live selection
  logic, builds the both-wrong set and a char confusion matrix. Keep this in
  sync with any change to `septima.Read`'s selection logic (it must
  replicate the live per-class threshold behavior or its numbers lie).
- `septima-punctsweep` — sweeps a punctuation-only confidence threshold
  in-memory over one detection pass, for confirming/re-confirming the
  "dead end" above without a retrain.
- `septima-labelaudit` — batch-reads training images and compares the
  model's reading against their label boxes, for auditing whether a training
  source's labels are trustworthy before deciding to keep/drop it.

## Repo quirks

- `old/` is the previous GoCV/traditional-CV implementation (gitignored),
  kept only for reference — not part of the active build.
- `preprocess/` (top-level, pure `image` stdlib, no GoCV/OpenCV) is a
  ssocr-style optional pipeline (crop/rotate/mirror/invert/grayscale/
  threshold/otsu) applied before detection via `Options.Pipeline`. It covers
  core ops only, not ssocr's full flag surface — the old GoCV versions of the
  rest (shear, morphology, CLAHE, etc.) are in `old/preprocess/*.go` for
  reference if ever extended.
- `models/*.onnx` and `internal/ortlib/lib/*` are **not** tracked in git —
  they're fetched by `scripts/fetch-models.sh` / `scripts/fetch-ortlib.sh`
  (models pinned by tag+sha256 in `models/MODELS_VERSION`, ORT lib pulled
  straight from the upstream onnxruntime GitHub release). Run both before
  `go build`/`go test` on a fresh clone, or the build will fail (models) or
  `go:embed` will fail (ORT lib on the platforms it's embedded for). This
  replaced Git LFS, which was burning the free GitHub LFS bandwidth quota on
  every CI checkout.
