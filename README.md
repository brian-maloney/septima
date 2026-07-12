# septima

Seven-segment display recognition in Go, using a local two-stage YOLO pipeline.
Inference runs on CPU or GPU via ONNX Runtime; models are trained offline with
Ultralytics. This is a rewrite of an earlier GoCV/traditional-CV approach (now
in the git-ignored `old/`), which proved brittle.

## Pipeline

```
image ─▶ digit detector (YOLO) on full frame ─────────────┐
    └─▶ panel detector (YOLO) ─▶ crop + pad ─▶ digit detector ─┴─▶ pick higher
        mean-confidence candidate ─▶ assemble (cluster rows by y, sort L→R,
        map class→char) ─▶ Result
```

- **Adaptive localization** runs the digit detector twice — once on the full
  frame and once on a detected-panel crop — and keeps whichever reading has the
  higher *mean* detection confidence. The crop rescues a small panel in a large
  frame (e.g. the tank gauge in a 1920×1080 photo); the full frame wins for
  already-framed displays, where cropping would over-scale the glyphs. Mean (not
  summed) confidence is used so a candidate can't win by padding its reading with
  weak phantom detections.
- **Digit detection** finds each glyph (`0-9 . : -`).
- **Assembly** groups detections into rows and reads them left-to-right. The tank
  panel's split 4-digit field (`12 | 15` → `1215`) falls out of x-ordering.

## Layout

| Path | Purpose |
|------|---------|
| `septima.go`, `result.go`, `options.go` | Public API (`ReadFile`/`Read` → `Result`) |
| `internal/onnx` | ONNX Runtime session wrapper (binding wired at M3) |
| `internal/imageproc` | Letterbox, CHW tensor, crop (pure Go, no OpenCV) |
| `internal/detect` | YOLO decode + NMS, panel/digit model wrappers |
| `internal/assemble` | Row clustering, left-to-right read, string build |
| `internal/ortlib` | Embedded per-platform ONNX Runtime shared library (fetched, not tracked — see `scripts/fetch-ortlib.sh`) |
| `models/` | `classes.json` (tracked) + `panel.onnx`/`digits.onnx` (fetched, not tracked — see "Model weights") — also embedded into `cmd/septima` |
| `scripts/fetch-models.sh`, `scripts/fetch-ortlib.sh`, `scripts/publish-models.sh` | Fetch/publish the model weights and ONNX Runtime lib (see "Model weights") |
| `cmd/septima` | CLI: read one image (self-contained; no `models/` dir or `SEPTIMA_ORT_LIB` needed) |
| `cmd/septima-bench` | Eval against a dir's `ground_truth.json` |
| `training/` | Python (Ultralytics) data prep, training, ONNX export |
| `tests/`, `tanktests/` | Fixtures + ground truth (tanktests = primary use case) |

## Building & running (Go)

`models/*.onnx` (trained weights) and `internal/ortlib/lib/*` (the ONNX
Runtime shared library) aren't checked into git — they're fetched on demand:

```sh
scripts/fetch-models.sh   # models/panel.onnx, models/digits.onnx — from the
                           # GitHub release pinned in models/MODELS_VERSION
scripts/fetch-ortlib.sh   # ONNX Runtime shared lib for the host platform —
                           # from the upstream onnxruntime GitHub release
```

Run both before `go build`/`go test`; CI does the same (see
`.github/workflows/`). Model weights are versioned independently of the
software (retraining doesn't imply a new software release) — see "Model
weights" below for how that's tracked and how to publish a retrain.

Inference uses [`onnxruntime_go`](https://github.com/yalue/onnxruntime_go), which
loads the ONNX Runtime **shared library at runtime** (cgo build; no link-time
dependency). `cmd/septima` embeds both the panel/digit ONNX models and the
platform's ONNX Runtime shared library via `go:embed` (see `models/embed.go` and
`internal/ortlib`), so the built binary is self-contained — no `models/` directory
or `SEPTIMA_ORT_LIB` needed at runtime. Every other `cmd/` tool (`septima-bench`
etc.) still resolves models/ORT from disk as before: the engine auto-discovers the
library from the Python `onnxruntime` wheel in `training/.venv` (or a copy under
`models/`), walking up from the working directory. Override explicitly with
`SEPTIMA_ORT_LIB=/path/to/libonnxruntime.dylib` and/or `-models DIR` /
`SEPTIMA_MODEL_DIR`, which take precedence over the embedded defaults too. CPU
works everywhere; CoreML/CUDA optional.

```sh
go build ./...
go test ./internal/...
# opt-in ORT runtime smoke test (load → run → output shape):
SEPTIMA_TEST_ONNX=/path/to/any-yolo-640.onnx go test ./internal/onnx -run TestRunModel -v

go run ./cmd/septima path/to/image.jpg                     # self-contained, no flags needed
go run ./cmd/septima -models models path/to/image.jpg      # override with a different models dir
go run ./cmd/septima -version                              # prints "dev" outside a tagged release build
go run ./cmd/septima-bench tanktests                       # exact + per-char accuracy
```

## Model weights

`models/panel.onnx` and `models/digits.onnx` are published as GitHub release
assets, separately from the software's `vX.Y.Z` tags — retraining doesn't
imply a new software release. `models/MODELS_VERSION` pins the release tag
and a sha256 per file:

```
TAG=models-v1
digits.onnx=sha256:...
panel.onnx=sha256:...
```

`scripts/fetch-models.sh` reads that pin, downloads from
`github.com/brian-maloney/septima/releases/download/<TAG>/<file>`, and
verifies the checksum; it no-ops if the on-disk file already matches. After a
retrain that passes the verification loop in `AGENTS.md`, publish it with:

```sh
scripts/publish-models.sh [--notes-file FILE]
```

This creates the next `models-vN` release, uploads the two `.onnx` files, and
rewrites `models/MODELS_VERSION` with the new tag/checksums — review and
commit that file (and `models/classes.json`, if the class list changed) as
one commit. `models/classes.json` itself stays tracked directly in git (it's
a small text file, not a build artifact), so keep it in sync with whichever
model version is pinned.

## Training (Python, offline)

```sh
python3 -m venv training/.venv && source training/.venv/bin/activate
pip install -r training/requirements.txt
# training/datasets/prepare.py  — merge public 7-seg YOLO datasets
# training/synth/render.py       — synthetic 7-seg generator (LCD + LED)
# training/train_digits.py        — Ultralytics YOLO-nano, device=mps
# training/export_onnx.py         — export to models/*.onnx
```

Data strategy (priority): public 7-seg YOLO datasets → hand-annotated real images
→ synthetic renders that fill the LCD dark-on-light look and the exact tank framing.

## Status

- **M0 scaffold — done.** Pure-Go logic (letterbox, NMS, row assembly) implemented
  and unit-tested.
- **M1 data — done.** 6,325 digit + 1,501 panel training images (public + synthetic).
- **M2 train — done.** `digits.onnx` trained (YOLO11n, val mAP50 0.991), then
  fine-tuned on real tank crops via `cmd/septima-annotate`.
- **M3 Go inference — done.** `onnxruntime_go` two-stage pipeline.
- **M4 tank — done. 32/32 (100%) exact, 100% char** on `tanktests/` via classical
  bright-panel crop + fine-tuned digit model. (ML `panel.onnx` deferred to M5.)
- **M5 generalize to `tests/`** — next: clocks/calculator/pumps need an ML panel
  detector and end-to-end decimals/colons.

## License

MIT — see [LICENSE](LICENSE). `cmd/septima` embeds ONNX Runtime (MIT,
© Microsoft Corporation) and its bundled third-party components — see
[NOTICE.md](NOTICE.md) for attribution.
