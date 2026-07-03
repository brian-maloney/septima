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
| `models/` | `panel.onnx`, `digits.onnx`, `classes.json` |
| `cmd/septima` | CLI: read one image |
| `cmd/septima-bench` | Eval against a dir's `ground_truth.json` |
| `training/` | Python (Ultralytics) data prep, training, ONNX export |
| `tests/`, `tanktests/` | Fixtures + ground truth (tanktests = primary use case) |

## Building & running (Go)

`models/*.onnx` are tracked via [Git LFS](https://git-lfs.com); install it
(`brew install git-lfs && git lfs install`) before cloning, or run
`git lfs pull` after a plain clone.

Inference uses [`onnxruntime_go`](https://github.com/yalue/onnxruntime_go), which
loads the ONNX Runtime **shared library at runtime** (cgo build; no link-time
dependency). The engine auto-discovers the library from the Python `onnxruntime`
wheel in `training/.venv` (or a copy under `models/`), walking up from the working
directory. Override explicitly with `SEPTIMA_ORT_LIB=/path/to/libonnxruntime.dylib`.
CPU works everywhere; CoreML/CUDA optional.

```sh
go build ./...
go test ./internal/...
# opt-in ORT runtime smoke test (load → run → output shape):
SEPTIMA_TEST_ONNX=/path/to/any-yolo-640.onnx go test ./internal/onnx -run TestRunModel -v

go run ./cmd/septima -models models path/to/image.jpg     # needs models/digits.onnx
go run ./cmd/septima -version                              # prints "dev" outside a tagged release build
go run ./cmd/septima-bench tanktests                       # exact + per-char accuracy
```

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

MIT — see [LICENSE](LICENSE).
