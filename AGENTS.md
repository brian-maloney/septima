# Septima — Agent Handoff Notes

## Project overview

**Septima** is a Go-native seven-segment display OCR library and CLI.  It is built on top of [gocv](https://github.com/hybridgroup/gocv) (OpenCV 4 CGO bindings) and targets two reference implementations:

- **ssocr** (C/Imlib2): full preprocessing pipeline (dilation, erosion, gray-stretch, shear, mirror, adaptive thresholding, multiple charsets, decimal/colon/minus recognition).
- **SegoDec** (Python/OpenCV): hands-off, CLAHE-based, with fixed segment-sample points and lookup table.

Goal: as reliable and hands-off as SegoDec, as full-featured as ssocr, fully Go-native.

---

## Repository layout

```
septima/
├── go.mod / go.sum            # github.com/vond/septima, gocv v0.43.0
├── septima.go                 # ReadFile / Read / ReadMat public API + full pipeline
├── options.go                 # Options struct + functional option helpers
├── result.go                  # Result, Row, Digit, DebugInfo types
├── preprocess/                # ssocr-compatible preprocessing ops (all implement Op interface)
│   ├── op.go                  # Pipeline type, Apply method
│   ├── threshold.go           # MakeMonoOp, IterThresholdOp, DynamicThresholdOp, OtsuThresholdOp, RGB/channel ops
│   ├── morphology.go          # DilationOp, ErosionOp, OpeningOp, ClosingOp, RemoveIsolatedOp
│   ├── geometry.go            # CropOp, RotateOp, ShearOp, MirrorOp, WhiteBorderOp
│   ├── color.go               # GrayscaleOp, InvertOp, GrayStretchOp
│   └── filter.go              # SetPixelsFilterOp, KeepPixelsFilterOp, CLAHEOp
├── detect/                    # Display location + digit segmentation
│   ├── display.go             # FindDisplayROI — contour-based, rectangularity filter
│   ├── perspective.go         # RectifyPerspective — Hough lines + warpPerspective
│   ├── polarity.go            # DetectPolarity — Otsu-based white-fraction on pre-CLAHE gray
│   ├── rows.go                # SplitRows — horizontal projection, density filter, forceBandCount
│   └── digits.go              # SegmentDigits — CC + vertical merge + colon detection
├── decode/                    # Segment mask → character
│   ├── lookup.go              # segList (ordered, deterministic), nearestMatch, exactMatch
│   ├── segments.go            # SampleSegments — 7 zone windows on 30×50 canvas, density threshold 0.40
│   ├── charset.go             # CharsetFull/Digits/Decimal/Hex/TTRobot, Decode()
│   └── special.go             # AspectClassify (1/- by h/w), AsciiArtSegments
├── dnn/                       # ONNX fallback stub (no trained model yet)
│   └── classifier.go
├── profile/                   # Named display presets
│   ├── profiles.go            # Get, AutoSelect, registry
│   └── builtin/*.json         # alarm_clock, microwave_clock, multimeter, gas_pump,
│                              # tank_gauge, security_token, calculator, generic
├── cmd/septima/main.go        # CLI — ssocr-compatible flags + positional pipeline ops
├── cmd/septima-bench/main.go  # Aggregate accuracy runner
└── tests/
    ├── ground_truth.json      # 8 test images with expected values, display types, rows
    ├── *.jpg / *.jpeg / *.webp
    └── septima_test.go        # Table-driven: TestRecognizeAuto + TestRecognizeHinted
```

---

## Pipeline (septima.go: ReadMat)

```
full image
  → FindDisplayROI           (contour-based, rectangularity ≥ 0.35, h ≥ 6% imgH)
  → PadROI(5px)
  → working = ROI crop

  → autoPreprocess(working)  (CLAHE clipLimit=3.0, 8×8 tiles, single-channel)
  → rawGray = toGray(working) (pre-CLAHE, for polarity detection only)
  → DetectPolarity(rawGray)  (Otsu on rawGray: white >50% → dark-on-light)
  → binaryRaw = adaptiveThreshold(gray)
      ├── IterThresholdOp (k-means, default)
      └── fallback: AdaptiveThreshold if white ratio outside [8%, 92%]
  → polarity inversion: DarkOnLight → BitwiseNot; else clone
  → zeroBorder(binaryInv)    (hard margin Rows/30 + CC-based flood to remove bezel)
  → morphClean(binary)
      ├── RemoveIsolatedOp
      └── ClosingOp{N:1} only when Rows ≥ 60 (skip for tiny VFD ROIs)

  → SplitRows(cleaned, expectedRows)
      ├── horizontal projection
      ├── findBands (threshold = cols/100)
      ├── mergeBands (gap < 20% avg height)
      ├── filterByDensity (keep bands ≥ 30% of max-density band)
      └── forceBandCount (peak-projection score, not total pixels)

  for each row band:
    → SegmentDigits(rowMask, rowOffset, digitOpts)
        ├── ConnectedComponentsWithStats
        ├── medianCompHeight (comps > rowH/6)
        ├── classify: IsDecimal if h < DecHRatio×medH AND w/h < DecWRatio×3
        ├── mergeXOverlapping (digitComps only, requires ≥40% x-overlap)
        ├── second-pass reclassify against post-merge medianH
        ├── detectColons (pair two decimals: same x, dy in [medH/8, medH])
        └── detectColonsSingleDot (single decimal + white pixels at +medH/3)
    → post-height filter: remove non-decimal/non-colon boxes < 35% maxH
    → per digit:
        ├── IsDecimal → '.' (skip if charset=digits)
        ├── IsColon   → ':' (skip if charset≠full)
        ├── AspectClassify: h/w > oneRatio(4.0) → '1'
        └── NormalizeDigitImage → SampleSegments(30×50, 7 zones, threshold 0.40)
            → exactMatch → nearestMatch (deterministic ordered segList)
            → Decode(charset)
            → DNN fallback if confidence < 0.60 (no model yet)
```

---

## Key design decisions & bug history

### Polarity detection
- **Wrong approach**: binary white-fraction on CLAHE-enhanced image — CLAHE over-equalizes uniform backgrounds, creating artifical dark pixels → white fraction can be misleadingly low.
- **Right approach**: Otsu on the **pre-CLAHE raw gray** (`rawGray`). For medium-gray LCD backgrounds (tank gauge ~130-150) Otsu correctly returns >50% white → dark-on-light.

### Border removal
- After polarity inversion, the LCD bezel/frame inverts to white and creates a large connected component spanning the full image width. This merges all digits into one CC component.
- **Solution**: `zeroBorder` — hard-zero a margin (`Rows/30`), then CC-scan the inner edge to flood-fill any remaining border-touching white regions.
- **Margin size matters**: Too large (Rows/20) clips digit segments near the ROI edge (e.g. gas pump "9" top-right bar). Too small misses the bezel. Rows/30 ≈ 3.3% is the current sweet spot.

### Morphological closing
- Needed to bridge tiny intra-segment gaps within a single LCD segment bar (multimeter, tank).
- **But** for small VFD displays (microwave ROI ~62px tall), the 3×3 kernel merges the two colon dots (3px each, ~10px apart) into one 10px blob — too tall to be classified as decimal.
- **Solution**: skip closing when `Rows < 60`.

### Segment sampling threshold
- Global threshold 0.40 works for most displays.
- Known failure: gas pump 2's italic "9" has segment b density ≈ 0.206 (below 0.40) → decoded as "5". No single global threshold resolves this without breaking the multimeter "0" (segment g at 0.267 should be OFF).
- **No fix yet**. DNN fallback is the right solution but requires a trained model.

### Colon detection
- `detectColons`: two stacked decimal dots → merged colon.  `minDy = medH/8` (was `/5` — too large for big LED displays where dots are 40px apart out of 300px digit height).
- `detectColonsSingleDot`: handles VFD microwave case where lower colon dot merges into adjacent "2" digit's CC component. Samples white pixels at `cy + medH/3`.

### Determinism
- `nearestMatch` previously iterated a `map[byte]rune` (random Go map order) → different results on ties. Fixed by using an ordered `[]segEntry` slice (`segList`), digits first.

### minArea scaling
- `minArea = Rows*Rows/2000` (not `pixels/800`).  Reason: decimal dots scale with digit height, not total image area. A 1310×785 ROI has pixels/800=1285 which filters the gas pump decimal (area=526).

### Multi-row value encoding
- `ground_truth.json` `value` field for multi-row displays (gas pump) now contains the full joined text: `"29.29\n13.318"` so that `got.Text == c.Value` works correctly in the test.

---

## Current test status (as of 2026-06-14, third session)

| Image | Display | Expected | Auto | Hinted |
|---|---|---|---|---|
| 2013meax1g981.jpg | multimeter | 0.68 | ✅ 0.68 | ✅ 0.68 |
| images.jpeg | microwave_clock | 21:24 | ✅ 21:24 | ✅ 21:24 |
| jai5qyznvjky.jpg | gas_pump | 29.29\n13.318 | ✅ 29.29\n13.318 | ✅ 29.29\n13.318 |
| spr-dreamsky….jpeg | alarm_clock | 2:47 | ✅ 2:47 | ✅ 2:47 |
| 0502.jpg | tank_gauge | 1077 | ❌ 1377 | ✅ 1077 |
| dVv50.jpg | security_token | 156311 | ❌ -04- | ❌ 68 |
| getting-weird….webp | gas_pump | 86.47\n14.659 | ❌ :8- | ❌ .8-\n2 |
| 68f79706….jpeg | calculator | 123456789012 | ❌ 4- | ❌ 4- |

**Auto: 4/8 — Hinted: 5/8** (was 4/8 auto and 4/8 hinted in the second session).
Tank gauge hinted now passes; tank gauge auto is closer (one-digit error) but still failing.

### Fixes that landed in the second session

- **Second-pass decimal reclassification in `detect/digits.go`** — after `mergeXOverlapping`, recompute the maximum post-merge digit height; any leftover merged "digit" component that is short enough to be a decimal is re-bucketed.  Guarded by `postMergeMaxH > medH*13/10` so it only fires when digits really did split into half-halves (otherwise it produced spurious decimals on noisy displays).  Fixes gas pump 2 auto's missing "13.318" decimal.
- **`isDecimalShape` helper** — replaces the inline `aspectRatio < DecWRatio*3` check with a clamp of the upper aspect bound at 1.5, so square colon dots (aspect ≈ 1.0) are recognized as decimals even when a profile uses a small `DecWRatio`.  Necessary for alarm-clock colon detection.
- **`shouldMerge` Y-containment merge** — `mergeXOverlapping` now also merges a narrow stub whose Y span is ≥80% contained inside a wider neighbour with non-negative X overlap.  Fixes a-half-of-"2" + isolated b-segment being treated as two boxes ("2" + "1").
- **Profile-driven `HasDecimal`/`HasColon` filter (`septima.go`)** — `applyProfile` propagates these into `Options`, and the per-row pipeline drops unpaired decimals when the profile says `has_decimal=false` (alarm-clock AM/PM dots, alarm-bell icons, etc.).  Auto-mode equivalent: when a row contains a colon, drop any remaining isolated decimals (clock-style heuristic).
- **`detect/rows.go` `filterByStructure`** — for auto-mode (`expectedRows==0`) bands, require the tallest CC to be ≥40% of band height AND ≥30% of the best band's tallest CC.  Discards scatter-noise bands that pass the density filter.  Fixes alarm-clock auto's phantom second row.

### Fixes that landed in the third session (tank gauge hinted)

- **`findDigitRowYRange` + per-row dense-Y clamp (`septima.go`)** — compute the row band's dense y-range from the aggregate horizontal projection (longest run of rows whose white-pixel count is ≥ `maxP/8`, clamped to ≥2).  Clamp each digit bbox's y-range to this strip and drop colon/decimal boxes that fall entirely outside it.  Fixes the tank-gauge "0" CC, which connects upward through a thin pixel path to the "CAPACITY" label after closing — the over-tall CC squeezed the digit into the bottom of its normalized canvas and made segment sampling fail (mask 0x10 → '1').  After clamp the mask is 0x3F → '0'.  Also drops the bright specks above the digit row that the colon detector pairs into spurious colons.
- **X-proximity filter for cols/decimals (`septima.go`)** — a real colon or decimal sits next to a digit.  Drop col/dec boxes whose centre-x is more than one median digit width outside the digit cluster.  Removes the speck-pair colon on the left margin of the tank gauge ROI (which sits inside the dense y-range but far from the digit cluster).
- **Y-alignment filter for digits (`septima.go`)** — drop non-decimal/non-colon boxes whose centre-y is more than `medianH/4` from the median digit centre.  Removes the "GAL" CC, which trims to about 70 % of digit height (so the 35 % rule keeps it) but rides ~40 px above the digit centreline.

### Files touched

- Second session: `detect/digits.go`, `detect/rows.go`, `septima.go`, `options.go`, `AGENTS.md`.
- Third session: `septima.go` (added `abs`, `findDigitRowYRange`, dense-Y clamp + drop, X-proximity filter, Y-alignment filter), `AGENTS.md` (this update).

No unit-test regressions: all 42 pure-Go tests still pass.  No integration-test regressions: every previously-passing image still passes; tank-gauge hinted is now a new pass.

---

## Unit test suite (as of 2026-06-14)

42 pure-Go unit tests across three files (no image files required):

| File | Tests | What it covers |
|---|---|---|
| `decode/decode_test.go` | 12 | `hammingByte`, exact/nearest segment match (all 17 entries), `Decode` with every charset, `AspectClassify` edge cases, `AsciiArtSegments`, segment constant integrity |
| `profile/profile_test.go` | 10 | `Get` for all 8 builtins, unknown-name fallback to generic, detailed field checks, `scoreProfile` (aspect + digit count), `AutoSelect` |
| `options_test.go` | 18 | `defaultOptions`, every `With*` functional option, `applyProfile` for multimeter/microwave_clock/unknown, `toDecodeCharset`, combined option application |

Run with:
```bash
PKG_CONFIG_PATH=$(brew --prefix opencv)/lib/pkgconfig go test ./decode/ ./profile/ github.com/vond/septima -timeout 60s
```

---

## Remaining work / known issues

### Alarm clock (spr-dreamsky…jpeg)
- Auto "21:.47" — colon correct, digits correct, but extra "1" (from PM indicator or alarm bell icon) and extra "." noise.
- Hinted "21147" — colon missing vs auto. Appears the single-band forced selection via `expected_rows=1` is losing the colon detection. The auto's multi-band processing surfaces the colon from band 0; the hinted single-band result doesn't pair the colon dots.
- The "7" is now decoded correctly (was "1" before; fixing zeroBorder margin fixed it).
- **Next step**: debug why `detectColons` (which successfully fires in auto) fails in the hinted single-band pass. Likely the `medH2` from merged digits is different when only one band is selected.

### Tank gauge (0502.jpg)
- **Hinted now passes "1077"** — fixed in the third session via the dense-Y clamp + X-proximity + Y-alignment filters.
- Auto returns "1377" — one-digit error: the "0" CC's right edge merges with a noise speck (via a thin pixel path absent in the hinted band crop), widening the bbox to `w=59` vs the hinted `w=50`.  The wider canvas shifts the segment-sampling windows so the right-vertical zone misses the "0" stroke and the decoder lands on "3" (mask 0x0d).
- **Next step**: try cropping the row mask to the dense y-range *before* `SegmentDigits` runs.  A naive crop regressed microwave (lost the "1" of "21:24") — likely because the small ROI's per-row counts are themselves marginal against `maxP/8`.  A more conservative crop (e.g., only when the trim removes ≥30 % of the band height, or only when the row band is large) might give auto-mode tank without touching microwave.

### Gas pump 1 (getting-weird….webp)
- ROI finds the correct display area, but heavy glass reflections destroy the binary — adaptive threshold fires but still noisy.
- Only "8" typically decoded correctly from "86.47".
- **Next step**: try applying CLAHE with much stronger parameters, or apply shear to correct the italic font before thresholding.

### Gas pump 2 auto (jai5qyznvjky.jpg)
- Row 1 "29.29" passes. Row 2 "13.318" is missing the decimal → "13318".
- The decimal (area=526) is above `minArea=308` and `h=23 > minH=13`, so it should be found. The second-pass reclassification (using post-merge medianH≈188) should mark it IsDecimal=true (23 < 0.20×188=37.6). Somehow this works in hinted (expected_rows=2) but not in auto.
- **Next step**: add debug box printing to row 2 in auto mode to see if the decimal is classified as IsDecimal.

### RSA SecurID token (dVv50.jpg)
- Small LCD token display. Bar graph on the left must be excluded.
- Currently getting only "0" or similar — ROI or polarity issue.
- **Next step**: check the ROI and binary debug images.

### Calculator (68f79706….jpeg)
- 12 digits, perspective tilt, hand in frame, glare.
- Currently getting "2" and "4" (a few correct digits in a sea of noise).
- **Next step**: implement perspective correction (`RectifyPerspective` exists but is not yet wired into the main pipeline for the calculator case). The `detect/perspective.go` code is present.

### DNN fallback
- `dnn/classifier.go` stub is present. Path: `dnn.SetModelPath(path)`, then `dnn.Classify(digitGray)`.
- No trained model exists. To train: use `internal/render` (not yet implemented) to generate synthetic 7-seg digit images at various fonts/slants/noise, export to 28×28 grayscale PNGs, train a small CNN (or load a pre-trained ONNX 7-seg classifier).
- The gas pump 2 "9"→"5" (segment b density=0.206) is the primary motivating case.

---

## Build & run

```bash
# Dependencies
brew install opencv pkg-config   # macOS; see gocv docs for Linux/Windows
go env -w CGO_CXXFLAGS_ALLOW='.*'

# Build
PKG_CONFIG_PATH=$(brew --prefix opencv)/lib/pkgconfig go build ./...

# Run tests
PKG_CONFIG_PATH=$(brew --prefix opencv)/lib/pkgconfig go test ./tests/ -v -timeout 180s

# CLI (ssocr-compatible)
go run ./cmd/septima tests/images.jpeg
go run ./cmd/septima -D /tmp/debug --profile microwave_clock tests/images.jpeg

# Aggregate bench
go run ./cmd/septima-bench tests/
```

---

## Profile notes

Profiles in `profile/builtin/*.json` populate `Options` defaults. Rules:
- `dec_h_ratio: 0.0` / `dec_w_ratio: 0.0` means "inherit default" (not zero-override) — handled by `if p.DecHRatio > 0` in `applyProfile`.
- Clock displays (alarm_clock, microwave_clock) need `charset: "full"` to allow ':' output.
- `expected_rows: 1` triggers `forceBandCount` which selects the single peak-densest band.

Known profile charset issue: `applyProfile` in `septima.go` has no `case "full":` — the "full" value is intentionally a no-op because `CharsetFull` is already the default. Do not add a case for it unless you want to explicitly enforce re-setting it after user opts.
