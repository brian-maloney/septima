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

## Current test status (as of 2026-06-16, seventh session)

| Image | Display | Expected | Auto | Hinted |
|---|---|---|---|---|
| 2013meax1g981.jpg | multimeter | 0.68 | ✅ 0.68 | ✅ 0.68 |
| images.jpeg | microwave_clock | 21:24 | ✅ 21:24 | ✅ 21:24 |
| jai5qyznvjky.jpg | gas_pump | 29.29\n13.318 | ✅ 29.29\n13.318 | ✅ 29.29\n13.318 |
| spr-dreamsky….jpeg | alarm_clock | 2:47 | ✅ 2:47 | ✅ 2:47 |
| 0502.jpg | tank_gauge | 1077 | ✅ 1077 | ✅ 1077 |
| dVv50.jpg | security_token | 156311 | ✅ 156311 | ✅ 156311 |
| getting-weird….webp | gas_pump | 86.47\n14.659 | ❌ 8b1\n011054 | ❌ 8.6.1.\n011054 |
| 68f79706….jpeg | calculator | 123456789012 | ❌ 122456888981:1 | ❌ 1224568981 |

**Auto: 6/8 — Hinted: 6/8** (security token auto now passes as well; the
sixth-session note that the auto path still produced `15631:1` was stale).

Calculator decode improved substantially with perspective-rectification
wiring (see "Fixes that landed in the seventh session" below).  All 12
digits now survive into the cleaned binary, but the digit segmenter only
emits 10 boxes (hinted) / 14 (auto), with `3` misread as `2`, `7` lost
into the adjacent `8` CC, `0` misread as `8`, and the trailing `12` pair
merged into one box.  These are segmentation/decode issues, not ROI
issues.

Gas pump 1 unchanged from sixth session: tilt is 4.6° so the new
perspective gate (15°) doesn't trigger; the glare/italic decode problem
remains.

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

### Fix that landed in the fourth session (tank gauge auto)

- **`shouldMerge` X-overlap branch — stacked-pair area gate (`detect/digits.go`)** — for X-overlapping pairs with no Y-overlap (so they're vertically stacked), additionally require min/max area ratio ≥ 25 %.  Rejects label-text fragments that x-align with a digit but sit clearly above/below it.  Fixes 0502 auto: the "CAPACITY" label fragments above the "0" were merging into its bbox via X-overlap alone, widening the bbox and pushing the segment-sample zones off-target so the "0" decoded as "3".

### Fix that landed in the fifth session (security token LCD ROI)

- **`detect.RefineDisplayROI` + plumbing (`detect/display.go`, `septima.go`)** — after `FindDisplayROI` returns the whole device, search for a tighter LCD sub-rectangle inside it using Canny edges + contour analysis.  Pick the longest-perimeter non-edge-touching contour with aspect 1.5–12 and bbFrac 5–55 %; require `peri ≥ 1000` or `peri ≥ 1.3 × bboxPeri`.  The bbox-perimeter ratio is the key discriminator — an LCD's contour weaves through interior digit edges and is much longer than the bezel rectangle alone, while incidental rectangular sub-features (label stickers, plastic recesses) have peri ≈ bboxPeri so they fail the gate.
  - `septima.go` then shrinks the bbox inward (Y/6, X/12) to exclude the bezel itself.  Without the shrink, polarity inversion makes the bezel WHITE and `zeroBorder` either floods it (taking connected digits) or leaves bezel-noise inside the ROI.
  - When refinement triggers, `septima.go` (a) forces polarity to `DarkOnLight` if `Polarity == Auto` (embedded LCDs are virtually always dark-on-light, and polarity detection on a tight LCD crop is histogram-sensitive — see fourth-session note), and (b) uses a new `zeroBorderHard` (hard margin only, no CC flood-fill) so bezel-edge artefacts don't drag connected digits into oblivion.
  - Non-interference verified: multimeter, microwave, alarm_clock, tank_gauge produce only edge-touching contour candidates → no refinement.  Gas pump 2 has small non-edge candidates with `peri < 300` → fails the peri ≥ 1000 threshold → no refinement.  Calculator's candidates are all edge-touching → no refinement.  Gas pump 1 *does* trigger refinement, and the result is a much cleaner 2-row ROI; the downstream decode is still failing but on a better input.

### Fixes that landed in the sixth session (security token hinted)

- **`splitWideMergedDigits` in `detect/digits.go`** — When polarity inversion
  leaves bezel artefacts that bridge adjacent digits into one wide CC, split
  it via column-projection valleys.  Algorithm: compute per-column white-
  pixel counts inside the CC's bbox; identify contiguous "active" runs
  where count ≥ 30 % of the CC's per-column maximum AND the run is wider
  than medW/3 AND the in-run peak count reaches ≥80 % of the CC's row
  height; each surviving run becomes a sub-CC.  The peak-height floor is
  the key discriminator that rejects bezel-noise sub-peaks while keeping
  real digit strokes (which span full row height).
  - Sub-box padding is just 1 pixel so the resulting sub-box's aspect h/w
    correctly reflects the stroke shape — for split '1' strokes the aspect
    is above OneRatio and AspectClassify handles them directly, avoiding
    the segment-sampling failure mode where a centred-in-bbox stroke lights
    up the wrong zones.
  - For the RSA SecurID image, the trailing "11" merged into one
    181×147 CC.  Column projection finds three peaks (a narrow bezel-noise
    peak that fails the peak-height floor, and the two real '1' strokes
    that pass) → split into two sub-CCs, each decoded as '1'.
- **Edge-bargraph filter in `septima.go`** (post-`SegmentDigits` block) —
  drop the leftmost (or rightmost) non-decimal CC of a row when:
  (a) aspect h/w > opts.OneRatio (so AspectClassify would emit '1'),
  (b) width < 55 % of the upper-half-median digit width, AND
  (c) gap to inward neighbour > 80 % of the upper-half-median digit width.
  The "upper-half-median width" (median of widths above the overall median)
  is biased toward the row's "wide-digit" width and is robust to legitimate
  thin '1' digits in the row and to split sub-boxes — both of those are
  narrow themselves but won't drag the upper-half-median down.  This filter
  removes the time-remaining bar-graph on the RSA SecurID fob without
  breaking the thin '1' digits in tank gauge "1077" or gas pump "13.318",
  because those '1's sit at normal digit spacing from their neighbours
  (gap test fails).

### Fixes that landed in the seventh session (calculator rectification)

- **`detect.RectifyPerspectiveDetailed(src, roi)` in `detect/perspective.go`**
  — new entry point that runs the existing Hough-based quad detection and
  returns both the warped Mat and a `RectifyInfo{Applied, MaxTiltDeg}`.
  Tilt is the maximum *acute* angle between any quad edge and its nominal
  axis (horizontal for top/bottom, vertical for left/right).  The old
  `RectifyPerspective` is kept as a thin wrapper that ignores the info.
  Also fixed a tilt-math bug — the previous `atan2(dy, dx)` formulation
  produced angles up to 180° for backward-oriented edges (e.g., the RSA
  fob measured "172°" rather than the actual ~8°).  The replacement uses
  `atan2(|dy|, |dx|)` so the result is always in `[0, 90]` degrees.
- **`cropToDarkBand(src)` in `septima.go`** — given an image (typically the
  rectified device-level ROI), find the longest contiguous run of rows
  whose row-mean intensity is below `(min+max)/2`.  Pads by `bandH/8`
  and returns `(0, rows)` (no crop) when:
  (a) `max - min < 64` (no clear contrast band), or
  (b) the longest band is shorter than `rows/12` (no plausible LCD strip).
  This is the dark-band heuristic that trims the bright case area away
  from the dark LCD strip after rectification.  Restricted to the
  dark-on-light path — for light-on-dark displays the LCD would be the
  BRIGHTEST band, not the darkest.
- **Perspective fallback in `septima.go`** — after `RefineDisplayROI` runs,
  if it found nothing AND `info.MaxTiltDeg >= 15.0` AND
  `opts.Polarity != PolarityLightOnDark`, replace `working` with the
  rectified Mat, run `cropToDarkBand`, and set `rectified = true`.  The
  15° gate is deliberately high — at 5° the quad detector misfit on
  weakly-tilted bezels and regressed alarm clock (12.36° tilt), tank
  gauge (5.40°), and security_token (25.30°, refinement already handles
  it).  Calculator measures 18.13°, well above the gate.
- The `refined || rectified` path forces `PolarityDarkOnLight` when
  `Polarity == Auto` and uses `zeroBorderHard` instead of the CC-flood
  `zeroBorder`.  The head-on, no-refinement, no-rectification path
  continues to use the existing `zeroBorder`.
- Result on calculator: was `4-` for both auto and hinted; now
  `122456888981:1` (auto) / `1224568981` (hinted).  Not yet passing —
  the cleaned binary shows all 12 digits clearly but the segmenter only
  emits 10–14 boxes (some digits merge with neighbours, others get
  segment-misread).  These are downstream issues, not a rectification
  defect.

### Files touched

- Second session: `detect/digits.go`, `detect/rows.go`, `septima.go`, `options.go`, `AGENTS.md`.
- Third session: `septima.go` (added `abs`, `findDigitRowYRange`, dense-Y clamp + drop, X-proximity filter, Y-alignment filter), `AGENTS.md`.
- Fourth session: `detect/digits.go` (`shouldMerge` stacked-pair area gate), `AGENTS.md`.
- Fifth session: `detect/display.go` (`RefineDisplayROI`), `septima.go` (refinement plumbing, `zeroBorderHard`, polarity hint), `AGENTS.md`.
- Sixth session: `detect/digits.go` (`splitWideMergedDigits`, `splitCCByProjection`), `septima.go` (edge-bargraph filter), `AGENTS.md`.
- Seventh session: `detect/perspective.go` (`RectifyPerspectiveDetailed`, `RectifyInfo`, fixed tilt math), `septima.go` (rectification fallback, `cropToDarkBand`), `AGENTS.md` (this update).

No unit-test regressions: all 42 pure-Go tests still pass.  No integration-test regressions: every previously-passing image still passes.  Security token: from `-..A..8F-`/`08` (auto/hinted) → `115631:`/`115631` — right multiset, wrong order.

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
- **Both auto and hinted now pass "1077"** as of the fourth session.

### Gas pump 1 (getting-weird….webp)
- LCD ROI refinement (fifth session) now isolates the two-row display cleanly, but glare/italics still destroy the per-digit binary so decode is wrong.  Output went from `:8-` / `.8-\n2` (single-row garbage) → `..8.b.1\n011054` / `8.6.1.\n011054` (two rows of nearer-but-still-wrong digits).
- **Next step**: try applying CLAHE with much stronger parameters, or apply shear to correct the italic font before thresholding.  Per-digit confidence + DNN fallback would also help.

### Gas pump 2 auto (jai5qyznvjky.jpg)
- Now passes both auto and hinted as of the fourth session.

### RSA SecurID token (dVv50.jpg)
- **Hinted now passes** as of the sixth session (the `security_token` profile
  uses `charset=digits`, which drops the spurious ':').
- Auto path still fails at `15631:1`.  Root cause for the remaining ':' is
  `detectColonsSingleDot`: at the trailing "1" digit's column position, the
  expected lower-partner position lands inside the digit stroke, which is
  densely white → the speck above is mis-upgraded to a colon.  An earlier
  attempt added "stroke detector" sentinels (require the rows immediately
  above/below the partner sample to be sparse) which dropped the colon but
  unmasked five pre-existing decimal specks (formerly suppressed by the
  `hasColon → drop decimals` heuristic), so the output got worse.  The
  decimals at the top/bottom edges of the auto ROI are within the digit
  cluster x-range and within the dense-y range, so the existing X-proximity
  and dense-y filters don't catch them.
- **Original diagnosis from fifth session was partially wrong**: the
  trailing "11" digits are merged into a single CC at the RAW CC level
  (bezel noise bridges the strokes before `shouldMerge` ever runs), not by
  `shouldMerge`.  Splitting must happen as a CC-level operation rather
  than by tuning the merge rules.  This is what `splitWideMergedDigits`
  does in the sixth-session fix.
- **Next step** (auto): make `detectColonsSingleDot` reject partners that
  are continuous digit strokes (sample above/below the partner band).  THEN
  filter the now-exposed bezel decimals — possibly by (a) dropping decimal
  candidates within a few pixels of the row's y-edge, or (b) clamping the
  per-row dense-y range using a stricter projection threshold (e.g.,
  maxP/4 instead of maxP/8).  Both have to leave gas-pump-style legitimate
  bottom-anchored decimals intact.

### Calculator (68f79706….jpeg)
- 12 digits, perspective tilt (18.13°), hand in frame, glare.
- Seventh session: perspective rectification + `cropToDarkBand` wired in
  as a fallback when refinement fails.  Output went from `4-` to
  `1224568981` (hinted) / `122456888981:1` (auto).  All 12 digits now
  visible in the cleaned binary.
- **Remaining defects (downstream of the binary pipeline):**
  - `3` segment-misread as `2` — likely the top-right segment doesn't
    register cleanly; check `SampleSegments` zone sampling on this digit
    image (`digit_r0_x173.png` when running with `-D`).
  - `7` lost: the `7` and adjacent `8` are a single CC at the bbox level,
    so `splitWideMergedDigits` never sees a wide-enough parent to
    consider splitting.  The bezel-noise bridge connecting them survives
    `morphClean`.
  - `0` segment-misread as `8` — probably the middle segment ghosting from
    bezel noise.
  - Trailing `12` pair merged into one box decoded as `1` (or
    aspect-classified).  Same root cause as the `7`-`8` merge.
- **Next step**: the binary CC bridging between `7`-`8` and `1`-`2` at the
  trailing end likely shows up as thin horizontal artefacts at the
  digit-baseline.  A vertical-projection-based "valley restorer" run on
  the cleaned binary before `SegmentDigits` could re-cut them.  Or invest
  in DNN per-digit classification (`dnn` package scaffolding exists) for
  the segment-misreads.

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
