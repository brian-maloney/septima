// Package septima provides seven-segment display OCR.
// It is designed to be as reliable and hands-off as SegoDec while exposing the
// full preprocessing pipeline of ssocr.
package septima

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gocv.io/x/gocv"

	"github.com/brian-maloney/septima/decode"
	"github.com/brian-maloney/septima/detect"
	"github.com/brian-maloney/septima/dnn"
	"github.com/brian-maloney/septima/preprocess"
	"github.com/brian-maloney/septima/profile"
)

// ReadFile reads an image from disk and recognizes its seven-segment display.
func ReadFile(path string, opts ...Option) (*Result, error) {
	m := gocv.IMRead(path, gocv.IMReadColor)
	if m.Empty() {
		return nil, fmt.Errorf("septima: cannot read image %q", path)
	}
	defer m.Close()
	return ReadMat(m, opts...)
}

// Read recognizes a seven-segment display from a standard Go image.
func Read(img image.Image, opts ...Option) (*Result, error) {
	m, err := gocv.ImageToMatRGB(img)
	if err != nil {
		return nil, fmt.Errorf("septima: image conversion: %w", err)
	}
	defer m.Close()
	return ReadMat(m, opts...)
}

// ReadMat recognizes a seven-segment display from a gocv.Mat.
// The caller retains ownership of m; ReadMat does not close it.
func ReadMat(m gocv.Mat, userOpts ...Option) (*Result, error) {
	opts := defaultOptions()
	for _, o := range userOpts {
		o(&opts)
	}

	// Apply profile defaults (profile settings are overridden by explicit opts)
	if opts.Profile != "" {
		applyProfile(&opts, opts.Profile)
	}

	dbg := &DebugInfo{}
	save := func(name string, mat gocv.Mat) {
		if opts.DebugDir == "" {
			return
		}
		_ = os.MkdirAll(opts.DebugDir, 0o755)
		path := filepath.Join(opts.DebugDir, name+".png")
		gocv.IMWrite(path, mat)
		dbg.Stages = append(dbg.Stages, DebugStage{Name: name, Path: path})
	}

	// ── Stage 1: obtain the working image (ROI crop or full frame) ──────────
	// refined records whether RefineDisplayROI swapped in a sub-rectangle.
	// When true, the working image is the LCD-interior crop and we can both
	//   (a) assume dark-on-light polarity (embedded LCDs almost always are),
	//       useful because polarity detection on a tightly-cropped LCD is
	//       sensitive to histogram shape and may flip the wrong way, and
	//   (b) skip the CC-based zeroBorder flood, which would chase digit-to-
	//       edge bridges and destroy digits along with the bezel artefact.
	var working gocv.Mat
	refined := false
	rectified := false
	if opts.ROI != nil {
		region := m.Region(*opts.ROI)
		working = region.Clone()
		region.Close()
	} else {
		roi := detect.FindDisplayROI(m)
		roi = detect.PadROI(roi, 5, m.Cols(), m.Rows())
		region := m.Region(roi)
		outer := region.Clone()
		region.Close()

		// Optional refinement: look for a tighter LCD sub-rectangle inside
		// the device-level ROI (e.g., RSA SecurID's LCD strip embedded in a
		// plastic body).  Only swap if a clear LCD sub-region is found that is
		// meaningfully smaller than the parent.  Shrink the bbox inward to
		// exclude the LCD bezel.
		if sub := detect.RefineDisplayROI(outer); !sub.Empty() {
			subArea := sub.Dx() * sub.Dy()
			outerArea := outer.Cols() * outer.Rows()
			if outerArea > 0 && subArea*100/outerArea <= 55 {
				shrinkY := sub.Dy() / 6
				if shrinkY < 4 {
					shrinkY = 4
				}
				shrinkX := sub.Dx() / 12
				if shrinkX < 4 {
					shrinkX = 4
				}
				inside := image.Rect(
					sub.Min.X+shrinkX, sub.Min.Y+shrinkY,
					sub.Max.X-shrinkX, sub.Max.Y-shrinkY,
				)
				if inside.Dx() > 40 && inside.Dy() > 20 {
					subRegion := outer.Region(inside)
					working = subRegion.Clone()
					subRegion.Close()
					outer.Close()
					refined = true
				} else {
					working = outer
				}
			} else {
				working = outer
			}
		} else {
			working = outer
		}

		// Fallback: when refinement found no LCD strip and the device is
		// photographed at a strong tilt, attempt perspective rectification
		// of the device-level ROI.  Calculator-style shots (large bezel,
		// thin LCD strip in the middle, no internal contour structure for
		// RefineDisplayROI to latch onto) need this path.  The tilt gate is
		// deliberately high (>= 15°) — moderately-tilted shots like alarm
		// clocks (~12°) decode correctly without rectification, and the
		// quad-detection occasionally fits the wrong lines on those.
		if !refined && opts.Polarity != PolarityLightOnDark {
			rectMat, info := detect.RectifyPerspectiveDetailed(m, roi)
			if info.Applied && info.MaxTiltDeg >= 15.0 {
				// Trim to the dark LCD band: the quad detected here is
				// typically the whole device, so the rectified output
				// still contains bright case pixels above and below the
				// LCD strip that confuse row segmentation.  Restricting
				// to dark-on-light polarity (and only when no LightOnDark
				// hint is set) keeps the dark-band heuristic correct.
				working.Close()
				y0, y1 := cropToDarkBand(rectMat)
				if y0 > 0 || y1 < rectMat.Rows() {
					bandRegion := rectMat.Region(image.Rect(0, y0, rectMat.Cols(), y1))
					working = bandRegion.Clone()
					bandRegion.Close()
					rectMat.Close()
				} else {
					working = rectMat
				}
				rectified = true
			} else {
				rectMat.Close()
			}
		}

		if (refined || rectified) && opts.Polarity == PolarityAuto {
			opts.Polarity = PolarityDarkOnLight
		}
	}
	defer working.Close()
	save("01_roi", working)

	// ── Stage 2: explicit preprocessing pipeline or automatic ──────────────
	var preprocessed gocv.Mat
	if len(opts.Pipeline) > 0 {
		result, err := preprocess.Pipeline(opts.Pipeline).Apply(working)
		if err != nil {
			return nil, fmt.Errorf("septima: pipeline: %w", err)
		}
		preprocessed = result
	} else {
		preprocessed = autoPreprocess(working)
	}
	defer preprocessed.Close()
	save("02_preprocessed", preprocessed)

	// ── Stage 3: grayscale + polarity detection ───────────────────────────────
	// Detect polarity on the raw (pre-CLAHE) grayscale so CLAHE equalization
	// artifacts don't fool the histogram analysis.
	rawGray := gocv.NewMat()
	defer rawGray.Close()
	if working.Channels() > 1 {
		gocv.CvtColor(working, &rawGray, gocv.ColorBGRToGray)
	} else {
		rawGray = working.Clone()
	}

	polarity := opts.Polarity
	if polarity == PolarityAuto {
		dp := detect.DetectPolarity(rawGray)
		if dp == detect.PolarityLightOnDark {
			polarity = PolarityLightOnDark
		} else {
			polarity = PolarityDarkOnLight
		}
	}

	// CLAHE-enhanced gray for thresholding (better contrast for segment edges).
	gray := gocv.NewMat()
	defer gray.Close()
	if preprocessed.Channels() > 1 {
		gocv.CvtColor(preprocessed, &gray, gocv.ColorBGRToGray)
	} else {
		gray = preprocessed.Clone()
	}
	save("03_gray", gray)

	// ── Stage 4: threshold → binary ────────────────────────────────────────
	// ThresholdBinary: pixels above threshold → 255, below → 0.
	binaryRaw := adaptiveThreshold(gray)
	defer binaryRaw.Close()
	save("04_binary_raw", binaryRaw)

	// ── Stage 5: polarity normalization ────────────────────────────────────
	// Ensure digit pixels = 255 (white), background = 0 (black).
	var binaryInv gocv.Mat
	if polarity == PolarityDarkOnLight {
		// binaryRaw has background=255, digits=0 → invert.
		binaryInv = gocv.NewMat()
		gocv.BitwiseNot(binaryRaw, &binaryInv)
	} else {
		// Light-on-dark: digits are already bright in binaryRaw.
		binaryInv = binaryRaw.Clone()
	}
	defer binaryInv.Close()

	// For refined ROIs we've already cropped tightly inside the LCD bezel, so
	// the CC-based flood-fill that bezel removal needs would be too aggressive
	// (small bezel-edge artefacts touching the border are often connected to
	// real digits via thin pixel bridges, and flooding the artefact destroys
	// the digit).  Use the hard-margin-only variant in that case.
	var binary gocv.Mat
	if refined || rectified {
		binary = zeroBorderHard(binaryInv)
	} else {
		binary = zeroBorder(binaryInv)
	}
	defer binary.Close()
	save("04_binary", binary)

	// ── Stage 6: morphological cleanup ────────────────────────────────────
	cleaned := morphClean(binary)
	defer cleaned.Close()
	save("05_cleaned", cleaned)

	// ── Stage 7: row segmentation ─────────────────────────────────────────
	rowRegions := detect.SplitRows(cleaned, opts.ExpectedRows)
	defer func() {
		for i := range rowRegions {
			rowRegions[i].Mask.Close()
		}
	}()

	// ── Stage 8–10: per-row digit segmentation + decode ───────────────────
	charset := toDecodeCharset(opts.Charset)
	// Scale minimum component area with ROI height squared, because decimal dots
	// and colon dots scale with digit height (not total image area).
	// This avoids filtering tiny colon dots in small ROIs (microwave, ~62px tall)
	// while still filtering single-pixel noise in large ROIs.
	minArea := cleaned.Rows() * cleaned.Rows() / 2000
	if minArea < 4 {
		minArea = 4
	}
	minH := cleaned.Rows() / 60
	if minH < 2 {
		minH = 2
	}
	digitOpts := detect.DigitOptions{
		MinArea:   minArea,
		MinWidth:  2,
		MinHeight: minH,
		DecHRatio: opts.DecHRatio,
		DecWRatio: opts.DecWRatio,
		IgnorePix: 2,
	}

	var rows []Row
	for ri, rr := range rowRegions {
		digitBoxes := detect.SegmentDigits(rr.Mask, rr.Bounds.Min.Y, digitOpts)

		// Establish the row band's dense y-range from the aggregate horizontal
		// projection.  Two uses follow:
		//   1. Clamp each digit bbox to this strip — a digit CC can extend far
		//      above the digit when label text connects to it via a thin pixel
		//      path (e.g., tank-gauge "CAPACITY" connecting down to the "0").
		//   2. Drop colon/decimal boxes that fall entirely outside the strip —
		//      stray bright specs in label rows can be paired by
		//      detectColonsSingleDot into spurious colons.
		denseY0, denseY1 := findDigitRowYRange(rr.Mask, rr.Bounds.Min.Y)
		{
			var kept []detect.DigitBox
			for _, db := range digitBoxes {
				if db.IsDecimal || db.IsColon {
					if db.Bounds.Max.Y < denseY0 || db.Bounds.Min.Y > denseY1 {
						continue
					}
					kept = append(kept, db)
					continue
				}
				bb := db.Bounds
				y0 := bb.Min.Y
				if y0 < denseY0 {
					y0 = denseY0
				}
				y1 := bb.Max.Y
				if y1 > denseY1 {
					y1 = denseY1
				}
				if y1 > y0 {
					db.Bounds = image.Rect(bb.Min.X, y0, bb.Max.X, y1)
				}
				kept = append(kept, db)
			}
			digitBoxes = kept
		}
		// Post-filter: remove boxes whose height is < 35% of the tallest non-decimal
		// digit in this row.  This eliminates label text, "GAL", "CAPACITY" etc.
		// Decimal/colon candidates are exempt (they're inherently short).
		{
			maxH := 0
			for _, db := range digitBoxes {
				if !db.IsDecimal && !db.IsColon && db.Bounds.Dy() > maxH {
					maxH = db.Bounds.Dy()
				}
			}
			if maxH > 0 {
				threshold := int(float64(maxH) * 0.35)
				var filtered []detect.DigitBox
				for _, db := range digitBoxes {
					if db.IsDecimal || db.IsColon || db.Bounds.Dy() >= threshold {
						filtered = append(filtered, db)
					}
				}
				digitBoxes = filtered
			}
		}

		// Edge-bargraph filter: a non-digit element at the LCD's left or
		// right margin can survive as a tall narrow CC whose aspect h/w >
		// opts.OneRatio, so AspectClassify would emit a phantom '1'.
		// Example: the time-remaining bar-graph on the RSA SecurID fob.
		//
		// Identify it as the leftmost (or rightmost) non-decimal CC that
		// is BOTH significantly narrower than the row's "wide-digit" width
		// AND separated from its inward neighbour by a gap that exceeds
		// typical inter-digit spacing.  Using the median of the upper half
		// of widths as the reference makes the filter robust against the
		// existence of legitimately narrow '1' digits in the same row —
		// a real '1' fails the gap test because it sits at normal digit
		// spacing from its neighbour (tank gauge "1077", gas pump "13.318").
		{
			var edgeIdxs []int
			var widths []int
			for i, db := range digitBoxes {
				if db.IsDecimal || db.IsColon {
					continue
				}
				edgeIdxs = append(edgeIdxs, i)
				widths = append(widths, db.Bounds.Dx())
			}
			if len(edgeIdxs) >= 3 {
				sortedW := append([]int(nil), widths...)
				sort.Ints(sortedW)
				// Median of the upper half of widths — biases toward the
				// width of "real" digits (5, 6, 8, 0…) and away from thin
				// '1' digits and bezel-noise sub-boxes.
				upper := sortedW[len(sortedW)/2:]
				wideW := upper[len(upper)/2]
				sort.SliceStable(edgeIdxs, func(i, j int) bool {
					return digitBoxes[edgeIdxs[i]].Bounds.Min.X <
						digitBoxes[edgeIdxs[j]].Bounds.Min.X
				})
				toDrop := map[int]bool{}
				check := func(edgeI, neighbourI int, leftEdge bool) {
					eb := digitBoxes[edgeI].Bounds
					nb := digitBoxes[neighbourI].Bounds
					var gap int
					if leftEdge {
						gap = nb.Min.X - eb.Max.X
					} else {
						gap = eb.Min.X - nb.Max.X
					}
					w := eb.Dx()
					h := eb.Dy()
					if w == 0 {
						return
					}
					aspectHW := float64(h) / float64(w)
					if aspectHW > opts.OneRatio &&
						w*100 < wideW*55 &&
						gap*10 > wideW*8 {
						toDrop[edgeI] = true
					}
				}
				check(edgeIdxs[0], edgeIdxs[1], true)
				check(edgeIdxs[len(edgeIdxs)-1], edgeIdxs[len(edgeIdxs)-2], false)
				if len(toDrop) > 0 {
					var filtered []detect.DigitBox
					for i, db := range digitBoxes {
						if !toDrop[i] {
							filtered = append(filtered, db)
						}
					}
					digitBoxes = filtered
				}
			}
		}

		// X-proximity filter for colons/decimals: a real colon or decimal
		// sits next to a digit (between two of them, or just past the last
		// one).  Stray bright specs that pass the colon/decimal classifier
		// but live far from the digit cluster — e.g., the speck pair on the
		// left margin of the tank gauge display — should be dropped.
		{
			medW := 0
			digitMinX, digitMaxX := -1, -1
			for _, db := range digitBoxes {
				if db.IsDecimal || db.IsColon {
					continue
				}
				if w := db.Bounds.Dx(); w > medW {
					medW = w
				}
				if digitMinX < 0 || db.Bounds.Min.X < digitMinX {
					digitMinX = db.Bounds.Min.X
				}
				if db.Bounds.Max.X > digitMaxX {
					digitMaxX = db.Bounds.Max.X
				}
			}
			if medW > 0 && digitMinX >= 0 {
				margin := medW
				var kept []detect.DigitBox
				for _, db := range digitBoxes {
					if db.IsDecimal || db.IsColon {
						cx := (db.Bounds.Min.X + db.Bounds.Max.X) / 2
						if cx < digitMinX-margin || cx > digitMaxX+margin {
							continue
						}
					}
					kept = append(kept, db)
				}
				digitBoxes = kept
			}
		}

		// Y-alignment filter: a real row of 7-segment digits has all digits
		// vertically aligned (their bounding-box centres on a common Y).
		// Anything significantly above or below that centreline is label text
		// or icon noise that survived earlier filters (e.g., tank-gauge "GAL"
		// fragment, which is shorter and rides higher than the digit row).
		// Decimal/colon boxes are exempt — they sit at the edges by design.
		{
			var centres []int
			var medH int
			for _, db := range digitBoxes {
				if db.IsDecimal || db.IsColon {
					continue
				}
				centres = append(centres, (db.Bounds.Min.Y+db.Bounds.Max.Y)/2)
				if h := db.Bounds.Dy(); h > medH {
					medH = h
				}
			}
			if len(centres) >= 3 && medH > 0 {
				sortedC := append([]int(nil), centres...)
				sort.Ints(sortedC)
				medianY := sortedC[len(sortedC)/2]
				tolerance := medH / 4
				if tolerance < 6 {
					tolerance = 6
				}
				var filtered []detect.DigitBox
				for _, db := range digitBoxes {
					if db.IsDecimal || db.IsColon {
						filtered = append(filtered, db)
						continue
					}
					cy := (db.Bounds.Min.Y + db.Bounds.Max.Y) / 2
					if abs(cy-medianY) <= tolerance {
						filtered = append(filtered, db)
					}
				}
				digitBoxes = filtered
			}
		}

		// Bezel-line cluster filter: top/bottom edges of the LCD often leak
		// through polarity inversion as a horizontal row of small decimal-
		// shaped CCs (e.g., the RSA SecurID auto path has 3 such specks at
		// the top edge and 3 at the bottom).  If 3 or more decimal/colon
		// boxes cluster within a narrow y-band, treat the cluster as bezel
		// noise and drop them all.  Real displays have at most 1–2 decimals
		// per row, and colons are pre-paired into a single box, so legitimate
		// clusters of 3+ on the same y-line don't occur in practice.
		{
			medH := 0
			for _, db := range digitBoxes {
				if !db.IsDecimal && !db.IsColon {
					if h := db.Bounds.Dy(); h > medH {
						medH = h
					}
				}
			}
			if medH > 0 {
				tol := medH / 10
				if tol < 4 {
					tol = 4
				}
				type smallBox struct {
					idx int
					cy  int
				}
				var smalls []smallBox
				for i, db := range digitBoxes {
					if db.IsDecimal || db.IsColon {
						smalls = append(smalls, smallBox{
							idx: i,
							cy:  (db.Bounds.Min.Y + db.Bounds.Max.Y) / 2,
						})
					}
				}
				sort.Slice(smalls, func(i, j int) bool {
					return smalls[i].cy < smalls[j].cy
				})
				toDrop := map[int]bool{}
				n := len(smalls)
				i := 0
				for i < n {
					j := i
					for j+1 < n && smalls[j+1].cy-smalls[i].cy <= tol {
						j++
					}
					if j-i+1 >= 3 {
						for k := i; k <= j; k++ {
							toDrop[smalls[k].idx] = true
						}
					}
					i = j + 1
				}
				if len(toDrop) > 0 {
					var filtered []detect.DigitBox
					for i, db := range digitBoxes {
						if !toDrop[i] {
							filtered = append(filtered, db)
						}
					}
					digitBoxes = filtered
				}
			}
		}

		// Profile-driven filter: when a profile says the display has no
		// decimal points, drop any isolated decimals (typically AM/PM indicator
		// dots, alarm-bell icons, or stray reflection specks).  Same idea for
		// unexpected colons.
		if opts.decimalExpectationSet {
			var filtered []detect.DigitBox
			for _, db := range digitBoxes {
				if db.IsDecimal && !opts.HasDecimal {
					continue
				}
				if db.IsColon && !opts.HasColon {
					continue
				}
				filtered = append(filtered, db)
			}
			digitBoxes = filtered
		} else {
			// No profile guidance: a row that contains a colon almost always
			// belongs to a clock-style display.  Lone decimal dots in such a
			// row are usually AM/PM indicators, not decimal points, so drop them.
			hasColon := false
			for _, db := range digitBoxes {
				if db.IsColon {
					hasColon = true
					break
				}
			}
			if hasColon {
				var filtered []detect.DigitBox
				for _, db := range digitBoxes {
					if db.IsDecimal {
						continue
					}
					filtered = append(filtered, db)
				}
				digitBoxes = filtered
			}
		}

		// Compute median height for aspect classification
		medianH := 0
		for _, db := range digitBoxes {
			if !db.IsDecimal && !db.IsColon {
				h := db.Bounds.Dy()
				if h > medianH {
					medianH = h
				}
			}
		}

		var digits []Digit
		var sb strings.Builder
		minConf := 1.0

		for _, db := range digitBoxes {
			d := Digit{Box: db.Bounds}

			if db.IsDecimal {
				// Only output '.' for charsets that include it.
				if charset == decode.CharsetDigits {
					continue // skip decimals in digit-only mode
				}
				d.Char = '.'
				d.Confidence = 1.0
				d.Source = SourceGeometric
			} else if db.IsColon {
				// Only output ':' for full charset.
				if charset != decode.CharsetFull {
					continue // skip colons in restricted charsets
				}
				d.Char = ':'
				d.Confidence = 1.0
				d.Source = SourceGeometric
			} else {
				// Aspect-ratio quick classify
				r := decode.AspectClassify(db.Bounds.Dx(), db.Bounds.Dy(), medianH,
					opts.OneRatio, opts.MinusRatio)
				if r != 0 {
					d.Char = r
					d.Confidence = 0.85
					d.Source = SourceGeometric
				} else {
					// Full segment sampling
					digitImg := decode.NormalizeDigitImage(cleaned, db.Bounds)
					if opts.DebugDir != "" {
						gocv.IMWrite(fmt.Sprintf("%s/digit_r%d_x%d.png", opts.DebugDir, ri, db.Bounds.Min.X), digitImg)
					}
					mask, conf := decode.SampleSegments(digitImg)
					d.Segments = mask
					ch, conf2 := decode.Decode(mask, charset, false, false)
					d.Char = ch
					d.Confidence = (conf + conf2) / 2.0
					d.Source = SourceGeometric

					// DNN fallback if confidence is low
					if opts.EnableDNN && d.Confidence < opts.DNNThreshold {
						dnnCh, dnnConf, dnnErr := dnn.Classify(digitImg)
						if dnnErr == nil && dnnConf > d.Confidence {
							d.Char = dnnCh
							d.Confidence = (d.Confidence + dnnConf) / 2.0
							d.Source = SourceDNN
						}
					}
					digitImg.Close()
				}
			}

			// Add space if print-spaces is on and there's a gap
			if opts.PrintSpaces && len(digits) > 0 {
				prev := digits[len(digits)-1]
				gap := db.Bounds.Min.X - prev.Box.Max.X
				avgW := (db.Bounds.Dx() + prev.Box.Dx()) / 2
				if avgW > 0 && float64(gap)/float64(avgW) > opts.SpaceFactor {
					sb.WriteRune(' ')
				}
			}

			sb.WriteRune(d.Char)
			if d.Confidence < minConf {
				minConf = d.Confidence
			}
			digits = append(digits, d)
		}

		if len(digits) == 0 {
			continue
		}
		_ = ri
		rows = append(rows, Row{
			Text:       sb.String(),
			Digits:     digits,
			Box:        rr.Bounds,
			Confidence: minConf,
		})
	}

	// Build result
	var textParts []string
	overallConf := 1.0
	for _, row := range rows {
		textParts = append(textParts, row.Text)
		if row.Confidence < overallConf {
			overallConf = row.Confidence
		}
	}

	res := &Result{
		Rows:       rows,
		Text:       strings.Join(textParts, "\n"),
		Confidence: overallConf,
	}
	if opts.DebugDir != "" {
		res.Debug = dbg
	}
	return res, nil
}

// ── internal helpers ──────────────────────────────────────────────────────

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// findDigitRowYRange identifies the y-range within rowMask (a row-band binary
// image, top-left at rowOffset in the parent image) that contains the actual
// digit content, separating it from label text or noise above/below.
//
// Motivation: connected-component analysis on the band can return a digit
// bbox much taller than the digit itself when a thin pixel path links the
// digit upward to label text (e.g., tank-gauge "CAPACITY" letters connecting
// down to the top of a "0").  Looking at horizontal projection across the
// full row width is more reliable than per-bbox analysis, because individual
// digits can have legitimate empty-row gaps (the inter-segment gap inside
// "0" / "8") while the aggregate projection over several aligned digits is
// non-zero throughout the true digit row.
//
// Algorithm: compute the per-row sum of white pixels across the full row
// width, mark rows whose count clears max(2, maxCount/8) as active, and
// return the longest contiguous active run as the digit-content y-range in
// image coordinates.  Returns (rowOffset, rowOffset+rowMask.Rows()) when the
// run analysis is inconclusive.
func findDigitRowYRange(rowMask gocv.Mat, rowOffset int) (int, int) {
	h := rowMask.Rows()
	if h <= 0 {
		return rowOffset, rowOffset + h
	}
	proj := make([]int, h)
	maxP := 0
	cols := rowMask.Cols()
	for r := 0; r < h; r++ {
		count := 0
		for c := 0; c < cols; c++ {
			if rowMask.GetUCharAt(r, c) > 0 {
				count++
			}
		}
		proj[r] = count
		if count > maxP {
			maxP = count
		}
	}
	if maxP == 0 {
		return rowOffset, rowOffset + h
	}
	threshold := maxP / 8
	if threshold < 2 {
		threshold = 2
	}
	bestStart, bestLen := 0, 0
	curStart, curLen := 0, 0
	for r := 0; r < h; r++ {
		if proj[r] >= threshold {
			if curLen == 0 {
				curStart = r
			}
			curLen++
			if curLen > bestLen {
				bestStart, bestLen = curStart, curLen
			}
		} else {
			curLen = 0
		}
	}
	if bestLen <= 0 || bestLen == h {
		return rowOffset, rowOffset + h
	}
	return rowOffset + bestStart, rowOffset + bestStart + bestLen
}

// cropToDarkBand finds the darkest contiguous horizontal band in a (possibly
// colour) image and returns the y-range [y0, y1) of that band, or (0, rows)
// when no clear band is detected.  Used after perspective rectification to
// trim the device case area away from a dark LCD strip — the rectified ROI
// for a calculator/large-device shot still contains bright case pixels above
// and below the LCD that confuse row segmentation downstream.
//
// Algorithm: convert to gray, compute row-mean intensities, find the longest
// contiguous run where row-mean is below (rowMeanMax+rowMeanMin)/2 minus a
// small bias.  The band must be at least 8% of image height and must be at
// least 25% darker than the brightest row; otherwise return the full range
// (the input is already a clean LCD-only ROI or has no LCD band to find).
func cropToDarkBand(src gocv.Mat) (int, int) {
	rows := src.Rows()
	cols := src.Cols()
	if rows < 30 || cols < 30 {
		return 0, rows
	}
	gray := gocv.NewMat()
	defer gray.Close()
	if src.Channels() > 1 {
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
	} else {
		gray = src.Clone()
	}
	means := make([]float64, rows)
	minMean := 255.0
	maxMean := 0.0
	for r := 0; r < rows; r++ {
		sum := 0.0
		for c := 0; c < cols; c++ {
			sum += float64(gray.GetUCharAt(r, c))
		}
		m := sum / float64(cols)
		means[r] = m
		if m < minMean {
			minMean = m
		}
		if m > maxMean {
			maxMean = m
		}
	}
	if maxMean-minMean < 64 {
		return 0, rows
	}
	threshold := (minMean + maxMean) / 2
	bestStart, bestEnd := 0, 0
	curStart := -1
	for r := 0; r < rows; r++ {
		if means[r] < threshold {
			if curStart < 0 {
				curStart = r
			}
		} else {
			if curStart >= 0 {
				if r-curStart > bestEnd-bestStart {
					bestStart, bestEnd = curStart, r
				}
				curStart = -1
			}
		}
	}
	if curStart >= 0 && rows-curStart > bestEnd-bestStart {
		bestStart, bestEnd = curStart, rows
	}
	bandH := bestEnd - bestStart
	if bandH < rows/12 {
		return 0, rows
	}
	pad := bandH / 8
	if pad < 4 {
		pad = 4
	}
	y0 := bestStart - pad
	if y0 < 0 {
		y0 = 0
	}
	y1 := bestEnd + pad
	if y1 > rows {
		y1 = rows
	}
	return y0, y1
}

// zeroBorderHard zeros a hard margin around the image without doing any
// CC-based flood-fill.  Used for refined LCD ROIs where the caller has
// already cropped tightly inside the bezel: a flood-fill there would chase
// digit-to-edge bridges and destroy the digits along with the bezel artefact.
func zeroBorderHard(src gocv.Mat) gocv.Mat {
	dst := src.Clone()
	margin := src.Rows() / 30
	if margin < 3 {
		margin = 3
	}
	rows, cols := dst.Rows(), dst.Cols()
	for r := 0; r < margin; r++ {
		for c := 0; c < cols; c++ {
			dst.SetUCharAt(r, c, 0)
			dst.SetUCharAt(rows-1-r, c, 0)
		}
	}
	for c := 0; c < margin; c++ {
		for r := 0; r < rows; r++ {
			dst.SetUCharAt(r, c, 0)
			dst.SetUCharAt(r, cols-1-c, 0)
		}
	}
	return dst
}

// zeroBorder removes white regions that touch the image border — these are
// typically bezel/frame artifacts from the polarity inversion step.
//
// Algorithm: zero a hard margin, then use ConnectedComponents to find any
// remaining white region whose bounding box still touches the (shrunk) border,
// and paint those regions black.
func zeroBorder(src gocv.Mat) gocv.Mat {
	dst := src.Clone()

	// Hard margin: zero the outermost pixels unconditionally.
	margin := src.Rows() / 30 // ~3.3% of height — tight enough to remove bezel
	if margin < 3 {
		margin = 3
	}
	rows, cols := dst.Rows(), dst.Cols()
	for r := 0; r < margin; r++ {
		for c := 0; c < cols; c++ {
			dst.SetUCharAt(r, c, 0)
			dst.SetUCharAt(rows-1-r, c, 0)
		}
	}
	for c := 0; c < margin; c++ {
		for r := 0; r < rows; r++ {
			dst.SetUCharAt(r, c, 0)
			dst.SetUCharAt(r, cols-1-c, 0)
		}
	}

	// CC pass: find any remaining white region whose bounding rect touches the
	// (inner-edge) border and erase it.
	labels := gocv.NewMat()
	stats := gocv.NewMat()
	centroids := gocv.NewMat()
	defer labels.Close()
	defer stats.Close()
	defer centroids.Close()

	n := gocv.ConnectedComponentsWithStats(dst, &labels, &stats, &centroids)
	for i := 1; i < n; i++ {
		x := int(stats.GetIntAt(i, 0)) // ccLeft
		y := int(stats.GetIntAt(i, 1)) // ccTop
		w := int(stats.GetIntAt(i, 2)) // ccWidth
		h := int(stats.GetIntAt(i, 3)) // ccHeight
		// Touches the inner border edge?
		if x <= margin || y <= margin || x+w >= cols-margin || y+h >= rows-margin {
			// Paint this region black in dst using the labels image.
			for r := y; r < y+h && r < rows; r++ {
				for c := x; c < x+w && c < cols; c++ {
					if labels.GetIntAt(r, c) == int32(i) {
						dst.SetUCharAt(r, c, 0)
					}
				}
			}
		}
	}
	return dst
}

func autoPreprocess(src gocv.Mat) gocv.Mat {
	// CLAHE to boost contrast, then a gentle blur
	clahe := preprocess.CLAHEOp{ClipLimit: 3.0, TileWidth: 8, TileHeight: 8}
	out, err := clahe.Apply(src)
	if err != nil {
		return src.Clone()
	}
	return out
}

func adaptiveThreshold(gray gocv.Mat) gocv.Mat {
	// Try iterative k-means threshold (most stable for uniform-lit displays).
	op := preprocess.IterThresholdOp{ThresholdPct: 50}
	out, err := op.Apply(gray)
	if err != nil {
		out = gocv.NewMat()
		gocv.Threshold(gray, &out, 0, 255, gocv.ThresholdBinary+gocv.ThresholdOtsu)
	}

	// Quality check: count the ratio of white pixels.
	// A good binary has between 10% and 80% white pixels.
	// If the result is dominated by one value, it means the global threshold
	// failed (e.g., high-reflection image) — fall back to adaptive threshold.
	total := float64(out.Rows() * out.Cols())
	white := out.Sum().Val1 / 255.0
	ratio := white / total
	if ratio < 0.08 || ratio > 0.92 {
		out.Close()
		// Adaptive (local) threshold handles non-uniform illumination.
		blockSize := gray.Cols() / 8
		if blockSize < 11 {
			blockSize = 11
		}
		if blockSize%2 == 0 {
			blockSize++
		}
		adaptive := gocv.NewMat()
		gocv.AdaptiveThreshold(gray, &adaptive, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, blockSize, 3)
		return adaptive
	}
	return out
}

func morphClean(binary gocv.Mat) gocv.Mat {
	// Remove isolated noise pixels.
	rm := preprocess.RemoveIsolatedOp{}
	out, err := rm.Apply(binary)
	if err != nil {
		return binary.Clone()
	}

	// Apply closing only for larger images (≥ 60px tall) where individual LCD
	// segments are fragmented by tiny intra-segment gaps.  For small ROIs like
	// the microwave clock (~28px), closing would merge the tiny colon dots
	// (3-4px each) into one 10px blob that can't be detected as a colon.
	if binary.Rows() >= 60 {
		cl := preprocess.ClosingOp{N: 1}
		out2, err := cl.Apply(out)
		out.Close()
		if err != nil {
			return binary.Clone()
		}
		return out2
	}
	return out
}

func toDetectPolarity(p Polarity) detect.Polarity {
	switch p {
	case PolarityLightOnDark:
		return detect.PolarityLightOnDark
	case PolarityDarkOnLight:
		return detect.PolarityDarkOnLight
	default:
		return detect.PolarityAuto
	}
}

func toDecodeCharset(c Charset) decode.CharsetID {
	switch c {
	case CharsetDigits:
		return decode.CharsetDigits
	case CharsetDecimal:
		return decode.CharsetDecimal
	case CharsetHex:
		return decode.CharsetHex
	case CharsetTTRobot:
		return decode.CharsetTTRobot
	default:
		return decode.CharsetFull
	}
}

func applyProfile(opts *Options, name string) {
	p := profile.Get(name)
	switch p.Polarity {
	case "dark_on_light":
		opts.Polarity = PolarityDarkOnLight
	case "light_on_dark":
		opts.Polarity = PolarityLightOnDark
	}
	switch p.Charset {
	case "digits":
		opts.Charset = CharsetDigits
	case "decimal":
		opts.Charset = CharsetDecimal
	case "hex":
		opts.Charset = CharsetHex
	}
	if p.ExpectedRows > 0 {
		opts.ExpectedRows = p.ExpectedRows
	}
	if p.OneRatio > 0 {
		opts.OneRatio = p.OneRatio
	}
	if p.MinusRatio > 0 {
		opts.MinusRatio = p.MinusRatio
	}
	if p.DecHRatio > 0 {
		opts.DecHRatio = p.DecHRatio
	}
	if p.DecWRatio > 0 {
		opts.DecWRatio = p.DecWRatio
	}
	// HasColon/HasDecimal are taken straight from the profile; the
	// decimalExpectationSet flag distinguishes "profile applied" from
	// "no profile / legacy permissive behaviour".
	opts.HasColon = p.HasColon
	opts.HasDecimal = p.HasDecimal
	opts.decimalExpectationSet = true
}
