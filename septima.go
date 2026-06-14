// Package septima provides seven-segment display OCR.
// It is designed to be as reliable and hands-off as SegoDec while exposing the
// full preprocessing pipeline of ssocr.
package septima

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"

	"github.com/vond/septima/decode"
	"github.com/vond/septima/detect"
	"github.com/vond/septima/dnn"
	"github.com/vond/septima/preprocess"
	"github.com/vond/septima/profile"
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
	var working gocv.Mat
	if opts.ROI != nil {
		region := m.Region(*opts.ROI)
		working = region.Clone()
		region.Close()
	} else {
		roi := detect.FindDisplayROI(m)
		roi = detect.PadROI(roi, 5, m.Cols(), m.Rows())
		region := m.Region(roi)
		working = region.Clone()
		region.Close()
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

	binary := zeroBorder(binaryInv)
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

// zeroBorder removes white regions that touch the image border — these are
// typically bezel/frame artifacts from the polarity inversion step.
//
// Algorithm: zero a hard margin, then use ConnectedComponents to find any
// remaining white region whose bounding box still touches the (shrunk) border,
// and paint those regions black.
func zeroBorder(src gocv.Mat) gocv.Mat {
	dst := src.Clone()

	// Hard margin: zero the outermost pixels unconditionally.
	margin := src.Rows() / 20
	if margin < 4 {
		margin = 4
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
}
