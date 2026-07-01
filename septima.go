// Package septima recognizes seven-segment displays using a two-stage local
// YOLO pipeline: a panel detector locates the display, then a digit detector
// reads the glyphs within the cropped panel. Inference runs locally via ONNX
// Runtime (CPU or GPU); models are trained offline with Ultralytics.
package septima

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"sort"
	"strings"

	_ "golang.org/x/image/webp"

	"github.com/brian-maloney/septima/internal/assemble"
	"github.com/brian-maloney/septima/internal/detect"
	"github.com/brian-maloney/septima/internal/imageproc"
)

// ReadFile recognizes the display in the image at path.
func ReadFile(path string, opts ...Option) (Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return Result{}, fmt.Errorf("septima: open %s: %w", path, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return Result{}, fmt.Errorf("septima: decode %s: %w", path, err)
	}
	return Read(img, opts...)
}

// Read recognizes the display in an already-decoded image.
func Read(img image.Image, opts ...Option) (Result, error) {
	o := applyOptions(opts)
	modelDir := resolveModelDir(o.ModelDir)

	classes, err := detect.LoadClasses(modelDir)
	if err != nil {
		return Result{}, fmt.Errorf("septima: %w", err)
	}

	digits, err := detect.OpenModel(modelPath(modelDir, "digits.onnx"), len(classes.DigitClasses), classes.InputSize)
	if err != nil {
		return Result{}, fmt.Errorf("septima: open digit model: %w", err)
	}
	defer digits.Close()

	// Punctuation is detected at a lower confidence bar than digits (see
	// Options.PunctThreshold), so decode at the lower of the two floors and apply
	// the real per-class bars afterwards.
	punct := punctClassSet(classes.DigitClasses)
	floor := math.Min(o.ConfThreshold, o.PunctThreshold)

	// Adaptive localization. Candidate A: run the digit detector on the full
	// frame — best when the display already fills the frame, since cropping then
	// over-scales the glyphs out of the detector's training distribution.
	rawFull, err := digits.Detect(img, floor, o.IOUThreshold)
	if err != nil {
		return Result{}, fmt.Errorf("septima: digit detection: %w", err)
	}
	fullDets := applyClassThresholds(rawFull, punct, o.ConfThreshold, o.PunctThreshold)

	// Candidate B: localize the panel (trained panel.onnx, else the bright-panel
	// heuristic) and detect within the crop — best when the display is a small
	// part of a larger scene (the tank), where full-frame glyphs are too small.
	var cropDets []detect.Detection
	haveCrop := false
	if !o.SkipPanel {
		if region, ok := locatePanel(img, modelDir, classes, o); ok {
			region = imageproc.PadRect(region, img.Bounds(), 0.30)
			if cd, derr := digits.Detect(imageproc.Crop(img, region), floor, o.IOUThreshold); derr == nil {
				for i := range cd {
					cd[i].Box = cd[i].Box.Add(region.Min)
				}
				cropDets = applyClassThresholds(cd, punct, o.ConfThreshold, o.PunctThreshold)
				haveCrop = true
			}
		}
	}

	// Keep whichever candidate the detector is more confident about, by MEAN
	// detection confidence rather than summed score. A summed score rewards a
	// candidate that pads its reading with extra weak detections (e.g. full-frame
	// hallucinating phantom leading digits on the RSA token); the mean penalizes
	// that dilution, so it picks the cleaner reading. A genuine rescue crop (the
	// tank, where full-frame glyphs are too small to detect at all) still wins,
	// since its non-empty detections beat the full frame's near-zero mean.
	//
	// Exception: when both candidates read the SAME digit sequence and one of
	// them additionally detected punctuation, the mean signal is biased AGAINST
	// the more complete reading (a correct low-confidence '.' drags the mean
	// down), so prefer the punctuation-bearing reading directly.
	reading := finalizeReading(fullDets, classes)
	if haveCrop {
		cropReading := finalizeReading(cropDets, classes)
		switch punctAgreementPick(reading.Text, cropReading.Text) {
		case pickFirst:
			// keep the full-frame reading
		case pickSecond:
			reading = cropReading
		default:
			if meanScore(cropDets) > meanScore(fullDets) {
				reading = cropReading
			}
		}
	}
	return toResult(reading), nil
}

// punctAgreementPick decides between two candidate readings that agree on the
// digit sequence but disagree on punctuation. The candidate with MORE
// punctuation is preferred — both framings saw the same digits, and the richer
// one cleared the punctuation confidence bar on a mark the other missed —
// provided its rows stay well-formed (a second '.' in one row is a phantom
// tell, e.g. "10.8.00"). Returns pickNone when the rule does not apply and the
// caller should fall back to the confidence-based selection.
func punctAgreementPick(a, b string) int {
	if a == b || digitsOnly(a) != digitsOnly(b) {
		return pickNone
	}
	switch {
	case punctCount(a) > punctCount(b) && wellFormedRows(a):
		return pickFirst
	case punctCount(b) > punctCount(a) && wellFormedRows(b):
		return pickSecond
	}
	return pickNone
}

const (
	pickNone = iota
	pickFirst
	pickSecond
)

// digitsOnly strips '.'/':' (keeping row structure), the same digit-sequence
// view septima-bench's digits-only metric uses.
func digitsOnly(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '.' || r == ':' {
			return -1
		}
		return r
	}, s)
}

func punctCount(s string) int { return strings.Count(s, ".") + strings.Count(s, ":") }

// wellFormedRows reports whether every row reads like one number: at most one
// '.' and at most one ':' per row.
func wellFormedRows(s string) bool {
	for _, row := range strings.Split(s, "\n") {
		if strings.Count(row, ".") > 1 || strings.Count(row, ":") > 1 {
			return false
		}
	}
	return true
}

// finalizeReading applies the shared post-processing pipeline (class-agnostic
// dedup + colon-dot merge) and assembles the detections into a reading.
func finalizeReading(dets []detect.Detection, classes detect.Classes) assemble.Reading {
	// Class-agnostic dedup removes duplicate boxes the per-class NMS misses
	// (e.g. the same glyph detected as both '9' and '4', or a narrow '1' box
	// nested in a wider one).
	dets = detect.DedupeAcrossClasses(dets, 0.5)

	// A colon is two stacked dots the detector often reports as two decimals —
	// merge such pairs back into a single ':'. Then drop any stray '.' the
	// colon-trained model fires inside a ':' box (which would read "2:47" as
	// "2:.47").
	if dot, col := classIndex(classes.DigitClasses, '.'), classIndex(classes.DigitClasses, ':'); dot >= 0 && col >= 0 {
		dets = detect.MergeColonDots(dets, dot, col)
		dets = detect.SuppressDotsInsideColon(dets, dot, col)
	}
	return assemble.Assemble(dets, classes.DigitClasses)
}

func toResult(r assemble.Reading) Result {
	res := Result{Text: r.Text, Confidence: r.Confidence}
	for _, row := range r.Rows {
		gr := Row{Text: row.Text, Box: row.Box, Confidence: row.Confidence}
		for _, ch := range row.Chars {
			gr.Digits = append(gr.Digits, Digit{Char: ch.R, Box: ch.Box, Confidence: ch.Confidence})
		}
		res.Rows = append(res.Rows, gr)
	}
	return res
}

// locatePanel finds the display region: the trained panel.onnx if available and
// it detects something, else the classical bright-panel heuristic.
func locatePanel(img image.Image, modelDir string, classes detect.Classes, o Options) (image.Rectangle, bool) {
	if panel, err := detect.OpenModel(modelPath(modelDir, "panel.onnx"), len(classes.PanelClasses), classes.InputSize); err == nil {
		defer panel.Close()
		if dets, derr := panel.Detect(img, o.ConfThreshold, o.IOUThreshold); derr == nil && len(dets) > 0 {
			return bestDetection(dets).Box, true
		}
	}
	return detect.FindBrightPanel(img)
}

// digitScore sums detection confidences.
func digitScore(dets []detect.Detection) float64 {
	var s float64
	for _, d := range dets {
		s += d.Score
	}
	return s
}

// meanScore is the average detection confidence — the localization-candidate
// selection signal. Unlike a summed score it does not reward a candidate for
// padding its reading with extra low-confidence (often phantom) detections. An
// empty candidate scores 0.
func meanScore(dets []detect.Detection) float64 {
	if len(dets) == 0 {
		return 0
	}
	return digitScore(dets) / float64(len(dets))
}

// punctClassSet returns the set of class indices that are punctuation glyphs
// ('.', ':', '-'), which use the lower PunctThreshold rather than ConfThreshold.
func punctClassSet(names []string) map[int]bool {
	m := map[int]bool{}
	for i, n := range names {
		if n == "." || n == ":" || n == "-" {
			m[i] = true
		}
	}
	return m
}

// applyClassThresholds keeps punctuation detections scoring at least punctT and
// every other (digit) detection scoring at least digitT. Detection runs at the
// lower of the two floors, so this is where the real per-class bars are applied.
func applyClassThresholds(dets []detect.Detection, punct map[int]bool, digitT, punctT float64) []detect.Detection {
	out := dets[:0:0]
	for _, d := range dets {
		thr := digitT
		if punct[d.Class] {
			thr = punctT
		}
		if d.Score >= thr {
			out = append(out, d)
		}
	}
	return out
}

// classIndex returns the class index whose single-rune label is r, or -1.
func classIndex(names []string, r rune) int {
	for i, n := range names {
		if len(n) == 1 && rune(n[0]) == r {
			return i
		}
	}
	return -1
}

func bestDetection(dets []detect.Detection) detect.Detection {
	sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	return dets[0]
}

func resolveModelDir(opt string) string {
	if opt != "" {
		return opt
	}
	if env := os.Getenv("SEPTIMA_MODEL_DIR"); env != "" {
		return env
	}
	return "models"
}

func modelPath(dir, name string) string { return dir + string(os.PathSeparator) + name }
