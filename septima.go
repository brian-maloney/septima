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
	"os"
	"sort"

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

	// Adaptive localization. Candidate A: run the digit detector on the full
	// frame — best when the display already fills the frame, since cropping then
	// over-scales the glyphs out of the detector's training distribution.
	dets, err := digits.Detect(img, o.ConfThreshold, o.IOUThreshold)
	if err != nil {
		return Result{}, fmt.Errorf("septima: digit detection: %w", err)
	}

	// Candidate B: localize the panel (trained panel.onnx, else the bright-panel
	// heuristic) and detect within the crop — best when the display is a small
	// part of a larger scene (the tank), where full-frame glyphs are too small.
	// Keep whichever candidate the detector is more confident about.
	if !o.SkipPanel {
		if region, ok := locatePanel(img, modelDir, classes, o); ok {
			region = imageproc.PadRect(region, img.Bounds(), 0.30)
			if cropDets, derr := digits.Detect(imageproc.Crop(img, region), o.ConfThreshold, o.IOUThreshold); derr == nil {
				for i := range cropDets {
					cropDets[i].Box = cropDets[i].Box.Add(region.Min)
				}
				// Keep whichever candidate the detector is more confident about.
				// A needed crop (tank, where full-frame glyphs are tiny) wins by a
				// wide margin; an already-framed display favors the full frame.
				if digitScore(cropDets) > digitScore(dets) {
					dets = cropDets
				}
			}
		}
	}

	// Class-agnostic dedup removes duplicate boxes the per-class NMS misses
	// (e.g. the same glyph detected as both '9' and '4', or a narrow '1' box
	// nested in a wider one).
	dets = detect.DedupeAcrossClasses(dets, 0.5)

	// A colon is two stacked dots the detector often reports as two decimals —
	// merge such pairs back into a single ':'.
	if dot, col := classIndex(classes.DigitClasses, '.'), classIndex(classes.DigitClasses, ':'); dot >= 0 && col >= 0 {
		dets = detect.MergeColonDots(dets, dot, col)
	}

	reading := assemble.Assemble(dets, classes.DigitClasses)
	return toResult(reading), nil
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

// digitScore sums detection confidences — used to pick the localization
// candidate the detector is most confident about.
func digitScore(dets []detect.Detection) float64 {
	var s float64
	for _, d := range dets {
		s += d.Score
	}
	return s
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
