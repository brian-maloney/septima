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

	// Stage 1: locate the display panel and crop to it (with padding). Prefer the
	// trained panel.onnx; if it is unavailable or finds nothing, fall back to the
	// classical bright-panel heuristic; failing that, use the full frame.
	panelImg := img
	offset := image.Point{}
	region, found := image.Rectangle{}, false
	if panel, err := detect.OpenModel(modelPath(modelDir, "panel.onnx"), len(classes.PanelClasses), classes.InputSize); err == nil {
		defer panel.Close()
		if dets, derr := panel.Detect(img, o.ConfThreshold, o.IOUThreshold); derr == nil && len(dets) > 0 {
			region, found = bestDetection(dets).Box, true
		}
	}
	if !found {
		region, found = detect.FindBrightPanel(img)
	}
	if found {
		region = imageproc.PadRect(region, img.Bounds(), 0.08)
		panelImg = imageproc.Crop(img, region)
		offset = region.Min
	}

	// Stage 2: detect glyphs within the panel crop.
	digits, err := detect.OpenModel(modelPath(modelDir, "digits.onnx"), len(classes.DigitClasses), classes.InputSize)
	if err != nil {
		return Result{}, fmt.Errorf("septima: open digit model: %w", err)
	}
	defer digits.Close()

	dets, err := digits.Detect(panelImg, o.ConfThreshold, o.IOUThreshold)
	if err != nil {
		return Result{}, fmt.Errorf("septima: digit detection: %w", err)
	}
	// Map crop-space boxes back into original image coordinates.
	for i := range dets {
		dets[i].Box = dets[i].Box.Add(offset)
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
