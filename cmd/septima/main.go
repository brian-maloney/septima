// Command septima reads a seven-segment display from a single image and prints
// the recognized value.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/brian-maloney/septima"
	"github.com/brian-maloney/septima/internal/onnx"
	"github.com/brian-maloney/septima/internal/ortlib"
	"github.com/brian-maloney/septima/models"
	"github.com/brian-maloney/septima/preprocess"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "septima:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	var (
		opts      []septima.Option
		pipeline  preprocess.Pipeline
		imagePath string
		verbose   bool
		showVer   bool
		modelDir  string
		profile   string
		conf      = 0.25
		skipPanel bool
	)

	i := 0
	for i < len(args) {
		arg := args[i]
		switch {
		case arg == "-h" || arg == "--help":
			printUsage()
			return nil
		case arg == "-version" || arg == "--version" || arg == "-V":
			showVer = true
		case arg == "-v" || arg == "--verbose":
			verbose = true
		case arg == "-models":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			modelDir = args[i]
		case arg == "-profile":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			profile = args[i]
		case arg == "-conf":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			v, err := strconv.ParseFloat(args[i], 64)
			if err != nil {
				return fmt.Errorf("invalid -conf %q", args[i])
			}
			conf = v
		case arg == "-skip-panel" || arg == "--skip-panel":
			skipPanel = true

		// ssocr-style image manipulation pipeline ops, applied in order before
		// detection. Useful for pre-cropping to the panel (pair with
		// -skip-panel) or cleaning up glare/contrast.
		case arg == "crop":
			if i+4 >= len(args) {
				return fmt.Errorf("crop requires X Y W H")
			}
			x, err1 := strconv.Atoi(args[i+1])
			y, err2 := strconv.Atoi(args[i+2])
			w, err3 := strconv.Atoi(args[i+3])
			h, err4 := strconv.Atoi(args[i+4])
			if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
				return fmt.Errorf("crop requires integer X Y W H")
			}
			pipeline = append(pipeline, preprocess.CropOp{X: x, Y: y, W: w, H: h})
			i += 4
		case arg == "rotate":
			if i+1 >= len(args) {
				return fmt.Errorf("rotate requires DEGREES")
			}
			deg, err := strconv.ParseFloat(args[i+1], 64)
			if err != nil {
				return fmt.Errorf("rotate requires a numeric DEGREES")
			}
			pipeline = append(pipeline, preprocess.RotateOp{Degrees: deg})
			i++
		case arg == "mirror":
			if i+1 >= len(args) {
				return fmt.Errorf("mirror requires horiz|vert")
			}
			pipeline = append(pipeline, preprocess.MirrorOp{Horiz: args[i+1] == "horiz"})
			i++
		case arg == "invert":
			pipeline = append(pipeline, preprocess.InvertOp{})
		case arg == "grayscale":
			pipeline = append(pipeline, preprocess.GrayscaleOp{})
		case arg == "threshold":
			if i+1 >= len(args) {
				return fmt.Errorf("threshold requires PCT")
			}
			pct, err := strconv.ParseFloat(args[i+1], 64)
			if err != nil {
				return fmt.Errorf("threshold requires a numeric PCT")
			}
			pipeline = append(pipeline, preprocess.ThresholdOp{ThresholdPct: pct})
			i++
		case arg == "otsu_threshold":
			pipeline = append(pipeline, preprocess.OtsuThresholdOp{})

		default:
			if strings.HasPrefix(arg, "-") {
				return fmt.Errorf("unknown flag %q", arg)
			}
			imagePath = arg
		}
		i++
	}

	if showVer {
		fmt.Println("septima", septima.Version)
		return nil
	}
	if imagePath == "" {
		printUsage()
		return fmt.Errorf("no image file specified")
	}

	if err := models.Verify(); err != nil {
		return err
	}
	onnx.SetEmbeddedLibSource(func() ([]byte, string, bool) {
		return ortlib.Bytes, ortlib.Filename, ortlib.Available()
	})

	opts = append(opts, septima.WithConfThreshold(conf))
	opts = append(opts, septima.WithEmbeddedModels(models.PanelONNX, models.DigitsONNX, models.ClassesJSON))
	if modelDir != "" {
		opts = append(opts, septima.WithModelDir(modelDir))
	}
	if profile != "" {
		opts = append(opts, septima.WithProfile(profile))
	}
	if skipPanel {
		opts = append(opts, septima.WithSkipPanel(true))
	}
	if len(pipeline) > 0 {
		opts = append(opts, septima.WithPipeline(pipeline...))
	}

	res, err := septima.ReadFile(imagePath, opts...)
	if err != nil {
		return err
	}

	fmt.Println(res.Text)
	if verbose {
		fmt.Fprintf(os.Stderr, "confidence: %.3f\n", res.Confidence)
		for i, row := range res.Rows {
			fmt.Fprintf(os.Stderr, "row %d: %q (conf %.3f, %d digits)\n", i, row.Text, row.Confidence, len(row.Digits))
		}
	}
	return nil
}

func printUsage() {
	prog := filepath.Base(os.Args[0])
	fmt.Fprintf(os.Stderr, `Usage: %s [flags] [pipeline ops...] IMAGE

Flags:
  -models DIR       directory containing panel.onnx, digits.onnx, classes.json
                    (optional — omit to use this binary's built-in models)
  -profile NAME     display-type hint (e.g. tank_gauge)
  -conf N           detection confidence threshold (default 0.25)
  -skip-panel       bypass stage-1 panel localization (use when IMAGE is
                    already cropped to the display)
  -v                print per-digit detail
  -version          print version and exit

Pipeline ops (ssocr-style; applied in order, before detection):
  crop X Y W H      crop to a fixed region
  rotate DEGREES    rotate clockwise about the center
  mirror horiz|vert flip the image
  invert            invert light/dark
  grayscale         convert to grayscale
  threshold PCT     binarize at PCT%% between the image's min/max luminance
  otsu_threshold    binarize using Otsu's automatic threshold
`, prog)
}
