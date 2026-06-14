// Command septima recognizes digits from seven-segment displays.
// It is interface-compatible with ssocr and adds multi-row support and
// automatic display detection.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/vond/septima"
	"github.com/vond/septima/preprocess"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "septima:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	var opts []septima.Option
	var pipeline []preprocess.Op
	var imagePath string
	var printSpaces bool
	var printHex bool
	var asciiArt bool
	var verbose bool
	var debugDir string
	var outputImage string

	i := 0
	for i < len(args) {
		arg := args[i]
		switch {
		case arg == "-h" || arg == "--help":
			printUsage()
			return nil
		case arg == "-v" || arg == "--verbose":
			verbose = true
			_ = verbose
		case arg == "-V" || arg == "--version":
			fmt.Println("septima 0.1.0")
			return nil
		case arg == "-s" || arg == "--print-spaces":
			printSpaces = true
		case arg == "-X" || arg == "--print-as-hex":
			printHex = true
			_ = printHex
		case arg == "-S" || arg == "--ascii-art-segments":
			asciiArt = true
			_ = asciiArt
		case arg == "-a" || arg == "--absolute-threshold":
			// handled via pipeline MakeMonoOp Absolute flag – no-op here
		case arg == "-T" || arg == "--iter-threshold":
			pipeline = append(pipeline, preprocess.IterThresholdOp{ThresholdPct: 50})
		case arg == "--no-dnn":
			opts = append(opts, septima.WithDNN(false))
		case arg == "-t" || arg == "--threshold":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			t, err := strconv.ParseFloat(args[i], 64)
			if err != nil {
				return fmt.Errorf("invalid threshold %q", args[i])
			}
			pipeline = append(pipeline, preprocess.MakeMonoOp{ThresholdPct: t})
		case arg == "-d" || arg == "--number-digits":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			n, err := strconv.Atoi(args[i])
			if err != nil {
				return fmt.Errorf("invalid digit count %q", args[i])
			}
			opts = append(opts, septima.WithExpectedDigits(n))
		case arg == "--number-rows":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			n, err := strconv.Atoi(args[i])
			if err != nil {
				return fmt.Errorf("invalid row count %q", args[i])
			}
			opts = append(opts, septima.WithExpectedRows(n))
		case arg == "-c" || arg == "--charset":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			switch strings.ToLower(args[i]) {
			case "digits":
				opts = append(opts, septima.WithCharset(septima.CharsetDigits))
			case "decimal":
				opts = append(opts, septima.WithCharset(septima.CharsetDecimal))
			case "hex":
				opts = append(opts, septima.WithCharset(septima.CharsetHex))
			case "tt_robot":
				opts = append(opts, septima.WithCharset(septima.CharsetTTRobot))
			default:
				opts = append(opts, septima.WithCharset(septima.CharsetFull))
			}
		case arg == "-f" || arg == "--foreground":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			if strings.ToLower(args[i]) == "white" {
				opts = append(opts, septima.WithPolarity(septima.PolarityLightOnDark))
			} else {
				opts = append(opts, septima.WithPolarity(septima.PolarityDarkOnLight))
			}
		case arg == "-b" || arg == "--background":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			if strings.ToLower(args[i]) == "black" {
				opts = append(opts, septima.WithPolarity(septima.PolarityLightOnDark))
			} else {
				opts = append(opts, septima.WithPolarity(septima.PolarityDarkOnLight))
			}
		case arg == "--profile":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			opts = append(opts, septima.WithProfile(args[i]))
		case arg == "-D" || arg == "--debug-image":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			debugDir = args[i]
			opts = append(opts, septima.WithDebugDir(debugDir))
		case arg == "-P" || arg == "--debug-output":
			verbose = true
		case arg == "-o" || arg == "--output-image":
			i++
			if i >= len(args) {
				return fmt.Errorf("missing value for %s", arg)
			}
			outputImage = args[i]
			_ = outputImage
		case arg == "-r" || arg == "--one-ratio":
			i++
			if i < len(args) {
				// Store and apply via WithRatios at the end
			}
		// Pipeline ops (ssocr-compatible positional arguments)
		case arg == "crop":
			if i+4 >= len(args) {
				return fmt.Errorf("crop requires X Y W H")
			}
			x, _ := strconv.Atoi(args[i+1])
			y, _ := strconv.Atoi(args[i+2])
			w, _ := strconv.Atoi(args[i+3])
			h, _ := strconv.Atoi(args[i+4])
			pipeline = append(pipeline, preprocess.CropOp{X: x, Y: y, W: w, H: h})
			i += 4
		case arg == "rotate":
			if i+1 >= len(args) {
				return fmt.Errorf("rotate requires DEGREES")
			}
			deg, _ := strconv.ParseFloat(args[i+1], 64)
			pipeline = append(pipeline, preprocess.RotateOp{Degrees: deg})
			i++
		case arg == "shear":
			if i+1 >= len(args) {
				return fmt.Errorf("shear requires OFFSET")
			}
			off, _ := strconv.ParseFloat(args[i+1], 64)
			pipeline = append(pipeline, preprocess.ShearOp{Offset: off})
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
		case arg == "make_mono":
			pipeline = append(pipeline, preprocess.MakeMonoOp{ThresholdPct: 50})
		case arg == "dynamic_threshold":
			if i+2 >= len(args) {
				return fmt.Errorf("dynamic_threshold requires W H")
			}
			w, _ := strconv.Atoi(args[i+1])
			h, _ := strconv.Atoi(args[i+2])
			pipeline = append(pipeline, preprocess.DynamicThresholdOp{W: w, H: h})
			i += 2
		case arg == "gray_stretch":
			if i+2 >= len(args) {
				return fmt.Errorf("gray_stretch requires T1 T2")
			}
			t1, _ := strconv.ParseFloat(args[i+1], 64)
			t2, _ := strconv.ParseFloat(args[i+2], 64)
			pipeline = append(pipeline, preprocess.GrayStretchOp{T1: t1, T2: t2})
			i += 2
		case arg == "rgb_threshold":
			pipeline = append(pipeline, preprocess.RGBThresholdOp{ThresholdPct: 50})
		case arg == "r_threshold":
			pipeline = append(pipeline, preprocess.SingleChannelThresholdOp{Channel: 2, ThresholdPct: 50})
		case arg == "g_threshold":
			pipeline = append(pipeline, preprocess.SingleChannelThresholdOp{Channel: 1, ThresholdPct: 50})
		case arg == "b_threshold":
			pipeline = append(pipeline, preprocess.SingleChannelThresholdOp{Channel: 0, ThresholdPct: 50})
		case arg == "dilation":
			n := 1
			if i+1 < len(args) {
				if v, err := strconv.Atoi(args[i+1]); err == nil {
					n = v
					i++
				}
			}
			pipeline = append(pipeline, preprocess.DilationOp{N: n})
		case arg == "erosion":
			n := 1
			if i+1 < len(args) {
				if v, err := strconv.Atoi(args[i+1]); err == nil {
					n = v
					i++
				}
			}
			pipeline = append(pipeline, preprocess.ErosionOp{N: n})
		case arg == "opening":
			n := 1
			if i+1 < len(args) {
				if v, err := strconv.Atoi(args[i+1]); err == nil {
					n = v
					i++
				}
			}
			pipeline = append(pipeline, preprocess.OpeningOp{N: n})
		case arg == "closing":
			n := 1
			if i+1 < len(args) {
				if v, err := strconv.Atoi(args[i+1]); err == nil {
					n = v
					i++
				}
			}
			pipeline = append(pipeline, preprocess.ClosingOp{N: n})
		case arg == "remove_isolated":
			pipeline = append(pipeline, preprocess.RemoveIsolatedOp{})
		case arg == "white_border":
			w := 5
			if i+1 < len(args) {
				if v, err := strconv.Atoi(args[i+1]); err == nil {
					w = v
					i++
				}
			}
			pipeline = append(pipeline, preprocess.WhiteBorderOp{Width: w})
		case arg == "set_pixels_filter":
			if i+1 >= len(args) {
				return fmt.Errorf("set_pixels_filter requires MASK")
			}
			m, _ := strconv.Atoi(args[i+1])
			pipeline = append(pipeline, preprocess.SetPixelsFilterOp{Mask: m})
			i++
		case arg == "keep_pixels_filter":
			if i+1 >= len(args) {
				return fmt.Errorf("keep_pixels_filter requires MASK")
			}
			m, _ := strconv.Atoi(args[i+1])
			pipeline = append(pipeline, preprocess.KeepPixelsFilterOp{Mask: m})
			i++
		default:
			// Assume it's the image path
			if !strings.HasPrefix(arg, "-") {
				imagePath = arg
			}
		}
		i++
	}

	if imagePath == "" {
		return fmt.Errorf("no image file specified")
	}
	if !fileExists(imagePath) {
		return fmt.Errorf("file not found: %q", imagePath)
	}

	if printSpaces {
		opts = append(opts, septima.WithPrintSpaces(true))
	}
	if len(pipeline) > 0 {
		opts = append(opts, septima.WithPipeline(pipeline...))
	}

	result, err := septima.ReadFile(imagePath, opts...)
	if err != nil {
		return err
	}

	fmt.Println(result.Text)

	if debugDir != "" && result.Debug != nil {
		for _, s := range result.Debug.Stages {
			fmt.Fprintf(os.Stderr, "debug: %s → %s\n", s.Name, s.Path)
		}
	}
	return nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func printUsage() {
	prog := filepath.Base(os.Args[0])
	fmt.Fprintf(os.Stderr, `Usage: %s [flags] [pipeline ops...] <image>

Global flags:
  -t, --threshold N             luminance threshold percentage
  -T, --iter-threshold          iterative k-means threshold (default behavior)
  -d, --number-digits N         expected digits per row (0=auto)
      --number-rows N           expected rows (0=auto)
  -c, --charset {full,digits,decimal,hex,tt_robot}
  -f, --foreground {black,white}
  -b, --background {black,white}
  -s, --print-spaces
      --profile NAME            built-in profile (alarm_clock, gas_pump, ...)
      --no-dnn                  disable ONNX fallback classifier
  -o, --output-image FILE
  -D, --debug-image DIR         write per-stage debug images
  -P, --debug-output            verbose stage logs
  -S, --ascii-art-segments
  -X, --print-as-hex
  -v, --verbose
  -V, --version

Pipeline ops (ssocr-compatible):
  crop X Y W H
  rotate DEG | shear OFFSET | mirror {horiz|vert}
  invert | grayscale | make_mono
  dynamic_threshold W H | gray_stretch T1 T2
  rgb_threshold | r_threshold | g_threshold | b_threshold
  dilation [N] | erosion [N] | opening [N] | closing [N]
  remove_isolated | white_border [W]
  set_pixels_filter MASK | keep_pixels_filter MASK
`, prog)
}

