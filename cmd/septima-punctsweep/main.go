// Command septima-punctsweep sweeps a punctuation-specific confidence threshold
// over a benchmark directory and reports exact-match accuracy at each value.
//
// Rationale: the dominant real error is the phantom decimal point — the detector
// fires low-confidence '.'/':'/'-' on clutter. Rather than retraining (which made
// it worse), we make the PIPELINE more skeptical of punctuation: keep a digit at
// the base confidence, but require a punctuation glyph to clear a higher bar.
//
// Detection runs once per image (full-frame + panel-crop candidates); for each
// threshold the punctuation detections below it are dropped and the reading is
// re-finalized and re-selected, so the whole sweep is a single inference pass.
// T equal to the base conf (0.25) reproduces the current pipeline exactly.
//
// Usage:
//
//	go run ./cmd/septima-punctsweep [-thresholds 0.25,0.3,...] BENCHDIR
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	_ "golang.org/x/image/webp"

	"github.com/brian-maloney/septima/internal/assemble"
	"github.com/brian-maloney/septima/internal/detect"
	"github.com/brian-maloney/septima/internal/imageproc"
)

const (
	confThreshold = 0.25
	iouThreshold  = 0.45
	cropPad       = 0.30
)

// decodeFloor is the low confidence floor used at decode time so that sub-0.25
// punctuation detections exist and the punctuation bar can be swept BELOW the
// base. Digits are always held to digitThreshold; only the punctuation bar moves.
const (
	decodeFloor    = 0.05
	digitThreshold = 0.25
)

type groundTruth struct {
	Images []struct {
		File  string `json:"file"`
		Value string `json:"value"`
	} `json:"images"`
}

// cand holds one localization candidate's raw detections (full-frame or crop).
type cand struct {
	dets []detect.Detection
	have bool
}

func main() {
	modelDir := flag.String("models", "models", "model directory")
	grid := flag.String("thresholds", "0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.30,0.35", "punctuation conf thresholds")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: septima-punctsweep [-thresholds ...] BENCHDIR")
		os.Exit(2)
	}
	dir := flag.Arg(0)
	thresholds := parseGrid(*grid)

	gt, err := loadGT(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	classes, err := detect.LoadClasses(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	punct := punctClasses(classes.DigitClasses)

	digits, err := detect.OpenModel(filepath.Join(*modelDir, "digits.onnx"), len(classes.DigitClasses), classes.DigitSize())
	if err != nil {
		fmt.Fprintln(os.Stderr, "open digits:", err)
		os.Exit(1)
	}
	defer digits.Close()
	panel, _ := detect.OpenModel(filepath.Join(*modelDir, "panel.onnx"), len(classes.PanelClasses), classes.PanelSize())
	if panel != nil {
		defer panel.Close()
	}

	// One inference pass: cache each image's two candidate detection sets + GT.
	type sample struct {
		gt         string
		full, crop cand
	}
	var samples []sample
	for _, c := range gt.Images {
		img, err := decode(filepath.Join(dir, c.File))
		if err != nil {
			continue
		}
		s := sample{gt: c.Value}
		if d, err := digits.Detect(img, decodeFloor, iouThreshold); err == nil {
			s.full = cand{d, true}
		}
		if region, ok := locatePanel(img, panel, classes); ok {
			region = imageproc.PadRect(region, img.Bounds(), cropPad)
			if d, err := digits.Detect(imageproc.Crop(img, region), decodeFloor, iouThreshold); err == nil {
				for i := range d {
					d[i].Box = d[i].Box.Add(region.Min)
				}
				s.crop = cand{d, true}
			}
		}
		samples = append(samples, s)
	}

	fmt.Printf("punctuation-confidence sweep on %s (%d images, base conf %.2f)\n", dir, len(samples), confThreshold)
	fmt.Printf("  %-8s %-12s %s\n", "punctT", "exact", "(T=0.25 = current pipeline)")
	for _, t := range thresholds {
		exact := 0
		for _, s := range samples {
			if readingFor(s.full, s.crop, classes, punct, t) == s.gt {
				exact++
			}
		}
		fmt.Printf("  %-8.2f %d/%d (%.1f%%)\n", t, exact, len(samples), 100*float64(exact)/float64(len(samples)))
	}
}

// readingFor reproduces septima.Read for one image at punctuation threshold t:
// drop punctuation below t in each candidate, mean-select, finalize.
func readingFor(full, crop cand, classes detect.Classes, punct map[int]bool, t float64) string {
	fd := filterPunct(full.dets, punct, t)
	dets := fd
	if crop.have {
		cd := filterPunct(crop.dets, punct, t)
		if mean(cd) > mean(fd) {
			dets = cd
		}
	}
	return finalize(dets, classes).Text
}

// filterPunct keeps punctuation detections scoring >= t and digits scoring >=
// digitThreshold. Decoding happens at decodeFloor (0.05), so this is where the
// real per-class thresholds are applied — letting t drop below the base 0.25 to
// recover genuine but low-confidence decimals/colons.
func filterPunct(dets []detect.Detection, punct map[int]bool, t float64) []detect.Detection {
	out := dets[:0:0]
	for _, d := range dets {
		if punct[d.Class] {
			if d.Score < t {
				continue
			}
		} else if d.Score < digitThreshold {
			continue
		}
		out = append(out, d)
	}
	return out
}

func finalize(dets []detect.Detection, classes detect.Classes) assemble.Reading {
	dets = detect.DedupeAcrossClasses(dets, 0.5)
	if dot, col := classIndex(classes.DigitClasses, '.'), classIndex(classes.DigitClasses, ':'); dot >= 0 && col >= 0 {
		dets = detect.MergeColonDots(dets, dot, col)
	}
	return assemble.Assemble(dets, classes.DigitClasses)
}

func locatePanel(img image.Image, panel *detect.Model, classes detect.Classes) (image.Rectangle, bool) {
	if panel != nil {
		if dets, err := panel.Detect(img, confThreshold, iouThreshold); err == nil && len(dets) > 0 {
			sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
			return dets[0].Box, true
		}
	}
	return detect.FindBrightPanel(img)
}

func mean(dets []detect.Detection) float64 {
	if len(dets) == 0 {
		return 0
	}
	var s float64
	for _, d := range dets {
		s += d.Score
	}
	return s / float64(len(dets))
}

func punctClasses(names []string) map[int]bool {
	m := map[int]bool{}
	for i, n := range names {
		if n == "." || n == ":" || n == "-" {
			m[i] = true
		}
	}
	return m
}

func classIndex(names []string, r rune) int {
	for i, n := range names {
		if len(n) == 1 && rune(n[0]) == r {
			return i
		}
	}
	return -1
}

func parseGrid(s string) []float64 {
	var out []float64
	for _, p := range strings.Split(s, ",") {
		if v, err := strconv.ParseFloat(strings.TrimSpace(p), 64); err == nil {
			out = append(out, v)
		}
	}
	return out
}

func loadGT(dir string) (groundTruth, error) {
	var gt groundTruth
	data, err := os.ReadFile(filepath.Join(dir, "ground_truth.json"))
	if err != nil {
		return gt, err
	}
	return gt, json.Unmarshal(data, &gt)
}

func decode(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}
