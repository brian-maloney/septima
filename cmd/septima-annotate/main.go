// Command septima-annotate bootstraps a real-image fine-tuning set.
//
// For each image in a directory's ground_truth.json it runs the live two-stage
// pipeline (bright-panel crop + digit detection + post-processing). Where the
// decoded string already matches the ground truth, the detected boxes are
// trustworthy labels, so it writes the panel crop + YOLO labels into a fine-tune
// dataset (oversampling the train split). Where it disagrees, it drops the crop
// and a predicted-label starting point into a review/ folder for quick manual
// correction.
//
// Usage:
//
//	go run ./cmd/septima-annotate -in tanktests
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"image"
	"image/jpeg"
	_ "image/png"
	"os"

	_ "golang.org/x/image/webp"
	"path/filepath"
	"strings"

	"github.com/brian-maloney/septima/internal/assemble"
	"github.com/brian-maloney/septima/internal/detect"
	"github.com/brian-maloney/septima/internal/imageproc"
)

type groundTruth struct {
	Images []imageCase `json:"images"`
}

type imageCase struct {
	File  string `json:"file"`
	Value string `json:"value"`
}

func main() {
	var (
		inDir      = flag.String("in", "", "directory with images + ground_truth.json")
		outDir     = flag.String("out", "training/data/real_tank", "fine-tune dataset root")
		modelDir   = flag.String("models", "models", "dir with digits.onnx + classes.json")
		panelModel = flag.String("panel-model", "", "use this panel.onnx for cropping instead of the bright-panel heuristic (matches the live pipeline)")
		conf       = flag.Float64("conf", 0.25, "detection confidence")
		iou        = flag.Float64("iou", 0.45, "NMS IoU")
		valFrac    = flag.Float64("val-frac", 0.2, "fraction of matched images held out for val")
		repeat     = flag.Int("repeat", 6, "oversample factor for matched train images")
	)
	flag.Parse()
	if *inDir == "" {
		fmt.Fprintln(os.Stderr, "usage: septima-annotate -in DIR [flags]")
		os.Exit(2)
	}

	classes, err := detect.LoadClasses(*modelDir)
	must(err)
	names := classes.DigitClasses

	var panel *detect.Model
	if *panelModel != "" {
		panel, err = detect.OpenModel(*panelModel, len(classes.PanelClasses), classes.InputSize)
		must(err)
		defer panel.Close()
	}

	model, err := detect.OpenModel(filepath.Join(*modelDir, "digits.onnx"), len(names), classes.InputSize)
	must(err)
	defer model.Close()

	data, err := os.ReadFile(filepath.Join(*inDir, "ground_truth.json"))
	must(err)
	var gt groundTruth
	must(json.Unmarshal(data, &gt))

	reviewDir := filepath.Join(*outDir, "review")
	for _, sub := range []string{"train/images", "train/labels", "val/images", "val/labels"} {
		must(os.MkdirAll(filepath.Join(*outDir, sub), 0o755))
	}
	must(os.MkdirAll(reviewDir, 0o755))

	var matched, reviewed int
	var reviewList []string
	for _, c := range gt.Images {
		path := filepath.Join(*inDir, c.File)
		img, err := loadImage(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", c.File, err)
			continue
		}

		crop, dets := detectOnPanel(img, panel, model, *conf, *iou)
		reading := assemble.Assemble(dets, names)
		stem := "tank_" + strings.TrimSuffix(c.File, filepath.Ext(c.File))
		labels := yoloLabels(dets, crop.Bounds())

		if reading.Text == c.Value {
			matched++
			if isVal(c.File, *valFrac) {
				writeSample(*outDir, "val", stem, crop, labels, 1)
			} else {
				writeSample(*outDir, "train", stem, crop, labels, *repeat)
			}
		} else {
			reviewed++
			reviewList = append(reviewList, fmt.Sprintf("%s: got %q want %q", c.File, reading.Text, c.Value))
			writeImage(filepath.Join(reviewDir, stem+".jpg"), crop)
			// Predicted labels as a correction starting point.
			os.WriteFile(filepath.Join(reviewDir, stem+".txt"), []byte(strings.Join(labels, "\n")+"\n"), 0o644)
			os.WriteFile(filepath.Join(reviewDir, stem+".expected.txt"),
				[]byte(fmt.Sprintf("got: %s\nwant: %s\n", reading.Text, c.Value)), 0o644)
		}
	}

	writeFinetuneYAML(*outDir, names)

	fmt.Printf("matched (auto-labeled): %d\nto review (manual fix):  %d\n", matched, reviewed)
	if len(reviewList) > 0 {
		fmt.Println("review:")
		for _, r := range reviewList {
			fmt.Println("  " + r)
		}
		fmt.Printf("  crops + predicted labels in %s\n", reviewDir)
	}
}

// detectOnPanel mirrors septima.Read's stage-1/stage-2 path and returns the
// panel crop and the post-processed detections in crop coordinates.
// If panel is non-nil its detections are used to locate the crop; otherwise
// the bright-panel heuristic is used.
func detectOnPanel(img image.Image, panel, digits *detect.Model, conf, iou float64) (*image.RGBA, []detect.Detection) {
	var region image.Rectangle
	if panel != nil {
		dets, err := panel.Detect(img, conf, iou)
		must(err)
		if len(dets) > 0 {
			best := dets[0]
			for _, d := range dets {
				if d.Score > best.Score {
					best = d
				}
			}
			region = best.Box
		}
	}
	if region.Empty() {
		var ok bool
		region, ok = detect.FindBrightPanel(img)
		if !ok {
			region = img.Bounds()
		}
	}
	region = imageproc.PadRect(region, img.Bounds(), 0.08)
	crop := imageproc.Crop(img, region)

	dets, err := digits.Detect(crop, conf, iou)
	must(err)
	dets = detect.DedupeAcrossClasses(dets, 0.5)
	return crop, dets
}

func yoloLabels(dets []detect.Detection, b image.Rectangle) []string {
	w, h := float64(b.Dx()), float64(b.Dy())
	var lines []string
	for _, d := range dets {
		cx := float64(d.Box.Min.X+d.Box.Max.X) / 2 / w
		cy := float64(d.Box.Min.Y+d.Box.Max.Y) / 2 / h
		bw := float64(d.Box.Dx()) / w
		bh := float64(d.Box.Dy()) / h
		lines = append(lines, fmt.Sprintf("%d %.6f %.6f %.6f %.6f", d.Class, cx, cy, bw, bh))
	}
	return lines
}

func writeSample(root, split, stem string, img *image.RGBA, labels []string, repeat int) {
	for k := 0; k < repeat; k++ {
		name := stem
		if repeat > 1 {
			name = fmt.Sprintf("%s_r%d", stem, k)
		}
		writeImage(filepath.Join(root, split, "images", name+".jpg"), img)
		os.WriteFile(filepath.Join(root, split, "labels", name+".txt"),
			[]byte(strings.Join(labels, "\n")+"\n"), 0o644)
	}
}

func writeImage(path string, img image.Image) {
	f, err := os.Create(path)
	must(err)
	defer f.Close()
	must(jpeg.Encode(f, img, &jpeg.Options{Quality: 92}))
}

// writeFinetuneYAML writes a data config combining the existing synthetic/public
// digit training data with the real crops (real oversampled), validating on the
// held-out real split so the metric tracks real-image accuracy.
func writeFinetuneYAML(outDir string, names []string) {
	dataRoot, _ := filepath.Abs(filepath.Dir(outDir)) // training/data
	real := filepath.Base(outDir)                     // real_tank
	var nb strings.Builder
	for i, n := range names {
		fmt.Fprintf(&nb, "  %d: %q\n", i, n)
	}
	yaml := fmt.Sprintf(`# Fine-tune config: synthetic/public digits + oversampled real crops.
# Validation uses the synthetic digits val split (always present); the true
# real-image metric is the Go bench (cmd/septima-bench tanktests).
path: %s
train:
  - digits/train/images
  - %s/train/images
val:
  - digits/val/images
names:
%s`, dataRoot, real, nb.String())
	must(os.WriteFile(filepath.Join(dataRoot, "data_finetune.yaml"), []byte(yaml), 0o644))
	fmt.Printf("wrote %s\n", filepath.Join(dataRoot, "data_finetune.yaml"))
}

func isVal(file string, frac float64) bool {
	h := fnv.New32a()
	h.Write([]byte(file))
	return float64(h.Sum32()%1000)/1000.0 < frac
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
