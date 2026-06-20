// Command septima-panel-bootstrap adds real panel-localization training data for
// bright-on-dark displays. For each image it runs FindBrightPanel (which is
// reliable on a lit LCD against a dark frame, e.g. the tank/propane gauges) and
// writes the full frame plus the panel box (class 0) into the panel dataset.
// This anchors those displays so a trained panel.onnx reproduces the crop that
// already reads correctly, instead of regressing them.
//
// Usage:
//   go run ./cmd/septima-panel-bootstrap -in tanktests -repeat 4
//   go run ./cmd/septima-panel-bootstrap -in tests              # picks up 0502 (others skipped)
package main

import (
	"flag"
	"fmt"
	"hash/fnv"
	"image"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	_ "golang.org/x/image/webp"

	"github.com/brian-maloney/septima/internal/detect"
)

func main() {
	var (
		inDir   = flag.String("in", "", "directory of images")
		outDir  = flag.String("out", "training/data/panel", "panel dataset root")
		prefix  = flag.String("prefix", "realpanel", "filename prefix for written samples")
		repeat  = flag.Int("repeat", 4, "oversample factor for train images")
		valFrac = flag.Float64("val-frac", 0.15, "fraction held out for val")
	)
	flag.Parse()
	if *inDir == "" {
		fmt.Fprintln(os.Stderr, "usage: septima-panel-bootstrap -in DIR [flags]")
		os.Exit(2)
	}

	entries, err := os.ReadDir(*inDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	for _, sub := range []string{"train/images", "train/labels", "val/images", "val/labels"} {
		os.MkdirAll(filepath.Join(*outDir, sub), 0o755)
	}

	var written, skipped int
	for _, e := range entries {
		if e.IsDir() || !isImage(e.Name()) {
			continue
		}
		path := filepath.Join(*inDir, e.Name())
		img, err := loadImage(path)
		if err != nil {
			continue
		}
		region, ok := detect.FindBrightPanel(img)
		if !ok {
			skipped++
			continue
		}
		b := img.Bounds()
		w, h := float64(b.Dx()), float64(b.Dy())
		cx := float64(region.Min.X+region.Max.X) / 2 / w
		cy := float64(region.Min.Y+region.Max.Y) / 2 / h
		bw, bh := float64(region.Dx())/w, float64(region.Dy())/h
		label := fmt.Sprintf("0 %.6f %.6f %.6f %.6f", cx, cy, bw, bh)
		stem := *prefix + "_" + strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))

		if isVal(e.Name(), *valFrac) {
			writeSample(*outDir, "val", stem, img, label, 1)
		} else {
			writeSample(*outDir, "train", stem, img, label, *repeat)
		}
		written++
	}
	fmt.Printf("panel-bootstrap: wrote %d images (skipped %d with no bright panel) -> %s\n",
		written, skipped, *outDir)
}

func writeSample(root, split, stem string, img image.Image, label string, repeat int) {
	for k := 0; k < repeat; k++ {
		name := stem
		if repeat > 1 {
			name = fmt.Sprintf("%s_r%d", stem, k)
		}
		f, err := os.Create(filepath.Join(root, split, "images", name+".jpg"))
		if err != nil {
			continue
		}
		jpeg.Encode(f, img, &jpeg.Options{Quality: 90})
		f.Close()
		os.WriteFile(filepath.Join(root, split, "labels", name+".txt"), []byte(label+"\n"), 0o644)
	}
}

func isImage(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png", ".webp":
		return true
	}
	return false
}

func isVal(name string, frac float64) bool {
	h := fnv.New32a()
	h.Write([]byte(name))
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
