// Command septima-diag runs a single detection stage on an image and dumps the
// raw detections (and an annotated PNG), for diagnosing the pipeline. It runs
// one model directly — no panel crop, no assembly — so you can see exactly what
// the detector produces.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/brian-maloney/septima/internal/detect"
)

func main() {
	var (
		modelPath = flag.String("model", "models/digits.onnx", "ONNX model")
		modelDir  = flag.String("models", "models", "dir with classes.json")
		stage     = flag.String("stage", "digits", "digits|panel (which class table)")
		conf      = flag.Float64("conf", 0.25, "confidence threshold")
		iou       = flag.Float64("iou", 0.45, "NMS IoU threshold")
		cropStr   = flag.String("crop", "", "optional crop x0,y0,x1,y1 before detect")
		usePanel  = flag.Bool("panel", false, "crop via FindBrightPanel heuristic first")
		out       = flag.String("out", "", "write annotated PNG here")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: septima-diag [flags] IMAGE")
		flag.PrintDefaults()
		os.Exit(2)
	}

	classes, err := detect.LoadClasses(*modelDir)
	must(err)
	names := classes.DigitClasses
	if *stage == "panel" {
		names = classes.PanelClasses
	}

	f, err := os.Open(flag.Arg(0))
	must(err)
	defer f.Close()
	img, _, err := image.Decode(f)
	must(err)

	work := img
	off := image.Point{}
	if *cropStr != "" {
		r := parseRect(*cropStr)
		work = cropTo(img, r)
		off = r.Min
		fmt.Fprintf(os.Stderr, "cropped to %v\n", r)
	} else if *usePanel {
		if r, ok := detect.FindBrightPanel(img); ok {
			work = cropTo(img, r)
			off = r.Min
			fmt.Fprintf(os.Stderr, "panel heuristic -> %v\n", r)
		} else {
			fmt.Fprintln(os.Stderr, "panel heuristic found nothing; using full frame")
		}
	}

	model, err := detect.OpenModel(*modelPath, len(names), classes.InputSize)
	must(err)
	defer model.Close()

	dets, err := model.Detect(work, *conf, *iou)
	must(err)
	for i := range dets {
		dets[i].Box = dets[i].Box.Add(off)
	}
	sort.Slice(dets, func(i, j int) bool { return dets[i].Box.Min.X < dets[j].Box.Min.X })

	fmt.Fprintf(os.Stderr, "%d detections (left-to-right):\n", len(dets))
	var chars []string
	for _, d := range dets {
		ch := "?"
		if d.Class >= 0 && d.Class < len(names) {
			ch = names[d.Class]
		}
		chars = append(chars, ch)
		fmt.Fprintf(os.Stderr, "  %-2s score %.3f  box %v  (w=%d h=%d)\n",
			ch, d.Score, d.Box, d.Box.Dx(), d.Box.Dy())
	}
	fmt.Println(strings.Join(chars, ""))

	if *out != "" {
		writeAnnotated(img, dets, names, *out)
		fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
	}
}

func writeAnnotated(img image.Image, dets []detect.Detection, names []string, path string) {
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, img, b.Min, draw.Src)
	red := color.RGBA{255, 30, 30, 255}
	for _, d := range dets {
		drawRect(rgba, d.Box, red, 3)
	}
	f, err := os.Create(path)
	must(err)
	defer f.Close()
	must(png.Encode(f, rgba))
}

func drawRect(img *image.RGBA, r image.Rectangle, c color.RGBA, w int) {
	r = r.Intersect(img.Bounds())
	for t := 0; t < w; t++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.SetRGBA(x, r.Min.Y+t, c)
			img.SetRGBA(x, r.Max.Y-1-t, c)
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			img.SetRGBA(r.Min.X+t, y, c)
			img.SetRGBA(r.Max.X-1-t, y, c)
		}
	}
}

func cropTo(img image.Image, r image.Rectangle) image.Image {
	r = r.Intersect(img.Bounds())
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(dst, dst.Bounds(), img, r.Min, draw.Src)
	return dst
}

func parseRect(s string) image.Rectangle {
	p := strings.Split(s, ",")
	if len(p) != 4 {
		fmt.Fprintln(os.Stderr, "crop must be x0,y0,x1,y1")
		os.Exit(2)
	}
	n := make([]int, 4)
	for i := range p {
		v, err := strconv.Atoi(strings.TrimSpace(p[i]))
		must(err)
		n[i] = v
	}
	return image.Rect(n[0], n[1], n[2], n[3])
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
