// Command septima-bench evaluates the recognizer against a directory's
// ground_truth.json, reporting exact-string accuracy and mean per-character
// (Levenshtein) accuracy.
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

	"github.com/brian-maloney/septima"
)

type groundTruth struct {
	Description string      `json:"description"`
	Images      []imageCase `json:"images"`
}

type imageCase struct {
	File        string   `json:"file"`
	Value       string   `json:"value"`
	DisplayType string   `json:"display_type"`
	Rows        []string `json:"rows"`
	Notes       string   `json:"notes"`
	Source      string   `json:"source"`
	MedianPx    float64  `json:"median_px"`
}

// srcStat accumulates per-source pass/fail counts for the breakdown table.
type srcStat struct {
	total, exact, digitsExact int
}

func main() {
	var (
		modelDir = flag.String("models", "", "directory containing the ONNX models and classes.json")
		hinted   = flag.Bool("hinted", false, "pass display_type/rows hints from ground truth")
		noPanel  = flag.Bool("nopanel", false, "diagnostic: skip panel stage, run digits on the whole image")
		oracle   = flag.Bool("oracle", false, "diagnostic: crop to the ground-truth digit-box union (perfect localization)")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: septima-bench [flags] DIR")
		flag.PrintDefaults()
		os.Exit(2)
	}
	dir := flag.Arg(0)

	data, err := os.ReadFile(filepath.Join(dir, "ground_truth.json"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	var gt groundTruth
	if err := json.Unmarshal(data, &gt); err != nil {
		fmt.Fprintln(os.Stderr, "parse error:", err)
		os.Exit(1)
	}

	var exact, digitsExact, total, missing int
	var charAccSum float64
	bySource := map[string]*srcStat{}
	var srcOrder []string
	// Resolution buckets: small crops (median digit < smallPx tall) are at/near the
	// detector's resolution floor, so their failures are largely benchmark
	// contamination rather than model deficiency. Splitting them out shows the
	// recognition rate on genuinely readable displays.
	const smallPx = 30
	var readable, small srcStat
	for _, c := range gt.Images {
		path := filepath.Join(dir, c.File)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			missing++
			continue
		}
		total++

		opts := []septima.Option{}
		if *modelDir != "" {
			opts = append(opts, septima.WithModelDir(*modelDir))
		}
		if *hinted {
			opts = append(opts, septima.WithProfile(c.DisplayType))
			if len(c.Rows) > 0 {
				opts = append(opts, septima.WithExpectedRows(len(c.Rows)))
			}
		}

		var res septima.Result
		var err error
		switch {
		case *oracle:
			// Crop to the GT digit-box union (perfect localization), then digits.
			opts = append(opts, septima.WithSkipPanel(true))
			res, err = readOracle(dir, path, opts)
		case *noPanel:
			opts = append(opts, septima.WithSkipPanel(true))
			res, err = readImage(path, opts)
		default:
			res, err = septima.ReadFile(path, opts...)
		}
		got := res.Text
		status := "ok"
		if err != nil {
			got = ""
			status = "ERR: " + err.Error()
		}
		ca := charAccuracy(got, c.Value)
		charAccSum += ca
		digitsOK := digitsOnly(got) == digitsOnly(c.Value)
		if digitsOK {
			digitsExact++
		}
		src := c.Source
		if src == "" {
			src = "(unknown)"
		}
		st := bySource[src]
		if st == nil {
			st = &srcStat{}
			bySource[src] = st
			srcOrder = append(srcOrder, src)
		}
		st.total++
		if digitsOK {
			st.digitsExact++
		}
		if got == c.Value {
			st.exact++
		}
		bucket := &readable
		if c.MedianPx > 0 && c.MedianPx < smallPx {
			bucket = &small
		}
		bucket.total++
		if digitsOK {
			bucket.digitsExact++
		}
		if got == c.Value {
			bucket.exact++
		}
		if got == c.Value {
			exact++
			fmt.Printf("PASS  %-16s %q\n", c.File, got)
		} else {
			// [digits-ok] flags a failure whose digit sequence is correct and that
			// differs only in punctuation placement ('.'/':') — often GT-label noise
			// rather than a recognition error (decimal points are inconsistently
			// annotated across the source datasets).
			tag := ""
			if digitsOK {
				tag = " [digits-ok]"
			}
			fmt.Printf("FAIL  %-16s got %q want %q (char %.0f%%)%s %s\n", c.File, got, c.Value, ca*100, tag, status)
		}
	}

	fmt.Println("----")
	if total == 0 {
		fmt.Printf("no images found (%d listed, all missing)\n", missing)
		return
	}
	fmt.Printf("exact: %d/%d (%.1f%%)   mean char acc: %.1f%%   missing: %d\n",
		exact, total, 100*float64(exact)/float64(total), 100*charAccSum/float64(total), missing)
	fmt.Printf("digits-only exact: %d/%d (%.1f%%)   <- ignores '.'/':' placement (decimal-label noise)\n",
		digitsExact, total, 100*float64(digitsExact)/float64(total))

	if small.total > 0 {
		fmt.Printf("by resolution: readable (>=%dpx) %d/%d (%.1f%%)   small (<%dpx) %d/%d (%.1f%%)\n",
			smallPx, readable.exact, readable.total, 100*float64(readable.exact)/float64(readable.total),
			smallPx, small.exact, small.total, 100*float64(small.exact)/float64(small.total))
	}

	// Per-source breakdown: which datasets pass, and how much of each source's
	// failure is punctuation/GT-label noise (the strict-vs-digits-only gap) versus
	// genuine digit-recognition error. Sorted worst strict pass-rate first.
	if len(srcOrder) > 1 {
		sort.Slice(srcOrder, func(i, j int) bool {
			a, b := bySource[srcOrder[i]], bySource[srcOrder[j]]
			ra := float64(a.exact) / float64(a.total)
			rb := float64(b.exact) / float64(b.total)
			if ra != rb {
				return ra < rb
			}
			return srcOrder[i] < srcOrder[j]
		})
		fmt.Println("---- by source (worst strict first) ----")
		for _, s := range srcOrder {
			st := bySource[s]
			fmt.Printf("%-26s strict %3d/%-3d (%5.1f%%)   digits-only %3d/%-3d (%5.1f%%)\n",
				s, st.exact, st.total, 100*float64(st.exact)/float64(st.total),
				st.digitsExact, st.total, 100*float64(st.digitsExact)/float64(st.total))
		}
	}
}

// digitsOnly strips the punctuation whose placement is ambiguously/inconsistently
// annotated across the source datasets ('.' decimal point and ':' colon), leaving
// the digit sequence, sign, and row structure. Comparing on this isolates true
// digit recognition from decimal/colon label noise.
func digitsOnly(s string) string {
	return strings.NewReplacer(".", "", ":", "").Replace(s)
}

func decodeImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func readImage(path string, opts []septima.Option) (septima.Result, error) {
	img, err := decodeImage(path)
	if err != nil {
		return septima.Result{}, err
	}
	return septima.Read(img, opts...)
}

// readOracle crops to the union of the ground-truth digit boxes (perfect
// localization) and runs the digit stage on that crop.
func readOracle(dir, path string, opts []septima.Option) (septima.Result, error) {
	img, err := decodeImage(path)
	if err != nil {
		return septima.Result{}, err
	}
	stem := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	box, ok := gtUnionBox(filepath.Join(dir, "labels", stem+".txt"), img.Bounds())
	if !ok {
		return septima.Read(img, opts...)
	}
	return septima.Read(cropImage(img, box), opts...)
}

func gtUnionBox(lblPath string, b image.Rectangle) (image.Rectangle, bool) {
	data, err := os.ReadFile(lblPath)
	if err != nil {
		return image.Rectangle{}, false
	}
	w, h := float64(b.Dx()), float64(b.Dy())
	minX, minY, maxX, maxY := 1e18, 1e18, -1e18, -1e18
	any := false
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) < 5 {
			continue
		}
		cx, _ := strconv.ParseFloat(f[1], 64)
		cy, _ := strconv.ParseFloat(f[2], 64)
		bw, _ := strconv.ParseFloat(f[3], 64)
		bh, _ := strconv.ParseFloat(f[4], 64)
		x0, x1 := (cx-bw/2)*w, (cx+bw/2)*w
		y0, y1 := (cy-bh/2)*h, (cy+bh/2)*h
		minX, minY = min(minX, x0), min(minY, y0)
		maxX, maxY = max(maxX, x1), max(maxY, y1)
		any = true
	}
	if !any {
		return image.Rectangle{}, false
	}
	pw, ph := (maxX-minX)*0.10, (maxY-minY)*0.10
	r := image.Rect(int(minX-pw), int(minY-ph), int(maxX+pw), int(maxY+ph))
	return r.Intersect(b), true
}

func cropImage(img image.Image, r image.Rectangle) image.Image {
	r = r.Intersect(img.Bounds())
	if r.Empty() {
		return img
	}
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			dst.Set(x, y, img.At(r.Min.X+x, r.Min.Y+y))
		}
	}
	return dst
}

// charAccuracy returns 1 - normalized Levenshtein distance between got and want.
func charAccuracy(got, want string) float64 {
	if want == "" {
		if got == "" {
			return 1
		}
		return 0
	}
	d := levenshtein([]rune(got), []rune(want))
	acc := 1 - float64(d)/float64(len([]rune(want)))
	if acc < 0 {
		return 0
	}
	return acc
}

func levenshtein(a, b []rune) int {
	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min3(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(b)]
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}
