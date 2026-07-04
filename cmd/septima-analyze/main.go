// Command septima-analyze runs an error-analysis pass over a benchmark
// directory (ground_truth.json + images/). For every image it reproduces the
// two localization candidates the live pipeline chooses between — full-frame
// digit detection and panel-crop digit detection — and records each candidate's
// reading. From that it reports:
//
//   - a reproduction of the live (mean-selected), oracle-selected, and
//     per-candidate exact-match accuracy, as a sanity check against known numbers;
//   - the BOTH-WRONG set (neither candidate equals ground truth) — the recognition
//     ceiling the project notes flagged as the real bottleneck;
//   - a character confusion matrix over the both-wrong cases (substitutions,
//     phantom insertions, missed deletions), to show WHICH errors dominate;
//   - a condition-tag histogram (colon/decimal/minus/multi-row/small-image/
//     length-mismatch/empty) to show under what conditions they happen.
//
// Full per-image detail is written to analyze_bothwrong.tsv next to the binary's
// working directory for offline inspection.
//
// Usage:
//
//	go run ./cmd/septima-analyze [-models DIR] BENCHDIR
//	go run ./cmd/septima-analyze training/data/digits/test
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	_ "golang.org/x/image/webp"

	"github.com/brian-maloney/septima/internal/assemble"
	"github.com/brian-maloney/septima/internal/detect"
	"github.com/brian-maloney/septima/internal/imageproc"
)

const (
	confThreshold  = 0.25 // matches septima defaultOptions (digit floor)
	punctThreshold = 0.20 // matches septima defaultOptions (punctuation floor)
	iouThreshold   = 0.45
	cropPad        = 0.30 // matches septima.Read panel-crop padding
	smallImgDim    = 400  // min(W,H) below this is tagged small-img
)

type groundTruth struct {
	Images []struct {
		File  string `json:"file"`
		Value string `json:"value"`
	} `json:"images"`
}

// record is one image's two-candidate outcome.
type record struct {
	file               string
	gt                 string
	full, crop         string  // candidate readings
	fullMean, cropMean float64 // mean detection confidence (the live selector)
	w, h               int
}

// best returns the candidate reading closest to GT (higher char accuracy) and
// whether it came from the crop path.
func (r record) best() (string, bool) {
	if charAcc(r.crop, r.gt) > charAcc(r.full, r.gt) {
		return r.crop, true
	}
	return r.full, false
}

// live reproduces septima.Read's candidate selection: the punct-agreement
// override (matches septima.punctAgreementPick) first, then mean confidence.
func (r record) live() string {
	if r.full != r.crop && digitsOnly(r.full) == digitsOnly(r.crop) {
		switch {
		case punctCount(r.full) > punctCount(r.crop) && wellFormed(r.full):
			return r.full
		case punctCount(r.crop) > punctCount(r.full) && wellFormed(r.crop):
			return r.crop
		}
	}
	if r.cropMean > r.fullMean {
		return r.crop
	}
	return r.full
}

func main() {
	modelDir := flag.String("models", "models", "directory with ONNX models and classes.json")
	topN := flag.Int("top", 25, "how many confusion pairs to print")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: septima-analyze [-models DIR] BENCHDIR")
		os.Exit(2)
	}
	dir := flag.Arg(0)

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
	digits, err := detect.OpenModel(filepath.Join(*modelDir, "digits.onnx"), len(classes.DigitClasses), classes.DigitSize())
	if err != nil {
		fmt.Fprintln(os.Stderr, "open digits:", err)
		os.Exit(1)
	}
	defer digits.Close()
	// Panel model is optional; nil means fall back to the bright-panel heuristic.
	panel, _ := detect.OpenModel(filepath.Join(*modelDir, "panel.onnx"), len(classes.PanelClasses), classes.PanelSize())
	if panel != nil {
		defer panel.Close()
	}

	// Punctuation uses a lower confidence bar than digits, matching septima.Read:
	// decode at the lower floor, then apply per-class bars before finalize.
	punctSet := punctClassSet(classes.DigitClasses)
	floor := math.Min(confThreshold, punctThreshold)

	var recs []record
	for _, c := range gt.Images {
		path := filepath.Join(dir, c.File)
		img, err := decode(path)
		if err != nil {
			continue
		}
		b := img.Bounds()
		rec := record{file: c.File, gt: c.Value, w: b.Dx(), h: b.Dy()}

		// Candidate A: full frame.
		if d, err := digits.Detect(img, floor, iouThreshold); err == nil {
			d = applyClassThresholds(d, punctSet, confThreshold, punctThreshold)
			rec.full = finalize(d, classes).Text
			rec.fullMean = meanScore(d)
		}
		// Candidate B: panel crop.
		if region, ok := locatePanel(img, panel, classes); ok {
			region = imageproc.PadRect(region, b, cropPad)
			if d, err := digits.Detect(imageproc.Crop(img, region), floor, iouThreshold); err == nil {
				for i := range d {
					d[i].Box = d[i].Box.Add(region.Min)
				}
				d = applyClassThresholds(d, punctSet, confThreshold, punctThreshold)
				rec.crop = finalize(d, classes).Text
				rec.cropMean = meanScore(d)
			}
		}
		recs = append(recs, rec)
	}

	report(recs, *topN)
}

// ----------------------------------------------------------------- pipeline replication

// finalize mirrors septima.finalizeReading.
func finalize(dets []detect.Detection, classes detect.Classes) assemble.Reading {
	dets = detect.DedupeAcrossClasses(dets, 0.5)
	dot, col := classIndex(classes.DigitClasses, '.'), classIndex(classes.DigitClasses, ':')
	if minus := classIndex(classes.DigitClasses, '-'); minus >= 0 {
		dets = detect.SuppressIndicatorMinus(dets, minus, dot, col)
	}
	if dot >= 0 && col >= 0 {
		dets = detect.MergeColonDots(dets, dot, col)
		dets = detect.SuppressDotsInsideColon(dets, dot, col)
	}
	return assemble.Assemble(dets, classes.DigitClasses)
}

// locatePanel mirrors septima.locatePanel: trained panel model if it fires, else
// the classical bright-panel heuristic.
func locatePanel(img image.Image, panel *detect.Model, classes detect.Classes) (image.Rectangle, bool) {
	if panel != nil {
		if dets, err := panel.Detect(img, confThreshold, iouThreshold); err == nil && len(dets) > 0 {
			sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
			return dets[0].Box, true
		}
	}
	return detect.FindBrightPanel(img)
}

// punctClassSet mirrors septima.punctClassSet.
func punctClassSet(names []string) map[int]bool {
	m := map[int]bool{}
	for i, n := range names {
		if n == "." || n == ":" || n == "-" {
			m[i] = true
		}
	}
	return m
}

// applyClassThresholds mirrors septima.applyClassThresholds: punctuation keeps
// detections >= punctT, digits >= digitT.
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

func meanScore(dets []detect.Detection) float64 {
	if len(dets) == 0 {
		return 0
	}
	var s float64
	for _, d := range dets {
		s += d.Score
	}
	return s / float64(len(dets))
}

func classIndex(names []string, r rune) int {
	for i, n := range names {
		if len(n) == 1 && rune(n[0]) == r {
			return i
		}
	}
	return -1
}

// ----------------------------------------------------------------- reporting

func report(recs []record, topN int) {
	var liveExact, oracleExact, fullExact, cropExact, bothWrong int
	var liveCharSum float64

	// Confusion accumulators (keyed by display rune, '~' = empty/edge).
	subs := map[[2]rune]int{} // want -> got
	phantom := map[rune]int{} // got char with no GT match (insertion)
	missed := map[rune]int{}  // GT char with no pred match (deletion)
	tags := map[string]int{}

	var bw []record
	for _, r := range recs {
		live := r.live()
		liveCharSum += charAcc(live, r.gt)
		if live == r.gt {
			liveExact++
		}
		if r.full == r.gt {
			fullExact++
		}
		if r.crop == r.gt {
			cropExact++
		}
		if r.full == r.gt || r.crop == r.gt {
			oracleExact++
			continue
		}
		// Both wrong.
		bothWrong++
		bw = append(bw, r)
		best, _ := r.best()
		accumulate(best, r.gt, subs, phantom, missed)
		for _, t := range conditionTags(r, best) {
			tags[t]++
		}
	}

	n := len(recs)
	fmt.Printf("=== reproduction (n=%d) ===\n", n)
	pct := func(x int) float64 { return 100 * float64(x) / float64(max(n, 1)) }
	fmt.Printf("live (selection)   exact : %d (%.1f%%)   char acc %.1f%%\n", liveExact, pct(liveExact), 100*liveCharSum/float64(max(n, 1)))
	fmt.Printf("full-frame only    exact : %d (%.1f%%)\n", fullExact, pct(fullExact))
	fmt.Printf("panel-crop only    exact : %d (%.1f%%)\n", cropExact, pct(cropExact))
	fmt.Printf("oracle-select      exact : %d (%.1f%%)   <- ceiling if selector were perfect\n", oracleExact, pct(oracleExact))
	fmt.Printf("BOTH-WRONG               : %d (%.1f%%)   <- recognition bottleneck\n", bothWrong, pct(bothWrong))

	fmt.Printf("\n=== both-wrong condition tags (n=%d) ===\n", bothWrong)
	for _, kv := range sortMapStr(tags) {
		fmt.Printf("  %-14s %4d  (%.0f%%)\n", kv.k, kv.v, 100*float64(kv.v)/float64(max(bothWrong, 1)))
	}

	fmt.Printf("\n=== top substitution confusions (want -> got) ===\n")
	type pair struct {
		w, g rune
		n    int
	}
	var ps []pair
	for k, v := range subs {
		ps = append(ps, pair{k[0], k[1], v})
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].n > ps[j].n })
	for i, p := range ps {
		if i >= topN {
			break
		}
		fmt.Printf("  %s -> %s : %d\n", disp(p.w), disp(p.g), p.n)
	}

	fmt.Printf("\n=== missed (deletions: GT char not read) ===\n")
	for _, kv := range sortMapRune(missed) {
		fmt.Printf("  %s : %d\n", disp(kv.k), kv.v)
	}
	fmt.Printf("\n=== phantom (insertions: read char not in GT) ===\n")
	for _, kv := range sortMapRune(phantom) {
		fmt.Printf("  %s : %d\n", disp(kv.k), kv.v)
	}

	writeDetail("analyze_bothwrong.tsv", bw)
	fmt.Printf("\nwrote per-image both-wrong detail to analyze_bothwrong.tsv (%d rows)\n", len(bw))

	// Selection losses: the live mean-confidence pick was wrong while the other
	// candidate read the GT exactly. These are the images a better selector
	// would recover (oracle-select minus live).
	fmt.Printf("\n=== selection losses (live wrong, other candidate exactly right) ===\n")
	for _, r := range recs {
		live := r.live()
		if live == r.gt {
			continue
		}
		other, otherMean, liveMean := r.full, r.fullMean, r.cropMean
		won := "crop"
		if live == r.full {
			other, otherMean, liveMean = r.crop, r.cropMean, r.fullMean
			won = "full"
		}
		if other != r.gt {
			continue
		}
		fmt.Printf("  %-50s gt=%-12q %s won mean %.3f vs %.3f  full=%q crop=%q\n",
			r.file, r.gt, won, liveMean, otherMean, r.full, r.crop)
	}

	// Margin distribution among correct picks, to gauge how much a selector
	// change could break: a flip rule endangers correct picks with small margins.
	fmt.Printf("\n=== mean-confidence margin |full-crop| when live is RIGHT and candidates disagree ===\n")
	var margins []float64
	for _, r := range recs {
		if r.live() == r.gt && r.full != r.crop {
			margins = append(margins, math.Abs(r.fullMean-r.cropMean))
		}
	}
	sort.Float64s(margins)
	if len(margins) > 0 {
		q := func(p float64) float64 { return margins[int(p*float64(len(margins)-1))] }
		fmt.Printf("  n=%d  p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f\n",
			len(margins), q(.10), q(.25), q(.50), q(.75), q(.90))
	}
}

// conditionTags labels a both-wrong case by GT/image properties so we can see
// under what conditions recognition fails.
func conditionTags(r record, best string) []string {
	var t []string
	if strings.ContainsRune(r.gt, ':') {
		t = append(t, "colon-gt")
	}
	if strings.ContainsRune(r.gt, '.') {
		t = append(t, "decimal-gt")
	}
	if strings.ContainsRune(r.gt, '-') {
		t = append(t, "minus-gt")
	}
	if strings.ContainsRune(r.gt, '\n') {
		t = append(t, "multirow")
	}
	if min(r.w, r.h) < smallImgDim {
		t = append(t, "small-img")
	}
	if best == "" {
		t = append(t, "empty-read")
	}
	if r.full == "" && r.crop == "" {
		t = append(t, "both-empty")
	}
	gl, bl := len([]rune(strip(r.gt))), len([]rune(strip(best)))
	switch {
	case best == "":
	case bl == gl:
		t = append(t, "same-len-sub") // pure recognition confusion (no count error)
	default:
		t = append(t, "len-mismatch") // localization/count error
	}
	return t
}

// accumulate aligns got against want and tallies edit operations.
func accumulate(got, want string, subs map[[2]rune]int, phantom, missed map[rune]int) {
	for _, op := range align([]rune(got), []rune(want)) {
		switch op.kind {
		case 's':
			subs[[2]rune{op.w, op.g}]++
		case 'i':
			phantom[op.g]++
		case 'd':
			missed[op.w]++
		}
	}
}

// ----------------------------------------------------------------- alignment

type op struct {
	kind byte // 'm' match, 's' sub, 'i' insertion (got extra), 'd' deletion (want missing)
	g, w rune
}

// align computes a minimal edit path between got and want via Levenshtein DP
// with traceback, returning the ordered operations.
func align(got, want []rune) []op {
	m, n := len(got), len(want)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			cost := 1
			if got[i-1] == want[j-1] {
				cost = 0
			}
			dp[i][j] = min3(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
		}
	}
	// Traceback from (m,n).
	var ops []op
	i, j := m, n
	for i > 0 || j > 0 {
		switch {
		case i > 0 && j > 0 && got[i-1] == want[j-1] && dp[i][j] == dp[i-1][j-1]:
			ops = append(ops, op{'m', got[i-1], want[j-1]})
			i, j = i-1, j-1
		case i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1]+1:
			ops = append(ops, op{'s', got[i-1], want[j-1]})
			i, j = i-1, j-1
		case i > 0 && dp[i][j] == dp[i-1][j]+1:
			ops = append(ops, op{'i', got[i-1], 0}) // got has an extra char
			i--
		default:
			ops = append(ops, op{'d', 0, want[j-1]}) // want char missing from got
			j--
		}
	}
	// Reverse into reading order.
	for l, r := 0, len(ops)-1; l < r; l, r = l+1, r-1 {
		ops[l], ops[r] = ops[r], ops[l]
	}
	return ops
}

// ----------------------------------------------------------------- helpers

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

func writeDetail(path string, bw []record) {
	var sb strings.Builder
	sb.WriteString("file\tgt\tfull\tcrop\tbest_is_crop\n")
	for _, r := range bw {
		_, crop := r.best()
		sb.WriteString(fmt.Sprintf("%s\t%q\t%q\t%q\t%v\n",
			r.file, oneline(r.gt), oneline(r.full), oneline(r.crop), crop))
	}
	os.WriteFile(path, []byte(sb.String()), 0o644)
}

func oneline(s string) string { return strings.ReplaceAll(s, "\n", "\\n") }
func strip(s string) string   { return strings.ReplaceAll(s, "\n", "") }

// digitsOnly removes '.'/':' (keeps row structure) — matches septima-bench's
// digits-only metric.
func digitsOnly(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '.' || r == ':' {
			return -1
		}
		return r
	}, s)
}

func punctCount(s string) int { return strings.Count(s, ".") + strings.Count(s, ":") }

// wellFormed reports whether every row reads like one number: at most one '.'
// and at most one ':' per row. A second dot in a row is a phantom tell.
func wellFormed(s string) bool {
	for _, row := range strings.Split(s, "\n") {
		if strings.Count(row, ".") > 1 || strings.Count(row, ":") > 1 {
			return false
		}
	}
	return true
}

func disp(r rune) string {
	switch r {
	case 0:
		return "∅"
	case '\n':
		return "␊"
	}
	return string(r)
}

func charAcc(got, want string) float64 {
	if want == "" {
		if got == "" {
			return 1
		}
		return 0
	}
	d := align([]rune(got), []rune(want))
	edits := 0
	for _, o := range d {
		if o.kind != 'm' {
			edits++
		}
	}
	acc := 1 - float64(edits)/float64(len([]rune(want)))
	if acc < 0 {
		return 0
	}
	return acc
}

type kvStr struct {
	k string
	v int
}
type kvRune struct {
	k rune
	v int
}

func sortMapStr(m map[string]int) []kvStr {
	var s []kvStr
	for k, v := range m {
		s = append(s, kvStr{k, v})
	}
	sort.Slice(s, func(i, j int) bool { return s[i].v > s[j].v })
	return s
}

func sortMapRune(m map[rune]int) []kvRune {
	var s []kvRune
	for k, v := range m {
		s = append(s, kvRune{k, v})
	}
	sort.Slice(s, func(i, j int) bool { return s[i].v > s[j].v })
	return s
}

func min3(a, b, c int) int { return min(a, min(b, c)) }
