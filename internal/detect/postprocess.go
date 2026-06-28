// Package detect holds the two YOLO inference stages (panel + digits) and the
// shared post-processing used to turn raw model output into image-space boxes.
package detect

import (
	"image"
	"sort"

	"github.com/brian-maloney/septima/internal/imageproc"
)

// Detection is one model output box mapped into original-image coordinates.
type Detection struct {
	Class int
	Score float64
	Box   image.Rectangle
}

// DecodeYOLO converts a flat Ultralytics detection-head tensor into detections,
// applying the score threshold and mapping each box back through the letterbox
// transform into original image coordinates.
//
// The Ultralytics ONNX detection head exports shape [1, 4+nc, n] where the first
// 4 rows are cx,cy,w,h (in model-input pixels) and the remaining nc rows are
// per-class scores. data is that tensor flattened row-major; numClasses = nc and
// numBoxes = n. tr maps model-input coordinates back to the source image.
func DecodeYOLO(data []float32, numClasses, numBoxes int, tr imageproc.LetterboxTransform, confThreshold float64) []Detection {
	stride := 4 + numClasses
	if stride <= 4 || numBoxes <= 0 || len(data) < stride*numBoxes {
		return nil
	}
	at := func(row, col int) float32 { return data[row*numBoxes+col] }

	dets := make([]Detection, 0, 64)
	for i := 0; i < numBoxes; i++ {
		bestClass, bestScore := -1, float32(0)
		for c := 0; c < numClasses; c++ {
			s := at(4+c, i)
			if s > bestScore {
				bestScore, bestClass = s, c
			}
		}
		if bestClass < 0 || float64(bestScore) < confThreshold {
			continue
		}
		cx, cy := float64(at(0, i)), float64(at(1, i))
		w, h := float64(at(2, i)), float64(at(3, i))
		x0, y0 := tr.ToSource(cx-w/2, cy-h/2)
		x1, y1 := tr.ToSource(cx+w/2, cy+h/2)
		dets = append(dets, Detection{
			Class: bestClass,
			Score: float64(bestScore),
			Box:   image.Rect(int(x0+0.5), int(y0+0.5), int(x1+0.5), int(y1+0.5)),
		})
	}
	return dets
}

// NMS runs class-wise non-maximum suppression and returns the survivors sorted
// by descending score.
func NMS(dets []Detection, iouThreshold float64) []Detection {
	byClass := map[int][]Detection{}
	for _, d := range dets {
		byClass[d.Class] = append(byClass[d.Class], d)
	}
	var kept []Detection
	for _, group := range byClass {
		sort.Slice(group, func(i, j int) bool { return group[i].Score > group[j].Score })
		suppressed := make([]bool, len(group))
		for i := range group {
			if suppressed[i] {
				continue
			}
			kept = append(kept, group[i])
			for j := i + 1; j < len(group); j++ {
				if suppressed[j] {
					continue
				}
				if iou(group[i].Box, group[j].Box) > iouThreshold {
					suppressed[j] = true
				}
			}
		}
	}
	sort.Slice(kept, func(i, j int) bool { return kept[i].Score > kept[j].Score })
	return kept
}

// DedupeAcrossClasses runs a final class-agnostic NMS pass: when two boxes of
// different classes overlap beyond iouThreshold (e.g. the detector firing both
// '9' and '4' on the same glyph), only the higher-scoring one survives. Adjacent
// distinct digits don't overlap, so this is safe for tight displays.
func DedupeAcrossClasses(dets []Detection, iouThreshold float64) []Detection {
	sorted := append([]Detection(nil), dets...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
	var kept []Detection
	for _, d := range sorted {
		overlaps := false
		for _, k := range kept {
			// Suppress on IoU, or when either box's center sits inside the other
			// AND the two are either the same class or of similar height (catches
			// duplicate thin '1' boxes, including a narrow box nested in a wider
			// one, and two overlapping '.' boxes the detector fires on one dot).
			// The height guard keeps a real decimal/colon — short, and often
			// overlapping a tall neighbour digit's box corner — from being mistaken
			// for a duplicate of that digit; the same-class shortcut doesn't apply
			// there since digit and punctuation are different classes. A colon's two
			// dots are vertically offset enough that neither centre is inside the
			// other, so they survive for MergeColonDots.
			contained := centerInside(d.Box, k.Box) || centerInside(k.Box, d.Box)
			if iou(d.Box, k.Box) > iouThreshold || (contained && (d.Class == k.Class || heightSimilar(d.Box, k.Box))) {
				overlaps = true
				break
			}
		}
		if !overlaps {
			kept = append(kept, d)
		}
	}
	return kept
}

// MergeColonDots replaces a pair of vertically-stacked '.' detections (the two
// dots of a colon, which the digit detector commonly reports as two separate
// decimals) with a single ':' detection. It is constrained so it won't merge the
// decimals of two stacked rows: the dots must be x-aligned (within a dot width)
// and separated vertically by only a fraction of a digit's height.
func MergeColonDots(dets []Detection, dotClass, colonClass int) []Detection {
	medH := medianDigitHeight(dets, dotClass, colonClass)
	if medH <= 0 {
		return dets
	}
	used := make([]bool, len(dets))
	var out []Detection
	for i := 0; i < len(dets); i++ {
		if used[i] || dets[i].Class != dotClass {
			continue
		}
		for j := i + 1; j < len(dets); j++ {
			if used[j] || dets[j].Class != dotClass {
				continue
			}
			ax, ay := boxCenter(dets[i].Box)
			bx, by := boxCenter(dets[j].Box)
			xtol := maxInt(dets[i].Box.Dx(), dets[j].Box.Dx())
			ygap := absInt(ay - by)
			if absInt(ax-bx) <= xtol && ygap >= int(0.2*medH) && ygap <= int(0.75*medH) {
				score := dets[i].Score
				if dets[j].Score > score {
					score = dets[j].Score
				}
				out = append(out, Detection{Class: colonClass, Score: score, Box: dets[i].Box.Union(dets[j].Box)})
				used[i], used[j] = true, true
				break
			}
		}
	}
	for i := range dets {
		if !used[i] {
			out = append(out, dets[i])
		}
	}
	return out
}

// SuppressDotsInsideColon removes stray '.' detections that belong to a ':'
// detection's vertical dot-stack rather than being a real decimal point. A
// colon-trained model commonly fires an extra '.' on or just above/below a colon
// (e.g. a reflection or indicator pip), yielding "2:.47". Such a dot is x-aligned
// with the colon (its center-x lies within the colon's box) and vertically close
// (within one colon-height above or below). A genuine decimal point sits between
// digits at a distinct x and on the baseline, so it is never x-aligned with a
// colon — dropping these is safe and does not touch real decimals (verified
// against the decimal-bearing benchmark images).
func SuppressDotsInsideColon(dets []Detection, dotClass, colonClass int) []Detection {
	var colons []image.Rectangle
	for _, d := range dets {
		if d.Class == colonClass {
			colons = append(colons, d.Box)
		}
	}
	if len(colons) == 0 {
		return dets
	}
	out := dets[:0:0]
	for _, d := range dets {
		if d.Class == dotClass && dotInColonStack(d.Box, colons) {
			continue
		}
		out = append(out, d)
	}
	return out
}

// dotInColonStack reports whether a dot box is part of a colon's vertical stack:
// its center-x falls within a colon box and its center-y is within one
// colon-height above or below that box.
func dotInColonStack(dot image.Rectangle, colons []image.Rectangle) bool {
	cx := (dot.Min.X + dot.Max.X) / 2
	cy := (dot.Min.Y + dot.Max.Y) / 2
	for _, c := range colons {
		ch := c.Dy()
		if cx >= c.Min.X && cx < c.Max.X && cy >= c.Min.Y-ch && cy < c.Max.Y+ch {
			return true
		}
	}
	return false
}

func medianDigitHeight(dets []Detection, dotClass, colonClass int) float64 {
	var hs []int
	for _, d := range dets {
		if d.Class != dotClass && d.Class != colonClass {
			hs = append(hs, d.Box.Dy())
		}
	}
	if len(hs) == 0 {
		return 0
	}
	sort.Ints(hs)
	return float64(hs[len(hs)/2])
}

func boxCenter(r image.Rectangle) (int, int) {
	return (r.Min.X + r.Max.X) / 2, (r.Min.Y + r.Max.Y) / 2
}

func absInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// heightSimilar reports whether two boxes are close enough in height to be the
// same glyph (vs a short punctuation mark overlapping a tall digit).
func heightSimilar(a, b image.Rectangle) bool {
	ha, hb := a.Dy(), b.Dy()
	if ha == 0 || hb == 0 {
		return false
	}
	lo, hi := ha, hb
	if lo > hi {
		lo, hi = hi, lo
	}
	return float64(lo) >= 0.6*float64(hi)
}

// centerInside reports whether the center of a lies within b.
func centerInside(a, b image.Rectangle) bool {
	cx := (a.Min.X + a.Max.X) / 2
	cy := (a.Min.Y + a.Max.Y) / 2
	return cx >= b.Min.X && cx < b.Max.X && cy >= b.Min.Y && cy < b.Max.Y
}

func iou(a, b image.Rectangle) float64 {
	inter := a.Intersect(b)
	if inter.Empty() {
		return 0
	}
	ia := area(inter)
	ua := area(a) + area(b) - ia
	if ua <= 0 {
		return 0
	}
	return float64(ia) / float64(ua)
}

func area(r image.Rectangle) int {
	if r.Empty() {
		return 0
	}
	return r.Dx() * r.Dy()
}
