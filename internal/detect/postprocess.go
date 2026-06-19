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
			// (catches duplicate thin '1' boxes — including a narrow box nested in
			// a wider one — that overlap too little for IoU but clearly mark the
			// same glyph). Genuinely adjacent digits sit a full pitch apart, so
			// neither center falls inside the other.
			if iou(d.Box, k.Box) > iouThreshold || centerInside(d.Box, k.Box) || centerInside(k.Box, d.Box) {
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

// FilterPunctuationSpecks drops '.'/':'/'-' detections whose height is far below
// the median digit height — on real LCD photos these are usually dust or screen
// artifacts rather than real punctuation. Digit detections are never dropped.
// names maps class index to its single-rune label (e.g. "0".."9",".",":","-").
func FilterPunctuationSpecks(dets []Detection, names []string) []Detection {
	var digitH []int
	for _, d := range dets {
		if IsDigitClass(d.Class, names) {
			digitH = append(digitH, d.Box.Dy())
		}
	}
	if len(digitH) == 0 {
		return dets
	}
	sort.Ints(digitH)
	minH := int(0.4 * float64(digitH[len(digitH)/2]))

	out := dets[:0]
	for _, d := range dets {
		if !IsDigitClass(d.Class, names) && d.Box.Dy() < minH {
			continue
		}
		out = append(out, d)
	}
	return out
}

// IsDigitClass reports whether the class maps to a numeric glyph ('0'-'9').
func IsDigitClass(class int, names []string) bool {
	if class < 0 || class >= len(names) {
		return false
	}
	return len(names[class]) == 1 && names[class][0] >= '0' && names[class][0] <= '9'
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
