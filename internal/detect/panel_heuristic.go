package detect

import "image"

// FindBrightPanel locates a bright display panel against a darker background via
// a projection-based bounding box of the brightest region (Otsu threshold, then
// the row/column span carrying most of the bright pixels). It returns the panel
// rect and true when a plausible panel is found.
//
// This is a classical stage-1 fallback used when no panel.onnx is available —
// e.g. the tank gauge, a bright LCD on a near-black frame. It is deliberately
// conservative: it rejects results that are tiny or near-full-frame.
func FindBrightPanel(img image.Image) (image.Rectangle, bool) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 16 || h < 16 {
		return image.Rectangle{}, false
	}

	gray := make([]uint8, w*h)
	var hist [256]int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			v := uint8((299*(r>>8) + 587*(g>>8) + 114*(bl>>8)) / 1000)
			gray[y*w+x] = v
			hist[v]++
		}
	}
	t := otsuThreshold(hist[:], w*h)

	colSum := make([]int, w)
	rowSum := make([]int, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if gray[y*w+x] > t {
				colSum[x]++
				rowSum[y]++
			}
		}
	}
	x0, x1 := spanAboveFrac(colSum, 0.05)
	y0, y1 := spanAboveFrac(rowSum, 0.05)
	if x1 <= x0 || y1 <= y0 {
		return image.Rectangle{}, false
	}
	rect := image.Rect(b.Min.X+x0, b.Min.Y+y0, b.Min.X+x1+1, b.Min.Y+y1+1)

	af := float64(rect.Dx()*rect.Dy()) / float64(w*h)
	if af < 0.005 || af > 0.97 {
		return image.Rectangle{}, false
	}
	return rect, true
}

// otsuThreshold returns the gray level that maximizes between-class variance.
func otsuThreshold(hist []int, total int) uint8 {
	var sum float64
	for i, c := range hist {
		sum += float64(i * c)
	}
	var sumB, wB float64
	var maxVar float64
	best := 0
	for i := 0; i < 256; i++ {
		wB += float64(hist[i])
		if wB == 0 {
			continue
		}
		wF := float64(total) - wB
		if wF == 0 {
			break
		}
		sumB += float64(i * hist[i])
		mB := sumB / wB
		mF := (sum - sumB) / wF
		between := wB * wF * (mB - mF) * (mB - mF)
		if between > maxVar {
			maxVar = between
			best = i
		}
	}
	return uint8(best)
}

// spanAboveFrac returns the first and last indices whose value exceeds frac of
// the maximum — the dense extent of the signal, ignoring sparse specks.
func spanAboveFrac(sums []int, frac float64) (int, int) {
	max := 0
	for _, v := range sums {
		if v > max {
			max = v
		}
	}
	if max == 0 {
		return 0, -1
	}
	thr := float64(max) * frac
	lo, hi := -1, -1
	for i, v := range sums {
		if float64(v) > thr {
			if lo < 0 {
				lo = i
			}
			hi = i
		}
	}
	return lo, hi
}
