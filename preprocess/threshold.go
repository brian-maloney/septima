package preprocess

import (
	"fmt"
	"image"
	"image/color"
)

// ThresholdOp converts the image to a binary black/white image using a global
// luminance threshold. Matches ssocr's "-t/--threshold PCT" / "make_mono".
// By default the threshold is a percentage between the image's darkest and
// brightest pixel (matching ssocr's adaptive default); set Absolute to treat
// ThresholdPct as a fraction of the full 0-255 range instead.
type ThresholdOp struct {
	ThresholdPct float64 // 0-100; percentage between min and max luminance (or of 0-255 if Absolute)
	Absolute     bool
}

func (o ThresholdOp) Name() string { return fmt.Sprintf("threshold %g", o.ThresholdPct) }

func (o ThresholdOp) Apply(src image.Image) (image.Image, error) {
	gray := toGray(src)
	b := gray.Bounds()

	var t uint8
	if o.Absolute {
		t = uint8(clamp(o.ThresholdPct/100*255, 0, 255))
	} else {
		mn, mx := grayRange(gray)
		t = uint8(clamp(float64(mn)+o.ThresholdPct/100*float64(mx-mn), 0, 255))
	}

	return binarize(gray, b, t), nil
}

// OtsuThresholdOp applies Otsu's method to automatically pick a global
// threshold that best separates foreground from background. Matches ssocr's
// "otsu_threshold" pipeline op.
type OtsuThresholdOp struct{}

func (o OtsuThresholdOp) Name() string { return "otsu_threshold" }

func (o OtsuThresholdOp) Apply(src image.Image) (image.Image, error) {
	gray := toGray(src)
	b := gray.Bounds()
	t := otsuThreshold(gray)
	return binarize(gray, b, t), nil
}

func binarize(gray *image.Gray, b image.Rectangle, t uint8) *image.Gray {
	dst := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if gray.GrayAt(x, y).Y > t {
				dst.SetGray(x, y, color.Gray{Y: 255})
			} else {
				dst.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}
	return dst
}

func grayRange(gray *image.Gray) (min, max uint8) {
	b := gray.Bounds()
	min, max = 255, 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := gray.GrayAt(x, y).Y
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
	}
	return min, max
}

// otsuThreshold computes Otsu's optimal binarization threshold from the
// image's luminance histogram.
func otsuThreshold(gray *image.Gray) uint8 {
	var hist [256]int
	b := gray.Bounds()
	total := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			hist[gray.GrayAt(x, y).Y]++
			total++
		}
	}
	if total == 0 {
		return 128
	}

	var sumAll float64
	for i, c := range hist {
		sumAll += float64(i * c)
	}

	var sumB, wB float64
	var bestVar float64
	bestT := 0
	for t := 0; t < 256; t++ {
		wB += float64(hist[t])
		if wB == 0 {
			continue
		}
		wF := float64(total) - wB
		if wF == 0 {
			break
		}
		sumB += float64(t * hist[t])
		mB := sumB / wB
		mF := (sumAll - sumB) / wF
		betweenVar := wB * wF * (mB - mF) * (mB - mF)
		if betweenVar > bestVar {
			bestVar = betweenVar
			bestT = t
		}
	}
	return uint8(bestT)
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
