// Package detect locates the display and its individual digit regions.
package detect

import (
	"gocv.io/x/gocv"
)

// Polarity describes which pixels are "on" (digit) vs "off" (background).
type Polarity int

const (
	PolarityAuto       Polarity = iota
	PolarityDarkOnLight         // dark segments on light background (most LCDs)
	PolarityLightOnDark         // bright segments on dark background (LEDs, VFDs)
)

// DetectPolarity inspects the grayscale histogram to decide whether the display
// is light-on-dark or dark-on-light.
//
// Strategy: apply a quick Otsu binarization, count white vs. black pixels.
// If white > 50%, the background is bright (dark-on-light).
// This is more robust than a simple median check when the modal pixel value
// is close to 127 (e.g., a medium-gray LCD panel).
func DetectPolarity(gray gocv.Mat) Polarity {
	bin := gocv.NewMat()
	defer bin.Close()
	gocv.Threshold(gray, &bin, 0, 255, gocv.ThresholdBinary+gocv.ThresholdOtsu)

	total := float64(bin.Rows() * bin.Cols())
	if total == 0 {
		return PolarityDarkOnLight
	}
	white := bin.Sum().Val1 / 255.0
	if white/total > 0.50 {
		return PolarityDarkOnLight
	}
	return PolarityLightOnDark
}

// NormalizePolarity ensures the output has dark digits on a light background,
// inverting if necessary.
func NormalizePolarity(src gocv.Mat, p Polarity) gocv.Mat {
	pol := p
	if pol == PolarityAuto {
		pol = DetectPolarity(src)
	}
	if pol == PolarityLightOnDark {
		dst := gocv.NewMat()
		gocv.BitwiseNot(src, &dst)
		return dst
	}
	return src.Clone()
}
