package decode

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

// Canonical digit canvas dimensions.
const (
	canvasW = 30
	canvasH = 50
)

// DebugSegments enables per-digit segment sampling diagnostics.
var DebugSegments bool

// segmentWindows defines the sampling rectangle (x, y, w, h) for each of the
// 7 segments on the canonical 30×50 canvas.
//
// Bit order: a=0, b=1, c=2, d=3, e=4, f=5, g=6
var segmentWindows = [7][4]int{
	{4, 0, 22, 10},  // a: top horizontal    (top 20%)
	{21, 2, 8, 20},  // b: top-right vertical (top 50%, right 27%)
	{21, 28, 8, 20}, // c: bottom-right       (bottom 50%, right 27%)
	{4, 40, 22, 10}, // d: bottom horizontal  (bottom 20%)
	{1, 28, 8, 20},  // e: bottom-left        (bottom 50%, left 27%)
	{1, 2, 8, 20},   // f: top-left           (top 50%, left 27%)
	{4, 21, 22, 8},  // g: middle horizontal  (middle 16%)
}

var segNames = [7]string{"a", "b", "c", "d", "e", "f", "g"}

// segOnThreshold is the minimum white-pixel density for a zone to be
// considered "ON".  0.40 means at least 40% of zone pixels must be
// foreground.  This suppresses contamination from perpendicular segments
// crossing the zone boundaries (e.g. vertical bars through the 'g' zone).
const segOnThreshold = 0.40

// SampleSegments extracts the 7-bit segment mask from a digit image.
// binaryDigit must be an 8-bit single-channel image where digit pixels are
// bright (≥ 128) and background is dark.
func SampleSegments(binaryDigit gocv.Mat) (mask byte, confidence float64) {
	if binaryDigit.Empty() {
		return 0, 0
	}

	// Use NearestNeighbor to preserve the binary quality after resize.
	canvas := gocv.NewMat()
	defer canvas.Close()
	gocv.Resize(binaryDigit, &canvas, image.Point{X: canvasW, Y: canvasH}, 0, 0, gocv.InterpolationNearestNeighbor)

	totalConf := 0.0
	for bit, win := range segmentWindows {
		x, y, w, h := win[0], win[1], win[2], win[3]
		if x+w > canvasW {
			w = canvasW - x
		}
		if y+h > canvasH {
			h = canvasH - y
		}
		if w <= 0 || h <= 0 {
			continue
		}
		region := canvas.Region(image.Rect(x, y, x+w, y+h))
		density := regionDensity(region)
		region.Close()

		isOn := density > segOnThreshold
		if isOn {
			mask |= 1 << uint(bit)
		}

		if DebugSegments {
			on := " "
			if isOn {
				on = "ON"
			}
			fmt.Printf("    seg %s: %.3f %s\n", segNames[bit], density, on)
		}

		// Confidence: how far from the threshold the density is.
		conf := density / segOnThreshold
		if conf > 1.0 {
			conf = 1.0
		}
		if !isOn {
			conf = 1.0 - density/segOnThreshold
			if conf < 0 {
				conf = 0
			}
		}
		totalConf += conf
	}
	if DebugSegments {
		fmt.Printf("    mask=0x%02x\n", mask)
	}
	confidence = totalConf / 7.0
	return mask, confidence
}

// regionDensity returns the fraction of pixels in a region that are
// "bright" (≥ 128), for an 8-bit single-channel Mat.
func regionDensity(m gocv.Mat) float64 {
	n := m.Rows() * m.Cols()
	if n == 0 {
		return 0
	}
	s := m.Sum()
	whiteCount := s.Val1 / 255.0
	return whiteCount / float64(n)
}

// NormalizeDigitImage prepares a digit bounding-box crop for segment sampling.
func NormalizeDigitImage(src gocv.Mat, box image.Rectangle) gocv.Mat {
	bounds := image.Rect(0, 0, src.Cols(), src.Rows())
	box = box.Intersect(bounds)
	if box.Empty() {
		return gocv.NewMat()
	}
	region := src.Region(box)
	defer region.Close()

	var gray gocv.Mat
	if region.Channels() > 1 {
		gray = gocv.NewMat()
		gocv.CvtColor(region, &gray, gocv.ColorBGRToGray)
	} else {
		gray = region.Clone()
	}
	defer gray.Close()

	bin := gocv.NewMat()
	gocv.Threshold(gray, &bin, 0, 255, gocv.ThresholdBinary+gocv.ThresholdOtsu)
	return bin
}
