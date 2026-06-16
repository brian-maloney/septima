package detect

import (
	"image"
	"sort"

	"gocv.io/x/gocv"
)

// FindDisplayROI attempts to locate the bounding rectangle of the seven-segment
// display within the full input image.
func FindDisplayROI(src gocv.Mat) image.Rectangle {
	gray := gocv.NewMat()
	defer gray.Close()
	if src.Channels() > 1 {
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
	} else {
		gray = src.Clone()
	}

	// Blur heavily to merge nearby blobs into display-shaped regions.
	blurred := gocv.NewMat()
	defer blurred.Close()
	blurSize := gray.Cols() / 20
	if blurSize < 5 {
		blurSize = 5
	}
	if blurSize%2 == 0 {
		blurSize++
	}
	gocv.GaussianBlur(gray, &blurred, image.Point{X: blurSize, Y: blurSize}, 0, 0, gocv.BorderReflect)

	binary := gocv.NewMat()
	defer binary.Close()
	gocv.Threshold(blurred, &binary, 0, 255, gocv.ThresholdBinary+gocv.ThresholdOtsu)

	// Try bright-blob detection first, then dark-blob.
	if roi, ok := largestDisplayContour(binary, src.Cols(), src.Rows()); ok {
		return roi
	}
	inverted := gocv.NewMat()
	defer inverted.Close()
	gocv.BitwiseNot(binary, &inverted)
	if roi, ok := largestDisplayContour(inverted, src.Cols(), src.Rows()); ok {
		return roi
	}
	return image.Rect(0, 0, src.Cols(), src.Rows())
}

// largestDisplayContour finds the largest contour that looks like a display rectangle.
func largestDisplayContour(binary gocv.Mat, imgW, imgH int) (image.Rectangle, bool) {
	contours := gocv.FindContours(binary, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	type candidate struct {
		rect        image.Rectangle
		area        float64
		fillRatio   float64 // contour area / bounding-rect area (rectangularity)
	}
	var candidates []candidate
	imgArea := float64(imgW * imgH)

	for i := 0; i < contours.Size(); i++ {
		c := contours.At(i)
		rect := gocv.BoundingRect(c)
		w, h := float64(rect.Dx()), float64(rect.Dy())
		if w < 30 || h < 15 {
			continue
		}

		// Aspect ratio: displays are wider than tall, not too extreme.
		aspect := w / h
		if aspect < 1.2 || aspect > 14 {
			continue
		}

		// Minimum height: at least 6% of image height.
		// This rejects thin glare-bar artifacts.
		if h < float64(imgH)*0.06 {
			continue
		}

		// Bounding-box area as fraction of image.
		bbFrac := (w * h) / imgArea
		if bbFrac < 0.005 || bbFrac > 0.80 {
			continue
		}

		// Rectangularity: contour fill ratio (OpenCV contourArea / bbox area).
		// A real display rectangle scores > 0.4; a curved arc scores < 0.3.
		cArea := gocv.ContourArea(c)
		fillRatio := cArea / (w * h)
		if fillRatio < 0.35 {
			continue
		}

		candidates = append(candidates, candidate{
			rect:      rect,
			area:      w * h,
			fillRatio: fillRatio,
		})
	}

	if len(candidates) == 0 {
		return image.Rectangle{}, false
	}

	// Sort by area descending; prefer rectangular candidates.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].area > candidates[j].area
	})
	return candidates[0].rect, true
}

// RefineDisplayROI looks within roiMat for a tighter LCD-style sub-rectangle.
// It is intended as an optional refinement after FindDisplayROI when the
// initial detection captured a whole device (LCD + plastic body + logo).
//
// The classic failure mode this addresses: a small LCD strip embedded in a
// larger device (e.g., an RSA SecurID key fob).  FindDisplayROI's heavy blur
// merges the LCD with the rest of the device into a single roughly-rectangular
// contour and picks the whole device, dragging logos and label text into the
// downstream segmentation.
//
// Algorithm — edge-contour LCD detection:
//
//   1. Canny edge detection on the ROI grayscale.  An LCD's bezel produces
//      a strong rectangular outline, and the digits inside add further sharp
//      edges, all connected together once edges are dilated.
//   2. Light dilation (3×3) closes single-pixel gaps so the bezel + interior
//      edges form a single contour.
//   3. Find contours, keep those that are:
//        - not touching the ROI border (the device body or background),
//        - LCD-shaped aspect (1.5–12),
//        - substantial area (≥ 5 % of ROI),
//        - long perimeter (≥ 1000 px or ≥ 1.3× the bbox perimeter).  This
//          discriminates an LCD (whose contour wraps around the bezel AND
//          weaves through interior digit segments → very long perimeter)
//          from incidental rectangular sub-features whose contour just
//          traces an outline.
//   4. Pick the candidate with the longest perimeter.
//
// Returns the sub-rectangle in roiMat's coordinates, or the zero rectangle if
// no candidate qualifies (signalling the caller to keep the original ROI).
func RefineDisplayROI(roiMat gocv.Mat) image.Rectangle {
	if roiMat.Empty() {
		return image.Rectangle{}
	}
	gray := gocv.NewMat()
	defer gray.Close()
	if roiMat.Channels() > 1 {
		gocv.CvtColor(roiMat, &gray, gocv.ColorBGRToGray)
	} else {
		gray = roiMat.Clone()
	}

	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 50, 150)

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 3, Y: 3})
	defer kernel.Close()
	dil := gocv.NewMat()
	defer dil.Close()
	gocv.Dilate(edges, &dil, kernel)

	imgW, imgH := roiMat.Cols(), roiMat.Rows()
	imgArea := imgW * imgH

	contours := gocv.FindContours(dil, gocv.RetrievalList, gocv.ChainApproxSimple)
	defer contours.Close()

	margin := 5
	var bestRect image.Rectangle
	bestPeri := 0.0

	for i := 0; i < contours.Size(); i++ {
		c := contours.At(i)
		rect := gocv.BoundingRect(c)
		// Reject contours touching the ROI border: those are the device body
		// outline or background frame, not an embedded LCD.
		if rect.Min.X <= margin || rect.Min.Y <= margin ||
			rect.Max.X >= imgW-margin || rect.Max.Y >= imgH-margin {
			continue
		}
		w, h := rect.Dx(), rect.Dy()
		if w < 80 || h < 30 {
			continue
		}
		bbFrac := float64(w*h) / float64(imgArea)
		if bbFrac < 0.05 || bbFrac > 0.55 {
			continue
		}
		aspect := float64(w) / float64(h)
		if aspect < 1.5 || aspect > 12 {
			continue
		}
		peri := gocv.ArcLength(c, true)
		bboxPeri := 2.0 * float64(w+h)
		// Either an absolutely-long perimeter, or a contour that weaves
		// substantially more than the bounding-box rectangle would imply
		// (signature of digit-edges-inside-the-bezel).
		if peri < 1000 && peri < 1.3*bboxPeri {
			continue
		}
		if peri > bestPeri {
			bestPeri = peri
			bestRect = rect
		}
	}

	return bestRect
}


// PadROI adds a small padding to an ROI while clamping to image bounds.
func PadROI(roi image.Rectangle, pad int, imgW, imgH int) image.Rectangle {
	return image.Rect(
		max_int(0, roi.Min.X-pad),
		max_int(0, roi.Min.Y-pad),
		min_int(imgW, roi.Max.X+pad),
		min_int(imgH, roi.Max.Y+pad),
	)
}

func max_int(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min_int(a, b int) int {
	if a < b {
		return a
	}
	return b
}
