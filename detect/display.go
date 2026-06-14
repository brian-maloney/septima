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
