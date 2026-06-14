package preprocess

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

// SetPixelsFilterOp sets pixels that have at least MASK 8-connected neighbors set.
type SetPixelsFilterOp struct{ Mask int }

func (o SetPixelsFilterOp) Name() string { return fmt.Sprintf("set_pixels_filter %d", o.Mask) }

func (o SetPixelsFilterOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	dst := gocv.NewMat()
	src.CopyTo(&dst)
	rows, cols := src.Rows(), src.Cols()
	for r := 1; r < rows-1; r++ {
		for c := 1; c < cols-1; c++ {
			count := 0
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					if dr == 0 && dc == 0 {
						continue
					}
					if src.GetUCharAt(r+dr, c+dc) > 0 {
						count++
					}
				}
			}
			if count >= o.Mask {
				dst.SetUCharAt(r, c, 255)
			} else {
				dst.SetUCharAt(r, c, 0)
			}
		}
	}
	return dst, nil
}

// KeepPixelsFilterOp retains pixels that have at least MASK 8-connected neighbors set
// (excluding the pixel itself) — only set pixels are candidates.
type KeepPixelsFilterOp struct{ Mask int }

func (o KeepPixelsFilterOp) Name() string { return fmt.Sprintf("keep_pixels_filter %d", o.Mask) }

func (o KeepPixelsFilterOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	dst := gocv.NewMat()
	src.CopyTo(&dst)
	rows, cols := src.Rows(), src.Cols()
	for r := 1; r < rows-1; r++ {
		for c := 1; c < cols-1; c++ {
			if src.GetUCharAt(r, c) == 0 {
				dst.SetUCharAt(r, c, 0)
				continue
			}
			count := 0
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					if dr == 0 && dc == 0 {
						continue
					}
					if src.GetUCharAt(r+dr, c+dc) > 0 {
						count++
					}
				}
			}
			if count < o.Mask {
				dst.SetUCharAt(r, c, 0)
			}
		}
	}
	return dst, nil
}

// CLAHEOp applies Contrast Limited Adaptive Histogram Equalization (as used by SegoDec).
type CLAHEOp struct {
	ClipLimit  float64
	TileWidth  int
	TileHeight int
}

func (o CLAHEOp) Name() string { return "clahe" }

func (o CLAHEOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	gray := gocv.NewMat()
	defer gray.Close()
	if src.Channels() > 1 {
		gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
	} else {
		gray = src.Clone()
	}
	clip := o.ClipLimit
	if clip <= 0 {
		clip = 2.0
	}
	tw, th := o.TileWidth, o.TileHeight
	if tw <= 0 {
		tw = 8
	}
	if th <= 0 {
		th = 8
	}
	clahe := gocv.NewCLAHEWithParams(clip, image.Point{X: tw, Y: th})
	defer clahe.Close()
	dst := gocv.NewMat()
	if err := clahe.Apply(gray, &dst); err != nil {
		return gocv.NewMat(), err
	}
	return dst, nil
}
