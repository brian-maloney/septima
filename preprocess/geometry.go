package preprocess

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

// CropOp extracts a rectangular sub-image.
type CropOp struct{ X, Y, W, H int }

func (o CropOp) Name() string { return fmt.Sprintf("crop %d %d %d %d", o.X, o.Y, o.W, o.H) }

func (o CropOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	r := image.Rect(o.X, o.Y, o.X+o.W, o.Y+o.H)
	bounds := image.Rect(0, 0, src.Cols(), src.Rows())
	r = r.Intersect(bounds)
	if r.Empty() {
		return gocv.NewMat(), fmt.Errorf("crop region is outside image bounds")
	}
	region := src.Region(r)
	dst := region.Clone()
	region.Close()
	return dst, nil
}

// RotateOp rotates the image by degrees (clockwise).
type RotateOp struct{ Degrees float64 }

func (o RotateOp) Name() string { return fmt.Sprintf("rotate %.2f", o.Degrees) }

func (o RotateOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	cx := float64(src.Cols()) / 2.0
	cy := float64(src.Rows()) / 2.0
	M := gocv.GetRotationMatrix2D(image.Point{X: int(cx), Y: int(cy)}, -o.Degrees, 1.0)
	defer M.Close()
	dst := gocv.NewMat()
	sz := image.Point{X: src.Cols(), Y: src.Rows()}
	if err := gocv.WarpAffineWithParams(src, &dst, M, sz,
		gocv.InterpolationLinear, gocv.BorderConstant, color.RGBA{R: 255, G: 255, B: 255, A: 255}); err != nil {
		return gocv.NewMat(), err
	}
	return dst, nil
}

// ShearOp applies a horizontal shear offset.
type ShearOp struct{ Offset float64 }

func (o ShearOp) Name() string { return fmt.Sprintf("shear %.2f", o.Offset) }

func (o ShearOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	h := float64(src.Rows())
	M := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	defer M.Close()
	M.SetDoubleAt(0, 0, 1)
	M.SetDoubleAt(0, 1, o.Offset/h)
	M.SetDoubleAt(0, 2, 0)
	M.SetDoubleAt(1, 0, 0)
	M.SetDoubleAt(1, 1, 1)
	M.SetDoubleAt(1, 2, 0)
	dst := gocv.NewMat()
	sz := image.Point{X: src.Cols(), Y: src.Rows()}
	if err := gocv.WarpAffineWithParams(src, &dst, M, sz,
		gocv.InterpolationLinear, gocv.BorderConstant, color.RGBA{R: 255, G: 255, B: 255, A: 255}); err != nil {
		return gocv.NewMat(), err
	}
	return dst, nil
}

// MirrorOp flips the image.
type MirrorOp struct{ Horiz bool }

func (o MirrorOp) Name() string {
	if o.Horiz {
		return "mirror horiz"
	}
	return "mirror vert"
}

func (o MirrorOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	dst := gocv.NewMat()
	flipCode := 0
	if o.Horiz {
		flipCode = 1
	}
	gocv.Flip(src, &dst, flipCode)
	return dst, nil
}

// WhiteBorderOp adds a border of background color.
type WhiteBorderOp struct{ Width int }

func (o WhiteBorderOp) Name() string { return fmt.Sprintf("white_border %d", o.Width) }

func (o WhiteBorderOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	w := o.Width
	if w < 1 {
		w = 5
	}
	dst := gocv.NewMat()
	if err := gocv.CopyMakeBorder(src, &dst, w, w, w, w, gocv.BorderConstant,
		color.RGBA{R: 255, G: 255, B: 255, A: 255}); err != nil {
		return gocv.NewMat(), err
	}
	return dst, nil
}
