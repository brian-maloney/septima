package preprocess

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

func kernelOf(n int) gocv.Mat {
	if n < 1 {
		n = 1
	}
	sz := 2*n + 1
	return gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: sz, Y: sz})
}

// DilationOp expands foreground pixels.
type DilationOp struct{ N int }

func (o DilationOp) Name() string { return fmt.Sprintf("dilation %d", o.N) }

func (o DilationOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	k := kernelOf(o.N)
	defer k.Close()
	dst := gocv.NewMat()
	gocv.Dilate(src, &dst, k)
	return dst, nil
}

// ErosionOp shrinks foreground pixels.
type ErosionOp struct{ N int }

func (o ErosionOp) Name() string { return fmt.Sprintf("erosion %d", o.N) }

func (o ErosionOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	k := kernelOf(o.N)
	defer k.Close()
	dst := gocv.NewMat()
	gocv.Erode(src, &dst, k)
	return dst, nil
}

// OpeningOp is erosion then dilation (removes small blobs).
type OpeningOp struct{ N int }

func (o OpeningOp) Name() string { return fmt.Sprintf("opening %d", o.N) }

func (o OpeningOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	k := kernelOf(o.N)
	defer k.Close()
	dst := gocv.NewMat()
	gocv.MorphologyEx(src, &dst, gocv.MorphOpen, k)
	return dst, nil
}

// ClosingOp is dilation then erosion (fills small holes).
type ClosingOp struct{ N int }

func (o ClosingOp) Name() string { return fmt.Sprintf("closing %d", o.N) }

func (o ClosingOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	k := kernelOf(o.N)
	defer k.Close()
	dst := gocv.NewMat()
	gocv.MorphologyEx(src, &dst, gocv.MorphClose, k)
	return dst, nil
}

// RemoveIsolatedOp eliminates pixels with no 8-connected neighbors.
type RemoveIsolatedOp struct{}

func (o RemoveIsolatedOp) Name() string { return "remove_isolated" }

func (o RemoveIsolatedOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	k := kernelOf(1)
	defer k.Close()
	eroded := gocv.NewMat()
	defer eroded.Close()
	gocv.Erode(src, &eroded, k)
	dilated := gocv.NewMat()
	defer dilated.Close()
	gocv.Dilate(eroded, &dilated, k)
	dst := gocv.NewMat()
	gocv.BitwiseAnd(src, dilated, &dst)
	return dst, nil
}
