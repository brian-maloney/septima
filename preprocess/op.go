// Package preprocess implements the image processing pipeline primitives
// that correspond to ssocr's command-line operations.
package preprocess

import "gocv.io/x/gocv"

// Op is a single image processing operation in the pipeline.
type Op interface {
	Apply(src gocv.Mat) (gocv.Mat, error)
	Name() string
}

// Pipeline applies a sequence of Ops in order.
type Pipeline []Op

// Apply runs all ops in sequence.  Each op receives the output of the previous.
// All intermediate Mats are freed; only the final result is returned.
func (p Pipeline) Apply(src gocv.Mat) (gocv.Mat, error) {
	cur := src.Clone()
	for _, op := range p {
		next, err := op.Apply(cur)
		cur.Close()
		if err != nil {
			return gocv.NewMat(), err
		}
		cur = next
	}
	return cur, nil
}
