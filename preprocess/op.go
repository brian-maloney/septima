// Package preprocess implements a small set of pure-Go image manipulation
// operations for cleaning up an image before detection, corresponding to a
// core subset of ssocr's command-line pipeline (crop, rotate, mirror, invert,
// grayscale, threshold). It has no OpenCV/cgo dependency, matching the rest of
// the inference pipeline.
package preprocess

import (
	"fmt"
	"image"
)

// Op is a single image processing operation in the pipeline.
type Op interface {
	Apply(src image.Image) (image.Image, error)
	Name() string
}

// Pipeline applies a sequence of Ops in order.
type Pipeline []Op

// Apply runs all ops in sequence, each receiving the output of the previous.
func (p Pipeline) Apply(src image.Image) (image.Image, error) {
	cur := src
	for _, op := range p {
		next, err := op.Apply(cur)
		if err != nil {
			return nil, fmt.Errorf("preprocess: %s: %w", op.Name(), err)
		}
		cur = next
	}
	return cur, nil
}
