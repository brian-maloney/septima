package preprocess

import (
	"image"
	"image/color"
)

// InvertOp inverts the RGB channels, leaving alpha unchanged. Matches
// ssocr's "invert" pipeline op (swaps foreground/background polarity).
type InvertOp struct{}

func (o InvertOp) Name() string { return "invert" }

func (o InvertOp) Apply(src image.Image) (image.Image, error) {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, a := src.At(x, y).RGBA()
			dst.SetRGBA(x, y, color.RGBA{
				R: 255 - uint8(r>>8),
				G: 255 - uint8(g>>8),
				B: 255 - uint8(bl>>8),
				A: uint8(a >> 8),
			})
		}
	}
	return dst, nil
}

// GrayscaleOp converts the image to grayscale (luminance). Matches ssocr's
// "grayscale" pipeline op.
type GrayscaleOp struct{}

func (o GrayscaleOp) Name() string { return "grayscale" }

func (o GrayscaleOp) Apply(src image.Image) (image.Image, error) {
	b := src.Bounds()
	dst := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	return dst, nil
}

// toGray returns src as a *image.Gray, converting if necessary.
func toGray(src image.Image) *image.Gray {
	if g, ok := src.(*image.Gray); ok {
		return g
	}
	b := src.Bounds()
	dst := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
	return dst
}
