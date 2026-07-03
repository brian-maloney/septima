package preprocess

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// CropOp crops the image to the rectangle (X, Y, W, H), in source pixel
// coordinates. Matches ssocr's "crop X Y W H" pipeline op.
type CropOp struct {
	X, Y, W, H int
}

func (o CropOp) Name() string { return fmt.Sprintf("crop %d %d %d %d", o.X, o.Y, o.W, o.H) }

func (o CropOp) Apply(src image.Image) (image.Image, error) {
	r := image.Rect(o.X, o.Y, o.X+o.W, o.Y+o.H).Intersect(src.Bounds())
	if r.Empty() {
		return nil, fmt.Errorf("crop rectangle %v does not intersect image bounds %v", image.Rect(o.X, o.Y, o.X+o.W, o.Y+o.H), src.Bounds())
	}
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			dst.Set(x, y, src.At(r.Min.X+x, r.Min.Y+y))
		}
	}
	return dst, nil
}

// RotateOp rotates the image by Degrees (clockwise, positive) about its
// center. The output canvas is the same size as the input (matching ssocr's
// "rotate DEGREES"); pixels rotated in from outside the source bounds are
// filled white.
type RotateOp struct {
	Degrees float64
}

func (o RotateOp) Name() string { return fmt.Sprintf("rotate %g", o.Degrees) }

func (o RotateOp) Apply(src image.Image) (image.Image, error) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, w, h))

	theta := o.Degrees * math.Pi / 180
	sin, cos := math.Sin(theta), math.Cos(theta)
	cx, cy := float64(w)/2, float64(h)/2

	white := color.RGBA{255, 255, 255, 255}
	for dy := 0; dy < h; dy++ {
		for dx := 0; dx < w; dx++ {
			// Inverse-rotate the destination pixel to find its source.
			ox, oy := float64(dx)-cx, float64(dy)-cy
			sx := ox*cos + oy*sin + cx
			sy := -ox*sin + oy*cos + cy
			ix, iy := int(math.Round(sx)), int(math.Round(sy))
			if ix < 0 || iy < 0 || ix >= w || iy >= h {
				dst.Set(dx, dy, white)
				continue
			}
			dst.Set(dx, dy, src.At(b.Min.X+ix, b.Min.Y+iy))
		}
	}
	return dst, nil
}

// MirrorOp flips the image horizontally or vertically.
type MirrorOp struct {
	Horiz bool // true = mirror left-right; false = mirror top-bottom
}

func (o MirrorOp) Name() string {
	if o.Horiz {
		return "mirror horiz"
	}
	return "mirror vert"
}

func (o MirrorOp) Apply(src image.Image) (image.Image, error) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sx, sy := x, y
			if o.Horiz {
				sx = w - 1 - x
			} else {
				sy = h - 1 - y
			}
			dst.Set(x, y, src.At(b.Min.X+sx, b.Min.Y+sy))
		}
	}
	return dst, nil
}
