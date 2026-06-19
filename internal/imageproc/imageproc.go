// Package imageproc holds pure-Go image helpers used by the inference pipeline:
// letterbox resizing, CHW float32 tensor construction, and cropping. No OpenCV.
package imageproc

import (
	"image"
	"image/color"
)

// LetterboxTransform maps coordinates from the (square, padded) model input back
// to the original image. It is produced by Letterbox.
type LetterboxTransform struct {
	Scale      float64 // source pixels * Scale = model pixels
	PadX, PadY float64 // padding added on the model-input canvas
}

// ToSource maps a model-input coordinate back to original image space.
func (t LetterboxTransform) ToSource(x, y float64) (float64, float64) {
	if t.Scale == 0 {
		return x, y
	}
	return (x - t.PadX) / t.Scale, (y - t.PadY) / t.Scale
}

// Letterbox resizes img to a square size×size canvas preserving aspect ratio,
// padding the remainder with a neutral gray (114, matching Ultralytics). It
// returns the canvas plus the transform needed to map model coordinates back to
// the original image.
func Letterbox(img image.Image, size int) (*image.RGBA, LetterboxTransform) {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	fillGray(dst, 114)
	if sw == 0 || sh == 0 {
		return dst, LetterboxTransform{Scale: 1}
	}

	scale := float64(size) / float64(max(sw, sh))
	nw, nh := int(float64(sw)*scale+0.5), int(float64(sh)*scale+0.5)
	padX := float64(size-nw) / 2
	padY := float64(size-nh) / 2

	// Nearest-neighbour resize into the padded region. Adequate for detection;
	// can be upgraded to bilinear (x/image/draw) if needed.
	for dy := 0; dy < nh; dy++ {
		sy := b.Min.Y + int(float64(dy)/scale)
		if sy >= b.Max.Y {
			sy = b.Max.Y - 1
		}
		for dx := 0; dx < nw; dx++ {
			sx := b.Min.X + int(float64(dx)/scale)
			if sx >= b.Max.X {
				sx = b.Max.X - 1
			}
			dst.Set(int(padX)+dx, int(padY)+dy, img.At(sx, sy))
		}
	}
	return dst, LetterboxTransform{Scale: scale, PadX: padX, PadY: padY}
}

// ToCHW converts an RGBA image into a normalized CHW float32 tensor (R,G,B
// planes, values /255), matching the Ultralytics input convention.
func ToCHW(img *image.RGBA) []float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]float32, 3*w*h)
	plane := w * h
	idx := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			out[idx] = float32(r>>8) / 255
			out[plane+idx] = float32(g>>8) / 255
			out[2*plane+idx] = float32(bl>>8) / 255
			idx++
		}
	}
	return out
}

// Crop returns the sub-image within r intersected with img's bounds, as a fresh
// RGBA copy. An empty intersection yields a 1×1 image.
func Crop(img image.Image, r image.Rectangle) *image.RGBA {
	r = r.Intersect(img.Bounds())
	if r.Empty() {
		r = image.Rect(0, 0, 1, 1)
	}
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			dst.Set(x, y, img.At(r.Min.X+x, r.Min.Y+y))
		}
	}
	return dst
}

// PadRect expands r by frac of its size on every side, clamped to bounds.
func PadRect(r, bounds image.Rectangle, frac float64) image.Rectangle {
	dx := int(float64(r.Dx()) * frac)
	dy := int(float64(r.Dy()) * frac)
	return image.Rect(r.Min.X-dx, r.Min.Y-dy, r.Max.X+dx, r.Max.Y+dy).Intersect(bounds)
}

func fillGray(img *image.RGBA, v uint8) {
	c := color.RGBA{v, v, v, 255}
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			img.SetRGBA(x, y, c)
		}
	}
}
