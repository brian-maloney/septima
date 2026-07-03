package preprocess

import (
	"image"
	"image/color"
	"testing"
)

func checkerboard(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if (x+y)%2 == 0 {
				img.Set(x, y, color.RGBA{0, 0, 0, 255})
			} else {
				img.Set(x, y, color.RGBA{255, 255, 255, 255})
			}
		}
	}
	return img
}

func TestCropOp(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	src.Set(5, 5, color.RGBA{1, 2, 3, 255})

	out, err := CropOp{X: 4, Y: 4, W: 3, H: 3}.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if out.Bounds().Dx() != 3 || out.Bounds().Dy() != 3 {
		t.Fatalf("got bounds %v, want 3x3", out.Bounds())
	}
	r, g, b, _ := out.At(1, 1).RGBA()
	if uint8(r>>8) != 1 || uint8(g>>8) != 2 || uint8(b>>8) != 3 {
		t.Fatalf("cropped pixel = (%d,%d,%d), want (1,2,3)", r>>8, g>>8, b>>8)
	}
}

func TestCropOpOutOfBounds(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	if _, err := (CropOp{X: 100, Y: 100, W: 5, H: 5}).Apply(src); err == nil {
		t.Fatal("expected error for out-of-bounds crop")
	}
}

func TestMirrorHoriz(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 2, 1))
	src.Set(0, 0, color.RGBA{1, 0, 0, 255})
	src.Set(1, 0, color.RGBA{2, 0, 0, 255})

	out, err := MirrorOp{Horiz: true}.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	r0, _, _, _ := out.At(0, 0).RGBA()
	r1, _, _, _ := out.At(1, 0).RGBA()
	if uint8(r0>>8) != 2 || uint8(r1>>8) != 1 {
		t.Fatalf("mirrored row = (%d,%d), want (2,1)", r0>>8, r1>>8)
	}
}

func TestInvertOp(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 1, 1))
	src.Set(0, 0, color.RGBA{10, 20, 30, 255})

	out, err := InvertOp{}.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	r, g, b, a := out.At(0, 0).RGBA()
	if uint8(r>>8) != 245 || uint8(g>>8) != 235 || uint8(b>>8) != 225 || uint8(a>>8) != 255 {
		t.Fatalf("inverted pixel = (%d,%d,%d,%d), want (245,235,225,255)", r>>8, g>>8, b>>8, a>>8)
	}
}

func TestGrayscaleOp(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 1, 1))
	src.Set(0, 0, color.RGBA{255, 255, 255, 255})

	out, err := GrayscaleOp{}.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if _, ok := out.(*image.Gray); !ok {
		t.Fatalf("got %T, want *image.Gray", out)
	}
}

func TestThresholdOpBinarizes(t *testing.T) {
	src := checkerboard(4, 4)
	out, err := ThresholdOp{ThresholdPct: 50}.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			v := out.At(x, y).(color.Gray).Y
			if v != 0 && v != 255 {
				t.Fatalf("pixel (%d,%d) = %d, want 0 or 255", x, y, v)
			}
		}
	}
}

func TestOtsuThresholdSeparatesBimodal(t *testing.T) {
	// Half the image dark, half bright: Otsu should land near the midpoint
	// and reproduce the two original clusters exactly.
	img := image.NewGray(image.Rect(0, 0, 10, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 10; x++ {
			if x < 5 {
				img.SetGray(x, y, color.Gray{Y: 10})
			} else {
				img.SetGray(x, y, color.Gray{Y: 240})
			}
		}
	}
	out, err := OtsuThresholdOp{}.Apply(img)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	for y := 0; y < 2; y++ {
		for x := 0; x < 10; x++ {
			v := out.At(x, y).(color.Gray).Y
			want := uint8(0)
			if x >= 5 {
				want = 255
			}
			if v != want {
				t.Fatalf("pixel (%d,%d) = %d, want %d", x, y, v, want)
			}
		}
	}
}

func TestRotate90PreservesSize(t *testing.T) {
	src := checkerboard(8, 8)
	out, err := RotateOp{Degrees: 90}.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if out.Bounds().Dx() != 8 || out.Bounds().Dy() != 8 {
		t.Fatalf("got bounds %v, want 8x8", out.Bounds())
	}
}

func TestPipelineAppliesInOrder(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	p := Pipeline{
		CropOp{X: 0, Y: 0, W: 5, H: 5},
		GrayscaleOp{},
	}
	out, err := p.Apply(src)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if out.Bounds().Dx() != 5 || out.Bounds().Dy() != 5 {
		t.Fatalf("got bounds %v, want 5x5", out.Bounds())
	}
	if _, ok := out.(*image.Gray); !ok {
		t.Fatalf("got %T, want *image.Gray", out)
	}
}

func TestPipelineErrorStopsChain(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	p := Pipeline{CropOp{X: 100, Y: 100, W: 5, H: 5}}
	if _, err := p.Apply(src); err == nil {
		t.Fatal("expected error to propagate from Pipeline.Apply")
	}
}
