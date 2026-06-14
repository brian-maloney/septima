package preprocess

import (
	"fmt"

	"gocv.io/x/gocv"
)

// LuminanceFormula selects the grayscale conversion formula.
type LuminanceFormula int

const (
	LuminanceRec601   LuminanceFormula = iota
	LuminanceRec709
	LuminanceLinear
	LuminanceMinimum
	LuminanceMaximum
	LuminanceRed
	LuminanceGreen
	LuminanceBlue
)

// GrayscaleOp converts to a single-channel luminance image.
type GrayscaleOp struct{ Formula LuminanceFormula }

func (o GrayscaleOp) Name() string { return fmt.Sprintf("grayscale %d", o.Formula) }

func (o GrayscaleOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Channels() == 1 {
		return src.Clone(), nil
	}
	switch o.Formula {
	case LuminanceLinear:
		channels := gocv.Split(src)
		defer func() {
			for i := range channels {
				channels[i].Close()
			}
		}()
		dst := gocv.NewMat()
		gocv.AddWeighted(channels[0], 1.0/3.0, channels[1], 1.0/3.0, 0, &dst)
		tmp := gocv.NewMat()
		gocv.AddWeighted(dst, 1.0, channels[2], 1.0/3.0, 0, &tmp)
		dst.Close()
		return tmp, nil
	case LuminanceMinimum:
		channels := gocv.Split(src)
		defer func() {
			for i := range channels {
				channels[i].Close()
			}
		}()
		dst := gocv.NewMat()
		gocv.Min(channels[0], channels[1], &dst)
		tmp := gocv.NewMat()
		gocv.Min(dst, channels[2], &tmp)
		dst.Close()
		return tmp, nil
	case LuminanceMaximum:
		channels := gocv.Split(src)
		defer func() {
			for i := range channels {
				channels[i].Close()
			}
		}()
		dst := gocv.NewMat()
		gocv.Max(channels[0], channels[1], &dst)
		tmp := gocv.NewMat()
		gocv.Max(dst, channels[2], &tmp)
		dst.Close()
		return tmp, nil
	case LuminanceRed:
		channels := gocv.Split(src)
		defer func() {
			for i := range channels {
				if i != 2 {
					channels[i].Close()
				}
			}
		}()
		return channels[2].Clone(), nil
	case LuminanceGreen:
		channels := gocv.Split(src)
		defer func() {
			for i := range channels {
				if i != 1 {
					channels[i].Close()
				}
			}
		}()
		return channels[1].Clone(), nil
	case LuminanceBlue:
		channels := gocv.Split(src)
		defer func() {
			for i := range channels {
				if i != 0 {
					channels[i].Close()
				}
			}
		}()
		return channels[0].Clone(), nil
	default: // Rec601, Rec709
		dst := gocv.NewMat()
		gocv.CvtColor(src, &dst, gocv.ColorBGRToGray)
		return dst, nil
	}
}

// InvertOp swaps foreground and background.
type InvertOp struct{}

func (o InvertOp) Name() string { return "invert" }

func (o InvertOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	dst := gocv.NewMat()
	gocv.BitwiseNot(src, &dst)
	return dst, nil
}

// GrayStretchOp remaps luminance interval [T1,T2] to [0,255].
type GrayStretchOp struct {
	T1, T2     float64
	Percentage bool
}

func (o GrayStretchOp) Name() string {
	return fmt.Sprintf("gray_stretch %.1f %.1f", o.T1, o.T2)
}

func (o GrayStretchOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	gray := toGray(src)
	defer gray.Close()

	t1, t2 := o.T1, o.T2
	if o.Percentage {
		mn, mx, _, _ := gocv.MinMaxLoc(gray)
		t1 = float64(mn) + t1/100.0*float64(mx-mn)
		t2 = float64(mn) + t2/100.0*float64(mx-mn)
	}

	alpha := float32(255.0 / (t2 - t1))
	beta := float32(-t1 * float64(alpha))
	dst := gocv.NewMat()
	gray.ConvertToWithParams(&dst, gocv.MatTypeCV8U, alpha, beta)
	return dst, nil
}
