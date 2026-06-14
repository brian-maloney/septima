package preprocess

import (
	"fmt"
	"math"

	"gocv.io/x/gocv"
)

// MakeMonoOp converts to a binary image using a global luminance threshold.
type MakeMonoOp struct {
	ThresholdPct float64 // 0-100; default 50
	Absolute     bool    // if true, do not adjust threshold to the image range
}

func (o MakeMonoOp) Name() string { return "make_mono" }

func (o MakeMonoOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	gray := toGray(src)
	defer gray.Close()

	mn, mx, _, _ := gocv.MinMaxLoc(gray)
	var t float32
	if o.Absolute {
		t = float32(o.ThresholdPct / 100.0 * 255.0)
	} else {
		t = mn + float32(o.ThresholdPct/100.0)*(mx-mn)
	}

	dst := gocv.NewMat()
	gocv.Threshold(gray, &dst, t, 255, gocv.ThresholdBinary)
	return dst, nil
}

// IterThresholdOp uses 1D k-means (ssocr's default) to find an optimal threshold.
type IterThresholdOp struct {
	ThresholdPct float64 // starting percentage hint (50 default)
}

func (o IterThresholdOp) Name() string { return "iter_threshold" }

func (o IterThresholdOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	gray := toGray(src)
	defer gray.Close()

	mn, mx, _, _ := gocv.MinMaxLoc(gray)
	t := mn + float32(o.ThresholdPct/100.0)*(mx-mn)

	for {
		var sumLow, sumHigh float64
		var cntLow, cntHigh int
		for row := 0; row < gray.Rows(); row++ {
			for col := 0; col < gray.Cols(); col++ {
				v := float32(gray.GetUCharAt(row, col))
				if v < t {
					sumLow += float64(v)
					cntLow++
				} else {
					sumHigh += float64(v)
					cntHigh++
				}
			}
		}
		meanLow := 0.0
		if cntLow > 0 {
			meanLow = sumLow / float64(cntLow)
		}
		meanHigh := 255.0
		if cntHigh > 0 {
			meanHigh = sumHigh / float64(cntHigh)
		}
		newT := float32((meanLow + meanHigh) / 2.0)
		if math.Abs(float64(newT-t)) < 0.5 {
			break
		}
		t = newT
	}

	dst := gocv.NewMat()
	gocv.Threshold(gray, &dst, t, 255, gocv.ThresholdBinary)
	return dst, nil
}

// DynamicThresholdOp applies local adaptive thresholding with a given window.
type DynamicThresholdOp struct {
	W, H int
}

func (o DynamicThresholdOp) Name() string { return fmt.Sprintf("dynamic_threshold %d %d", o.W, o.H) }

func (o DynamicThresholdOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	gray := toGray(src)
	defer gray.Close()

	blockSize := o.W
	if blockSize%2 == 0 {
		blockSize++
	}
	if blockSize < 3 {
		blockSize = 3
	}
	dst := gocv.NewMat()
	gocv.AdaptiveThreshold(gray, &dst, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, blockSize, 2)
	return dst, nil
}

// OtsuThresholdOp applies Otsu's method for automatic global thresholding.
type OtsuThresholdOp struct{}

func (o OtsuThresholdOp) Name() string { return "otsu_threshold" }

func (o OtsuThresholdOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	gray := toGray(src)
	defer gray.Close()
	dst := gocv.NewMat()
	gocv.Threshold(gray, &dst, 0, 255, gocv.ThresholdBinary+gocv.ThresholdOtsu)
	return dst, nil
}

// RGBThresholdOp thresholds each color channel independently.
type RGBThresholdOp struct {
	ThresholdPct float64
}

func (o RGBThresholdOp) Name() string { return "rgb_threshold" }

func (o RGBThresholdOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Channels() < 3 {
		return MakeMonoOp{ThresholdPct: o.ThresholdPct}.Apply(src)
	}
	channels := gocv.Split(src)
	defer func() {
		for i := range channels {
			channels[i].Close()
		}
	}()
	t := float32(o.ThresholdPct / 100.0 * 255.0)
	masks := make([]gocv.Mat, len(channels))
	for i, ch := range channels {
		m := gocv.NewMat()
		gocv.Threshold(ch, &m, t, 255, gocv.ThresholdBinary)
		masks[i] = m
	}
	defer func() {
		for i := range masks {
			masks[i].Close()
		}
	}()
	dst := gocv.NewMat()
	gocv.BitwiseAnd(masks[0], masks[1], &dst)
	tmp := gocv.NewMat()
	gocv.BitwiseAnd(dst, masks[2], &tmp)
	dst.Close()
	return tmp, nil
}

// SingleChannelThresholdOp thresholds one color channel (R, G, or B).
type SingleChannelThresholdOp struct {
	Channel      int // 0=B, 1=G, 2=R (OpenCV channel order)
	ThresholdPct float64
}

func (o SingleChannelThresholdOp) Name() string {
	names := []string{"b_threshold", "g_threshold", "r_threshold"}
	if o.Channel >= 0 && o.Channel < 3 {
		return names[o.Channel]
	}
	return "channel_threshold"
}

func (o SingleChannelThresholdOp) Apply(src gocv.Mat) (gocv.Mat, error) {
	var ch gocv.Mat
	if src.Channels() >= 3 {
		channels := gocv.Split(src)
		for i, c := range channels {
			if i != o.Channel {
				c.Close()
			}
		}
		ch = channels[o.Channel]
	} else {
		ch = src.Clone()
	}
	defer ch.Close()
	dst := gocv.NewMat()
	t := float32(o.ThresholdPct / 100.0 * 255.0)
	gocv.Threshold(ch, &dst, t, 255, gocv.ThresholdBinary)
	return dst, nil
}

// toGray converts a Mat to single-channel grayscale if needed.
func toGray(src gocv.Mat) gocv.Mat {
	if src.Channels() == 1 {
		return src.Clone()
	}
	dst := gocv.NewMat()
	gocv.CvtColor(src, &dst, gocv.ColorBGRToGray)
	return dst
}
