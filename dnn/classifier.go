// Package dnn provides an optional ONNX-based fallback digit classifier.
// When a geometric decode yields low confidence, this classifier re-examines
// the digit crop and returns a posterior probability over 0-9 + special chars.
package dnn

import (
	"fmt"
	"image"
	"sync"

	"gocv.io/x/gocv"
)

const modelPath = "" // set at build time or override via SetModelPath

var (
	mu      sync.Mutex
	net     gocv.Net
	netOnce sync.Once
	netErr  error
	netFile string
)

// SetModelPath overrides the default embedded model path.
func SetModelPath(path string) {
	mu.Lock()
	defer mu.Unlock()
	netFile = path
}

func loadNet() error {
	file := netFile
	if file == "" {
		return fmt.Errorf("dnn: no model path configured (use dnn.SetModelPath)")
	}
	net = gocv.ReadNetFromONNX(file)
	if net.Empty() {
		return fmt.Errorf("dnn: failed to load model from %s", file)
	}
	return nil
}

// Classify runs the ONNX model on a single digit crop and returns the most
// likely character and a confidence in [0,1].
func Classify(digitGray gocv.Mat) (rune, float64, error) {
	netOnce.Do(func() { netErr = loadNet() })
	if netErr != nil {
		return '?', 0, netErr
	}

	// Resize to 28×28 (model input size)
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(digitGray, &resized, image.Point{X: 28, Y: 28}, 0, 0, gocv.InterpolationLinear)

	blob := gocv.BlobFromImage(resized, 1.0/255.0, image.Point{X: 28, Y: 28},
		gocv.NewScalar(0, 0, 0, 0), false, false)
	defer blob.Close()

	mu.Lock()
	net.SetInput(blob, "")
	output := net.Forward("")
	mu.Unlock()
	defer output.Close()

	// output is 1×N float32 probabilities
	bestIdx := 0
	bestConf := float32(0)
	for i := 0; i < output.Cols(); i++ {
		v := output.GetFloatAt(0, i)
		if v > bestConf {
			bestConf = v
			bestIdx = i
		}
	}

	chars := []rune("0123456789.-abcdef")
	if bestIdx >= len(chars) {
		return '?', float64(bestConf), nil
	}
	return chars[bestIdx], float64(bestConf), nil
}
