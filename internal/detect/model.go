package detect

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/brian-maloney/septima/internal/imageproc"
	"github.com/brian-maloney/septima/internal/onnx"
)

// Classes describes the model input size and class label tables, loaded from
// models/classes.json. PanelClasses labels the stage-1 detector; DigitClasses
// labels the stage-2 detector (each entry a single-rune string).
type Classes struct {
	InputSize    int      `json:"input_size"`
	PanelClasses []string `json:"panel_classes"`
	DigitClasses []string `json:"digit_classes"`
}

// LoadClasses reads classes.json from modelDir.
func LoadClasses(modelDir string) (Classes, error) {
	var c Classes
	data, err := os.ReadFile(filepath.Join(modelDir, "classes.json"))
	if err != nil {
		return c, fmt.Errorf("load classes.json: %w", err)
	}
	if err := json.Unmarshal(data, &c); err != nil {
		return c, fmt.Errorf("parse classes.json: %w", err)
	}
	if c.InputSize == 0 {
		c.InputSize = 640
	}
	return c, nil
}

// Model is one YOLO detection stage: a session plus its class table.
type Model struct {
	session   onnx.Session
	numClass  int
	inputSize int
}

// OpenModel loads an ONNX detection model expecting numClasses classes at the
// given square input size.
func OpenModel(path string, numClasses, inputSize int) (*Model, error) {
	sess, err := onnx.Open(path)
	if err != nil {
		return nil, err
	}
	return &Model{session: sess, numClass: numClasses, inputSize: inputSize}, nil
}

// Close releases the underlying session.
func (m *Model) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	return m.session.Close()
}

// Detect runs the model on img and returns NMS-filtered detections in original
// image coordinates.
func (m *Model) Detect(img image.Image, confThreshold, iouThreshold float64) ([]Detection, error) {
	canvas, tr := imageproc.Letterbox(img, m.inputSize)
	tensor := imageproc.ToCHW(canvas)
	shape := []int64{1, 3, int64(m.inputSize), int64(m.inputSize)}
	out, outShape, err := m.session.Run(tensor, shape)
	if err != nil {
		return nil, err
	}
	numBoxes, err := boxesFromShape(outShape, m.numClass)
	if err != nil {
		return nil, err
	}
	dets := DecodeYOLO(out, m.numClass, numBoxes, tr, confThreshold)
	return NMS(dets, iouThreshold), nil
}

// boxesFromShape extracts the number of boxes from an Ultralytics output shape
// of [1, 4+nc, numBoxes].
func boxesFromShape(shape []int64, numClasses int) (int, error) {
	if len(shape) != 3 || shape[1] != int64(4+numClasses) {
		return 0, fmt.Errorf("unexpected output shape %v for %d classes", shape, numClasses)
	}
	return int(shape[2]), nil
}
