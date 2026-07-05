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

// defaultInputSize is the YOLO square input size assumed when classes.json names
// no size at all.
const defaultInputSize = 640

// Classes describes the model input sizes and class label tables, loaded from
// models/classes.json. PanelClasses labels the stage-1 detector; DigitClasses
// labels the stage-2 detector (each entry a single-rune string).
//
// The panel and digit models can be exported at different input sizes (e.g. a
// 640 panel detector feeding a 1280 digit detector), so the sizes are per stage.
// InputSize is the legacy shared field: it is used as the fallback for either
// stage whose specific size is absent, keeping older classes.json files working.
// Read the resolved sizes through PanelSize / DigitSize, never the raw fields.
type Classes struct {
	InputSize      int      `json:"input_size"`
	PanelInputSize int      `json:"panel_input_size"`
	DigitInputSize int      `json:"digit_input_size"`
	PanelClasses   []string `json:"panel_classes"`
	DigitClasses   []string `json:"digit_classes"`
}

// PanelSize is the stage-1 (panel) detector input size, falling back to the
// legacy shared input_size and finally the YOLO default of 640.
func (c Classes) PanelSize() int { return resolveInputSize(c.PanelInputSize, c.InputSize) }

// DigitSize is the stage-2 (digit) detector input size, with the same fallback
// chain as PanelSize.
func (c Classes) DigitSize() int { return resolveInputSize(c.DigitInputSize, c.InputSize) }

// resolveInputSize prefers the stage-specific size, then the shared legacy size,
// then the default — treating any non-positive value as unset.
func resolveInputSize(stage, shared int) int {
	if stage > 0 {
		return stage
	}
	if shared > 0 {
		return shared
	}
	return defaultInputSize
}

// LoadClasses reads classes.json from modelDir.
func LoadClasses(modelDir string) (Classes, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "classes.json"))
	if err != nil {
		return Classes{}, fmt.Errorf("load classes.json: %w", err)
	}
	return ParseClasses(data)
}

// ParseClasses unmarshals classes.json contents already read into memory
// (e.g. a go:embed'd classes.json), sharing LoadClasses's JSON schema.
func ParseClasses(data []byte) (Classes, error) {
	var c Classes
	if err := json.Unmarshal(data, &c); err != nil {
		return c, fmt.Errorf("parse classes.json: %w", err)
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

// OpenModelFromBytes loads an ONNX detection model from in-memory data (e.g. a
// go:embed'd model) expecting numClasses classes at the given square input size.
func OpenModelFromBytes(data []byte, numClasses, inputSize int) (*Model, error) {
	sess, err := onnx.OpenFromBytes(data)
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
