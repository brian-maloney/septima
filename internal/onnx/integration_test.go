package onnx

import (
	"os"
	"testing"
)

// TestRunModel is an opt-in runtime smoke test for the ONNX Runtime binding.
// It requires the ORT shared library and a model path:
//
//	SEPTIMA_ORT_LIB=/path/to/libonnxruntime.dylib \
//	SEPTIMA_TEST_ONNX=/path/to/model.onnx \
//	go test ./internal/onnx -run TestRunModel -v
//
// With a stock 640-input YOLO model it validates load → run → output shape.
func TestRunModel(t *testing.T) {
	model := os.Getenv("SEPTIMA_TEST_ONNX")
	if model == "" {
		t.Skip("set SEPTIMA_TEST_ONNX (and SEPTIMA_ORT_LIB) to run the ORT smoke test")
	}
	sess, err := Open(model)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	const size = 640
	input := make([]float32, 1*3*size*size) // zeros
	out, shape, err := sess.Run(input, []int64{1, 3, size, size})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(shape) != 3 || shape[0] != 1 {
		t.Fatalf("output shape = %v, want [1, 4+nc, n]", shape)
	}
	want := int(shape[1] * shape[2])
	if len(out) != want {
		t.Fatalf("output len = %d, want %d (shape %v)", len(out), want, shape)
	}
	t.Logf("ok: output shape %v (%d floats)", shape, len(out))
}
