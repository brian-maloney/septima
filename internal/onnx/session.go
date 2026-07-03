// Package onnx wraps an ONNX Runtime session for YOLO inference via
// github.com/yalue/onnxruntime_go.
//
// The ONNX Runtime shared library is loaded at runtime (not linked at build).
// Point the engine at it with the SEPTIMA_ORT_LIB environment variable, e.g.
// the libonnxruntime.*.dylib/.so that ships inside the Python onnxruntime wheel,
// or a standalone download (onnxruntime.dll on Windows). CPU works everywhere;
// CoreML/CUDA are optional.
package onnx

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Default input/output tensor names produced by Ultralytics YOLO ONNX export.
const (
	defaultInputName  = "images"
	defaultOutputName = "output0"
)

var (
	initOnce sync.Once
	initErr  error
)

func ensureInit() error {
	initOnce.Do(func() {
		if p := locateORT(); p != "" {
			ort.SetSharedLibraryPath(p)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// locateORT resolves the ONNX Runtime shared library path. SEPTIMA_ORT_LIB wins;
// otherwise it probes common locations (the Python onnxruntime wheel bundled in
// the training venv, or a lib dropped next to the models), walking up from both
// the working directory and the executable so it works from any subdir. Empty
// means "let ORT use its built-in default search".
func locateORT() string {
	if p := os.Getenv("SEPTIMA_ORT_LIB"); p != "" {
		return p
	}
	rel := []string{
		"training/.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib",
		"training/.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.so*",
		"models/libonnxruntime*.dylib",
		"models/libonnxruntime*.so*",
		"models/onnxruntime*.dll",
	}
	var bases []string
	if cwd, err := os.Getwd(); err == nil {
		bases = append(bases, cwd)
	}
	if exe, err := os.Executable(); err == nil {
		bases = append(bases, filepath.Dir(exe))
	}
	for _, base := range bases {
		for dir := base; ; {
			for _, pat := range rel {
				if hits, _ := filepath.Glob(filepath.Join(dir, pat)); len(hits) > 0 {
					return hits[0]
				}
			}
			parent := filepath.Dir(dir)
			if parent == dir { // reached filesystem root
				break
			}
			dir = parent
		}
	}
	return ""
}

// Session runs a single-input/single-output detection model.
type Session interface {
	// Run executes the model on a CHW float32 tensor of the given shape
	// (typically [1,3,H,W]) and returns the flattened output and its shape.
	Run(input []float32, shape []int64) (output []float32, outShape []int64, err error)
	// Close releases the session.
	Close() error
}

type session struct {
	s *ort.DynamicAdvancedSession
}

// Open loads an ONNX model from path.
func Open(path string) (Session, error) {
	if err := ensureInit(); err != nil {
		return nil, fmt.Errorf("init onnxruntime (set SEPTIMA_ORT_LIB to the libonnxruntime path): %w", err)
	}
	s, err := ort.NewDynamicAdvancedSession(path, []string{defaultInputName}, []string{defaultOutputName}, nil)
	if err != nil {
		return nil, fmt.Errorf("open model %s: %w", path, err)
	}
	return &session{s: s}, nil
}

func (se *session) Run(input []float32, shape []int64) ([]float32, []int64, error) {
	in, err := ort.NewTensor(ort.NewShape(shape...), input)
	if err != nil {
		return nil, nil, fmt.Errorf("build input tensor: %w", err)
	}
	defer in.Destroy()

	outputs := []ort.Value{nil} // nil -> ORT allocates the output
	if err := se.s.Run([]ort.Value{in}, outputs); err != nil {
		return nil, nil, fmt.Errorf("run: %w", err)
	}
	out := outputs[0]
	defer out.Destroy()

	t, ok := out.(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("unexpected output type %T (want float32 tensor)", out)
	}
	// Copy out of ORT-owned memory before the tensor is destroyed.
	src := t.GetData()
	data := make([]float32, len(src))
	copy(data, src)
	outShape := append([]int64(nil), out.GetShape()...)
	return data, outShape, nil
}

func (se *session) Close() error {
	if se == nil || se.s == nil {
		return nil
	}
	return se.s.Destroy()
}
