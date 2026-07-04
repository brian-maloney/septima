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

// embeddedLibSource, when set, supplies an embedded ONNX Runtime shared
// library to fall back on when locateORT finds nothing on disk. nil by
// default: internal/onnx has zero import-time dependency on internal/ortlib.
// cmd/septima's main() calls SetEmbeddedLibSource, closing over
// ortlib.Bytes/ortlib.Filename/ortlib.Available().
var embeddedLibSource func() (data []byte, filename string, ok bool)

// SetEmbeddedLibSource registers a fallback source of ONNX Runtime shared
// library bytes, used only when SEPTIMA_ORT_LIB is unset and no library is
// found on disk. Must be called before the first model is opened.
func SetEmbeddedLibSource(f func() (data []byte, filename string, ok bool)) {
	embeddedLibSource = f
}

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
		p := locateORT()
		if p == "" && embeddedLibSource != nil {
			if data, filename, ok := embeddedLibSource(); ok {
				extracted, err := extractEmbeddedLib(data, filename)
				if err != nil {
					initErr = fmt.Errorf("extract embedded onnxruntime library: %w", err)
					return
				}
				p = extracted
			}
		}
		if p != "" {
			ort.SetSharedLibraryPath(p)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// extractEmbeddedLib writes data to a stable per-machine cache path, skipping
// the write if a file of the right size is already there (extraction happens
// once per machine, not once per invocation). Writes via temp-file-then-
// rename for atomicity. No execute bit is needed: dlopen/LoadLibrary only
// require read access.
func extractEmbeddedLib(data []byte, filename string) (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil || dir == "" {
		dir = os.TempDir()
	}
	dir = filepath.Join(dir, "septima", "ortlib")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(dir, filename)
	if info, err := os.Stat(dest); err == nil && info.Size() == int64(len(data)) {
		return dest, nil
	}
	tmp, err := os.CreateTemp(dir, filename+".tmp*")
	if err != nil {
		return "", err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return "", err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return "", err
	}
	if err := os.Rename(tmpPath, dest); err != nil {
		os.Remove(tmpPath)
		return "", err
	}
	return dest, nil
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

// OpenFromBytes loads an ONNX model from in-memory data, e.g. a go:embed'd model.
func OpenFromBytes(data []byte) (Session, error) {
	if err := ensureInit(); err != nil {
		return nil, fmt.Errorf("init onnxruntime (set SEPTIMA_ORT_LIB to the libonnxruntime path): %w", err)
	}
	s, err := ort.NewDynamicAdvancedSessionWithONNXData(data, []string{defaultInputName}, []string{defaultOutputName}, nil)
	if err != nil {
		return nil, fmt.Errorf("open embedded model: %w", err)
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
