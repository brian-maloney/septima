// Package ortlib embeds the ONNX Runtime shared library for each supported
// platform so cmd/septima can run without any external files. Exactly one of
// the lib_<goos>_<goarch>.go files is compiled in for a given build (matched
// by Go's implicit GOOS/GOARCH file-suffix rules); lib_other.go is the
// fallback for every unsupported platform and carries no go:embed directive,
// so go build/go vet always succeed regardless of target.
package ortlib

// Available reports whether this platform has an embedded ONNX Runtime
// library baked in.
func Available() bool { return len(Bytes) > 0 }
