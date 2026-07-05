//go:build !(darwin && arm64) && !(linux && amd64) && !(linux && arm64) && !(windows && amd64)

package ortlib

// Bytes is nil on platforms with no embedded ONNX Runtime library. No
// go:embed directive here, deliberately: that keeps go build/go vet working
// on any GOOS/GOARCH, since there is no lib file to point at.
var Bytes []byte

// Filename is unused when Bytes is empty.
const Filename = ""
