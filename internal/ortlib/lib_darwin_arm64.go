package ortlib

import _ "embed"

//go:embed lib/libonnxruntime-darwin-arm64-1.27.0.dylib
var Bytes []byte

// Filename is the name the embedded bytes are extracted to on disk. It bakes
// in the ONNX Runtime version so a version bump naturally produces a new
// cache filename rather than requiring a content-hash comparison.
const Filename = "libonnxruntime-darwin-arm64-1.27.0.dylib"
