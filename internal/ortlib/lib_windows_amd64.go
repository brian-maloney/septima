package ortlib

import _ "embed"

//go:embed lib/onnxruntime-windows-amd64-1.27.0.dll
var Bytes []byte

// Filename is the name the embedded bytes are extracted to on disk. It bakes
// in the ONNX Runtime version so a version bump naturally produces a new
// cache filename rather than requiring a content-hash comparison.
const Filename = "onnxruntime-windows-amd64-1.27.0.dll"
