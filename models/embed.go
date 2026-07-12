// Package models embeds the default panel/digit ONNX models and their class
// table so cmd/septima can run as a single self-contained binary. It lives
// physically inside models/ because go:embed patterns cannot reach outside
// the source file's own directory.
package models

import (
	_ "embed"
	"fmt"
)

//go:embed panel.onnx
var PanelONNX []byte

//go:embed digits.onnx
var DigitsONNX []byte

//go:embed classes.json
var ClassesJSON []byte

// minPlausibleONNXBytes is well above the size of a stray placeholder file,
// which is what these embeds contain if the repo was built without
// scripts/fetch-models.sh resolving the real model weights first.
const minPlausibleONNXBytes = 1 << 20 // 1 MiB

// Verify catches the "forgot to fetch model weights" failure mode with a
// clear error instead of a cryptic ONNX Runtime parse failure at model-open
// time.
func Verify() error {
	if len(PanelONNX) < minPlausibleONNXBytes || len(DigitsONNX) < minPlausibleONNXBytes {
		return fmt.Errorf("embedded model data looks incomplete (panel.onnx=%d bytes, digits.onnx=%d bytes) — "+
			"this binary was likely built without scripts/fetch-models.sh resolving models/*.onnx first",
			len(PanelONNX), len(DigitsONNX))
	}
	return nil
}
