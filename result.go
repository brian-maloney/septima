package septima

import "image"

// Digit is a single recognized character from the display.
type Digit struct {
	Char       rune            // the decoded character ('0'-'9', '.', ':', '-')
	Box        image.Rectangle // bounding box in original image coordinates
	Confidence float64         // detector score for this glyph
}

// Row is one horizontal line of digits found in the display.
type Row struct {
	Text       string
	Digits     []Digit
	Box        image.Rectangle
	Confidence float64 // mean digit confidence in the row
}

// Result is the top-level output of a recognition call.
type Result struct {
	// Rows contains one entry per detected row of digits, top to bottom.
	Rows []Row
	// Text is the rows joined by "\n" — convenient for single-row callers.
	Text string
	// Confidence is the minimum per-row confidence (0 when no rows).
	Confidence float64
	// Debug is non-nil when WithDebugDir was set.
	Debug *DebugInfo
}

// DebugInfo holds diagnostic artifacts written to DebugDir when set.
type DebugInfo struct {
	Stages []DebugStage
}

// DebugStage captures the name and file path of one intermediate image.
type DebugStage struct {
	Name string
	Path string
}
