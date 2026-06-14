package septima

import "image"

// Source indicates which recognition method produced a digit.
type Source int

const (
	SourceGeometric Source = iota
	SourceDNN
	SourceBoth
)

// DebugInfo holds per-stage diagnostic images written to DebugDir when set.
type DebugInfo struct {
	Stages []DebugStage
}

// DebugStage captures the name and file path of one intermediate image.
type DebugStage struct {
	Name string
	Path string
}

// Digit is a single recognized character from the display.
type Digit struct {
	Char       rune
	Segments   byte // 7-bit mask: bit0=a(top) … bit6=g(middle)
	Box        image.Rectangle
	Confidence float64
	Source     Source
}

// Row is one horizontal line of digits found in the display.
type Row struct {
	Text       string
	Digits     []Digit
	Box        image.Rectangle
	Confidence float64
}

// Result is the top-level output of a recognition call.
type Result struct {
	// Rows contains one entry per detected row of digits.
	Rows []Row
	// Text is the rows joined by "\n" — convenient for single-row callers.
	Text string
	// Confidence is the minimum per-row confidence.
	Confidence float64
	// Debug is non-nil when WithDebugDir was set.
	Debug *DebugInfo
}
