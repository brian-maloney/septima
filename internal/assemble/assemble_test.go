package assemble

import (
	"image"
	"testing"

	"github.com/brian-maloney/septima/internal/detect"
)

var tankClasses = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ":", "-"}

func det(class, x0, y0, x1, y1 int, score float64) detect.Detection {
	return detect.Detection{Class: class, Score: score, Box: image.Rect(x0, y0, x1, y1)}
}

// The tank display renders 4-digit values as two visually separated groups
// (e.g. "12 | 15"); they must still read as one left-to-right row.
func TestAssembleTwoGroupRow(t *testing.T) {
	dets := []detect.Detection{
		det(5, 80, 0, 100, 40, 0.9), // '5' (placed out of order on purpose)
		det(1, 0, 0, 20, 40, 0.9),   // '1'
		det(2, 25, 0, 45, 40, 0.8),  // '2'
		det(1, 55, 0, 75, 40, 0.85), // '1'
	}
	got := Assemble(dets, tankClasses)
	if got.Text != "1215" {
		t.Fatalf("Text = %q, want %q", got.Text, "1215")
	}
	if len(got.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(got.Rows))
	}
}

func TestAssembleTwoRows(t *testing.T) {
	dets := []detect.Detection{
		// top row "86" at y 0-40
		det(8, 0, 0, 20, 40, 0.9),
		det(6, 25, 0, 45, 40, 0.9),
		// bottom row "47" at y 100-140
		det(4, 0, 100, 20, 140, 0.9),
		det(7, 25, 100, 45, 140, 0.9),
	}
	got := Assemble(dets, tankClasses)
	if got.Text != "86\n47" {
		t.Fatalf("Text = %q, want %q", got.Text, "86\n47")
	}
	if len(got.Rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(got.Rows))
	}
}

func TestAssembleEmpty(t *testing.T) {
	if got := Assemble(nil, tankClasses); got.Text != "" || len(got.Rows) != 0 {
		t.Fatalf("empty input should yield empty reading, got %+v", got)
	}
}

func TestAssembleConfidenceIsMinRow(t *testing.T) {
	dets := []detect.Detection{
		det(1, 0, 0, 20, 40, 0.9),
		det(2, 25, 0, 45, 40, 0.5), // drags row 1 mean down
		det(3, 0, 100, 20, 140, 0.8),
	}
	got := Assemble(dets, tankClasses)
	if got.Confidence > 0.71 || got.Confidence < 0.69 { // row1 mean = 0.7
		t.Fatalf("confidence = %.3f, want ~0.70 (min row mean)", got.Confidence)
	}
}
