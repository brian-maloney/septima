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

// A strongly tilted single row (e.g. a calculator photographed at an angle) must
// stay one row — its baseline slopes so endpoints don't vertically overlap, but
// adjacent digits stay close (single-linkage).
func TestAssembleTiltedRow(t *testing.T) {
	var dets []detect.Detection
	y := 100
	for i := 0; i < 12; i++ {
		x := 100 + i*60
		dets = append(dets, det((i%9)+1, x, y, x+55, y+130, 0.9))
		y += 15 // baseline climbs ~15px per digit -> 165px total span > one digit
	}
	got := Assemble(dets, tankClasses)
	if len(got.Rows) != 1 {
		t.Fatalf("tilted single row split into %d rows: %q", len(got.Rows), got.Text)
	}
	if len([]rune(got.Text)) != 12 {
		t.Fatalf("expected 12 chars in one row, got %q", got.Text)
	}
}

func TestTrimEdgePunctuation(t *testing.T) {
	mk := func(s string) []Char {
		var c []Char
		for _, r := range s {
			c = append(c, Char{R: r})
		}
		return c
	}
	str := func(c []Char) string {
		var b []rune
		for _, ch := range c {
			b = append(b, ch.R)
		}
		return string(b)
	}
	cases := map[string]string{
		".153.": "153", // leading + trailing dot
		":24":   "24",  // leading colon
		"812.":  "812", // trailing dot
		"-5":    "-5",  // leading minus kept
		"5-":    "5",   // trailing minus dropped
		".:":    "",    // punctuation only
		"0.68":  "0.68", // interior decimal untouched
	}
	for in, want := range cases {
		if got := str(trimEdgePunctuation(mk(in))); got != want {
			t.Errorf("trimEdgePunctuation(%q) = %q, want %q", in, got, want)
		}
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
