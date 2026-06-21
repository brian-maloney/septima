package detect

import (
	"image"
	"testing"

	"github.com/brian-maloney/septima/internal/imageproc"
)

func TestNMSSuppressesOverlap(t *testing.T) {
	dets := []Detection{
		{Class: 0, Score: 0.9, Box: image.Rect(0, 0, 10, 10)},
		{Class: 0, Score: 0.6, Box: image.Rect(1, 1, 11, 11)}, // heavy overlap, lower score
		{Class: 0, Score: 0.8, Box: image.Rect(100, 100, 110, 110)},
	}
	got := NMS(dets, 0.45)
	if len(got) != 2 {
		t.Fatalf("got %d detections, want 2", len(got))
	}
	if got[0].Score != 0.9 {
		t.Errorf("highest score should sort first, got %.2f", got[0].Score)
	}
}

func TestNMSKeepsDifferentClasses(t *testing.T) {
	dets := []Detection{
		{Class: 0, Score: 0.9, Box: image.Rect(0, 0, 10, 10)},
		{Class: 1, Score: 0.8, Box: image.Rect(0, 0, 10, 10)}, // same box, different class
	}
	if got := NMS(dets, 0.45); len(got) != 2 {
		t.Fatalf("class-wise NMS should keep both, got %d", len(got))
	}
}

func TestDedupeNestedBoxes(t *testing.T) {
	// A narrow '1' box nested inside a wider one — same glyph, low IoU. One survives.
	dets := []Detection{
		{Class: 1, Score: 0.638, Box: image.Rect(1013, 525, 1057, 613)}, // wide
		{Class: 1, Score: 0.851, Box: image.Rect(1041, 525, 1056, 613)}, // narrow, nested
	}
	got := DedupeAcrossClasses(dets, 0.5)
	if len(got) != 1 {
		t.Fatalf("nested duplicate boxes should collapse to 1, got %d", len(got))
	}
	if got[0].Score != 0.851 {
		t.Errorf("should keep higher-scoring box, got score %.3f", got[0].Score)
	}
}

func TestDedupeKeepsAdjacentDigits(t *testing.T) {
	// Three '1's a full pitch (~62px) apart — none should be suppressed.
	dets := []Detection{
		{Class: 1, Score: 0.94, Box: image.Rect(943, 526, 993, 616)},
		{Class: 1, Score: 0.85, Box: image.Rect(1041, 525, 1056, 613)},
		{Class: 1, Score: 0.88, Box: image.Rect(1103, 522, 1120, 610)},
	}
	if got := DedupeAcrossClasses(dets, 0.5); len(got) != 3 {
		t.Fatalf("adjacent distinct digits must all survive, got %d", len(got))
	}
}

func TestDedupeAcrossClassesSameBox(t *testing.T) {
	// Same glyph detected as both '9' and '4' — keep the higher score.
	dets := []Detection{
		{Class: 9, Score: 0.70, Box: image.Rect(0, 0, 50, 90)},
		{Class: 4, Score: 0.86, Box: image.Rect(0, 0, 50, 90)},
	}
	got := DedupeAcrossClasses(dets, 0.5)
	if len(got) != 1 || got[0].Class != 4 {
		t.Fatalf("same-box different-class should keep higher score (class 4), got %+v", got)
	}
}

func TestDedupeKeepsDecimalOverlappingDigit(t *testing.T) {
	// A real '.' sits at the bottom-right of a '0', so its center falls inside
	// the digit's box — but it's short, not a duplicate, and must be kept.
	dets := []Detection{
		{Class: 0, Score: 0.84, Box: image.Rect(287, 105, 359, 254)}, // '0', h=149
		{Class: 10, Score: 0.31, Box: image.Rect(348, 229, 369, 260)}, // '.', h=31, center inside '0'
	}
	if got := DedupeAcrossClasses(dets, 0.5); len(got) != 2 {
		t.Fatalf("decimal overlapping a digit must survive, got %d detections", len(got))
	}
}

func TestMergeColonDots(t *testing.T) {
	const dot, colon = 10, 11
	// Two stacked dots (same x, separated vertically within a digit height) + two
	// digits ~349px tall -> the dots become one ':'.
	dets := []Detection{
		{Class: 2, Score: 0.9, Box: image.Rect(517, 292, 706, 641)},
		{Class: dot, Score: 0.66, Box: image.Rect(710, 583, 753, 635)},
		{Class: dot, Score: 0.65, Box: image.Rect(716, 393, 748, 428)},
		{Class: 4, Score: 0.9, Box: image.Rect(760, 302, 930, 634)},
	}
	got := MergeColonDots(dets, dot, colon)
	colons, dots := 0, 0
	for _, d := range got {
		switch d.Class {
		case colon:
			colons++
		case dot:
			dots++
		}
	}
	if colons != 1 || dots != 0 {
		t.Fatalf("expected 1 colon and 0 dots, got %d colons %d dots", colons, dots)
	}
}

func TestMergeColonDotsKeepsSeparateRowsDecimals(t *testing.T) {
	const dot, colon = 10, 11
	// Two decimals a full row apart (different rows of a gas pump) must NOT merge.
	dets := []Detection{
		{Class: 8, Score: 0.9, Box: image.Rect(0, 0, 50, 90)},   // digit, row 1
		{Class: dot, Score: 0.6, Box: image.Rect(55, 70, 65, 88)}, // '.' row 1
		{Class: 1, Score: 0.9, Box: image.Rect(0, 300, 50, 390)}, // digit, row 2
		{Class: dot, Score: 0.6, Box: image.Rect(55, 370, 65, 388)}, // '.' row 2 (x-aligned, far below)
	}
	got := MergeColonDots(dets, dot, colon)
	for _, d := range got {
		if d.Class == colon {
			t.Fatalf("decimals on separate rows must not merge into a colon")
		}
	}
}

func TestDecodeYOLOMapsThroughTransform(t *testing.T) {
	// 2 classes, 1 box. Layout: [cx,cy,w,h, score0, score1] each as one column.
	// Box centered at (50,50) size 20x20 in model space, class 1 wins.
	numClasses, numBoxes := 2, 1
	data := []float32{
		50, // cx
		50, // cy
		20, // w
		20, // h
		0.1, // class 0 score
		0.8, // class 1 score
	}
	// Identity transform (scale 1, no pad) -> source coords == model coords.
	tr := imageproc.LetterboxTransform{Scale: 1}
	dets := DecodeYOLO(data, numClasses, numBoxes, tr, 0.25)
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets))
	}
	if dets[0].Class != 1 {
		t.Errorf("class = %d, want 1", dets[0].Class)
	}
	want := image.Rect(40, 40, 60, 60)
	if dets[0].Box != want {
		t.Errorf("box = %v, want %v", dets[0].Box, want)
	}
}

func TestDecodeYOLOScaleAndPad(t *testing.T) {
	// scale 0.5, pad (10,10): source = (model - 10) / 0.5
	tr := imageproc.LetterboxTransform{Scale: 0.5, PadX: 10, PadY: 10}
	data := []float32{30, 30, 20, 20, 0.9}
	dets := DecodeYOLO(data, 1, 1, tr, 0.25)
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets))
	}
	// x0 model = 20 -> (20-10)/0.5 = 20 ; x1 model = 40 -> (40-10)/0.5 = 60
	want := image.Rect(20, 20, 60, 60)
	if dets[0].Box != want {
		t.Errorf("box = %v, want %v", dets[0].Box, want)
	}
}
