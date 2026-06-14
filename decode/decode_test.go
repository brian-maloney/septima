package decode

import (
	"strings"
	"testing"
)

func TestHammingByte(t *testing.T) {
	tests := []struct {
		a, b byte
		want int
	}{
		{0, 0, 0},
		{0b1111111, 0b0000000, 7},
		{0b1010101, 0b1010101, 0},
		{0b1010101, 0b0101010, 7},
		{0b1111111, 0b1111110, 1},
		{0b1111111, 0b1111100, 2},
	}
	for _, tt := range tests {
		if got := hammingByte(tt.a, tt.b); got != tt.want {
			t.Errorf("hammingByte(%07b, %07b) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestExactMatch(t *testing.T) {
	known := []struct {
		mask byte
		want rune
	}{
		{SegA | SegB | SegC | SegD | SegE | SegF, '0'},
		{SegB | SegC, '1'},
		{SegA | SegB | SegD | SegE | SegG, '2'},
		{SegA | SegB | SegC | SegD | SegG, '3'},
		{SegB | SegC | SegF | SegG, '4'},
		{SegA | SegC | SegD | SegF | SegG, '5'},
		{SegA | SegC | SegD | SegE | SegF | SegG, '6'},
		{SegA | SegB | SegC, '7'},
		{SegA | SegB | SegC | SegD | SegE | SegF | SegG, '8'},
		{SegA | SegB | SegC | SegD | SegF | SegG, '9'},
		{SegG, '-'},
		{SegA | SegB | SegC | SegE | SegF | SegG, 'A'},
		{SegC | SegD | SegE | SegF | SegG, 'b'},
		{SegA | SegD | SegE | SegF, 'C'},
		{SegB | SegC | SegD | SegE | SegG, 'd'},
		{SegA | SegD | SegE | SegF | SegG, 'E'},
		{SegA | SegE | SegF | SegG, 'F'},
	}
	for _, tt := range known {
		r, conf := exactMatch(tt.mask)
		if r != tt.want {
			t.Errorf("exactMatch(%07b): got %q, want %q", tt.mask, r, tt.want)
		}
		if conf != 1.0 {
			t.Errorf("exactMatch(%07b): conf = %v, want 1.0", tt.mask, conf)
		}
	}
}

func TestExactMatchUnknown(t *testing.T) {
	// 0b1010011 is not in the table
	r, conf := exactMatch(0b1010011)
	if r != '?' {
		t.Errorf("exactMatch(unknown): got %q, want '?'", r)
	}
	if conf != 0 {
		t.Errorf("exactMatch(unknown): conf = %v, want 0", conf)
	}
}

func TestNearestMatch(t *testing.T) {
	// Exact matches should give conf=1.0
	r, conf := nearestMatch(SegB | SegC)
	if r != '1' {
		t.Errorf("nearestMatch '1': got %q, want '1'", r)
	}
	if conf != 1.0 {
		t.Errorf("nearestMatch '1': conf = %v, want 1.0", conf)
	}

	// '0' mask minus SegA (1 bit off) → nearest is still '0'
	// '0' = A|B|C|D|E|F = 63; remove SegA → 62 = B|C|D|E|F
	mask0MinusA := byte(SegB | SegC | SegD | SegE | SegF)
	r2, conf2 := nearestMatch(mask0MinusA)
	if r2 != '0' {
		t.Errorf("nearestMatch near-'0': got %q, want '0'", r2)
	}
	// 1 bit off out of 7 → conf = 6/7 ≈ 0.857
	if conf2 < 0.85 {
		t.Errorf("nearestMatch near-'0': conf = %v, want >= 0.85", conf2)
	}

	// All segments on (127) → exact '8'
	r3, conf3 := nearestMatch(SegA | SegB | SegC | SegD | SegE | SegF | SegG)
	if r3 != '8' {
		t.Errorf("nearestMatch '8': got %q, want '8'", r3)
	}
	if conf3 != 1.0 {
		t.Errorf("nearestMatch '8': conf = %v, want 1.0", conf3)
	}
}

func TestNearestMatchDeterminism(t *testing.T) {
	// Run nearestMatch multiple times on the same input and verify the result
	// is always the same (no map-iteration randomness).
	mask := byte(0b0110110) // some ambiguous mask
	first, firstConf := nearestMatch(mask)
	for i := 0; i < 20; i++ {
		r, c := nearestMatch(mask)
		if r != first || c != firstConf {
			t.Errorf("nearestMatch not deterministic: first=%q %v, got=%q %v on iter %d",
				first, firstConf, r, c, i)
		}
	}
}

func TestDecode(t *testing.T) {
	tests := []struct {
		mask    byte
		cs      CharsetID
		want    rune
		minConf float64
	}{
		// Exact '0' in full charset
		{SegA | SegB | SegC | SegD | SegE | SegF, CharsetFull, '0', 1.0},
		// Exact '5' in decimal charset
		{SegA | SegC | SegD | SegF | SegG, CharsetDecimal, '5', 1.0},
		// Exact '9' in digits charset
		{SegA | SegB | SegC | SegD | SegF | SegG, CharsetDigits, '9', 1.0},
		// Hex 'A' in full charset
		{SegA | SegB | SegC | SegE | SegF | SegG, CharsetFull, 'A', 1.0},
		// Hex 'A' in digits charset: nearest digit is '8' (1 bit off)
		{SegA | SegB | SegC | SegE | SegF | SegG, CharsetDigits, '8', 0.0},
		// '-' in decimal charset
		{SegG, CharsetDecimal, '-', 1.0},
	}
	for _, tt := range tests {
		r, conf := Decode(tt.mask, tt.cs, false, false)
		if r != tt.want {
			t.Errorf("Decode(%07b, charset=%d): got %q, want %q (conf=%v)",
				tt.mask, tt.cs, r, tt.want, conf)
		}
		if conf < tt.minConf {
			t.Errorf("Decode(%07b, charset=%d): conf = %v, want >= %v",
				tt.mask, tt.cs, conf, tt.minConf)
		}
	}
}

func TestDecodeDecimalPoint(t *testing.T) {
	// Decimal point in full charset → '.' at full confidence
	r, conf := Decode(0, CharsetFull, true, false)
	if r != '.' || conf != 1.0 {
		t.Errorf("decimal/full: got %q conf %v, want '.' 1.0", r, conf)
	}

	// Decimal point in decimal charset → '.' at full confidence
	r2, conf2 := Decode(0, CharsetDecimal, true, false)
	if r2 != '.' || conf2 != 1.0 {
		t.Errorf("decimal/decimal: got %q conf %v, want '.' 1.0", r2, conf2)
	}

	// Decimal point in digits charset → '.' at reduced confidence
	r3, conf3 := Decode(0, CharsetDigits, true, false)
	if r3 != '.' {
		t.Errorf("decimal/digits: got %q, want '.'", r3)
	}
	if conf3 != 0.5 {
		t.Errorf("decimal/digits: conf = %v, want 0.5", conf3)
	}
}

func TestDecodeColon(t *testing.T) {
	r, conf := Decode(0, CharsetFull, false, true)
	if r != ':' || conf != 1.0 {
		t.Errorf("colon: got %q conf %v, want ':' 1.0", r, conf)
	}
}

func TestDecodeAllDigitsExact(t *testing.T) {
	digitMasks := []struct {
		mask byte
		want rune
	}{
		{SegA | SegB | SegC | SegD | SegE | SegF, '0'},
		{SegB | SegC, '1'},
		{SegA | SegB | SegD | SegE | SegG, '2'},
		{SegA | SegB | SegC | SegD | SegG, '3'},
		{SegB | SegC | SegF | SegG, '4'},
		{SegA | SegC | SegD | SegF | SegG, '5'},
		{SegA | SegC | SegD | SegE | SegF | SegG, '6'},
		{SegA | SegB | SegC, '7'},
		{SegA | SegB | SegC | SegD | SegE | SegF | SegG, '8'},
		{SegA | SegB | SegC | SegD | SegF | SegG, '9'},
	}
	for _, tt := range digitMasks {
		r, conf := Decode(tt.mask, CharsetDigits, false, false)
		if r != tt.want {
			t.Errorf("Decode(%q mask) = %q, want %q", string(tt.want), r, tt.want)
		}
		if conf != 1.0 {
			t.Errorf("Decode(%q mask) conf = %v, want 1.0", string(tt.want), conf)
		}
	}
}

func TestAspectClassify(t *testing.T) {
	const oneRatio = 4.0
	const minusRatio = 0.5

	// Zero dimensions → 0
	if r := AspectClassify(0, 40, 40, oneRatio, minusRatio); r != 0 {
		t.Errorf("zero width: got %q, want 0", r)
	}
	if r := AspectClassify(10, 0, 40, oneRatio, minusRatio); r != 0 {
		t.Errorf("zero height: got %q, want 0", r)
	}

	// Tall narrow box (h/w = 5 > 4) → '1'
	if r := AspectClassify(10, 50, 40, oneRatio, minusRatio); r != '1' {
		t.Errorf("tall narrow: got %q, want '1'", r)
	}

	// Short wide box (w/h = 5 > 1/0.5=2, h < medianH/2) → '-'
	if r := AspectClassify(40, 8, 40, oneRatio, minusRatio); r != '-' {
		t.Errorf("short wide: got %q, want '-'", r)
	}

	// Typical digit (h/w ≈ 2) → 0 (use segment sampling)
	if r := AspectClassify(20, 40, 40, oneRatio, minusRatio); r != 0 {
		t.Errorf("typical digit: got %q, want 0", r)
	}

	// Borderline: h/w exactly at oneRatio = 4 → NOT classified (must be strictly greater)
	if r := AspectClassify(10, 40, 40, oneRatio, minusRatio); r != 0 {
		t.Errorf("at exact oneRatio: got %q, want 0", r)
	}

	// Short but not wide enough (w/h = 1.5 < 2) → not '-'
	if r := AspectClassify(15, 10, 40, oneRatio, minusRatio); r != 0 {
		t.Errorf("short but not wide: got %q, want 0", r)
	}

	// Wide but not short enough (h >= medianH/2) → not '-'
	if r := AspectClassify(40, 21, 40, oneRatio, minusRatio); r != 0 {
		t.Errorf("wide but too tall for minus: got %q, want 0", r)
	}
}

func TestAsciiArtSegments(t *testing.T) {
	// '8' has all segments on — the art should have '#' everywhere
	art8 := AsciiArtSegments(SegA | SegB | SegC | SegD | SegE | SegF | SegG)
	if !strings.Contains(art8, "#") {
		t.Errorf("AsciiArtSegments('8'): no '#' found in output:\n%s", art8)
	}
	lines := strings.Split(strings.TrimRight(art8, "\n"), "\n")
	if len(lines) != 5 {
		t.Errorf("AsciiArtSegments: expected 5 lines, got %d:\n%s", len(lines), art8)
	}

	// '-' has only SegG (middle) → second line blank, third has '#', fifth blank
	artMinus := AsciiArtSegments(SegG)
	linesM := strings.Split(strings.TrimRight(artMinus, "\n"), "\n")
	if len(linesM) != 5 {
		t.Errorf("AsciiArtSegments('-'): expected 5 lines, got %d", len(linesM))
	}
	// Top (line 0) should have no '#' (SegA is off)
	if strings.Contains(linesM[0], "#") {
		t.Errorf("AsciiArtSegments('-'): top line should be blank, got %q", linesM[0])
	}
	// Middle (line 2) should have '#' (SegG is on)
	if !strings.Contains(linesM[2], "#") {
		t.Errorf("AsciiArtSegments('-'): middle line should have '#', got %q", linesM[2])
	}

	// All off (mask=0) → no '#' at all
	artOff := AsciiArtSegments(0)
	if strings.Contains(artOff, "#") {
		t.Errorf("AsciiArtSegments(0): expected no '#', got:\n%s", artOff)
	}
}

func TestSegmentConstants(t *testing.T) {
	// Verify the segment bit constants are distinct single bits
	segs := []byte{SegA, SegB, SegC, SegD, SegE, SegF, SegG}
	for i, si := range segs {
		if si == 0 || (si&(si-1)) != 0 {
			t.Errorf("Seg%c (index %d) = %d is not a single power of 2", 'A'+i, i, si)
		}
		for j, sj := range segs {
			if i != j && si&sj != 0 {
				t.Errorf("Seg%c and Seg%c overlap", 'A'+i, 'A'+j)
			}
		}
	}
}
