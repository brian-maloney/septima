// Package decode turns binary digit images into character values.
package decode

// Segment bit positions (bit 0 = segment a, ..., bit 6 = segment g).
// Standard 7-segment labeling:
//
//	 _
//	|_|   a=top, b=top-right, c=bottom-right, d=bottom, e=bottom-left, f=top-left, g=middle
//	|_|
//
// Bit: a=0, b=1, c=2, d=3, e=4, f=5, g=6
const (
	SegA byte = 1 << 0 // top
	SegB byte = 1 << 1 // top-right
	SegC byte = 1 << 2 // bottom-right
	SegD byte = 1 << 3 // bottom
	SegE byte = 1 << 4 // bottom-left
	SegF byte = 1 << 5 // top-left
	SegG byte = 1 << 6 // middle
)

// segEntry is one entry in the segment lookup table.
type segEntry struct {
	mask byte
	r    rune
}

// segList is an ordered (deterministic) lookup list.
// Ordering: digits first (0-9), then hex letters, then special chars.
// When two entries have the same Hamming distance from a query mask, the
// first one in the list wins — so more "common" characters take priority.
var segList = []segEntry{
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
	// Hex letters
	{SegA | SegB | SegC | SegE | SegF | SegG, 'A'},
	{SegC | SegD | SegE | SegF | SegG, 'b'},
	{SegA | SegD | SegE | SegF, 'C'},
	{SegB | SegC | SegD | SegE | SegG, 'd'},
	{SegA | SegD | SegE | SegF | SegG, 'E'},
	{SegA | SegE | SegF | SegG, 'F'},
	// Special
	{SegG, '-'},
}

// segTable is kept for O(1) exact-match lookup.
var segTable = func() map[byte]rune {
	m := make(map[byte]rune, len(segList))
	for _, e := range segList {
		m[e.mask] = e.r
	}
	return m
}()

// nearestMatch finds the table entry whose segment mask has the fewest
// differing bits from the given mask.
//
// The search is done over segList (ordered slice) so results are fully
// deterministic — no Go map iteration randomness.
func nearestMatch(mask byte) (rune, float64) {
	bestRune := rune('?')
	bestDist := 8 // max possible Hamming distance for 7 bits

	for _, e := range segList {
		d := hammingByte(mask, e.mask)
		if d < bestDist {
			bestDist = d
			bestRune = e.r
		}
	}

	conf := float64(7-bestDist) / 7.0
	return bestRune, conf
}

// exactMatch returns the rune for an exact segment mask or ('?', 0).
func exactMatch(mask byte) (rune, float64) {
	if r, ok := segTable[mask]; ok {
		return r, 1.0
	}
	return '?', 0
}

func hammingByte(a, b byte) int {
	x := a ^ b
	count := 0
	for x != 0 {
		count += int(x & 1)
		x >>= 1
	}
	return count
}
