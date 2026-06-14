package decode

// CharsetID identifies which set of characters are valid outputs.
type CharsetID int

const (
	CharsetFull    CharsetID = iota // 0-9, -.:', a-f
	CharsetDigits                   // 0-9 only
	CharsetDecimal                  // 0-9, '.', '-'
	CharsetHex                      // 0-9, a-f, '.', '-'
	CharsetTTRobot                  // 0-9, '-', a b c d h l n p r t v
)

// allowedRunes returns the set of runes that are valid for the given charset.
func allowedRunes(cs CharsetID) map[rune]bool {
	switch cs {
	case CharsetDigits:
		return runeSet("0123456789")
	case CharsetDecimal:
		return runeSet("0123456789.-")
	case CharsetHex:
		return runeSet("0123456789abcdefABCDEF.-")
	case CharsetTTRobot:
		return runeSet("0123456789-abcdhlnprtv")
	default: // CharsetFull
		return nil // nil = allow everything in the segment table
	}
}

func runeSet(s string) map[rune]bool {
	m := make(map[rune]bool, len(s))
	for _, r := range s {
		m[r] = true
	}
	return m
}

// Decode converts a segment mask to a rune, filtered by charset.
// Returns '?' and 0 confidence if no match is found.
func Decode(mask byte, cs CharsetID, isDecimalPoint, isColon bool) (rune, float64) {
	if isDecimalPoint {
		if cs == CharsetDecimal || cs == CharsetFull {
			return '.', 1.0
		}
		return '.', 0.5
	}
	if isColon {
		return ':', 1.0
	}

	allowed := allowedRunes(cs)

	// Try exact match first
	r, conf := exactMatch(mask)
	if r != '?' {
		if allowed == nil || allowed[r] {
			return r, conf
		}
	}

	// Try nearest match, filtered by charset
	r, conf = nearestMatch(mask)
	if allowed != nil && !allowed[r] {
		// Find the nearest match within the allowed set
		bestR := rune('?')
		bestConf := 0.0
		for m2, r2 := range segTable {
			if allowed[r2] {
				d := hammingByte(mask, m2)
				c := float64(7-d) / 7.0
				if c > bestConf {
					bestConf = c
					bestR = r2
				}
			}
		}
		return bestR, bestConf
	}
	return r, conf
}
