package decode

// AspectClassify returns '1' or '-' if the digit box aspect ratio matches,
// or 0 meaning "use normal segment sampling instead."
//
// oneRatio:   height/width threshold — if h/w > oneRatio, it's a '1'
// minusRatio: width/height threshold — if w/h > minusRatio and the box is
//             in the lower half height-wise, it's a '-'
// AspectClassify returns '1' or '-' if the digit box aspect ratio is diagnostic,
// or 0 meaning "use segment sampling instead."
//
// oneRatio:   height/width threshold — if h/w > oneRatio, it's a '1' (narrow sliver).
//             Default 4.0.  A typical digit is h/w ≈ 2; a '1' is h/w ≈ 5+.
// minusRatio: width/height below which height is small and box is short → '-'.
func AspectClassify(w, h int, medianH int, oneRatio, minusRatio float64) rune {
	if w == 0 || h == 0 {
		return 0
	}
	hw := float64(h) / float64(w)
	wh := float64(w) / float64(h)

	if hw > oneRatio { // very tall and thin → digit '1'
		return '1'
	}
	if wh > 1.0/minusRatio && h < medianH/2 {
		return '-'
	}
	return 0
}

// AsciiArtSegments returns a printable ASCII representation of a 7-bit segment mask.
func AsciiArtSegments(mask byte) string {
	on := func(seg byte) string {
		if mask&seg != 0 {
			return "#"
		}
		return " "
	}
	top := on(SegA)
	topR := on(SegB)
	topL := on(SegF)
	mid := on(SegG)
	botR := on(SegC)
	botL := on(SegE)
	bot := on(SegD)

	return " " + top + top + top + "\n" +
		topL + "   " + topR + "\n" +
		" " + mid + mid + mid + "\n" +
		botL + "   " + botR + "\n" +
		" " + bot + bot + bot + "\n"
}
