package detect

import (
	"image"
	"sort"

	"gocv.io/x/gocv"
)

// Connected components stat column indices (OpenCV CC_STAT_*)
const (
	ccLeft   = 0
	ccTop    = 1
	ccWidth  = 2
	ccHeight = 3
	ccArea   = 4
)

// DigitBox describes a single character region within a row.
type DigitBox struct {
	Bounds    image.Rectangle
	IsDecimal bool
	IsColon   bool
}

// DigitOptions holds thresholds for digit segmentation.
type DigitOptions struct {
	MinArea   int
	MinWidth  int
	MinHeight int
	DecHRatio float64
	DecWRatio float64
	IgnorePix int // unused; kept for API compatibility
}

// DefaultDigitOptions returns sensible defaults.
func DefaultDigitOptions() DigitOptions {
	return DigitOptions{
		MinArea:   30,
		MinWidth:  3,
		MinHeight: 5,
		DecHRatio: 0.45,
		DecWRatio: 0.45,
		IgnorePix: 2,
	}
}

// SegmentDigits locates character bounding boxes within a binary row image.
//
// Algorithm (two-pass CC + vertical merge):
//
//  1. Find all white connected components in the row.
//  2. Classify each as "digit" (tall) or "decimal" (short, roughly square).
//  3. Merge X-overlapping digit components — these are top/bottom halves of
//     the same 7-segment character separated by the inter-segment gap.
//  4. Sort everything by horizontal centre; detect colons.
func SegmentDigits(rowBinary gocv.Mat, rowOffset int, opts DigitOptions) []DigitBox {
	labels := gocv.NewMat()
	stats := gocv.NewMat()
	centroids := gocv.NewMat()
	defer labels.Close()
	defer stats.Close()
	defer centroids.Close()

	n := gocv.ConnectedComponentsWithStats(rowBinary, &labels, &stats, &centroids)

	// Gather component info.
	var comps []classifiedComp
	for i := 1; i < n; i++ { // skip label 0 (background)
		x := int(stats.GetIntAt(i, ccLeft))
		y := int(stats.GetIntAt(i, ccTop))
		w := int(stats.GetIntAt(i, ccWidth))
		h := int(stats.GetIntAt(i, ccHeight))
		area := int(stats.GetIntAt(i, ccArea))
		if area < opts.MinArea || w < opts.MinWidth || h < opts.MinHeight {
			continue
		}
		// Drop low, wide horizontal "bar" CCs: bezel-edge specks that
		// land just inside the inner margin and have aspect 2x or more
		// at a height well below any digit or decimal dot.  They survive
		// the area filter (a 27×4 sliver still has area=84) yet provoke
		// false "wide-CC" splits and interleave between digit halves
		// during X-overlap merging.
		if h*12 < rowBinary.Rows() && w >= h*2 {
			continue
		}
		comps = append(comps, classifiedComp{
			rect: image.Rect(x, y+rowOffset, x+w, y+rowOffset+h),
			area: area,
		})
	}

	rowH := rowBinary.Rows()

	// Some CCs span multiple digits because polarity inversion brought in
	// bezel noise that bridges adjacent digits into a single connected
	// component.  Detect via width > 1.5x median digit width AND being a tall
	// component (not a tiny decimal), then split at column-projection valleys.
	// Narrow column-peaks (less than ~medW/3 wide) and peaks that do not
	// extend close to full row height are discarded as bezel artefacts.
	comps = splitWideMergedDigits(comps, rowBinary, rowOffset)

	// Compute median height of "large" components (digit halves or full digits).
	medH := medianCompHeight(comps, rowH)

	// Classify: decimal candidates vs digit components.
	var digitComps, decimalComps []classifiedComp
	for _, c := range comps {
		h := c.rect.Dy()
		w := c.rect.Dx()
		heightRatio := float64(h) / float64(medH)
		aspectRatio := float64(w) / float64(h)
		if isDecimalShape(heightRatio, aspectRatio, opts) {
			c.isDecimal = true
			decimalComps = append(decimalComps, c)
		} else {
			digitComps = append(digitComps, c)
		}
	}

	// Merge X-overlapping digit components (split digit halves).
	mergedDigits := mergeXOverlapping(digitComps)

	// Second-pass reclassification: now that we know real digit heights,
	// a component initially classified as digit may actually be a decimal.
	// The pre-merge medH can underestimate digit height when digits are split
	// into two halves (e.g. "3" → top+bottom CCs each ~half the digit height).
	postMergeMaxH := 0
	for _, d := range mergedDigits {
		if h := d.rect.Dy(); h > postMergeMaxH {
			postMergeMaxH = h
		}
	}
	// Only reclassify when the post-merge height is noticeably larger than
	// the pre-merge median.  A 30% jump indicates that real digits were
	// split into two CCs each (typical for clean LCD digits at the inter-
	// segment gap), and the pre-merge medH was therefore too small to use as
	// the decimal threshold.  Without this guard, second-pass reclassification
	// turns small label fragments and reflection specks into spurious decimals.
	if postMergeMaxH > medH*13/10 {
		reclassified := mergedDigits[:0]
		for _, c := range mergedDigits {
			h := c.rect.Dy()
			w := c.rect.Dx()
			heightRatio := float64(h) / float64(postMergeMaxH)
			aspectRatio := float64(w) / float64(h)
			if isDecimalShape(heightRatio, aspectRatio, opts) {
				c.isDecimal = true
				decimalComps = append(decimalComps, c)
			} else {
				reclassified = append(reclassified, c)
			}
		}
		mergedDigits = reclassified
	}

	// Combine merged digits with decimal candidates.
	all := append(mergedDigits, decimalComps...)

	// Sort by horizontal centre.
	sort.Slice(all, func(i, j int) bool {
		ci := (all[i].rect.Min.X + all[i].rect.Max.X) / 2
		cj := (all[j].rect.Min.X + all[j].rect.Max.X) / 2
		return ci < cj
	})

	// Recompute median height from merged digits for colon detection.
	medH2 := 0
	for _, d := range mergedDigits {
		h := d.rect.Dy()
		if h > medH2 {
			medH2 = h
		}
	}
	if medH2 == 0 {
		medH2 = medH
	}

	// Build result.
	var boxes []DigitBox
	for _, c := range all {
		boxes = append(boxes, DigitBox{
			Bounds:    c.rect,
			IsDecimal: c.isDecimal,
		})
	}

	boxes = detectColons(boxes, medH2)
	// Second-chance: upgrade isolated decimal dots to colons if the partner
	// dot is present in the binary (even if merged into an adjacent digit CC).
	boxes = detectColonsSingleDot(boxes, rowBinary, rowOffset, medH2)
	return boxes
}

type classifiedComp struct {
	rect      image.Rectangle
	area      int
	isDecimal bool
}

// mergeXOverlapping merges components that substantially overlap in X,
// treating them as top and bottom halves of the same 7-segment digit.
//
// "Same digit halves" criterion: the two components overlap in X by at least
// 40% of the narrower component's width.  This correctly handles the 3-pixel
// inter-segment gap that splits digits like "0" into two pieces, while NOT
// merging adjacent different digits that merely share a narrow gap (≤3px).
func mergeXOverlapping(comps []classifiedComp) []classifiedComp {
	if len(comps) == 0 {
		return nil
	}
	sort.Slice(comps, func(i, j int) bool {
		return comps[i].rect.Min.X < comps[j].rect.Min.X
	})

	merged := []classifiedComp{comps[0]}
	for i := 1; i < len(comps); i++ {
		cur := &merged[len(merged)-1]
		next := comps[i]
		if shouldMerge(*cur, next) {
			cur.rect = image.Rect(
				min_int(cur.rect.Min.X, next.rect.Min.X),
				min_int(cur.rect.Min.Y, next.rect.Min.Y),
				max_int(cur.rect.Max.X, next.rect.Max.X),
				max_int(cur.rect.Max.Y, next.rect.Max.Y),
			)
			cur.area += next.area
		} else {
			merged = append(merged, next)
		}
	}
	return merged
}

// shouldMerge decides whether two digit-components are halves of the same
// 7-segment character.
//
// Two criteria, either of which is sufficient:
//
//  1. Substantial X-overlap (≥40% of the narrower width).  If the two
//     components have no Y-overlap (one is stacked above the other), also
//     require that the smaller component's pixel area is at least 25% of
//     the larger's — this rejects merging a small label-text fragment into
//     a tall digit body when both happen to be x-aligned but the label
//     sits clearly above the digit.  Stacked CCs with similar areas are
//     the typical top/bottom halves of a digit split by the inter-segment
//     gap, so they pass.
//
//  2. Narrow stub fully Y-contained within a wider neighbour AND the X edges
//     touch or overlap.  Handles a single b- or e-segment that didn't connect
//     to the main "L" shape of the digit (e.g., "2" with isolated top-right
//     vertical bar that is narrower than the rest of the digit).
func shouldMerge(cur, next classifiedComp) bool {
	overlap := cur.rect.Max.X - next.rect.Min.X
	minW := min_int(cur.rect.Dx(), next.rect.Dx())
	if minW > 0 && float64(overlap)/float64(minW) > 0.40 {
		yOverlap := min_int(cur.rect.Max.Y, next.rect.Max.Y) -
			max_int(cur.rect.Min.Y, next.rect.Min.Y)
		if yOverlap > 0 {
			return true
		}
		minA := min_int(cur.area, next.area)
		maxA := max_int(cur.area, next.area)
		if maxA > 0 && float64(minA)/float64(maxA) >= 0.25 {
			return true
		}
		return false
	}
	// Narrow-stub merge: require X overlap or near-touching, narrow next,
	// and Y span of next mostly contained in cur.
	if overlap < -2 { // > 2-pixel horizontal gap → not the same digit
		return false
	}
	if next.rect.Dx()*3 > cur.rect.Dx() { // not narrow enough
		return false
	}
	yOverlap := min_int(cur.rect.Max.Y, next.rect.Max.Y) -
		max_int(cur.rect.Min.Y, next.rect.Min.Y)
	minH := min_int(cur.rect.Dy(), next.rect.Dy())
	if minH <= 0 {
		return false
	}
	return float64(yOverlap)/float64(minH) > 0.80
}

func medianCompHeight(comps []classifiedComp, rowH int) int {
	var heights []int
	for i := range comps {
		h := comps[i].rect.Dy()
		if h > rowH/6 {
			heights = append(heights, h)
		}
	}
	if len(heights) == 0 {
		return rowH
	}
	sort.Ints(heights)
	return heights[len(heights)/2]
}

// detectColons upgrades decimal-classified components to colons in two ways:
//
//  1. Two-dot: find a pair of decimal components that are vertically stacked
//     at the right distance, merge them into a colon.
//
//  2. Single-dot: if only one decimal exists at a given x, sample the row
//     binary to check for white pixels at the expected lower-dot position.
//     This handles displays where the lower colon dot merges into an adjacent
//     digit's CC component.
func detectColons(boxes []DigitBox, medianH int) []DigitBox {
	// Pass 1: pair two stacked decimal dots.
	for i := range boxes {
		if !boxes[i].IsDecimal {
			continue
		}
		for j := range boxes {
			if j == i || !boxes[j].IsDecimal {
				continue
			}
			iCX := (boxes[i].Bounds.Min.X + boxes[i].Bounds.Max.X) / 2
			jCX := (boxes[j].Bounds.Min.X + boxes[j].Bounds.Max.X) / 2
			if abs_int(iCX-jCX) > medianH/4 {
				continue
			}
			iCY := (boxes[i].Bounds.Min.Y + boxes[i].Bounds.Max.Y) / 2
			jCY := (boxes[j].Bounds.Min.Y + boxes[j].Bounds.Max.Y) / 2
			dy := abs_int(iCY - jCY)
			minDy := medianH / 8
			if minDy < 3 {
				minDy = 3
			}
			if dy > minDy && dy < medianH {
				minX := min_int(boxes[i].Bounds.Min.X, boxes[j].Bounds.Min.X)
				maxX := max_int(boxes[i].Bounds.Max.X, boxes[j].Bounds.Max.X)
				minY := min_int(boxes[i].Bounds.Min.Y, boxes[j].Bounds.Min.Y)
				maxY := max_int(boxes[i].Bounds.Max.Y, boxes[j].Bounds.Max.Y)
				boxes[i].Bounds = image.Rect(minX, minY, maxX, maxY)
				boxes[i].IsColon = true
				boxes[i].IsDecimal = false
				boxes[j].Bounds = image.Rectangle{}
			}
		}
	}
	var result []DigitBox
	for _, b := range boxes {
		if !b.Bounds.Empty() {
			result = append(result, b)
		}
	}
	return result
}

// detectColonsSingleDot upgrades an isolated decimal dot to a colon if the
// row binary contains white pixels at the expected second-dot position.
// This handles the case where the lower colon dot merged into an adjacent
// digit's CC component (common in VFD clock displays).
func detectColonsSingleDot(boxes []DigitBox, rowBinary gocv.Mat, rowOffset, medianH int) []DigitBox {
	if medianH <= 0 {
		return boxes
	}
	expectedDy := medianH / 3

	for i := range boxes {
		if !boxes[i].IsDecimal {
			continue
		}
		cx := (boxes[i].Bounds.Min.X + boxes[i].Bounds.Max.X) / 2
		cy := (boxes[i].Bounds.Min.Y + boxes[i].Bounds.Max.Y) / 2

		// Look for the partner dot below this one.
		cyLower := cy + expectedDy - rowOffset // convert to row-relative coords
		w := boxes[i].Bounds.Dx()
		if w < 2 {
			w = 2
		}
		x0 := cx - w
		if x0 < 0 {
			x0 = 0
		}
		x1 := cx + w
		if x1 >= rowBinary.Cols() {
			x1 = rowBinary.Cols() - 1
		}

		// Sample a small band at the expected lower-dot y position.
		whiteCount := 0
		for dy := -2; dy <= 2; dy++ {
			r := cyLower + dy
			if r < 0 || r >= rowBinary.Rows() {
				continue
			}
			for c := x0; c <= x1; c++ {
				if rowBinary.GetUCharAt(r, c) > 0 {
					whiteCount++
				}
			}
		}
		sampleArea := 5 * (x1 - x0 + 1)
		if sampleArea == 0 || float64(whiteCount)/float64(sampleArea) <= 0.20 {
			continue
		}

		// Stroke-rejection check: sample narrow bands ABOVE and BELOW the
		// partner position at a tight x window centred on cx.  For a real
		// isolated colon dot, the regions ~medianH/6 above and below the
		// dot are empty (inter-dot gap and the space below the lower dot).
		// For a continuous vertical stroke (e.g., an upper-bezel speck whose
		// predicted partner lands inside the next digit's vertical stroke),
		// both regions remain densely white — that pattern is what we reject.
		dyOff := medianH / 6
		if dyOff < 6 {
			dyOff = 6
		}
		nx0 := cx - 1
		if nx0 < 0 {
			nx0 = 0
		}
		nx1 := cx + 1
		if nx1 >= rowBinary.Cols() {
			nx1 = rowBinary.Cols() - 1
		}
		strokeAbove, strokeBelow := 0, 0
		for dy := -1; dy <= 1; dy++ {
			rA := cyLower - dyOff + dy
			rB := cyLower + dyOff + dy
			if rA >= 0 && rA < rowBinary.Rows() {
				for c := nx0; c <= nx1; c++ {
					if rowBinary.GetUCharAt(rA, c) > 0 {
						strokeAbove++
					}
				}
			}
			if rB >= 0 && rB < rowBinary.Rows() {
				for c := nx0; c <= nx1; c++ {
					if rowBinary.GetUCharAt(rB, c) > 0 {
						strokeBelow++
					}
				}
			}
		}
		strokeArea := 3 * (nx1 - nx0 + 1)
		if strokeArea > 0 {
			rAbove := float64(strokeAbove) / float64(strokeArea)
			rBelow := float64(strokeBelow) / float64(strokeArea)
			// A real isolated colon dot has empty rows BOTH above (the gap
			// between dots) and below (the space below the lower dot).  If
			// either band remains densely white, the partner sample is part
			// of a continuous stroke (digit segment) — reject the upgrade.
			if rAbove > 0.50 || rBelow > 0.50 {
				continue
			}
		}

		boxes[i].IsColon = true
		boxes[i].IsDecimal = false
	}
	return boxes
}

// splitWideMergedDigits looks for CCs whose width substantially exceeds the
// median digit width (bezel noise sometimes bridges adjacent digits into one
// CC), and splits them along column-projection valleys.
//
// Algorithm: for each suspect CC, compute the per-column white-pixel count
// within its bbox; identify contiguous runs of columns whose count is
// ≥30% of the CC's per-column maximum AND wider than medW/3 AND whose
// in-run peak height is ≥80% of the CC's row height.  Output one sub-CC
// per run, with the CC's full y-range.  Narrow column-peaks and peaks that
// don't reach near-full height are discarded — they tend to be bezel
// artefacts bleeding into the CC, not real digits.
func splitWideMergedDigits(comps []classifiedComp, rowBinary gocv.Mat, rowOffset int) []classifiedComp {
	rowH := rowBinary.Rows()
	var widths []int
	for _, c := range comps {
		if c.rect.Dy() > rowH/2 {
			widths = append(widths, c.rect.Dx())
		}
	}
	if len(widths) == 0 {
		return comps
	}
	sort.Ints(widths)
	medW := widths[len(widths)/2]
	if medW <= 0 {
		return comps
	}

	out := make([]classifiedComp, 0, len(comps))
	for _, c := range comps {
		w := c.rect.Dx()
		h := c.rect.Dy()
		if w <= medW*3/2 || h <= rowH/2 {
			out = append(out, c)
			continue
		}
		subs := splitCCByProjection(c, rowBinary, rowOffset, medW)
		if len(subs) >= 2 {
			out = append(out, subs...)
		} else {
			// Couldn't find a confident split; keep original.
			out = append(out, c)
		}
	}
	return out
}

func splitCCByProjection(c classifiedComp, rowBinary gocv.Mat, rowOffset, medW int) []classifiedComp {
	yLo := c.rect.Min.Y - rowOffset
	yHi := c.rect.Max.Y - rowOffset
	if yLo < 0 {
		yLo = 0
	}
	if yHi > rowBinary.Rows() {
		yHi = rowBinary.Rows()
	}
	if yHi <= yLo {
		return nil
	}
	width := c.rect.Dx()
	cols := make([]int, width)
	maxCol := 0
	for cx := 0; cx < width; cx++ {
		count := 0
		for ry := yLo; ry < yHi; ry++ {
			if rowBinary.GetUCharAt(ry, c.rect.Min.X+cx) > 0 {
				count++
			}
		}
		cols[cx] = count
		if count > maxCol {
			maxCol = count
		}
	}
	if maxCol == 0 {
		return nil
	}
	threshold := maxCol * 30 / 100
	minRun := medW / 3
	if minRun < 5 {
		minRun = 5
	}
	rowHeight := yHi - yLo
	// Real digit strokes span close to the full row height; bezel-noise
	// peaks within the wide CC top out well below it.
	heightFloor := rowHeight * 80 / 100

	// Find contiguous "active" runs (count >= threshold) wider than minRun
	// whose in-run peak count reaches heightFloor.
	var subs []classifiedComp
	runStart := -1
	runPeak := 0
	for cx := 0; cx <= width; cx++ {
		active := cx < width && cols[cx] >= threshold
		if active {
			if runStart < 0 {
				runStart = cx
				runPeak = 0
			}
			if cols[cx] > runPeak {
				runPeak = cols[cx]
			}
			continue
		}
		if runStart >= 0 {
			runEnd := cx // exclusive
			if runEnd-runStart >= minRun && runPeak >= heightFloor {
				// Tight bbox around the active run so the bbox aspect
				// reflects the stroke shape, not the wide CC.  A typical
				// '1' stroke split out this way then has aspect h/w
				// above OneRatio and AspectClassify handles it directly.
				pad := 1
				x0 := runStart - pad
				if x0 < 0 {
					x0 = 0
				}
				x1 := runEnd + pad
				if x1 > width {
					x1 = width
				}
				area := 0
				for px := x0; px < x1; px++ {
					area += cols[px]
				}
				// Tight y-range over the sub-rect's columns: parent's
				// y bounds may span the full LCD height (e.g., when the
				// "wide CC" is actually a bezel-outline ring touching all
				// edges), so inheriting Min.Y/Max.Y inflates this sub-rect
				// and pollutes downstream height statistics.
				subY0, subY1 := tightYRange(rowBinary,
					c.rect.Min.X+x0, c.rect.Min.X+x1, yLo, yHi)
				if subY1 <= subY0 {
					subY0 = c.rect.Min.Y - rowOffset
					subY1 = c.rect.Max.Y - rowOffset
				}
				subs = append(subs, classifiedComp{
					rect: image.Rect(
						c.rect.Min.X+x0, subY0+rowOffset,
						c.rect.Min.X+x1, subY1+rowOffset,
					),
					area: area,
				})
			}
			runStart = -1
			runPeak = 0
		}
	}
	return subs
}

// tightYRange finds the longest contiguous y-run in [yLo, yHi) within
// columns [x0, x1) where every row carries a substantial amount of
// content.  "Substantial" means at least ~15% of the column window — that
// excludes thin one- or two-pixel perimeter strokes while preserving real
// digit strokes.  Returns the longest run so that a bezel-outline ring's
// top/bottom strokes (separated from the digit body by an empty band) do
// not get included in the y-range; only the contiguous digit body wins.
// Returns (yLo, yLo) when no run is found.
func tightYRange(bin gocv.Mat, x0, x1, yLo, yHi int) (int, int) {
	cols := bin.Cols()
	if x0 < 0 {
		x0 = 0
	}
	if x1 > cols {
		x1 = cols
	}
	if x1 <= x0 {
		return yLo, yLo
	}
	minCount := (x1 - x0) * 15 / 100
	if minCount < 2 {
		minCount = 2
	}
	bestStart, bestEnd := -1, -1
	curStart := -1
	for r := yLo; r < yHi; r++ {
		count := 0
		for c := x0; c < x1; c++ {
			if bin.GetUCharAt(r, c) > 0 {
				count++
				if count >= minCount {
					break
				}
			}
		}
		if count >= minCount {
			if curStart < 0 {
				curStart = r
			}
			if bestStart < 0 || r-curStart+1 > bestEnd-bestStart+1 {
				bestStart, bestEnd = curStart, r
			}
		} else {
			curStart = -1
		}
	}
	if bestStart < 0 {
		return yLo, yLo
	}
	return bestStart, bestEnd + 1
}

func abs_int(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// isDecimalShape returns true if a component is short enough to be a decimal
// dot or colon dot and not so wide that it must be a minus sign.
//
// heightRatio is h/refH (refH is the digit height reference: median or max).
// aspectRatio is w/h.  The lower bound on max aspect (1.5) ensures that
// square colon dots (aspect ≈ 1.0) are admitted even when a profile chose a
// very small DecWRatio.
func isDecimalShape(heightRatio, aspectRatio float64, opts DigitOptions) bool {
	if heightRatio >= opts.DecHRatio {
		return false
	}
	maxAspect := opts.DecWRatio * 3
	if maxAspect < 1.5 {
		maxAspect = 1.5
	}
	return aspectRatio < maxAspect
}
