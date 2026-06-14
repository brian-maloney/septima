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
		comps = append(comps, classifiedComp{
			rect: image.Rect(x, y+rowOffset, x+w, y+rowOffset+h),
			area: area,
		})
	}

	// Compute median height of "large" components (digit halves or full digits).
	rowH := rowBinary.Rows()
	medH := medianCompHeight(comps, rowH)

	// Classify: decimal candidates vs digit components.
	var digitComps, decimalComps []classifiedComp
	for _, c := range comps {
		h := c.rect.Dy()
		w := c.rect.Dx()
		heightRatio := float64(h) / float64(medH)
		aspectRatio := float64(w) / float64(h)
		if heightRatio < opts.DecHRatio && aspectRatio < opts.DecWRatio*3 {
			c.isDecimal = true
			decimalComps = append(decimalComps, c)
		} else {
			digitComps = append(digitComps, c)
		}
	}

	// Merge X-overlapping digit components (split digit halves).
	mergedDigits := mergeXOverlapping(digitComps)

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
		overlap := cur.rect.Max.X - next.rect.Min.X
		minW := min_int(cur.rect.Dx(), next.rect.Dx())
		// Only merge when overlap is a significant fraction of the narrower width.
		// Adjacent different digits have 0 or tiny negative overlap → don't merge.
		if minW > 0 && float64(overlap)/float64(minW) > 0.40 {
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
		// If the sample area has enough white pixels, treat this decimal as a colon.
		sampleArea := 5 * (x1 - x0 + 1)
		if sampleArea > 0 && float64(whiteCount)/float64(sampleArea) > 0.20 {
			boxes[i].IsColon = true
			boxes[i].IsDecimal = false
		}
	}
	return boxes
}

func abs_int(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
