package detect

import (
	"image"
	"sort"

	"gocv.io/x/gocv"
)

// RowRegion describes a horizontal band of digits.
type RowRegion struct {
	Bounds image.Rectangle
	Mask   gocv.Mat // binary sub-image (caller must Close)
}

// SplitRows divides a binary image into one or more horizontal bands, each
// containing a row of digits.  expectedRows == 0 means auto-detect.
func SplitRows(binary gocv.Mat, expectedRows int) []RowRegion {
	proj := horizontalProjection(binary)
	bands := findBands(proj, binary.Rows(), binary.Cols())
	bands = mergeBands(bands, proj)

	// Filter bands by average pixel density per row: keep only "rich" bands
	// whose average row density indicates real digit content (not sparse labels
	// or noise from reflections).
	bands = filterByDensity(bands, proj, binary.Cols())

	// If we have an expectation, enforce it.
	if expectedRows > 0 && len(bands) != expectedRows {
		bands = forceBandCount(bands, proj, expectedRows, binary.Rows())
	}

	var rows []RowRegion
	for _, b := range bands {
		rect := image.Rect(0, b.start, binary.Cols(), b.end)
		sub := binary.Region(rect)
		rows = append(rows, RowRegion{
			Bounds: rect,
			Mask:   sub.Clone(),
		})
		sub.Close()
	}
	return rows
}

type band struct{ start, end int }

func horizontalProjection(binary gocv.Mat) []int {
	proj := make([]int, binary.Rows())
	for r := 0; r < binary.Rows(); r++ {
		for c := 0; c < binary.Cols(); c++ {
			if binary.GetUCharAt(r, c) > 0 {
				proj[r]++
			}
		}
	}
	return proj
}

func findBands(proj []int, totalRows, cols int) []band {
	// Threshold: a row is "active" if it has at least 1% of columns filled.
	// This is intentionally generous — density filtering happens later.
	threshold := max_int(1, cols/100)
	var bands []band
	inBand := false
	start := 0
	for r, v := range proj {
		if v >= threshold && !inBand {
			inBand = true
			start = r
		} else if v < threshold && inBand {
			inBand = false
			if r-start > 3 {
				bands = append(bands, band{start: start, end: r})
			}
		}
	}
	if inBand {
		bands = append(bands, band{start: start, end: totalRows})
	}
	return bands
}

func mergeBands(bands []band, proj []int) []band {
	if len(bands) < 2 {
		return bands
	}
	var totalH int
	for _, b := range bands {
		totalH += b.end - b.start
	}
	avgH := totalH / len(bands)
	minGap := avgH / 5
	if minGap < 5 {
		minGap = 5
	}
	var merged []band
	cur := bands[0]
	for i := 1; i < len(bands); i++ {
		gap := bands[i].start - cur.end
		if gap <= minGap {
			cur.end = bands[i].end
		} else {
			merged = append(merged, cur)
			cur = bands[i]
		}
	}
	merged = append(merged, cur)
	return merged
}

// filterByDensity removes bands whose average per-row pixel count is less than
// 15% of the busiest band's average.  This eliminates label-text bands and
// reflection-noise bands while keeping real digit bands (which are pixel-rich).
func filterByDensity(bands []band, proj []int, cols int) []band {
	if len(bands) <= 1 {
		return bands
	}
	// Compute average pixel density per row for each band.
	densities := make([]float64, len(bands))
	maxDensity := 0.0
	for i, b := range bands {
		h := b.end - b.start
		if h <= 0 {
			continue
		}
		sum := 0
		for r := b.start; r < b.end; r++ {
			sum += proj[r]
		}
		d := float64(sum) / float64(h*cols)
		densities[i] = d
		if d > maxDensity {
			maxDensity = d
		}
	}
	// Keep bands with density ≥ 30% of the maximum.
	// This removes sparse label-text bands and noise bands while keeping
	// all rows of actual 7-segment digits (which are pixel-rich).
	var result []band
	for i, b := range bands {
		if densities[i] >= maxDensity*0.30 {
			result = append(result, b)
		}
	}
	if len(result) == 0 {
		return bands // safety fallback
	}
	return result
}

func forceBandCount(bands []band, proj []int, want, totalRows int) []band {
	if len(bands) == 0 {
		h := totalRows / want
		var result []band
		for i := 0; i < want; i++ {
			result = append(result, band{start: i * h, end: (i + 1) * h})
		}
		return result
	}
	if len(bands) > want {
		// Keep the densest bands (by total pixel count).
		type scored struct {
			b     band
			score int
		}
		var s []scored
		for _, b := range bands {
			total := 0
			for r := b.start; r < b.end; r++ {
				total += proj[r]
			}
			s = append(s, scored{b, total})
		}
		sort.Slice(s, func(i, j int) bool { return s[i].score > s[j].score })
		var result []band
		for i := 0; i < want && i < len(s); i++ {
			result = append(result, s[i].b)
		}
		sort.Slice(result, func(i, j int) bool { return result[i].start < result[j].start })
		return result
	}
	return bands
}
