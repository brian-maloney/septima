// Package assemble turns a set of glyph detections into an ordered reading:
// detections are clustered into rows by vertical position, sorted left-to-right
// within each row, and mapped through the class table into characters.
package assemble

import (
	"image"
	"sort"
	"strings"

	"github.com/brian-maloney/septima/internal/detect"
)

// Row is one assembled line of glyphs.
type Row struct {
	Text       string
	Chars      []Char
	Box        image.Rectangle
	Confidence float64
}

// Char is a single decoded glyph with its source box and score.
type Char struct {
	R          rune
	Box        image.Rectangle
	Confidence float64
}

// Reading is the full assembled output across all rows (top to bottom).
type Reading struct {
	Rows       []Row
	Text       string
	Confidence float64 // minimum row confidence
}

// Assemble groups detections into rows and decodes them. classNames maps a
// detection class index to its character string (each entry is a single rune
// such as "0", ".", ":", "-").
func Assemble(dets []detect.Detection, classNames []string) Reading {
	if len(dets) == 0 {
		return Reading{}
	}

	// Cluster into rows by vertical overlap. Sort by box center-y, then greedily
	// start a new row whenever a box's center falls outside the current row band.
	sorted := append([]detect.Detection(nil), dets...)
	sort.Slice(sorted, func(i, j int) bool { return centerY(sorted[i].Box) < centerY(sorted[j].Box) })

	medianH := medianHeight(sorted)
	tol := medianH * 0.6

	var rows [][]detect.Detection
	var cur []detect.Detection
	var bandCenter float64
	for _, d := range sorted {
		cy := centerY(d.Box)
		if len(cur) == 0 {
			cur = append(cur, d)
			bandCenter = cy
			continue
		}
		if cy-bandCenter > tol {
			rows = append(rows, cur)
			cur = []detect.Detection{d}
			bandCenter = cy
			continue
		}
		cur = append(cur, d)
		// Running mean keeps the band centered as the row fills in.
		bandCenter = (bandCenter*float64(len(cur)-1) + cy) / float64(len(cur))
	}
	if len(cur) > 0 {
		rows = append(rows, cur)
	}

	reading := Reading{Confidence: 1}
	var allText []string
	for _, group := range rows {
		sort.Slice(group, func(i, j int) bool { return centerX(group[i].Box) < centerX(group[j].Box) })
		var sb strings.Builder
		var chars []Char
		box := group[0].Box
		var confSum float64
		for _, d := range group {
			r := classRune(d.Class, classNames)
			if r == 0 {
				continue
			}
			sb.WriteRune(r)
			chars = append(chars, Char{R: r, Box: d.Box, Confidence: d.Score})
			box = box.Union(d.Box)
			confSum += d.Score
		}
		if len(chars) == 0 {
			continue
		}
		rowConf := confSum / float64(len(chars))
		row := Row{Text: sb.String(), Chars: chars, Box: box, Confidence: rowConf}
		reading.Rows = append(reading.Rows, row)
		allText = append(allText, row.Text)
		if rowConf < reading.Confidence {
			reading.Confidence = rowConf
		}
	}
	if len(reading.Rows) == 0 {
		return Reading{}
	}
	reading.Text = strings.Join(allText, "\n")
	return reading
}

func classRune(class int, names []string) rune {
	if class < 0 || class >= len(names) {
		return 0
	}
	for _, r := range names[class] {
		return r
	}
	return 0
}

func centerX(r image.Rectangle) float64 { return float64(r.Min.X+r.Max.X) / 2 }
func centerY(r image.Rectangle) float64 { return float64(r.Min.Y+r.Max.Y) / 2 }

func medianHeight(dets []detect.Detection) float64 {
	hs := make([]int, 0, len(dets))
	for _, d := range dets {
		hs = append(hs, d.Box.Dy())
	}
	sort.Ints(hs)
	if len(hs) == 0 {
		return 1
	}
	m := float64(hs[len(hs)/2])
	if m == 0 {
		return 1
	}
	return m
}
