package detect

import (
	"image"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

// RectifyPerspective attempts to correct perspective distortion in the ROI.
func RectifyPerspective(src gocv.Mat, roi image.Rectangle) gocv.Mat {
	region := src.Region(roi)
	defer region.Close()

	gray := gocv.NewMat()
	defer gray.Close()
	if region.Channels() > 1 {
		gocv.CvtColor(region, &gray, gocv.ColorBGRToGray)
	} else {
		gray = region.Clone()
	}

	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(gray, &edges, 50, 150)

	lines := gocv.NewMat()
	defer lines.Close()
	gocv.HoughLinesPWithParams(edges, &lines, 1, math.Pi/180, 50, 50, 10)

	quad := detectQuad(lines, region.Cols(), region.Rows())
	if quad == nil {
		return region.Clone()
	}
	return warpToRect(region, quad)
}

type point2f struct{ x, y float64 }

// houghLine holds the four endpoints of a Hough line segment.
type houghLine struct{ x1, y1, x2, y2 int32 }

func getHoughLines(lines gocv.Mat) []houghLine {
	var result []houghLine
	for i := 0; i < lines.Rows(); i++ {
		result = append(result, houghLine{
			x1: lines.GetIntAt(i, 0),
			y1: lines.GetIntAt(i, 1),
			x2: lines.GetIntAt(i, 2),
			y2: lines.GetIntAt(i, 3),
		})
	}
	return result
}

func detectQuad(lines gocv.Mat, w, h int) []point2f {
	if lines.Rows() < 4 {
		return nil
	}
	all := getHoughLines(lines)

	var hLines, vLines []houghLine
	for _, v := range all {
		dx := math.Abs(float64(v.x2 - v.x1))
		dy := math.Abs(float64(v.y2 - v.y1))
		if dx > dy*2 {
			hLines = append(hLines, v)
		} else if dy > dx*2 {
			vLines = append(vLines, v)
		}
	}

	if len(hLines) < 2 || len(vLines) < 2 {
		return nil
	}

	sort.Slice(hLines, func(i, j int) bool {
		return hlAvgY(hLines[i]) < hlAvgY(hLines[j])
	})
	sort.Slice(vLines, func(i, j int) bool {
		return hlAvgX(vLines[i]) < hlAvgX(vLines[j])
	})

	top := hLines[0]
	bot := hLines[len(hLines)-1]
	left := vLines[0]
	right := vLines[len(vLines)-1]

	tl := hlIntersect(hlEq(top), hlEq(left))
	tr := hlIntersect(hlEq(top), hlEq(right))
	br := hlIntersect(hlEq(bot), hlEq(right))
	bl := hlIntersect(hlEq(bot), hlEq(left))

	fw, fh := float64(w), float64(h)
	for _, p := range []point2f{tl, tr, br, bl} {
		if p.x < -fw*0.5 || p.x > fw*1.5 || p.y < -fh*0.5 || p.y > fh*1.5 {
			return nil
		}
	}
	return []point2f{tl, tr, br, bl}
}

type lineEquation struct{ a, b, c float64 }

func hlEq(v houghLine) lineEquation {
	x1, y1 := float64(v.x1), float64(v.y1)
	x2, y2 := float64(v.x2), float64(v.y2)
	a := y2 - y1
	b := x1 - x2
	c := x2*y1 - x1*y2
	return lineEquation{a, b, c}
}

func hlIntersect(l1, l2 lineEquation) point2f {
	det := l1.a*l2.b - l2.a*l1.b
	if math.Abs(det) < 1e-10 {
		return point2f{}
	}
	x := (-l1.c*l2.b + l2.c*l1.b) / det
	y := (-l1.a*l2.c + l2.a*l1.c) / det
	return point2f{x, y}
}

func hlAvgY(v houghLine) float64 { return float64(v.y1+v.y2) / 2.0 }
func hlAvgX(v houghLine) float64 { return float64(v.x1+v.x2) / 2.0 }

func warpToRect(src gocv.Mat, quad []point2f) gocv.Mat {
	w := dist(quad[0], quad[1])
	h := dist(quad[0], quad[3])
	if w < 10 || h < 10 {
		return src.Clone()
	}

	srcPts := gocv.NewPointVectorFromPoints([]image.Point{
		{X: int(quad[0].x), Y: int(quad[0].y)},
		{X: int(quad[1].x), Y: int(quad[1].y)},
		{X: int(quad[2].x), Y: int(quad[2].y)},
		{X: int(quad[3].x), Y: int(quad[3].y)},
	})
	defer srcPts.Close()

	dstPts := gocv.NewPointVectorFromPoints([]image.Point{
		{X: 0, Y: 0},
		{X: int(w), Y: 0},
		{X: int(w), Y: int(h)},
		{X: 0, Y: int(h)},
	})
	defer dstPts.Close()

	M := gocv.GetPerspectiveTransform(srcPts, dstPts)
	defer M.Close()

	dst := gocv.NewMat()
	gocv.WarpPerspective(src, &dst, M, image.Point{X: int(w), Y: int(h)})
	return dst
}

func dist(a, b point2f) float64 {
	dx := a.x - b.x
	dy := a.y - b.y
	return math.Sqrt(dx*dx + dy*dy)
}
