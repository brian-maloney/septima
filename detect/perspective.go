package detect

import (
	"image"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

// RectifyPerspective attempts to correct perspective distortion in the ROI.
// Returns a clone of the region when no quadrilateral is detected.
func RectifyPerspective(src gocv.Mat, roi image.Rectangle) gocv.Mat {
	out, _ := RectifyPerspectiveDetailed(src, roi)
	return out
}

// RectifyPerspectiveDetailed runs the rectification and also reports the
// maximum tilt of any detected quad edge from its nominal axis, in degrees.
// `applied` is true only when a quadrilateral was found and warping actually
// ran; callers can use the tilt magnitude to gate whether to use the warped
// output (small tilts are usually within the noise of Hough line fitting and
// the warped output then differs from the input only by a sub-pixel shift).
//
// When `applied` is false, the returned Mat is a plain clone of the region
// and the tilt is zero.
func RectifyPerspectiveDetailed(src gocv.Mat, roi image.Rectangle) (gocv.Mat, RectifyInfo) {
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
		return region.Clone(), RectifyInfo{}
	}
	info := RectifyInfo{Applied: true, MaxTiltDeg: quadMaxTiltDeg(quad)}
	return warpToRect(region, quad), info
}

// RectifyInfo reports how RectifyPerspective interpreted the ROI.
type RectifyInfo struct {
	// Applied is true when a quadrilateral was detected and warping ran.
	Applied bool
	// MaxTiltDeg is the largest absolute angular deviation of any quad edge
	// from its nominal axis (top/bottom from horizontal, left/right from
	// vertical), in degrees. Zero when no quad was detected.
	MaxTiltDeg float64
}

// quadMaxTiltDeg returns the maximum absolute tilt (in degrees) of the quad's
// four edges from their nominal axes. Top/bottom edges are compared against
// horizontal, left/right against vertical. Always returns an acute angle in
// [0, 90].
func quadMaxTiltDeg(quad []point2f) float64 {
	if len(quad) != 4 {
		return 0
	}
	tl, tr, br, bl := quad[0], quad[1], quad[2], quad[3]
	hTop := edgeTiltFromHorizontal(tl, tr)
	hBot := edgeTiltFromHorizontal(bl, br)
	vLeft := edgeTiltFromVertical(tl, bl)
	vRight := edgeTiltFromVertical(tr, br)
	max := hTop
	for _, t := range []float64{hBot, vLeft, vRight} {
		if t > max {
			max = t
		}
	}
	return max
}

// edgeTiltFromHorizontal returns the acute angle (degrees) between the line
// a→b and the horizontal axis.
func edgeTiltFromHorizontal(a, b point2f) float64 {
	dx := b.x - a.x
	dy := b.y - a.y
	rad := math.Atan2(math.Abs(dy), math.Abs(dx))
	return rad * 180 / math.Pi
}

// edgeTiltFromVertical returns the acute angle (degrees) between the line
// a→b and the vertical axis.
func edgeTiltFromVertical(a, b point2f) float64 {
	dx := b.x - a.x
	dy := b.y - a.y
	rad := math.Atan2(math.Abs(dx), math.Abs(dy))
	return rad * 180 / math.Pi
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
