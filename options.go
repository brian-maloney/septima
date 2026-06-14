package septima

import (
	"image"

	"github.com/vond/septima/preprocess"
)

// Charset controls which characters are considered during lookup.
type Charset int

const (
	CharsetFull    Charset = iota // 0-9, -.:', and hex a-f
	CharsetDigits                 // 0-9 only
	CharsetDecimal                // 0-9, '.', '-'
	CharsetHex                    // 0-9, a-f, '.', '-'
	CharsetTTRobot                // 0-9, '-', a b c d h l n p r t v
)

// Polarity describes whether the digits are darker or lighter than the background.
type Polarity int

const (
	PolarityAuto       Polarity = iota
	PolarityDarkOnLight         // standard: dark segments on light LCD
	PolarityLightOnDark         // LED/VFD: bright segments on dark background
)

// Options holds all tunable parameters.  The zero value means "auto-detect everything."
type Options struct {
	Charset        Charset
	Polarity       Polarity
	ExpectedDigits int // 0 = auto
	ExpectedRows   int // 0 = auto
	ROI            *image.Rectangle
	Pipeline       []preprocess.Op
	Profile        string
	EnableDNN      bool
	DNNThreshold   float64 // confidence below which DNN is tried (default 0.6)
	OneRatio       float64 // height/width below which a box is called "1"
	MinusRatio     float64 // width/height below which a box is called "-"
	DecHRatio      float64
	DecWRatio      float64
	DebugDir       string
	PrintSpaces    bool
	SpaceFactor    float64
}

func defaultOptions() Options {
	return Options{
		Charset:      CharsetFull,
		Polarity:     PolarityAuto,
		EnableDNN:    true,
		DNNThreshold: 0.6,
		OneRatio:     4.0,  // h/w > 4 → classify as '1'
		MinusRatio:   0.5,
		DecHRatio:    0.20, // decimal dot must be < 20% of median digit height
		DecWRatio:    0.40, // aspect ratio (w/h) check — allow square and short dots
		SpaceFactor:  2.5,
	}
}

// Option is a functional option applied to Options.
type Option func(*Options)

// WithCharset sets the recognized character set.
func WithCharset(c Charset) Option { return func(o *Options) { o.Charset = c } }

// WithPolarity forces polarity instead of auto-detecting.
func WithPolarity(p Polarity) Option { return func(o *Options) { o.Polarity = p } }

// WithExpectedDigits tells the engine how many digits to expect per row.
// Use 0 (default) for auto-detection.
func WithExpectedDigits(n int) Option { return func(o *Options) { o.ExpectedDigits = n } }

// WithExpectedRows tells the engine how many rows of digits to expect.
func WithExpectedRows(n int) Option { return func(o *Options) { o.ExpectedRows = n } }

// WithROI restricts recognition to a sub-rectangle of the input image,
// bypassing the display-detection stage entirely.
func WithROI(r image.Rectangle) Option { return func(o *Options) { o.ROI = &r } }

// WithPipeline replaces the automatic preprocessing chain with an explicit list of ops.
func WithPipeline(ops ...preprocess.Op) Option {
	return func(o *Options) { o.Pipeline = ops }
}

// WithProfile activates a named built-in display profile (e.g. "gas_pump").
func WithProfile(name string) Option { return func(o *Options) { o.Profile = name } }

// WithDNN enables or disables the ONNX DNN fallback classifier.
func WithDNN(enabled bool) Option { return func(o *Options) { o.EnableDNN = enabled } }

// WithDNNThreshold sets the per-digit confidence floor; digits below this are
// re-examined by the DNN classifier.
func WithDNNThreshold(t float64) Option { return func(o *Options) { o.DNNThreshold = t } }

// WithRatios overrides the aspect-ratio thresholds for "1", "-", and decimal point.
func WithRatios(one, minus, decH, decW float64) Option {
	return func(o *Options) {
		o.OneRatio = one
		o.MinusRatio = minus
		o.DecHRatio = decH
		o.DecWRatio = decW
	}
}

// WithDebugDir enables per-stage debug image output to the given directory.
func WithDebugDir(dir string) Option { return func(o *Options) { o.DebugDir = dir } }

// WithPrintSpaces enables insertion of spaces between digit groups.
func WithPrintSpaces(s bool) Option { return func(o *Options) { o.PrintSpaces = s } }
