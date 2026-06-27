package septima

// Options holds tunable parameters for a recognition call.
// The zero value is filled in by defaultOptions().
type Options struct {
	// ModelDir is the directory holding panel.onnx, digits.onnx and classes.json.
	// Empty means use the default ("models" relative to the working directory,
	// overridable via the SEPTIMA_MODEL_DIR environment variable).
	ModelDir string

	// Profile is an optional display-type hint (e.g. "tank_gauge", "gas_pump").
	// It biases panel/charset selection but is not required.
	Profile string

	// ExpectedRows tells the engine how many digit rows to expect (0 = auto).
	ExpectedRows int
	// ExpectedDigits tells the engine how many digits per row to expect (0 = auto).
	ExpectedDigits int

	// ConfThreshold is the minimum detector score to keep a digit detection.
	ConfThreshold float64
	// PunctThreshold is the minimum detector score to keep a punctuation
	// detection ('.', ':', '-'). It is set lower than ConfThreshold because the
	// decimal point is the model's weakest, lowest-confidence class: genuine dots
	// are routinely detected at ~0.18-0.22 and were being dropped by the 0.25
	// digit floor (e.g. a water-meter "60.00" read as "6000"). The tank stays
	// phantom-free at this lower bar thanks to its digits-only hard negatives.
	PunctThreshold float64
	// IOUThreshold is the IoU cutoff used by non-maximum suppression.
	IOUThreshold float64

	// DebugDir, when set, enables diagnostic image output.
	DebugDir string

	// SkipPanel bypasses stage-1 panel localization and runs the digit detector
	// on the image as given (used for diagnostics / pre-cropped inputs).
	SkipPanel bool
}

func defaultOptions() Options {
	return Options{
		ConfThreshold:  0.25,
		PunctThreshold: 0.20,
		IOUThreshold:   0.45,
	}
}

// Option is a functional option applied to Options.
type Option func(*Options)

// WithModelDir overrides the directory containing the ONNX models and classes.json.
func WithModelDir(dir string) Option { return func(o *Options) { o.ModelDir = dir } }

// WithProfile activates a named display-type hint (e.g. "tank_gauge").
func WithProfile(name string) Option { return func(o *Options) { o.Profile = name } }

// WithExpectedRows tells the engine how many rows of digits to expect (0 = auto).
func WithExpectedRows(n int) Option { return func(o *Options) { o.ExpectedRows = n } }

// WithExpectedDigits tells the engine how many digits to expect per row (0 = auto).
func WithExpectedDigits(n int) Option { return func(o *Options) { o.ExpectedDigits = n } }

// WithConfThreshold sets the minimum detector score to keep a digit detection.
func WithConfThreshold(t float64) Option { return func(o *Options) { o.ConfThreshold = t } }

// WithPunctThreshold sets the minimum detector score to keep a punctuation
// detection ('.', ':', '-'). Defaults below ConfThreshold to recover genuine
// low-confidence decimal points and colons.
func WithPunctThreshold(t float64) Option { return func(o *Options) { o.PunctThreshold = t } }

// WithIOUThreshold sets the IoU cutoff used by non-maximum suppression.
func WithIOUThreshold(t float64) Option { return func(o *Options) { o.IOUThreshold = t } }

// WithDebugDir enables diagnostic image output to the given directory.
func WithDebugDir(dir string) Option { return func(o *Options) { o.DebugDir = dir } }

// WithSkipPanel bypasses stage-1 panel localization (for diagnostics or when the
// input is already a tight display crop).
func WithSkipPanel(b bool) Option { return func(o *Options) { o.SkipPanel = b } }

func applyOptions(opts []Option) Options {
	o := defaultOptions()
	for _, fn := range opts {
		fn(&o)
	}
	return o
}
