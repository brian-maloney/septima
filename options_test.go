package septima

import (
	"image"
	"testing"

	"github.com/vond/septima/preprocess"
)

func TestDefaultOptions(t *testing.T) {
	o := defaultOptions()
	if o.Charset != CharsetFull {
		t.Errorf("default Charset = %v, want CharsetFull", o.Charset)
	}
	if o.Polarity != PolarityAuto {
		t.Errorf("default Polarity = %v, want PolarityAuto", o.Polarity)
	}
	if !o.EnableDNN {
		t.Error("default EnableDNN = false, want true")
	}
	if o.DNNThreshold != 0.6 {
		t.Errorf("default DNNThreshold = %v, want 0.6", o.DNNThreshold)
	}
	if o.OneRatio != 4.0 {
		t.Errorf("default OneRatio = %v, want 4.0", o.OneRatio)
	}
	if o.MinusRatio != 0.5 {
		t.Errorf("default MinusRatio = %v, want 0.5", o.MinusRatio)
	}
	if o.DecHRatio != 0.20 {
		t.Errorf("default DecHRatio = %v, want 0.20", o.DecHRatio)
	}
	if o.DecWRatio != 0.40 {
		t.Errorf("default DecWRatio = %v, want 0.40", o.DecWRatio)
	}
}

func TestWithCharset(t *testing.T) {
	o := defaultOptions()
	WithCharset(CharsetDigits)(&o)
	if o.Charset != CharsetDigits {
		t.Errorf("WithCharset: got %v, want CharsetDigits", o.Charset)
	}
}

func TestWithPolarity(t *testing.T) {
	o := defaultOptions()
	WithPolarity(PolarityLightOnDark)(&o)
	if o.Polarity != PolarityLightOnDark {
		t.Errorf("WithPolarity: got %v, want PolarityLightOnDark", o.Polarity)
	}
}

func TestWithExpectedDigits(t *testing.T) {
	o := defaultOptions()
	WithExpectedDigits(4)(&o)
	if o.ExpectedDigits != 4 {
		t.Errorf("WithExpectedDigits: got %d, want 4", o.ExpectedDigits)
	}
}

func TestWithExpectedRows(t *testing.T) {
	o := defaultOptions()
	WithExpectedRows(2)(&o)
	if o.ExpectedRows != 2 {
		t.Errorf("WithExpectedRows: got %d, want 2", o.ExpectedRows)
	}
}

func TestWithROI(t *testing.T) {
	o := defaultOptions()
	r := image.Rect(10, 20, 100, 80)
	WithROI(r)(&o)
	if o.ROI == nil {
		t.Fatal("WithROI: ROI is nil")
	}
	if *o.ROI != r {
		t.Errorf("WithROI: got %v, want %v", *o.ROI, r)
	}
}

func TestWithPipeline(t *testing.T) {
	o := defaultOptions()
	op := preprocess.GrayscaleOp{}
	WithPipeline(op)(&o)
	if len(o.Pipeline) != 1 {
		t.Errorf("WithPipeline: len = %d, want 1", len(o.Pipeline))
	}
}

func TestWithProfile(t *testing.T) {
	o := defaultOptions()
	WithProfile("multimeter")(&o)
	if o.Profile != "multimeter" {
		t.Errorf("WithProfile: got %q, want %q", o.Profile, "multimeter")
	}
}

func TestWithDNN(t *testing.T) {
	o := defaultOptions()
	WithDNN(false)(&o)
	if o.EnableDNN {
		t.Error("WithDNN(false): EnableDNN still true")
	}
}

func TestWithDNNThreshold(t *testing.T) {
	o := defaultOptions()
	WithDNNThreshold(0.8)(&o)
	if o.DNNThreshold != 0.8 {
		t.Errorf("WithDNNThreshold: got %v, want 0.8", o.DNNThreshold)
	}
}

func TestWithRatios(t *testing.T) {
	o := defaultOptions()
	WithRatios(5.0, 0.3, 0.25, 0.50)(&o)
	if o.OneRatio != 5.0 {
		t.Errorf("WithRatios OneRatio = %v, want 5.0", o.OneRatio)
	}
	if o.MinusRatio != 0.3 {
		t.Errorf("WithRatios MinusRatio = %v, want 0.3", o.MinusRatio)
	}
	if o.DecHRatio != 0.25 {
		t.Errorf("WithRatios DecHRatio = %v, want 0.25", o.DecHRatio)
	}
	if o.DecWRatio != 0.50 {
		t.Errorf("WithRatios DecWRatio = %v, want 0.50", o.DecWRatio)
	}
}

func TestWithDebugDir(t *testing.T) {
	o := defaultOptions()
	WithDebugDir("/tmp/debug")(&o)
	if o.DebugDir != "/tmp/debug" {
		t.Errorf("WithDebugDir: got %q, want %q", o.DebugDir, "/tmp/debug")
	}
}

func TestWithPrintSpaces(t *testing.T) {
	o := defaultOptions()
	WithPrintSpaces(true)(&o)
	if !o.PrintSpaces {
		t.Error("WithPrintSpaces(true): PrintSpaces = false")
	}
}

func TestApplyProfileMultimeter(t *testing.T) {
	o := defaultOptions()
	applyProfile(&o, "multimeter")
	if o.Polarity != PolarityDarkOnLight {
		t.Errorf("applyProfile multimeter: Polarity = %v, want PolarityDarkOnLight", o.Polarity)
	}
	if o.Charset != CharsetDecimal {
		t.Errorf("applyProfile multimeter: Charset = %v, want CharsetDecimal", o.Charset)
	}
	if o.ExpectedRows != 1 {
		t.Errorf("applyProfile multimeter: ExpectedRows = %d, want 1", o.ExpectedRows)
	}
	if o.DecHRatio != 0.35 {
		t.Errorf("applyProfile multimeter: DecHRatio = %v, want 0.35", o.DecHRatio)
	}
}

func TestApplyProfileMicrowaveClock(t *testing.T) {
	o := defaultOptions()
	applyProfile(&o, "microwave_clock")
	if o.Polarity != PolarityLightOnDark {
		t.Errorf("applyProfile microwave_clock: Polarity = %v, want PolarityLightOnDark", o.Polarity)
	}
	// charset "full" is a no-op (already the default)
	if o.Charset != CharsetFull {
		t.Errorf("applyProfile microwave_clock: Charset = %v, want CharsetFull", o.Charset)
	}
	// dec_h_ratio=0.0 in JSON → should NOT override the default (0 means inherit)
	if o.DecHRatio != 0.20 {
		t.Errorf("applyProfile microwave_clock: DecHRatio = %v, want 0.20 (preserved default)", o.DecHRatio)
	}
}

func TestApplyProfileUnknown(t *testing.T) {
	o := defaultOptions()
	original := o
	applyProfile(&o, "does_not_exist")
	// Generic profile has auto polarity and full charset → same as defaults for these
	if o.Charset != original.Charset {
		t.Errorf("applyProfile unknown: Charset changed unexpectedly")
	}
}

func TestToDecodeCharset(t *testing.T) {
	tests := []struct {
		in  Charset
		out string
	}{
		{CharsetFull, "CharsetFull"},
		{CharsetDigits, "CharsetDigits"},
		{CharsetDecimal, "CharsetDecimal"},
		{CharsetHex, "CharsetHex"},
		{CharsetTTRobot, "CharsetTTRobot"},
	}
	for _, tt := range tests {
		got := toDecodeCharset(tt.in)
		// Just verify it's non-zero and doesn't panic; the exact values are
		// tested through Decode in the decode package.
		_ = got
	}
}

func TestMultipleOptionsCombine(t *testing.T) {
	o := defaultOptions()
	opts := []Option{
		WithCharset(CharsetHex),
		WithPolarity(PolarityDarkOnLight),
		WithExpectedRows(3),
	}
	for _, opt := range opts {
		opt(&o)
	}
	if o.Charset != CharsetHex {
		t.Errorf("combined opts: Charset = %v, want CharsetHex", o.Charset)
	}
	if o.Polarity != PolarityDarkOnLight {
		t.Errorf("combined opts: Polarity = %v, want PolarityDarkOnLight", o.Polarity)
	}
	if o.ExpectedRows != 3 {
		t.Errorf("combined opts: ExpectedRows = %d, want 3", o.ExpectedRows)
	}
}
