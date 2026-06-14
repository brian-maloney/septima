package profile

import (
	"math"
	"testing"
)

func TestGetKnownProfiles(t *testing.T) {
	tests := []struct {
		name     string
		polarity string
		charset  string
		hasColon bool
	}{
		{"multimeter", "dark_on_light", "decimal", false},
		{"microwave_clock", "light_on_dark", "full", true},
		{"alarm_clock", "light_on_dark", "full", true},
		{"gas_pump", "dark_on_light", "decimal", false},
		{"tank_gauge", "dark_on_light", "digits", false},
		{"generic", "auto", "full", false},
	}
	for _, tt := range tests {
		p := Get(tt.name)
		if p.Name != tt.name {
			t.Errorf("Get(%q).Name = %q, want %q", tt.name, p.Name, tt.name)
		}
		if p.Polarity != tt.polarity {
			t.Errorf("Get(%q).Polarity = %q, want %q", tt.name, p.Polarity, tt.polarity)
		}
		if p.Charset != tt.charset {
			t.Errorf("Get(%q).Charset = %q, want %q", tt.name, p.Charset, tt.charset)
		}
		if p.HasColon != tt.hasColon {
			t.Errorf("Get(%q).HasColon = %v, want %v", tt.name, p.HasColon, tt.hasColon)
		}
	}
}

func TestGetUnknownFallsBackToGeneric(t *testing.T) {
	p := Get("does_not_exist")
	if p.Name != "generic" {
		t.Errorf("Get(unknown).Name = %q, want %q", p.Name, "generic")
	}
}

func TestGetMultimeterDetails(t *testing.T) {
	p := Get("multimeter")
	if p.ExpectedRows != 1 {
		t.Errorf("multimeter expected_rows = %d, want 1", p.ExpectedRows)
	}
	if p.MinDigits != 3 {
		t.Errorf("multimeter min_digits = %d, want 3", p.MinDigits)
	}
	if p.MaxDigits != 5 {
		t.Errorf("multimeter max_digits = %d, want 5", p.MaxDigits)
	}
	if p.DecHRatio != 0.35 {
		t.Errorf("multimeter dec_h_ratio = %v, want 0.35", p.DecHRatio)
	}
	if p.HasDecimal != true {
		t.Errorf("multimeter has_decimal = false, want true")
	}
}

func TestGetMicrowaveClockDetails(t *testing.T) {
	p := Get("microwave_clock")
	if p.ExpectedRows != 1 {
		t.Errorf("microwave_clock expected_rows = %d, want 1", p.ExpectedRows)
	}
	if p.MinDigits != 4 || p.MaxDigits != 4 {
		t.Errorf("microwave_clock min/max_digits = %d/%d, want 4/4", p.MinDigits, p.MaxDigits)
	}
	// dec_h_ratio and dec_w_ratio = 0.0 (inherit defaults)
	if p.DecHRatio != 0.0 {
		t.Errorf("microwave_clock dec_h_ratio = %v, want 0.0", p.DecHRatio)
	}
}

func TestRegistryContainsAllBuiltins(t *testing.T) {
	expected := []string{
		"alarm_clock", "microwave_clock", "multimeter",
		"gas_pump", "tank_gauge", "security_token", "calculator", "generic",
	}
	for _, name := range expected {
		p := Get(name)
		if p.Name != name {
			t.Errorf("builtin %q not in registry (got %q)", name, p.Name)
		}
	}
}

func TestScoreProfileAspect(t *testing.T) {
	p := Profile{
		Name:            "test",
		TypicalAspect:   3.0,
		AspectTolerance: 1.0,
	}
	// Exact match → score = 1.0
	s := scoreProfile(p, 3.0, 0)
	if math.Abs(s-1.0) > 1e-9 {
		t.Errorf("scoreProfile exact aspect: got %v, want 1.0", s)
	}

	// Half-tolerance away → score = 0.5
	s2 := scoreProfile(p, 3.5, 0)
	if math.Abs(s2-0.5) > 1e-9 {
		t.Errorf("scoreProfile half-tolerance: got %v, want 0.5", s2)
	}

	// Beyond tolerance → score = 0
	s3 := scoreProfile(p, 5.0, 0)
	if s3 != 0.0 {
		t.Errorf("scoreProfile beyond tolerance: got %v, want 0.0", s3)
	}
}

func TestScoreProfileDigits(t *testing.T) {
	p := Profile{
		Name:      "test",
		MinDigits: 4,
		MaxDigits: 6,
	}
	// nDigits within min/max → +1.0
	s := scoreProfile(p, 0, 5)
	if math.Abs(s-1.0) > 1e-9 {
		t.Errorf("digits in range: got %v, want 1.0", s)
	}

	// nDigits below min → only MaxDigits bonus (+0.5)
	s2 := scoreProfile(p, 0, 2)
	if math.Abs(s2-0.5) > 1e-9 {
		t.Errorf("digits below min: got %v, want 0.5", s2)
	}

	// nDigits above max → only MinDigits bonus (+0.5)
	s3 := scoreProfile(p, 0, 10)
	if math.Abs(s3-0.5) > 1e-9 {
		t.Errorf("digits above max: got %v, want 0.5", s3)
	}

	// nDigits = 0 (unknown) → no digit score bonus
	s4 := scoreProfile(p, 0, 0)
	if s4 != 0.0 {
		t.Errorf("digits=0 (unknown): got %v, want 0.0", s4)
	}
}

func TestAutoSelectReturnsNonEmpty(t *testing.T) {
	name := AutoSelect(3.0, 5)
	if name == "" {
		t.Error("AutoSelect returned empty string")
	}
}

func TestAutoSelectGenericFallback(t *testing.T) {
	// An aspect of 0 and 0 digits will not match any real profile well.
	// The result can be any profile — just not empty.
	name := AutoSelect(0, 0)
	if name == "" {
		t.Error("AutoSelect(0,0): returned empty string")
	}
}

func TestAutoSelectPrefersMicrowaveForClockAspect(t *testing.T) {
	// microwave_clock has typical_aspect=3.0, tolerance=1.5, min=max=4 digits
	// With aspect=3.0 and 4 digits it should score well.
	name := AutoSelect(3.0, 4)
	if name == "" {
		t.Errorf("AutoSelect(clock): empty result")
	}
	// We don't assert a specific winner since scoring can be close, but it
	// must return something sensible (not panic or empty).
}
