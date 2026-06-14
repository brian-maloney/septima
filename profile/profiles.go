// Package profile provides named display presets that bias auto-detection.
package profile

import (
	_ "embed"
	"encoding/json"
	"math"
)

//go:embed builtin/alarm_clock.json
var alarmClockJSON []byte

//go:embed builtin/microwave_clock.json
var microwaveClockJSON []byte

//go:embed builtin/multimeter.json
var multimeterJSON []byte

//go:embed builtin/gas_pump.json
var gasPumpJSON []byte

//go:embed builtin/tank_gauge.json
var tankGaugeJSON []byte

//go:embed builtin/security_token.json
var securityTokenJSON []byte

//go:embed builtin/calculator.json
var calculatorJSON []byte

//go:embed builtin/generic.json
var genericJSON []byte

// Profile describes a class of display and the options best suited for it.
type Profile struct {
	Name           string  `json:"name"`
	Polarity       string  `json:"polarity"`        // "dark_on_light" | "light_on_dark" | "auto"
	Charset        string  `json:"charset"`         // "full" | "digits" | "decimal" | "hex"
	ExpectedRows   int     `json:"expected_rows"`   // 0 = auto
	MinDigits      int     `json:"min_digits"`
	MaxDigits      int     `json:"max_digits"`
	HasColon       bool    `json:"has_colon"`
	HasDecimal     bool    `json:"has_decimal"`
	OneRatio       float64 `json:"one_ratio"`
	MinusRatio     float64 `json:"minus_ratio"`
	DecHRatio      float64 `json:"dec_h_ratio"`
	DecWRatio      float64 `json:"dec_w_ratio"`
	// Scoring hints used by AutoSelect
	TypicalAspect  float64 `json:"typical_aspect"`  // width/height of the display ROI
	AspectTolerance float64 `json:"aspect_tolerance"`
}

var registry map[string]Profile

func init() {
	registry = make(map[string]Profile)
	for _, raw := range [][]byte{
		alarmClockJSON, microwaveClockJSON, multimeterJSON, gasPumpJSON,
		tankGaugeJSON, securityTokenJSON, calculatorJSON, genericJSON,
	} {
		var p Profile
		if err := json.Unmarshal(raw, &p); err == nil && p.Name != "" {
			registry[p.Name] = p
		}
	}
}

// Get returns the named profile, or the generic fallback if not found.
func Get(name string) Profile {
	if p, ok := registry[name]; ok {
		return p
	}
	var p Profile
	_ = json.Unmarshal(genericJSON, &p)
	return p
}

// AutoSelect scores all profiles against the observed display characteristics
// and returns the best-matching profile name.
//
// aspect is the width/height ratio of the detected display ROI.
// nDigits is the number of digit boxes found (0 = unknown).
func AutoSelect(aspect float64, nDigits int) string {
	bestName := "generic"
	bestScore := -math.MaxFloat64
	for name, p := range registry {
		if name == "generic" {
			continue
		}
		score := scoreProfile(p, aspect, nDigits)
		if score > bestScore {
			bestScore = score
			bestName = name
		}
	}
	return bestName
}

func scoreProfile(p Profile, aspect float64, nDigits int) float64 {
	score := 0.0
	if p.TypicalAspect > 0 && p.AspectTolerance > 0 {
		diff := math.Abs(aspect - p.TypicalAspect)
		score += math.Max(0, 1.0-diff/p.AspectTolerance)
	}
	if nDigits > 0 {
		if p.MinDigits > 0 && nDigits >= p.MinDigits {
			score += 0.5
		}
		if p.MaxDigits > 0 && nDigits <= p.MaxDigits {
			score += 0.5
		}
	}
	return score
}
