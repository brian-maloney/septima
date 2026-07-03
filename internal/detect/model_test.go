package detect

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestClassesInputSizeResolution covers the per-stage input size fallback chain:
// stage-specific size, then the legacy shared input_size, then the 640 default.
func TestClassesInputSizeResolution(t *testing.T) {
	cases := []struct {
		name             string
		c                Classes
		wantPanel, wantD int
	}{
		{
			name:      "per-stage sizes win over shared",
			c:         Classes{InputSize: 640, PanelInputSize: 640, DigitInputSize: 1280},
			wantPanel: 640, wantD: 1280,
		},
		{
			name:      "missing stage sizes fall back to shared input_size",
			c:         Classes{InputSize: 896},
			wantPanel: 896, wantD: 896,
		},
		{
			name:      "one stage set, other falls back to shared",
			c:         Classes{InputSize: 640, DigitInputSize: 1280},
			wantPanel: 640, wantD: 1280,
		},
		{
			name:      "nothing set falls back to the 640 default",
			c:         Classes{},
			wantPanel: defaultInputSize, wantD: defaultInputSize,
		},
		{
			name:      "no shared size but per-stage set",
			c:         Classes{PanelInputSize: 640, DigitInputSize: 1280},
			wantPanel: 640, wantD: 1280,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.c.PanelSize(); got != tc.wantPanel {
				t.Errorf("PanelSize() = %d, want %d", got, tc.wantPanel)
			}
			if got := tc.c.DigitSize(); got != tc.wantD {
				t.Errorf("DigitSize() = %d, want %d", got, tc.wantD)
			}
		})
	}
}

// TestLoadClassesPerStageSizes verifies classes.json parses the per-stage keys
// and that a legacy file with only input_size still resolves both stages.
func TestLoadClassesPerStageSizes(t *testing.T) {
	write := func(t *testing.T, body string) string {
		t.Helper()
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "classes.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}

	t.Run("per-stage keys", func(t *testing.T) {
		dir := write(t, `{"input_size":640,"panel_input_size":640,"digit_input_size":1280,
			"panel_classes":["display"],"digit_classes":["0","1"]}`)
		c, err := LoadClasses(dir)
		if err != nil {
			t.Fatal(err)
		}
		if c.PanelSize() != 640 || c.DigitSize() != 1280 {
			t.Errorf("panel=%d digit=%d, want 640/1280", c.PanelSize(), c.DigitSize())
		}
	})

	t.Run("legacy shared-only file", func(t *testing.T) {
		dir := write(t, `{"input_size":640,"panel_classes":["display"],"digit_classes":["0","1"]}`)
		c, err := LoadClasses(dir)
		if err != nil {
			t.Fatal(err)
		}
		if c.PanelSize() != 640 || c.DigitSize() != 640 {
			t.Errorf("panel=%d digit=%d, want 640/640", c.PanelSize(), c.DigitSize())
		}
	})
}

// TestShippedClassesJSON guards the real models/classes.json: it must parse and
// name a positive input size for each stage so inference never silently sends a
// stage the wrong resolution.
func TestShippedClassesJSON(t *testing.T) {
	path := filepath.Join("..", "..", "models", "classes.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no shipped classes.json: %v", err)
	}
	var c Classes
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatalf("models/classes.json does not parse: %v", err)
	}
	if c.PanelSize() <= 0 || c.DigitSize() <= 0 {
		t.Errorf("resolved sizes must be positive: panel=%d digit=%d", c.PanelSize(), c.DigitSize())
	}
}
