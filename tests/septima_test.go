package tests

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/brian-maloney/septima"
)

type groundTruth struct {
	Description string      `json:"description"`
	Images      []imageCase `json:"images"`
}

type imageCase struct {
	File        string   `json:"file"`
	Value       string   `json:"value"`
	DisplayType string   `json:"display_type"`
	Rows        []string `json:"rows"`
	Notes       string   `json:"notes"`
}

func loadGroundTruth(t *testing.T) []imageCase {
	t.Helper()
	_, src, _, _ := runtime.Caller(0)
	dir := filepath.Dir(src)
	data, err := os.ReadFile(filepath.Join(dir, "ground_truth.json"))
	if err != nil {
		t.Fatalf("cannot read ground_truth.json: %v", err)
	}
	var gt groundTruth
	if err := json.Unmarshal(data, &gt); err != nil {
		t.Fatalf("parse error: %v", err)
	}
	return gt.Images
}

func imageDir(t *testing.T) string {
	t.Helper()
	_, src, _, _ := runtime.Caller(0)
	return filepath.Dir(src)
}

func TestRecognizeAuto(t *testing.T) {
	cases := loadGroundTruth(t)
	dir := imageDir(t)

	for _, c := range cases {
		c := c
		t.Run(c.File, func(t *testing.T) {
			path := filepath.Join(dir, c.File)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Skipf("image file not found: %s", path)
			}

			got, err := septima.ReadFile(path)
			if err != nil {
				t.Fatalf("ReadFile error: %v", err)
			}
			if got.Text != c.Value {
				t.Errorf("auto pass: got %q, want %q", got.Text, c.Value)
			}
		})
	}
}

func TestRecognizeHinted(t *testing.T) {
	cases := loadGroundTruth(t)
	dir := imageDir(t)

	for _, c := range cases {
		c := c
		t.Run(c.File, func(t *testing.T) {
			path := filepath.Join(dir, c.File)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Skipf("image file not found: %s", path)
			}

			opts := []septima.Option{septima.WithProfile(c.DisplayType)}
			if len(c.Rows) > 0 {
				opts = append(opts, septima.WithExpectedRows(len(c.Rows)))
			}

			got, err := septima.ReadFile(path, opts...)
			if err != nil {
				t.Fatalf("ReadFile error: %v", err)
			}
			if got.Text != c.Value {
				t.Errorf("hinted pass: got %q, want %q", got.Text, c.Value)
			}

			if len(c.Rows) > 0 {
				if len(got.Rows) != len(c.Rows) {
					t.Errorf("row count: got %d, want %d", len(got.Rows), len(c.Rows))
					return
				}
				for i, expected := range c.Rows {
					if got.Rows[i].Text != expected {
						t.Errorf("row[%d]: got %q, want %q", i, got.Rows[i].Text, expected)
					}
				}
			}
		})
	}
}
