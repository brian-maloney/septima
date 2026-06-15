// Command septima-bench runs all test images from a ground_truth.json and prints
// a pass/fail summary table.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

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

func main() {
	dir := "tests"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	gtPath := filepath.Join(dir, "ground_truth.json")
	data, err := os.ReadFile(gtPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench: cannot read %s: %v\n", gtPath, err)
		os.Exit(1)
	}
	var gt groundTruth
	if err := json.Unmarshal(data, &gt); err != nil {
		fmt.Fprintf(os.Stderr, "bench: parse error: %v\n", err)
		os.Exit(1)
	}

	type result struct {
		file       string
		expected   string
		gotAuto    string
		gotHinted  string
		autoPass   bool
		hintedPass bool
	}

	var results []result
	autoPass, hintedPass := 0, 0

	for _, c := range gt.Images {
		imgPath := filepath.Join(dir, c.File)
		r := result{file: c.File, expected: c.Value}

		// Auto pass
		got, err := septima.ReadFile(imgPath)
		if err == nil {
			r.gotAuto = got.Text
		} else {
			r.gotAuto = "ERR: " + err.Error()
		}
		r.autoPass = r.gotAuto == c.Value
		if r.autoPass {
			autoPass++
		}

		// Hinted pass
		hintOpts := []septima.Option{septima.WithProfile(c.DisplayType)}
		if len(c.Rows) > 0 {
			hintOpts = append(hintOpts, septima.WithExpectedRows(len(c.Rows)))
		}
		got2, err := septima.ReadFile(imgPath, hintOpts...)
		if err == nil {
			r.gotHinted = got2.Text
		} else {
			r.gotHinted = "ERR: " + err.Error()
		}
		r.hintedPass = r.gotHinted == c.Value
		if r.hintedPass {
			hintedPass++
		}

		results = append(results, r)
	}

	n := len(results)
	fmt.Printf("%-55s  %-12s  %-12s  %-12s  %s  %s\n",
		"File", "Expected", "Auto", "Hinted", "Auto", "Hint")
	fmt.Printf("%s\n", strings.Repeat("-", 110))
	for _, r := range results {
		autoMark := "✗"
		if r.autoPass {
			autoMark = "✓"
		}
		hintMark := "✗"
		if r.hintedPass {
			hintMark = "✓"
		}
		fmt.Printf("%-55s  %-12s  %-12s  %-12s  %s   %s\n",
			r.file, r.expected, truncate(r.gotAuto, 12), truncate(r.gotHinted, 12),
			autoMark, hintMark)
	}
	fmt.Printf("\nAuto:   %d/%d (%.0f%%)\n", autoPass, n, 100*float64(autoPass)/float64(n))
	fmt.Printf("Hinted: %d/%d (%.0f%%)\n", hintedPass, n, 100*float64(hintedPass)/float64(n))

	if autoPass < n || hintedPass < n {
		os.Exit(1)
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
