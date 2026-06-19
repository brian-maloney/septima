// Command septima-bench evaluates the recognizer against a directory's
// ground_truth.json, reporting exact-string accuracy and mean per-character
// (Levenshtein) accuracy.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

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
	var (
		modelDir = flag.String("models", "", "directory containing the ONNX models and classes.json")
		hinted   = flag.Bool("hinted", false, "pass display_type/rows hints from ground truth")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: septima-bench [flags] DIR")
		flag.PrintDefaults()
		os.Exit(2)
	}
	dir := flag.Arg(0)

	data, err := os.ReadFile(filepath.Join(dir, "ground_truth.json"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	var gt groundTruth
	if err := json.Unmarshal(data, &gt); err != nil {
		fmt.Fprintln(os.Stderr, "parse error:", err)
		os.Exit(1)
	}

	var exact, total, missing int
	var charAccSum float64
	for _, c := range gt.Images {
		path := filepath.Join(dir, c.File)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			missing++
			continue
		}
		total++

		opts := []septima.Option{}
		if *modelDir != "" {
			opts = append(opts, septima.WithModelDir(*modelDir))
		}
		if *hinted {
			opts = append(opts, septima.WithProfile(c.DisplayType))
			if len(c.Rows) > 0 {
				opts = append(opts, septima.WithExpectedRows(len(c.Rows)))
			}
		}

		res, err := septima.ReadFile(path, opts...)
		got := res.Text
		status := "ok"
		if err != nil {
			got = ""
			status = "ERR: " + err.Error()
		}
		ca := charAccuracy(got, c.Value)
		charAccSum += ca
		if got == c.Value {
			exact++
			fmt.Printf("PASS  %-16s %q\n", c.File, got)
		} else {
			fmt.Printf("FAIL  %-16s got %q want %q (char %.0f%%) %s\n", c.File, got, c.Value, ca*100, status)
		}
	}

	fmt.Println("----")
	if total == 0 {
		fmt.Printf("no images found (%d listed, all missing)\n", missing)
		return
	}
	fmt.Printf("exact: %d/%d (%.1f%%)   mean char acc: %.1f%%   missing: %d\n",
		exact, total, 100*float64(exact)/float64(total), 100*charAccSum/float64(total), missing)
}

// charAccuracy returns 1 - normalized Levenshtein distance between got and want.
func charAccuracy(got, want string) float64 {
	if want == "" {
		if got == "" {
			return 1
		}
		return 0
	}
	d := levenshtein([]rune(got), []rune(want))
	acc := 1 - float64(d)/float64(len([]rune(want)))
	if acc < 0 {
		return 0
	}
	return acc
}

func levenshtein(a, b []rune) int {
	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur := make([]int, len(b)+1)
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min3(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(b)]
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}
