// Command septima reads a seven-segment display from a single image and prints
// the recognized value.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/brian-maloney/septima"
)

func main() {
	var (
		modelDir = flag.String("models", "", "directory containing panel.onnx, digits.onnx, classes.json")
		profile  = flag.String("profile", "", "display-type hint (e.g. tank_gauge)")
		conf     = flag.Float64("conf", 0.25, "detection confidence threshold")
		verbose  = flag.Bool("v", false, "print per-digit detail")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: septima [flags] IMAGE")
		flag.PrintDefaults()
		os.Exit(2)
	}

	opts := []septima.Option{septima.WithConfThreshold(*conf)}
	if *modelDir != "" {
		opts = append(opts, septima.WithModelDir(*modelDir))
	}
	if *profile != "" {
		opts = append(opts, septima.WithProfile(*profile))
	}

	res, err := septima.ReadFile(flag.Arg(0), opts...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	fmt.Println(res.Text)
	if *verbose {
		fmt.Fprintf(os.Stderr, "confidence: %.3f\n", res.Confidence)
		for i, row := range res.Rows {
			fmt.Fprintf(os.Stderr, "row %d: %q (conf %.3f, %d digits)\n", i, row.Text, row.Confidence, len(row.Digits))
		}
	}
}
