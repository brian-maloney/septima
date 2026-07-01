// Command septima-labelaudit runs the live pipeline over image paths given on
// stdin (one per line) and prints "path<TAB>reading" per image, for comparing
// model readings against training-label box counts (partial-label audit).
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/brian-maloney/septima"
)

func main() {
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		path := strings.TrimSpace(sc.Text())
		if path == "" {
			continue
		}
		res, err := septima.ReadFile(path)
		if err != nil {
			fmt.Printf("%s\tERROR:%v\n", path, err)
			continue
		}
		fmt.Printf("%s\t%s\n", path, strings.ReplaceAll(res.Text, "\n", "|"))
	}
}
