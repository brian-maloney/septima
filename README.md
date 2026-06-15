# septima

Seven-segment display OCR for Go.  Reads a photo or screenshot and returns the
digits shown on the display as a string.  The CLI is interface-compatible with
[ssocr](https://www.unix-ag.uni-kl.de/~auerswal/ssocr/) and adds multi-row
support, automatic display detection, and a DNN fallback classifier.

## Prerequisites

- Go 1.21+
- OpenCV 4.x with the [gocv](https://gocv.io/) binding (`gocv.io/x/gocv v0.43`)

Install OpenCV via Homebrew on macOS:

```
brew install opencv
```

Then follow the [gocv install guide](https://gocv.io/getting-started/) to set
the required `CGO_*` environment variables.

## Build

```
make          # builds bin/septima and bin/septima-bench
make install  # installs both to $(GOPATH)/bin
```

## Usage

```
septima [flags] [pipeline ops...] <image>
```

### Examples

```sh
# Auto-detect everything
septima photo.jpg

# Clock display (colon expected)
septima --profile microwave_clock photo.jpg

# Tank gauge with explicit hint
septima --profile tank_gauge gauge.jpg

# Manual crop + threshold then read
septima crop 10 10 200 80 make_mono photo.jpg

# Debug: write per-stage images to /tmp/dbg/
septima -D /tmp/dbg photo.jpg
```

### Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--threshold N` | `-t` | Luminance threshold percentage |
| `--iter-threshold` | `-T` | Iterative k-means threshold (default) |
| `--number-digits N` | `-d` | Expected digits per row (0 = auto) |
| `--number-rows N` | | Expected rows (0 = auto) |
| `--charset NAME` | `-c` | `full` (default), `digits`, `decimal`, `hex`, `tt_robot` |
| `--foreground COLOR` | `-f` | `black` or `white` |
| `--background COLOR` | `-b` | `black` or `white` |
| `--print-spaces` | `-s` | Insert spaces between digit groups |
| `--profile NAME` | | Built-in display profile (see below) |
| `--no-dnn` | | Disable ONNX fallback classifier |
| `--debug-image DIR` | `-D` | Write per-stage images to DIR |
| `--version` | `-V` | Print version and exit |

### Pipeline ops (ssocr-compatible)

```
crop X Y W H
rotate DEG | shear OFFSET | mirror {horiz|vert}
invert | grayscale | make_mono
dynamic_threshold W H | gray_stretch T1 T2
rgb_threshold | r_threshold | g_threshold | b_threshold
dilation [N] | erosion [N] | opening [N] | closing [N]
remove_isolated | white_border [W]
set_pixels_filter MASK | keep_pixels_filter MASK
```

## Built-in profiles

Profiles tune the recognition pipeline for specific display types.  Pass the
profile name with `--profile`.

| Profile | Use case |
|---------|----------|
| `generic` | Default — auto-detects everything |
| `multimeter` | Digital multimeter display |
| `microwave_clock` | Microwave / oven clock (colon separator) |
| `alarm_clock` | Bedside alarm clock (colon separator) |
| `gas_pump` | Fuel pump display (two rows: price and unit) |
| `tank_gauge` | Liquid tank level gauge |
| `calculator` | Handheld calculator |
| `security_token` | RSA / OTP token LCD |

## Go library

```go
import "github.com/vond/septima"

// Simplest call — returns the text on the display.
result, err := septima.ReadFile("photo.jpg")
fmt.Println(result.Text)

// With options.
result, err = septima.ReadFile("photo.jpg",
    septima.WithProfile("gas_pump"),
    septima.WithCharset(septima.CharsetDecimal),
    septima.WithDebugDir("/tmp/dbg"),
)

// From a standard Go image.Image.
result, err = septima.Read(img, septima.WithPolarity(septima.PolarityLightOnDark))

// Multi-row result.
for _, row := range result.Rows {
    fmt.Printf("row: %s  conf: %.2f\n", row.Text, row.Confidence)
}
```

### Result types

```go
type Result struct {
    Rows       []Row      // one per detected row
    Text       string     // rows joined by "\n"
    Confidence float64    // minimum per-row confidence
    Debug      *DebugInfo // non-nil when WithDebugDir is set
}

type Row struct {
    Text       string
    Digits     []Digit
    Box        image.Rectangle
    Confidence float64
}

type Digit struct {
    Char       rune
    Segments   byte            // 7-bit segment mask (bit 0 = top segment)
    Box        image.Rectangle
    Confidence float64
    Source     Source          // SourceGeometric or SourceDNN
}
```

## Benchmark

Run the test suite against `tests/ground_truth.json`:

```
make bench
```

Or directly:

```
septima-bench tests/
```

The benchmark prints a pass/fail table for each image in auto mode and in
hinted mode (using the display type from ground truth), then exits non-zero if
any image fails.

## Tests

```
make test       # pure-Go unit tests (no OpenCV required)
go test ./...   # same
```

Integration tests in `tests/septima_test.go` require a built binary and OpenCV
at runtime.
