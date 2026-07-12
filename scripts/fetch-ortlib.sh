#!/usr/bin/env bash
# Downloads the ONNX Runtime shared library for the current (or given)
# platform straight from the upstream onnxruntime GitHub release, and places
# it at the path internal/ortlib's go:embed directives expect
# (internal/ortlib/lib/<dest_name>). Skips the download if that file is
# already present.
#
# Usage: scripts/fetch-ortlib.sh [GOOS] [GOARCH]
# Defaults to `go env GOOS`/`go env GOARCH`, i.e. the host platform — the
# right choice for local dev and for CI jobs that build natively (this repo
# never cross-compiles). Prints the resolved destination path to stdout;
# all other output goes to stderr, so callers can do:
#   path="$(scripts/fetch-ortlib.sh)"
#
# Bumping the ONNX Runtime version requires updating both the `version`
# variable below and the //go:embed paths in internal/ortlib/lib_*.go —
# they must name the same file.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_dir="$repo_root/internal/ortlib/lib"
version="1.27.0"

goos="${1:-$(go env GOOS)}"
goarch="${2:-$(go env GOARCH)}"

case "$goos/$goarch" in
  darwin/arm64)
    asset="onnxruntime-osx-arm64-$version.tgz"
    inner_dir="onnxruntime-osx-arm64-$version"
    inner_lib="lib/libonnxruntime.$version.dylib"
    dest_name="libonnxruntime-darwin-arm64-$version.dylib"
    ;;
  linux/amd64)
    asset="onnxruntime-linux-x64-$version.tgz"
    inner_dir="onnxruntime-linux-x64-$version"
    inner_lib="lib/libonnxruntime.so.$version"
    dest_name="libonnxruntime-linux-amd64-$version.so"
    ;;
  linux/arm64)
    asset="onnxruntime-linux-aarch64-$version.tgz"
    inner_dir="onnxruntime-linux-aarch64-$version"
    inner_lib="lib/libonnxruntime.so.$version"
    dest_name="libonnxruntime-linux-arm64-$version.so"
    ;;
  windows/amd64)
    asset="onnxruntime-win-x64-$version.zip"
    inner_dir="onnxruntime-win-x64-$version"
    inner_lib="lib/onnxruntime.dll"
    dest_name="onnxruntime-windows-amd64-$version.dll"
    ;;
  *)
    echo "fetch-ortlib: no embedded ONNX Runtime build for $goos/$goarch — cmd/septima falls back to on-disk/SEPTIMA_ORT_LIB discovery on this platform, nothing to do" >&2
    exit 0
    ;;
esac

dest="$dest_dir/$dest_name"
if [[ -f "$dest" ]]; then
  echo "fetch-ortlib: $dest_name already present" >&2
  echo "$dest"
  exit 0
fi

mkdir -p "$dest_dir"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

url="https://github.com/microsoft/onnxruntime/releases/download/v$version/$asset"
echo "fetch-ortlib: downloading $asset for $goos/$goarch" >&2
case "$asset" in
  *.tgz)
    curl -fsSL "$url" | tar xz -C "$work"
    ;;
  *.zip)
    curl -fsSL -o "$work/ort.zip" "$url"
    unzip -q "$work/ort.zip" -d "$work"
    ;;
esac

cp "$work/$inner_dir/$inner_lib" "$dest"
echo "fetch-ortlib: wrote $dest" >&2
echo "$dest"
