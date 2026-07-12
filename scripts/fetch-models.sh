#!/usr/bin/env bash
# Downloads models/panel.onnx and models/digits.onnx from the GitHub release
# pinned in models/MODELS_VERSION, verifying each against its sha256. Skips
# files already on disk that match the pin, so re-running is cheap.
#
# Model weights are published independently of the software's vX.Y.Z tags
# (retraining doesn't imply a new software release) — see
# scripts/publish-models.sh and AGENTS.md.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="$repo_root/models"
pin_file="$models_dir/MODELS_VERSION"
repo="${SEPTIMA_REPO:-brian-maloney/septima}"

if [[ ! -f "$pin_file" ]]; then
  echo "fetch-models: missing $pin_file" >&2
  exit 1
fi

tag="$(grep '^TAG=' "$pin_file" | tr -d '\r' | cut -d= -f2)"
if [[ -z "$tag" ]]; then
  echo "fetch-models: no TAG= line in $pin_file" >&2
  exit 1
fi

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

fetch_one() {
  local name="$1" want="$2" dest="$models_dir/$name"
  if [[ -f "$dest" ]] && [[ "sha256:$(sha256_of "$dest")" == "$want" ]]; then
    echo "fetch-models: $name up to date ($tag)" >&2
    return
  fi
  echo "fetch-models: downloading $name from release $tag" >&2
  local url="https://github.com/$repo/releases/download/$tag/$name"
  local tmp
  tmp="$(mktemp "$dest.XXXXXX")"
  if ! curl -fsSL -o "$tmp" "$url"; then
    rm -f "$tmp"
    echo "fetch-models: download failed for $url" >&2
    exit 1
  fi
  local got="sha256:$(sha256_of "$tmp")"
  if [[ "$got" != "$want" ]]; then
    rm -f "$tmp"
    echo "fetch-models: checksum mismatch for $name: want $want got $got" >&2
    exit 1
  fi
  mv "$tmp" "$dest"
}

while IFS='=' read -r name checksum; do
  [[ -z "$name" || "$name" == "TAG" || "$name" == \#* ]] && continue
  fetch_one "$name" "$checksum"
done < <(tr -d '\r' < "$pin_file")
