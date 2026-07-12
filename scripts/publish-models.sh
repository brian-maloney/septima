#!/usr/bin/env bash
# Maintainer script: publishes the current models/panel.onnx + models/digits.onnx
# as a new GitHub release, tagged independently of the software's vX.Y.Z
# releases (retraining doesn't imply a new software version — see AGENTS.md).
#
# Run this only after a retrain has passed the full verification loop in
# AGENTS.md. It creates a public release and uploads ~160MB of model
# weights — a visible, real action, not a dry run.
#
# Usage: scripts/publish-models.sh [--notes-file FILE]
#
# After it finishes, review and commit the updated models/MODELS_VERSION
# (and models/classes.json, if the class list changed) as one commit.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="$repo_root/models"
pin_file="$models_dir/MODELS_VERSION"
repo="${SEPTIMA_REPO:-brian-maloney/septima}"

notes_file=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --notes-file)
      notes_file="$2"
      shift 2
      ;;
    *)
      echo "publish-models: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

command -v gh >/dev/null 2>&1 || { echo "publish-models: requires the GitHub CLI (gh)" >&2; exit 1; }

for f in digits.onnx panel.onnx; do
  [[ -f "$models_dir/$f" ]] || { echo "publish-models: missing $models_dir/$f" >&2; exit 1; }
done

git -C "$repo_root" fetch --tags -q

last_num="$(git -C "$repo_root" tag -l 'models-v*' | sed 's/^models-v//' | sort -n | tail -1)"
next_num=$(( ${last_num:-0} + 1 ))
tag="models-v$next_num"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

echo "publish-models: creating release $tag in $repo"
gh_args=(release create "$tag"
  "$models_dir/digits.onnx" "$models_dir/panel.onnx"
  --repo "$repo"
  --title "$tag")
if [[ -n "$notes_file" ]]; then
  gh_args+=(--notes-file "$notes_file")
else
  gh_args+=(--notes "Model weights for septima. Pinned by models/MODELS_VERSION; fetched via scripts/fetch-models.sh. See AGENTS.md for the retraining/versioning workflow.")
fi
gh "${gh_args[@]}"

digits_sha="$(sha256_of "$models_dir/digits.onnx")"
panel_sha="$(sha256_of "$models_dir/panel.onnx")"

cat > "$pin_file" <<EOF
TAG=$tag
digits.onnx=sha256:$digits_sha
panel.onnx=sha256:$panel_sha
EOF

echo "publish-models: wrote $pin_file"
echo "publish-models: now commit $pin_file (and models/classes.json, if it changed)"
