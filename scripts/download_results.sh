#!/usr/bin/env bash
# Download the paper results bundle from the GitHub release and unpack it into the repo root.
#
# The archive holds exactly the run directories, steering vectors and analysis JSONs that the
# figure and analysis scripts read: results/neutral/..., artifacts/neutral/... and analysis_outputs/...
# Directory timestamps are kept, because some selectors pick the newest matching run directory.
#
# Usage (from anywhere):
#   bash scripts/download_results.sh                  # download, verify, extract
#   bash scripts/download_results.sh --archive FILE   # verify and extract an archive you already have
#   bash scripts/download_results.sh --manifest FILE  # use a local copy of the manifest asset
#   bash scripts/download_results.sh --keep           # keep the downloaded files in .cache/paper_results/
set -euo pipefail

RELEASE_URL="https://github.com/lucabaroni/be-sure-or-be-random/releases/download/v1.0"
ARCHIVE_NAME="paper_results.tar.gz"
ARCHIVE_SHA256="c40ca9de2481b2c3d4b17af1bfd2e2113b5a0fecf05a55b4c01c90b631621ea7"
ARCHIVE_BYTES=711879563
MANIFEST_NAME="paper_results_MANIFEST.tsv"
MANIFEST_SHA256="ce9f8b4907c09de3819bcd87896b326cf81ee2a59d2b1ae6df056c435cd9d946"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache/paper_results"
ARCHIVE=""
MANIFEST=""
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

check_sha() {  # file expected_sha label
  local got
  got="$(sha256_of "$1")"
  if [[ "$got" != "$2" ]]; then
    echo "checksum mismatch for $3: expected $2, got $got" >&2
    exit 1
  fi
  echo "ok  $3 (sha256 $got)"
}

fetch() {  # asset_name destination
  echo "downloading ${RELEASE_URL}/$1"
  curl -L --fail --retry 3 --retry-delay 5 -o "$2" "${RELEASE_URL}/$1"
}

mkdir -p "$CACHE_DIR"
if [[ -z "$ARCHIVE" ]]; then
  ARCHIVE="${CACHE_DIR}/${ARCHIVE_NAME}"
  if [[ ! -f "$ARCHIVE" ]] || [[ "$(sha256_of "$ARCHIVE")" != "$ARCHIVE_SHA256" ]]; then
    echo "archive size: $((ARCHIVE_BYTES / 1000000)) MB"
    fetch "$ARCHIVE_NAME" "$ARCHIVE"
  fi
fi
check_sha "$ARCHIVE" "$ARCHIVE_SHA256" "$ARCHIVE_NAME"

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="${CACHE_DIR}/${MANIFEST_NAME}"
  if [[ ! -f "$MANIFEST" ]] || [[ "$(sha256_of "$MANIFEST")" != "$MANIFEST_SHA256" ]]; then
    fetch "$MANIFEST_NAME" "$MANIFEST"
  fi
fi
check_sha "$MANIFEST" "$MANIFEST_SHA256" "$MANIFEST_NAME"

echo "extracting into ${REPO_ROOT}"
tar -xzf "$ARCHIVE" --no-same-owner -C "$REPO_ROOT"

echo "verifying extracted files against the manifest"
n=0
while IFS=$'\t' read -r path bytes sha; do
  [[ "$path" == "path" ]] && continue
  f="${REPO_ROOT}/${path}"
  if [[ ! -f "$f" ]]; then
    echo "missing after extraction: $path" >&2
    exit 1
  fi
  if [[ "$(sha256_of "$f")" != "$sha" ]]; then
    echo "checksum mismatch after extraction: $path" >&2
    exit 1
  fi
  n=$((n + 1))
done < "$MANIFEST"
echo "ok  ${n} files verified"

if [[ "$KEEP" -eq 0 ]]; then
  rm -f "${CACHE_DIR}/${ARCHIVE_NAME}" "${CACHE_DIR}/${MANIFEST_NAME}"
  rmdir "$CACHE_DIR" 2>/dev/null || true
  rmdir "${REPO_ROOT}/.cache" 2>/dev/null || true
fi
echo "done: results/, artifacts/ and analysis_outputs/ are in place"
