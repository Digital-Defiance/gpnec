#!/usr/bin/env bash
# Build a universal (arm64 + x86_64) GPNECCBridge release tarball for Homebrew / Lattice.
#
# Usage:
#   ./scripts/release_bridge.sh           # version from VERSION file
#   ./scripts/release_bridge.sh 0.1.0
#
# Output:
#   dist/gpnec-<ver>-macos-universal.tar.gz
#   dist/SHA256SUMS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(tr -d '[:space:]' < VERSION)"
fi
[[ -n "$VERSION" ]] || { echo "VERSION empty" >&2; exit 1; }

DIST="$ROOT/dist"
STAGE="$DIST/stage-gpnec-$VERSION"
TARBALL="$DIST/gpnec-${VERSION}-macos-universal.tar.gz"
mkdir -p "$DIST"
rm -rf "$STAGE"
mkdir -p "$STAGE/lib" "$STAGE/include"

echo "==> Building GPNECCBridge (arm64)…"
swift build -c release --product GPNECCBridge --arch arm64
ARM_BIN="$(swift build -c release --product GPNECCBridge --arch arm64 --show-bin-path)"
ARM_DYLIB="$ARM_BIN/libGPNECCBridge.dylib"
[[ -f "$ARM_DYLIB" ]] || { echo "missing $ARM_DYLIB" >&2; exit 1; }

echo "==> Building GPNECCBridge (x86_64)…"
swift build -c release --product GPNECCBridge --arch x86_64
X86_BIN="$(swift build -c release --product GPNECCBridge --arch x86_64 --show-bin-path)"
X86_DYLIB="$X86_BIN/libGPNECCBridge.dylib"
[[ -f "$X86_DYLIB" ]] || { echo "missing $X86_DYLIB" >&2; exit 1; }

UNIVERSAL="$STAGE/lib/libGPNECCBridge.dylib"
echo "==> lipo → $UNIVERSAL"
lipo -create "$ARM_DYLIB" "$X86_DYLIB" -output "$UNIVERSAL"
lipo -info "$UNIVERSAL"

cp "$ROOT/host/include/gpnec.h" "$STAGE/include/gpnec.h"
printf '%s\n' "$VERSION" > "$STAGE/VERSION"
cp "$ROOT/LICENSE" "$STAGE/LICENSE"
cp "$ROOT/README.md" "$STAGE/README.md"

echo "==> Packing $TARBALL"
tar -C "$STAGE" -czf "$TARBALL" lib include VERSION LICENSE README.md

(
  cd "$DIST"
  shasum -a 256 "gpnec-${VERSION}-macos-universal.tar.gz" > SHA256SUMS
  # Also emit a single-line helper for Homebrew formula bumps
  awk '{print $1}' SHA256SUMS > "gpnec-${VERSION}.sha256"
)

echo ""
echo "Release artifact ready:"
echo "  $TARBALL"
echo "  sha256: $(cat "$DIST/gpnec-${VERSION}.sha256")"
echo ""
echo "Next:"
echo "  gh release create v${VERSION} \"$TARBALL\" \"$DIST/SHA256SUMS\" \\"
echo "    --repo Digital-Defiance/gpnec --title \"GPNEC ${VERSION}\" --notes-file CHANGELOG.md"
