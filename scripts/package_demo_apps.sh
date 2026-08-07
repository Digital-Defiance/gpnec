#!/usr/bin/env bash
# Build universal GPNEC Fluid + Route demo apps and pack for Homebrew cask.
#
# Usage:
#   ./scripts/package_demo_apps.sh           # version from VERSION
#   ./scripts/package_demo_apps.sh 0.1.0
#
# Output:
#   dist/gpnec-demos-<ver>-macos-universal.zip
#   dist/gpnec-demos-<ver>.sha256
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(tr -d '[:space:]' < VERSION)"
fi
[[ -n "$VERSION" ]] || { echo "VERSION empty" >&2; exit 1; }

DIST="$ROOT/dist"
STAGE="$DIST/stage-demos-$VERSION"
ZIP="$DIST/gpnec-demos-${VERSION}-macos-universal.zip"
mkdir -p "$DIST"
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"

build_product_arch() {
  local product="$1" arch="$2"
  swift build -c release --product "$product" --arch "$arch" >&2
  swift build -c release --product "$product" --arch "$arch" --show-bin-path
}

make_app() {
  local product="$1"        # gpnec-fluid | gpnec-route
  local display_name="$2"   # GPNEC Fluid | GPNEC Route
  local bundle_id="$3"      # computer.gpnec.fluid
  local arm_bin="$4"
  local x86_bin="$5"
  local app_dir="$STAGE/${display_name}.app"

  rm -rf "$app_dir"
  mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources"

  cat > "$app_dir/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>${product}</string>
  <key>CFBundleIdentifier</key><string>${bundle_id}</string>
  <key>CFBundleName</key><string>${display_name}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

  local out_bin="$app_dir/Contents/MacOS/${product}"
  lipo -create \
    "${arm_bin}/${product}" \
    "${x86_bin}/${product}" \
    -output "$out_bin"
  chmod +x "$out_bin"
  lipo -info "$out_bin"

  # Ad-hoc sign so Gatekeeper is less angry for Homebrew users.
  codesign --force --deep --sign - "$app_dir" 2>/dev/null || true
}

echo "==> Building gpnec-fluid (arm64 + x86_64)…"
ARM_FLUID="$(build_product_arch gpnec-fluid arm64)"
X86_FLUID="$(build_product_arch gpnec-fluid x86_64)"

echo "==> Building gpnec-route (arm64 + x86_64)…"
ARM_ROUTE="$(build_product_arch gpnec-route arm64)"
X86_ROUTE="$(build_product_arch gpnec-route x86_64)"

make_app gpnec-fluid "GPNEC Fluid" "computer.gpnec.fluid" "$ARM_FLUID" "$X86_FLUID"
make_app gpnec-route "GPNEC Route" "computer.gpnec.route" "$ARM_ROUTE" "$X86_ROUTE"

printf '%s\n' "$VERSION" > "$STAGE/VERSION"
cp "$ROOT/LICENSE" "$STAGE/LICENSE"
cat > "$STAGE/README.txt" <<EOF
GPNEC Demos ${VERSION}

  GPNEC Fluid.app  — Lattice Boltzmann MTKView (click/drag droplets)
  GPNEC Route.app  — Euclidean vs Poincaré routing sandbox (CRASH BACKBONE)

Requires macOS 14+ with Metal (Apple Silicon or Intel with Metal).
Subspace Lattice Sensor Net lives in the game + \`brew install gpnec\` bridge.
EOF

echo "==> Zipping $ZIP"
(
  cd "$STAGE"
  zip -ry "$ZIP" "GPNEC Fluid.app" "GPNEC Route.app" VERSION LICENSE README.txt
)

(
  cd "$DIST"
  shasum -a 256 "gpnec-demos-${VERSION}-macos-universal.zip" | tee "gpnec-demos-${VERSION}.sha256"
)

echo ""
echo "Demo package ready:"
echo "  $ZIP"
echo "  sha256: $(awk '{print $1}' "$DIST/gpnec-demos-${VERSION}.sha256")"
