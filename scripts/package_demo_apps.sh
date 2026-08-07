#!/usr/bin/env bash
# Build universal GPNEC Fluid + Route demo apps and pack for Homebrew cask.
#
# Usage:
#   ./scripts/package_demo_apps.sh           # version from VERSION
#   ./scripts/package_demo_apps.sh 0.1.0
#   ./scripts/package_demo_apps.sh 0.1.0 --skip-notarize
#
# Signing / notarization (Developer ID — same env as Lattice / Warp):
#   APPLE_SIGNING_IDENTITY   "Developer ID Application: … (TEAMID)"
#   APPLE_API_KEY + APPLE_API_ISSUER + APPLE_API_KEY_PATH
#     OR APPLE_ID + APPLE_PASSWORD (app-specific) + APPLE_TEAM_ID
#
# Output:
#   dist/gpnec-demos-<ver>-macos-universal.zip
#   dist/gpnec-demos-<ver>.sha256
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION=""
SKIP_NOTARIZE=0
for arg in "$@"; do
  case "$arg" in
    --skip-notarize) SKIP_NOTARIZE=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "$VERSION" && "$arg" != --* ]]; then
        VERSION="$arg"
      fi
      ;;
  esac
done
if [[ -z "$VERSION" ]]; then
  VERSION="$(tr -d '[:space:]' < VERSION)"
fi
[[ -n "$VERSION" ]] || { echo "VERSION empty" >&2; exit 1; }

DIST="$ROOT/dist"
STAGE="$DIST/stage-demos-$VERSION"
ZIP="$DIST/gpnec-demos-${VERSION}-macos-universal.zip"
ENTITLEMENTS="$ROOT/scripts/Entitlements.DeveloperID.plist"
IDENTITY="${APPLE_SIGNING_IDENTITY:-}"

mkdir -p "$DIST"
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"

build_product_arch() {
  local product="$1" arch="$2"
  swift build -c release --product "$product" --arch "$arch" >&2
  swift build -c release --product "$product" --arch "$arch" --show-bin-path
}

sign_app() {
  local app_dir="$1"
  if [[ -z "$IDENTITY" ]]; then
    echo "  (ad-hoc sign — Gatekeeper will block downloads; set APPLE_SIGNING_IDENTITY)" >&2
    codesign --force --deep --sign - "$app_dir"
    return
  fi
  echo "  codesign → $IDENTITY" >&2
  codesign --force --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" \
    --sign "$IDENTITY" \
    "$app_dir/Contents/MacOS/"*
  codesign --force --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" \
    --sign "$IDENTITY" \
    "$app_dir"
  codesign --verify --deep --strict --verbose=2 "$app_dir" >&2
}

can_notarize() {
  [[ "$SKIP_NOTARIZE" -eq 0 && -n "$IDENTITY" ]] || return 1
  if [[ -n "${APPLE_API_KEY:-}" && -n "${APPLE_API_ISSUER:-}" && -n "${APPLE_API_KEY_PATH:-}" ]]; then
    return 0
  fi
  if [[ -n "${APPLE_ID:-}" && -n "${APPLE_PASSWORD:-}" && -n "${APPLE_TEAM_ID:-}" ]]; then
    return 0
  fi
  return 1
}

notarize_zip() {
  local zip_path="$1"
  local args=(submit "$zip_path" --wait --progress)
  if [[ -n "${APPLE_API_KEY:-}" && -n "${APPLE_API_ISSUER:-}" && -n "${APPLE_API_KEY_PATH:-}" ]]; then
    echo "==> Notarizing with App Store Connect API key…"
    args+=(--key "$APPLE_API_KEY_PATH" --key-id "$APPLE_API_KEY" --issuer "$APPLE_API_ISSUER")
  else
    echo "==> Notarizing with Apple ID…"
    args+=(--apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID")
  fi
  xcrun notarytool "${args[@]}"
}

staple_apps() {
  local app
  for app in "$STAGE/GPNEC Fluid.app" "$STAGE/GPNEC Route.app"; do
    echo "==> Stapling $(basename "$app")…"
    xcrun stapler staple "$app"
    xcrun stapler validate "$app"
  done
}

write_zip() {
  rm -f "$ZIP"
  (
    cd "$STAGE"
    zip -ry "$ZIP" "GPNEC Fluid.app" "GPNEC Route.app" VERSION LICENSE README.txt
  )
}

make_app() {
  local product="$1"
  local display_name="$2"
  local bundle_id="$3"
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
  sign_app "$app_dir"
}

echo "==> Building gpnec-fluid (arm64 + x86_64)…"
ARM_FLUID="$(build_product_arch gpnec-fluid arm64)"
X86_FLUID="$(build_product_arch gpnec-fluid x86_64)"

echo "==> Building gpnec-route (arm64 + x86_64)…"
ARM_ROUTE="$(build_product_arch gpnec-route arm64)"
X86_ROUTE="$(build_product_arch gpnec-route x86_64)"

make_app gpnec-fluid "GPNEC Fluid" "org.digitaldefiance.gpnec.fluid" "$ARM_FLUID" "$X86_FLUID"
make_app gpnec-route "GPNEC Route" "org.digitaldefiance.gpnec.route" "$ARM_ROUTE" "$X86_ROUTE"

printf '%s\n' "$VERSION" > "$STAGE/VERSION"
cp "$ROOT/LICENSE" "$STAGE/LICENSE"
cat > "$STAGE/README.txt" <<EOF
GPNEC Demos ${VERSION}

  GPNEC Fluid.app  — Lattice Boltzmann MTKView (click/drag droplets)
  GPNEC Route.app  — Euclidean vs Poincaré routing sandbox (CRASH BACKBONE)

Requires macOS 14+ with Metal.
Sensor Net for Subspace Lattice: brew install gpnec
EOF

echo "==> Zipping $ZIP"
write_zip

if can_notarize; then
  notarize_zip "$ZIP"
  staple_apps
  echo "==> Re-zipping stapled apps…"
  write_zip
elif [[ "$SKIP_NOTARIZE" -eq 1 ]]; then
  echo "==> Skipping notarization (--skip-notarize)"
else
  echo "==> Skipping notarization (need APPLE_SIGNING_IDENTITY + API key or Apple ID)"
fi

(
  cd "$DIST"
  shasum -a 256 "gpnec-demos-${VERSION}-macos-universal.zip" | tee "gpnec-demos-${VERSION}.sha256"
)

echo ""
echo "Demo package ready:"
echo "  $ZIP"
echo "  sha256: $(awk '{print $1}' "$DIST/gpnec-demos-${VERSION}.sha256")"
if can_notarize; then
  echo "  signed + notarized + stapled (Developer ID)"
fi
