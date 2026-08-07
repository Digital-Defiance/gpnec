#!/usr/bin/env bash
# Wrap the SPM binary in a minimal .app so Dock / window server treat it as a GUI app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
swift build --product gpnec-fluid "$@"
BIN="$(swift build --product gpnec-fluid --show-bin-path)/gpnec-fluid"
APP="$ROOT/.build/gpnec-fluid.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>gpnec-fluid</string>
  <key>CFBundleIdentifier</key><string>computer.gpnec.fluid</string>
  <key>CFBundleName</key><string>GPNEC Fluid</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.2.1</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
cp "$BIN" "$APP/Contents/MacOS/gpnec-fluid"
chmod +x "$APP/Contents/MacOS/gpnec-fluid"
echo "Built $APP"
echo "Launch with: open $APP"
