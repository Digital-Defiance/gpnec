#!/usr/bin/env bash
# Build Swift C-ABI bridge as a dylib for the Rust host.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/host/target/bridge"
mkdir -p "$OUT"

cd "$ROOT"
swift build -c release --product GPNECCBridge
BUILD=$(swift build -c release --show-bin-path)

# Package produces .a / modules; synthesize a dylib from the bridge + deps via swiftc if needed.
# Prefer copying the built module artifacts and linking through swift build's produced binary.
if [[ -f "$BUILD/libGPNECCBridge.dylib" ]]; then
  cp "$BUILD/libGPNECCBridge.dylib" "$OUT/"
elif [[ -f "$BUILD/GPNECCBridge.o" ]]; then
  echo "object found but no dylib; use SPM dynamic library product"
fi

# Ensure Package exposes a dynamic library — rebuild instruction:
# The Package.swift library product is static by default; emit-library via:
swiftc -emit-library -o "$OUT/libGPNECCBridge.dylib" \
  -module-name GPNECCBridge \
  -I "$BUILD/Modules" \
  -L "$BUILD" \
  -lGPNECCore -lGPNECAdapters \
  "$ROOT/Sources/GPNECCBridge/Bridge.swift" \
  -parse-as-library \
  -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  -framework Foundation \
  2>/dev/null || {
    # Fallback: tell Rust host to use DYLD via built package path for development CLI instead.
    echo "Note: prefer \`swift run gpnec\` for demos; dylib bridge is best-effort."
  }

echo "Bridge artifacts in $OUT"
ls -la "$OUT" || true
echo "Swift bin path: $BUILD"
ls -la "$BUILD" | head -40
