# GPNEC — General Purpose Non-Euclidean Computer

Hand-crafted **Metal / MPSGraph** state automaton for Apple Silicon. Not an ML framework: no CoreML, MLX, or PyTorch. The core loop is

\[
X_{t+1} = \Phi(X_t, W, \theta)
\]

where \(W\) encodes geometry (adjacency, streaming shifts, mesh edges) and \(\Phi\) is a custom `.metal` kernel (collision, diffusion, or greedy hyperbolic hop).

**Target platform:** macOS 14+ with Metal (validated on Apple M4 Max). The Swift package and native demos do not build on Windows/Linux.

---

## Quick start

```bash
# Tests (needs a Metal device)
swift test

# CLI adapter smoke runs
swift run gpnec --adapter lbm --steps 32
swift run gpnec --adapter subspace --steps 32
swift run gpnec --adapter hyperbolic --steps 16

# Visual demos (SPM binaries wrapped as .app for Dock / window server)
./demo-fluid.sh     # Lattice Boltzmann MTKView
./demo-network.sh   # Dual Euclidean / Poincaré routing sandbox
```

---

## What lives here

| Piece | Role |
|-------|------|
| `GPNECCore` | Double-buffered `MTLBuffer` ping-pong, `MetalContext`, `TensorEngine`, MPSGraph topology path |
| `GPNECAdapters` | Domain adapters: LBM fluid, subspace lattice, hyperbolic router |
| `GPNECRouting` | Dual embedding (Poincaré + landmark MDS), Metal greedy routers, crash verification |
| `GPNECFluidView` / `GPNECFluidApp` | Zero-copy LBM → `MTKView` (`gpnec-fluid`) |
| `GPNECRouteView` / `GPNECRouteApp` | Split-screen routing sandbox (`gpnec-route`) |
| `GPNECCBridge` | C ABI (`libGPNECCBridge.dylib`) for Rust / Tauri hosts |
| `GPNECCLI` (`gpnec`) | Adapter demos, LBM bench, `route-verify` |
| `host/` | Minimal Rust host over the C ABI |
| `scripts/` | Bridge build + `.app` wrappers for GUI products |
| `demo-fluid.sh` / `demo-network.sh` | One-shot build + open helpers |

Directives (`directive-1.md` … `directive-4.md`) are the product briefs that drove each phase.

---

## Adapters

| ID | CLI `--adapter` | Domain |
|----|-----------------|--------|
| A | `lbm` | D2Q9 Lattice Boltzmann (Metal collide → stream) |
| B | `subspace` | Non-Euclidean board + Sensor Net control diffusion |
| C | `hyperbolic` | Poincaré-disk mesh routing (BrightChain-style) |

Swap \(W\) and \(\Phi\); the engine API stays the same.

Directive 4’s **visual** dual-greedy sandbox (`gpnec-route`) is a richer routing stack in `GPNECRouting` / `GPNECRouteView`, separate from the smaller `hyperbolic` CLI adapter demo.

---

## Prerequisites

- macOS 14+
- Xcode / Command Line Tools (Swift 6, Metal toolchain)
- Optional: Rust (`rustup`) for `host/` and for Subspace Lattice desktop integration

```bash
swift --version
xcrun -sdk macosx metal --version
```

---

## Build & run

### CLI

```bash
swift test

# Adapter demos
swift run gpnec --adapter lbm --steps 32
swift run gpnec --adapter subspace --steps 32
swift run gpnec --adapter hyperbolic --steps 16

# LBM throughput — Metal vs external CPU baseline (default: metal,cpu)
swift run gpnec bench
swift run gpnec bench --backends metal,cpu,cpu-mt
swift run gpnec bench --sizes 32,64,128,256 --steps 128 --warmup 32

# Accuracy gates (exit ≠0 on failure)
swift run gpnec verify-lbm
swift run gpnec verify-lbm --size 128 --steps 64 --l2 1e-4
swift run gpnec verify-route
swift run gpnec verify          # verify-lbm then verify-route
swift run gpnec bench --csv > bench.csv

# Dual-route scenario (no UI) — Euclidean freeze vs Poincaré respawn post-crash
swift run gpnec route-verify
swift run gpnec help
```

**External baselines & accuracy**

| Backend | Role |
|---------|------|
| `cpu` | Single-thread Swift D2Q9 mirroring `PhiLBM.metal` — **accuracy gold** (not a tuned CFD baseline) |
| `cpu-mt` | Same physics, multi-thread rows — optional; often *slower* (memory-bound) |
| `dense` | Optional internal O(N²) MPSGraph cost model (not identical physics) |

| Command | What it proves |
|---------|----------------|
| `verify-lbm` | Metal LBM ≡ CPU D2Q9 (relative L2 + mass + rest equilibrium) |
| `verify-route` | Embedding + Metal ≡ CPU hops + **symmetric** crash (sandbox reported only) |
| `verify` / `verify-all` | Both of the above |
| `route-verify` | Crash stats; default `--policy symmetric`. Use `both` / `sandbox` for UI narrative |

`gpnec verify-lbm` compares full 10-channel state after N steps (relative L2 + max abs + mass + rest-equilibrium drift). Exit `0` / `verified: true` when Metal matches CPU within threshold (default L2 ≤ 1e-4).

`gpnec verify-route` proves Metal≡CPU hop lockstep, then a **symmetric** post-crash control (identical retry+respawn). The Route app’s Euclidean-freeze protocol is reported as sandbox narrative—not the publication gate.

CUDA / OpenFOAM are not same-device baselines on Apple Silicon. Publishable claim: Metal vs matched CPU D2Q9 accuracy (`verify-lbm`); router hop parity + fair crash control (`verify-route`).

### Fluid viewer (directive 2)

```bash
./demo-fluid.sh
# equivalent:
./scripts/run_fluid_app.sh && open .build/gpnec-fluid.app
```

- Default lattice **256×256**, Metal collide+stream, cylinder wake + inlet dye.
- Click / drag injects droplets.
- SwiftUI HUD: steps/frame budget, compute ms, display FPS.
- SPM CLI binaries default to `.accessory` activation; the script wraps the binary in a minimal `.app` so the window surfaces correctly.

### Routing sandbox (directive 4)

```bash
./demo-network.sh
# equivalent:
./scripts/run_route_app.sh && open .build/gpnec-route.app
```

**What you see**

| Side | Geometry | Metric | Packets |
|------|----------|--------|---------|
| Left | Landmark-MDS plane of the same graph | Euclidean (\(L_2\)) | Amber |
| Right | Poincaré disk (generative H² sample) | Poincaré distance | Cyan |

**Simulation defaults (app)**

| Parameter | Default | Notes |
|-----------|---------|--------|
| \(N\) | 10 000 | Nodes |
| \(K\) | 12 | Fixed-K neighbor list (symmetrized Poincaré K-NN) |
| Crash set | ~800 | Vertical cut through the Euclidean MDS mid-plane (high-betweenness nodes in the strip) |
| Packets | 2048 | Uniform random live source–destination pairs |
| Pre-crash stuck handling | Silent retry after ~40 dwell hops | Recycles local-minima slots so Euclidean throughput does not die |
| Post-crash | Euclidean **stops respawning**; stuck packets freeze | Poincaré keeps cycling new SD pairs |
| Dead-node packets | Hard drop | Physical router failure; only source of `dropped` |

**How to demo**

1. Wait ~7–10 s for embedding (Poincaré K-NN + betweenness + MDS).
2. Confirm both panels’ `delivered` counters climb (Euclidean slower is expected).
3. Press **CRASH BACKBONE**.
4. Amber traffic bottlenecks / freezes; cyan keeps delivering around the damage.

**Offline math check (no UI)**

```bash
swift run gpnec route-verify \
  --n 800 --k 10 --crash 250 --packets 512 \
  --pre 80 --post 120 --seed 42 --policy both
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n` | 10000 | Node count |
| `--k` | 12 | Neighbors per node |
| `--crash` | 800 | Size of MDS-cut crash set |
| `--packets` | 2048 | Concurrent SD pairs |
| `--pre` / `--post` | 80 / 120 | Greedy ticks before / after crash |
| `--seed` | 1 | Graph + traffic seed |
| `--betweenness-samples` | 256 | Sampling Brandes sources |
| `--max-hops` | 0 | Hop TTL; `0` = unlimited (drops only on dead nodes) |
| `--policy` | `symmetric` | `symmetric` = identical retry+respawn (fair); `sandbox` = UI euc freeze; `both` = print both |

Exit `0` when the selected policy gate passes. **Symmetric** is the scientific control. **Sandbox** matches the app (Euclidean freeze) and is a demo narrative, not an embedding-only proof.

### C ABI bridge (Rust / Tauri)

```bash
./scripts/build_bridge.sh
# Prefer the SPM dynamic product when present:
swift build -c release --product GPNECCBridge
# Dylib typically under:
#   .build/arm64-apple-macosx/{debug,release}/libGPNECCBridge.dylib
```

Headers: [`host/include/gpnec.h`](host/include/gpnec.h).

```bash
cd host && cargo run -- --adapter lbm --steps 32
```

Important bridge entry points for the board game:

| Symbol | Purpose |
|--------|---------|
| `gpnec_create_subspace(w, h)` | Sized subspace engine (e.g. 11×11) |
| `gpnec_seed_sensor_net` | Occupancy + control seed from host |
| `gpnec_step` | Diffuse / advance |
| `gpnec_read_channel(…, 1, …)` | Control field (Sensor Net) |
| `gpnec_control_l1` | \(\sum|\mathrm{control}|\) |

---

## Dual routing stack (directive 4) — internals

```
H² area sample in Poincaré disk
        │
        ▼
Fixed-K Poincaré nearest neighbors (symmetrized)
        │
        ├─► Landmark hop-distance MDS → Euclidean positions (left)
        └─► Same points → Poincaré metric (right)
        │
        ▼
Approx. betweenness (sampling Brandes)
        │
        ▼
Crash targets = high-betweenness nodes in Euclidean mid-strip
        │
        ▼
Metal kernel dual_greedy_route_step
  • hard-drop if current/dest node dead
  • greedy hop under L2 or Poincaré distance
  • local min → dwell; optional silent retry (flag bit 8)
```

Zero-copy viz: node / packet **point instancing** reads the same `MTLBuffer`s the routers use (`positions`, `alive`, `packets`).

---

## Homebrew (stable)

```bash
brew tap digital-defiance/tap

# Bridge for Subspace Lattice Sensor Net (dylib only)
brew install gpnec

# Fluid + Route demo apps (not Subspace Lattice — that's the game)
brew install --cask gpnec-demos
open -a "GPNEC Fluid"
open -a "GPNEC Route"
```

| Package | What you get |
|---------|----------------|
| `gpnec` (formula) | `libGPNECCBridge.dylib` + `gpnec.h` |
| `gpnec-demos` (cask) | **GPNEC Fluid.app** + **GPNEC Route.app** |

---

## Releasing

```bash
# 1. Bump VERSION + CHANGELOG.md
# 2. Commit, push main, tag:
git tag v0.2.0 && git push origin v0.2.0
# CI builds dist/gpnec-0.2.0-macos-universal.tar.gz and creates the GitHub Release.

# Or locally:
./scripts/release_bridge.sh 0.2.0
./scripts/package_demo_apps.sh 0.2.0
gh release create v0.2.0 \
  dist/gpnec-0.2.0-macos-universal.tar.gz \
  dist/gpnec-demos-0.2.0-macos-universal.zip \
  dist/SHA256SUMS \
  --repo Digital-Defiance/gpnec --title "GPNEC 0.2.0" --notes-file CHANGELOG.md
# Then bump sha256 in digital-defiance/homebrew-tap Formula/gpnec.rb + Casks/gpnec-demos.rb
```

---

## Signing & notarization (demos cask)

Gatekeeper requires **Developer ID** + **notarization** for apps downloaded via Homebrew.

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Digital Defiance (J6887N729S)"
# App Store Connect API key (preferred):
export APPLE_API_KEY=…
export APPLE_API_ISSUER=…
export APPLE_API_KEY_PATH=~/private_keys/AuthKey_….p8

./scripts/package_demo_apps.sh 0.2.0
# → dist/gpnec-demos-….zip  (signed, notarized, stapled)
```

Same credentials as Lattice / Warp. CI release workflow should inject these as secrets for tagged builds.

---

## Subspace Lattice integration (directive 3)

The board game lives in a **sibling repo** (Nx + Tauri), typically checked out next to this tree as `IWGF/subspace-lattice`. Rules and Sensor Net authority stay in `@subspace-lattice/core`. GPNEC only **diffuses** the control field for the WebGL heat map under the pieces.

### Do not use a git branch for “optional Apple Silicon”

A long-lived branch that deletes Metal/GPNEC code splits CI and still leaves Windows/Linux needing a fallback. Prefer **compile-time opt-in + runtime fallback** in the game repo:

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| **1. Cargo feature** | `gpnec` on `apps/desktop/src-tauri` (default on; disable with `--no-default-features`) | Windows / Linux / iOS / Android never need Metal. |
| **2. `cfg(target_os = "macos")`** | Engine module + `libloading` only on Darwin | Non-Mac targets get stubs (`gpnec_available → false`). |
| **3. Dynamic load** | `libloading` of `libGPNECCBridge.dylib` (never hard-link Metal into Tauri) | Missing dylib ≠ link failure. |
| **4. Runtime gate** | `canUseGpnecSensorNet()` + CPU bloom in `resolveSensorNetField` | Browser, non-Mac desktop, and failed Metal all keep a correct field. |

**Hardware detection alone is not enough for builds** — it only decides which path to run after a binary exists. Use features / `cfg` so non-Apple toolchains never compile Metal FFI.

### Data path (macOS Tauri + GPNEC)

```
pieces / Sensor Net sets
        │
        ▼
seed occupancy + control (Float32, length = N²)
        │
        ▼
Tauri command `sensor_net_sync`
        │
        ▼
libGPNECCBridge.dylib  →  subspace Φ (Metal)
        │
        ▼
control channel bytes → WebGL `SensorNetField`
```

On any other platform (or if the dylib / Metal init fails):

```
buildSensorNetField(...)  // CPU Jacobi bloom — same tensor shape
```

Env overrides / discovery for the dylib (game desktop):

| Source | Meaning |
|--------|---------|
| `GPNEC_BRIDGE_DYLIB` | Absolute path override |
| App `Resources/` | Bundled dylib (MAS / embedded releases) |
| **Homebrew** | `brew install gpnec` → `$(brew --prefix)/lib/libGPNECCBridge.dylib` |
| `GPNEC_ROOT` | Dev checkout; probes `.build/*/…` and `host/target/bridge/` |

### Homebrew: install GPNEC, get the coolness

Developer ID / `brew install --cask subspace-lattice` builds can load Homebrew dylibs
(`disable-library-validation`). After installing the formula, relaunch the game —
Metal Sensor Net diffusion turns on with no game rebuild.

```bash
brew tap digital-defiance/tap
brew install gpnec
open -a "Subspace Lattice"
```

Formula lives in the tap as `Formula/gpnec.rb`. The Lattice cask `depends_on formula: "gpnec"` so installing the game pulls the accelerator.

**Mac App Store** is sandboxed and cannot see `/opt/homebrew` — embed the dylib in
the `.app` for MAS. Web / Windows keep CPU bloom.

### Building the game without Apple Silicon

From the Subspace Lattice repo:

```bash
# Web / Functions / core — never touch Metal
yarn serve:web
yarn nx run-many -t lint test build typecheck

# Desktop on Windows / Linux — GPNEC stubs compile out
yarn tauri:dev
yarn tauri:build

# macOS desktop without GPNEC (optional feature off)
cd apps/desktop/src-tauri
cargo build --no-default-features
```

Ship Metal acceleration only in the macOS desktop artifact that bundles (or finds) `libGPNECCBridge.dylib`.

---

## Architecture constraints

From [`.cursor/rules/gpnec.mdc`](.cursor/rules/gpnec.mdc):

1. **Compute** = `MPSGraph` + custom Metal kernels only.
2. **No** CoreML / PyTorch / MLX.
3. **Zero-copy rendering** for native views: pass the live `MTLBuffer` into the fragment / point path (no CPU serialize for display).
4. **All Swift ↔ Rust FFI** goes through `GPNECCBridge`.

Directive 2 / 3 note the tension: native fluid/route apps keep GPU buffers; the Tauri WebGL path **must** copy a channel to the CPU once per sync (`gpnec_read_channel`). That copy is acceptable for an 11×11 board; do not generalize it into the fluid or route viewers.

---

## Package layout

```
Sources/
  GPNECCore/          engine, Metal context, Φ shaders (LBM / diffusion / hyperbolic)
  GPNECAdapters/      Fluid / Subspace / Hyperbolic + TopologyBuilder + FluidBench
  GPNECRouting/       DualEmbedding, DualGreedyRouter, DualRouteVerification, route shaders
  GPNECFluidView/     MTKView fluid shade + diagnostics
  GPNECRouteView/     MTKView routing sandbox + diagnostics
  GPNECCBridge/       @_cdecl C ABI
  GPNECCLI/           gpnec executable
  GPNECFluidApp/      gpnec-fluid (@main)
  GPNECRouteApp/      gpnec-route (@main)
Tests/GPNECCoreTests/ Engine smoke + DualRoute tests
host/                 Rust demo + include/gpnec.h
scripts/
  build_bridge.sh
  run_fluid_app.sh
  run_route_app.sh
demo-fluid.sh
demo-network.sh
directive-1.md … directive-4.md
```

### Products (`Package.swift`)

| Product | Type |
|---------|------|
| `GPNECCore`, `GPNECAdapters`, `GPNECRouting`, `GPNECFluidView`, `GPNECRouteView` | Libraries |
| `GPNECCBridge` | Dynamic library |
| `gpnec`, `gpnec-fluid`, `gpnec-route` | Executables |

---

## Design briefs

| File | Scope | Status in this repo |
|------|--------|---------------------|
| `directive-1.md` | PRD — engine math + three adapters | Core + adapters + CLI |
| `directive-2.md` | Zero-copy LBM → `MTKView` | `gpnec-fluid` |
| `directive-3.md` | Subspace Sensor Net → Tauri / React / WebGL | Bridge API + sibling game wiring |
| `directive-4.md` | Hyperbolic vs Euclidean routing sandbox | `gpnec-route` + `route-verify` |

## Technical report

LaTeX write-up of architecture, LBM/route verification protocols, measured
tables (M4 Max), and balanced scope limits:

```bash
cd papers/gpnec && latexmk -pdf gpnec.tex   # → gpnec.pdf
```

See `papers/gpnec/README.md` for regenerating CLI transcripts used in the tables.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `gpnec-fluid` / `gpnec-route` builds but no window | SPM accessory activation policy | Use `./demo-*.sh` or `scripts/run_*.sh` + `open .build/….app` |
| Route Euclidean `delivered` freezes ~few hundred before crash | Pre-crash local-minima filled all slots | Current builds silent-retry stuck packets; rebuild/reopen app |
| `route-verify` → `verified: false` | Weak pre-crash or no post-crash gap | Raise `--k` / `--pre`, or adjust `--crash` |
| `swift test` fails without GPU | No Metal device | Run on Apple Silicon Mac (or Mac with Metal) |
| Tauri cannot find bridge | Dylib not on loader path | Set `GPNEC_BRIDGE_DYLIB` or `GPNEC_ROOT` |

---

## License

MIT (see `host/Cargo.toml`). Game client licensing follows the Subspace Lattice / IWGF repos.
