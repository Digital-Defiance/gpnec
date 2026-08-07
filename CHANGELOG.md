# Changelog

## 0.2.1 —

- Fix Swift 6 concurrency in RouteSandbox so CI can build gpnec-route demos

## 0.2.0 — verification gates + fair routing control

- `gpnec verify-lbm` / `verify-route` / `verify` — Metal≡CPU accuracy gates (exit ≠0 on failure)
- Dual-route **symmetric** post-crash policy (identical retry+respawn) as the publication control; sandbox (Euclidean freeze) reported as UI narrative only
- `route-verify --policy symmetric|sandbox|both`
- External CPU D2Q9 baseline for LBM bench (`cpu` / optional `cpu-mt`) with scoped speedup framing
- Technical report: `papers/gpnec/` (methods, paired crash tables, bandwidth estimate, limitations)

## 0.1.0 — first release

- Core Metal / MPSGraph engine with double-buffered state
- Adapters: LBM fluid, subspace Sensor Net diffusion, hyperbolic router
- Dual Euclidean / Poincaré routing sandbox + `route-verify`
- Zero-copy fluid and route `MTKView` demos
- C ABI bridge (`libGPNECCBridge.dylib`) for Subspace Lattice / Rust hosts
- Homebrew: `brew install gpnec` (bridge) · `brew install --cask gpnec-demos` (Fluid + Route apps)
