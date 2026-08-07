Context: `@GPNECCBridge`, `@subspace`, and our Tauri frontend directory. I need to visualize the non-Euclidean rules of our board game. When a piece is placed, the backend calculates a `control field L1` tensor.

1. Outline the FFI plan to pass this tensor array from the Rust backend to the frontend efficiently.
2. Design the React/WebGL integration to render this data as a topographical heat map or warped grid beneath the game pieces. Detail the data handoff before writing the code.