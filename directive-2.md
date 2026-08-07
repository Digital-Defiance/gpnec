Context: `@GPNECCBridge` and our `lbm` adapter. I need to build a native macOS `MTKView` frontend to visualize the fluid dynamics. The compute pipeline is updating an `MTLBuffer` at 723 µs/step.

1. Outline a MECE plan to bind this exact output `MTLBuffer` to a Metal fragment shader for zero-copy rendering.
2. Include a real-time diagnostic overlay that compares the backend compute time against the display frame rate. Do not write the implementation code until I approve the architecture.