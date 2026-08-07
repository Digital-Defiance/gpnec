# PRD: Tensor-Mapped Graph Engine (Non-Euclidean State Automaton)

## 1. System Architecture & Philosophy

This project subverts standard machine learning hardware primitives to build a high-speed, general-case physics and graph simulation engine for Apple Silicon (M4 Max).

Instead of training a neural network, this engine uses a hand-crafted `MPSGraph` to execute a non-Euclidean state automaton. The core operation is a recurrent matrix multiplication and activation loop: $X_{t+1} = \Phi(X_t, W, \theta)$, where the weights ($W$) encode spatial geometry and the activation function ($\Phi$) enforces discrete local physics rules.

**Key Constraints:**

- **Zero-Copy Execution:** The engine must use double-buffered `MTLBuffer` ping-ponging. Frame 1's output buffer becomes Frame 2's input buffer. Memory must remain on the GPU/Unified Memory.
- **No ML Frameworks:** Do not use CoreML, MLX, or PyTorch. The entire pipeline must be written in raw Metal / `MPSGraph` via Swift or Objective-C++ bound to a Rust/C++ host.
- **Custom Activation Shaders:** The transition function $\Phi$ must be implemented as a custom Metal kernel (`.metal` file) hooked into the `MPSGraph` pipeline.

## 2. Core Engine Mathematical Spec

The core engine evaluates the following state across a batched pipeline:

- **State Tensor ($X_t$):** A multidimensional tensor representing the global state (e.g., node coordinates, particle densities). Dimensions: `[Batch, Nodes/Cells, Channels]`.
- **Topology Matrix ($W$):** A constant tensor representing the exact topology of the space (Adjacency Matrix, Shift Operators, or Distance Kernels).
- **Transition Kernel ($\Phi$):** A custom operation that evaluates local collision, distance calculation, or state resolution.

## 3. The Three Application Adapters

The engine must be abstracted so that swapping the `W` tensor and the `$\Phi$` kernel completely changes the domain. We will target three specific implementations:

### Adapter A: High-Speed Fluid Simulator (Lattice Boltzmann)

- **Domain:** Chaotic fluid dynamics mapped to tensor operations.
- **$X_t$:** Discrete particle velocity distribution functions on a 2D/3D lattice.
- **$W$ (Streaming):** Discrete shift operators that physically move distributions to adjacent lattice nodes.
- **$\Phi$ (Collision):** The BGK relaxation operator calculated locally per node to conserve mass and momentum.

### Adapter B: Subspace Lattice Game Engine

- **Domain:** Real-time state evaluation for a non-Euclidean board game.
- **$X_t$:** Board state tensor (piece positions) plus local manifold curvature vectors.
- **$W$ (Topology):** A non-Euclidean adjacency matrix defining valid piece movement and influence radius.
- **$\Phi$ (Rules/Diffusion):** A min-max evaluation or heat-diffusion kernel calculating positional control across the lattice.

### Adapter C: Hyperbolic Network Router (BrightChain Mesh)

- **Domain:** Decentralized, ultra-low-latency network routing in a Poincaré disk manifold.
- **$X_t$:** Tensor containing hyperbolic coordinates of active mesh nodes and destination vectors for in-flight BrightSpace data packets.
- **$W$ (Adjacency):** Active edge connections between BrightChain peers.
- **$\Phi$ (Greedy Distance Selector):** Calculates the hyperbolic distance between local neighbors and the packet destination, instantly passing the state to the mathematically closest node.
