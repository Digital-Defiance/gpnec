import Foundation
import GPNECCore

/// Adapter B — Non-Euclidean subspace lattice board with control diffusion.
public struct SubspaceLatticeAdapter: EngineAdapter {
    public let domain: AdapterDomain = .subspaceLattice
    public let width: Int
    public let height: Int
    public let batch: Int
    public let temperature: Float
    public let controlGain: Float

    public init(
        width: Int = 11,
        height: Int = 11,
        batch: Int = 1,
        /// Retain mix for diffused Sensor Net control (high = keep bloom).
        temperature: Float = 0.9,
        controlGain: Float = 2.0
    ) {
        self.width = width
        self.height = height
        self.batch = batch
        self.temperature = temperature
        self.controlGain = controlGain
    }

    public func makeEngine(context: MetalContext) throws -> TensorEngine {
        let nodes = width * height
        let channels = 4 // occupancy, control, curvature_x, curvature_y
        let shape = TensorShape(batch: batch, nodes: nodes, channels: channels)
        let W = TopologyBuilder.subspaceAdjacency(width: width, height: height)

        var state = [Float](repeating: 0, count: shape.elementCount)
        for b in 0..<batch {
            // Place opposing pieces in opposite corners; curvature from coordinates.
            let p1 = 1 * width + 1
            let p2 = (height - 2) * width + (width - 2)
            for n in 0..<nodes {
                let x = n % width
                let y = n / width
                let base = (b * nodes + n) * channels
                if n == p1 { state[base] = 1 }
                if n == p2 { state[base] = -1 }
                state[base + 1] = 0
                // Local manifold curvature vectors
                state[base + 2] = sinf(Float(x) * 0.4) * 0.5
                state[base + 3] = cosf(Float(y) * 0.4) * 0.5
            }
        }

        let uniforms = PhiUniforms(shape: shape, param0: temperature, param1: controlGain)
        let phi = try MetalPhiKernel(
            context: context,
            kernelName: "phi_subspace_diffusion",
            sourceFile: "PhiDiffusion.metal",
            uniforms: uniforms
        )
        return try TensorEngine(
            context: context,
            shape: shape,
            domain: domain,
            topology: W,
            phi: phi,
            initialState: state
        )
    }
}
