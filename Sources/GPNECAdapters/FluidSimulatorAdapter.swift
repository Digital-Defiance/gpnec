import Foundation
import GPNECCore
import Metal

/// Adapter A — true D2Q9 Lattice Boltzmann (Metal collide + stream, bounce-back obstacle).
public struct FluidSimulatorAdapter: EngineAdapter {
    public let domain: AdapterDomain = .latticeBoltzmann
    public let width: Int
    public let height: Int
    public let batch: Int
    public let tau: Float
    public let inletUx: Float

    public init(
        width: Int = 256,
        height: Int = 256,
        batch: Int = 1,
        tau: Float = 0.56,
        inletUx: Float = 0.08
    ) {
        self.width = width
        self.height = height
        self.batch = batch
        self.tau = tau
        self.inletUx = inletUx
    }

    public func makeEngine(context: MetalContext) throws -> TensorEngine {
        let nodes = width * height
        let channels = 10 // D2Q9 + dye
        let shape = TensorShape(batch: batch, nodes: nodes, channels: channels)

        let w9: [Float] = [
            4.0 / 9.0,
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        ]

        // Circular obstacle (cylinder in cross-flow) — classic von Kármán setup.
        var solid = [Float](repeating: 0, count: nodes)
        let cx = width / 4
        let cy = height / 2
        let radius = max(width / 16, 8)
        let r2 = Float(radius * radius)
        for y in 0..<height {
            for x in 0..<width {
                let dx = Float(x - cx)
                let dy = Float(y - cy)
                if dx * dx + dy * dy <= r2 {
                    solid[y * width + x] = 1
                }
            }
        }

        var state = [Float](repeating: 0, count: shape.elementCount)
        for b in 0..<batch {
            for y in 0..<height {
                for x in 0..<width {
                    let n = y * width + x
                    let base = (b * nodes + n) * channels
                    if solid[n] > 0.5 {
                        for q in 0..<9 { state[base + q] = w9[q] }
                        state[base + 9] = 0
                        continue
                    }
                    let rho: Float = 1.0
                    let ux = inletUx
                    let usq = ux * ux
                    for q in 0..<9 {
                        let ex: Float = [0, 1, 0, -1, 0, 1, -1, -1, 1][q]
                        // uy = 0 for uniform inlet IC; ey unused in eu.
                        let eu = ex * ux
                        state[base + q] = w9[q] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usq)
                    }
                    state[base + 9] = (x < 3 && (y % 32) < 6) ? 0.9 : 0
                }
            }
        }

        let uniforms = PhiUniforms(
            shape: shape,
            param0: tau,
            param1: Float(width),
            param2: Float(height),
            param3: inletUx
        )
        let collide = try MetalPhiKernel(
            context: context,
            kernelName: "lbm_bgk_collide",
            sourceFile: "PhiLBM.metal",
            uniforms: uniforms,
            auxData: solid
        )
        let stream = try MetalPhiKernel(
            context: context,
            kernelName: "lbm_stream_bounce",
            sourceFile: "PhiLBM.metal",
            uniforms: uniforms,
            auxData: solid
        )
        return try TensorEngine(
            context: context,
            shape: shape,
            domain: domain,
            collide: collide,
            stream: stream,
            initialState: state
        )
    }
}
