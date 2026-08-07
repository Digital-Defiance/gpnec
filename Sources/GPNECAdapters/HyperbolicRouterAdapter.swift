import Foundation
import GPNECCore

/// Adapter C — Hyperbolic (Poincaré disk) BrightChain mesh router.
public struct HyperbolicRouterAdapter: EngineAdapter {
    public let domain: AdapterDomain = .hyperbolicRouter
    public let nodeCount: Int
    public let batch: Int
    public let curvatureScale: Float

    public init(nodeCount: Int = 32, batch: Int = 1, curvatureScale: Float = 1.0) {
        self.nodeCount = nodeCount
        self.batch = batch
        self.curvatureScale = curvatureScale
    }

    public func makeEngine(context: MetalContext) throws -> TensorEngine {
        let channels = 6
        let shape = TensorShape(batch: batch, nodes: nodeCount, channels: channels)

        // Ring + chords. W is identity: greedy Φ walks the aux adjacency explicitly
        // so a single packet remains coherent (no mass dilution).
        let K = 3
        var edges: [(Int, Int)] = []
        var aux = [Float](repeating: -1, count: nodeCount * K)
        for i in 0..<nodeCount {
            let n0 = (i + 1) % nodeCount
            let n1 = (i + 3) % nodeCount
            let n2 = (i + nodeCount / 2) % nodeCount
            let nbrs = [n0, n1, n2]
            for (k, nb) in nbrs.enumerated() {
                aux[i * K + k] = Float(nb)
                edges.append((i, nb))
            }
        }
        _ = edges
        let W = TopologyBuilder.identity(nodes: nodeCount)

        var state = [Float](repeating: 0, count: shape.elementCount)
        let dest = Self.poincarePoint(index: nodeCount / 2, total: nodeCount)
        for b in 0..<batch {
            for n in 0..<nodeCount {
                let p = Self.poincarePoint(index: n, total: nodeCount)
                let base = (b * nodeCount + n) * channels
                state[base + 0] = p.0
                state[base + 1] = p.1
                state[base + 2] = dest.0
                state[base + 3] = dest.1
                state[base + 4] = (n == 0) ? 1 : 0
                state[base + 5] = 0
            }
        }

        let uniforms = PhiUniforms(
            shape: shape,
            param0: curvatureScale,
            param1: Float(K)
        )
        let phi = try MetalPhiKernel(
            context: context,
            kernelName: "phi_hyperbolic_greedy",
            sourceFile: "PhiHyperbolic.metal",
            uniforms: uniforms,
            auxData: aux
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

    private static func poincarePoint(index: Int, total: Int) -> (Float, Float) {
        let angle = 2 * Float.pi * Float(index) / Float(total)
        let r = 0.55 + 0.25 * sinf(Float(index) * 1.7)
        return (r * cosf(angle), r * sinf(angle))
    }
}
