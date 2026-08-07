import Foundation
import GPNECCore

/// Builds adapter-specific \(W\) and \(\Phi\) and returns a ready TensorEngine.
public protocol EngineAdapter: Sendable {
    var domain: AdapterDomain { get }
    func makeEngine(context: MetalContext) throws -> TensorEngine
}

// MARK: - Topology helpers

public enum TopologyBuilder {
    /// Dense identity.
    public static func identity(nodes: Int) -> [Float] {
        var W = [Float](repeating: 0, count: nodes * nodes)
        for i in 0..<nodes { W[i * nodes + i] = 1 }
        return W
    }

    /// D2Q9 streaming as a block-structured shift on a width×height lattice.
    /// Nodes are row-major; W is identity on the collision path — streaming is
    /// absorbed by permuting node indices per discrete velocity channel via
    /// a single averaged adjacency (4-connected + diagonals + self).
    public static func latticeBoltzmannStreaming(width: Int, height: Int) -> [Float] {
        let nodes = width * height
        var W = [Float](repeating: 0, count: nodes * nodes)
        let dirs: [(Int, Int)] = [
            (0, 0),
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, 1), (-1, -1), (1, -1),
        ]
        let weight: Float = 1.0 / Float(dirs.count)
        for y in 0..<height {
            for x in 0..<width {
                let src = y * width + x
                for (dx, dy) in dirs {
                    let nx = (x + dx + width) % width
                    let ny = (y + dy + height) % height
                    let dst = ny * width + nx
                    // Stream: pull from neighbors into node (column of W)
                    W[src * nodes + dst] += weight
                }
            }
        }
        return W
    }

    /// Non-Euclidean board adjacency with toroidal wrap and knight-like long edges.
    public static func subspaceAdjacency(width: Int, height: Int, influenceRadius: Int = 2) -> [Float] {
        let nodes = width * height
        var W = [Float](repeating: 0, count: nodes * nodes)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                var rowSum: Float = 0
                for dy in -influenceRadius...influenceRadius {
                    for dx in -influenceRadius...influenceRadius {
                        let dist = abs(dx) + abs(dy)
                        if dist == 0 || dist > influenceRadius { continue }
                        // Curved warp: add a Möbius-ish offset on even ranks
                        let warp = ((x + y) % 2 == 0) ? 1 : 0
                        let nx = (x + dx + warp + width) % width
                        let ny = (y + dy - warp + height) % height
                        let j = ny * width + nx
                        let w = 1.0 / Float(dist * dist)
                        W[i * nodes + j] += w
                        rowSum += w
                    }
                }
                // Self weight preserves occupancy mass
                W[i * nodes + i] = 1
                if rowSum > 0 {
                    for j in 0..<nodes where j != i {
                        W[i * nodes + j] /= rowSum
                        W[i * nodes + j] *= 0.35 // influence mix
                    }
                }
            }
        }
        return W
    }

    /// Graph adjacency for BrightChain peers (undirected), row-normalized.
    public static func meshAdjacency(edgeList: [(Int, Int)], nodes: Int) -> [Float] {
        var W = [Float](repeating: 0, count: nodes * nodes)
        for i in 0..<nodes { W[i * nodes + i] = 1 }
        for (a, b) in edgeList {
            precondition(a >= 0 && a < nodes && b >= 0 && b < nodes)
            W[a * nodes + b] = 1
            W[b * nodes + a] = 1
        }
        for i in 0..<nodes {
            var sum: Float = 0
            for j in 0..<nodes { sum += W[i * nodes + j] }
            if sum > 0 {
                for j in 0..<nodes { W[i * nodes + j] /= sum }
            }
        }
        return W
    }
}
