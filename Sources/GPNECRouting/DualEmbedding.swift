import Foundation
import simd

/// Undirected mesh with dual geometry: same edges, separate Poincaré + Euclidean layouts.
public struct DualEmbeddedGraph: Sendable {
    public let nodeCount: Int
    public let neighborsPerNode: Int
    /// Row-major fixed-K adjacency; unused slots = -1.
    public let neighbors: [Int32]
    /// Poincaré disk coordinates (hyperbolic greedy / right panel).
    public let poincarePositions: [SIMD2<Float>]
    /// 2D Euclidean layout of the same graph (greedy / left panel).
    public let euclideanPositions: [SIMD2<Float>]
    /// Approximate betweenness scores (sampling Brandes).
    public let betweenness: [Float]
    /// Top-C structural backbone indices (descending betweenness).
    public let crashTargets: [Int]

    /// Convenience: positions for a routing metric.
    public func positions(for metric: RouteMetric) -> [SIMD2<Float>] {
        switch metric {
        case .euclidean: return euclideanPositions
        case .poincare: return poincarePositions
        }
    }

    public init(
        nodeCount: Int,
        neighborsPerNode: Int,
        neighbors: [Int32],
        poincarePositions: [SIMD2<Float>],
        euclideanPositions: [SIMD2<Float>],
        betweenness: [Float],
        crashTargets: [Int]
    ) {
        self.nodeCount = nodeCount
        self.neighborsPerNode = neighborsPerNode
        self.neighbors = neighbors
        self.poincarePositions = poincarePositions
        self.euclideanPositions = euclideanPositions
        self.betweenness = betweenness
        self.crashTargets = crashTargets
    }
}

public struct DualEmbeddingConfig: Sendable {
    public var nodeCount: Int
    public var neighborsPerNode: Int
    public var crashCount: Int
    public var betweennessSamples: Int
    public var landmarkCount: Int
    public var seed: UInt64
    /// Outer Poincaré radius for H² sampling.
    public var diskRadius: Float

    public init(
        nodeCount: Int = 10_000,
        neighborsPerNode: Int = 12,
        crashCount: Int = 800,
        betweennessSamples: Int = 256,
        landmarkCount: Int = 48,
        seed: UInt64 = 0xC0FFEE,
        diskRadius: Float = 0.96
    ) {
        self.nodeCount = nodeCount
        self.neighborsPerNode = neighborsPerNode
        self.crashCount = crashCount
        self.betweennessSamples = betweennessSamples
        self.landmarkCount = landmarkCount
        self.seed = seed
        self.diskRadius = diskRadius
    }
}

public enum DualEmbeddingGenerator {
    /// Sample H² points, connect Poincaré K-NN (symmetrized), build landmark-MDS Euclidean
    /// layout of the same topology, score betweenness, select top-C crash targets.
    public static func generate(_ config: DualEmbeddingConfig) -> DualEmbeddedGraph {
        var rng = SplitMix64(seed: config.seed)
        let n = config.nodeCount
        let k = config.neighborsPerNode

        var poincare = [SIMD2<Float>](repeating: .zero, count: n)
        // Sample with hyperbolic area measure in the disk: ρ ∈ [0, R], density ∝ sinh(ρ).
        let maxRho = atanhf(min(config.diskRadius, 0.995))
        for i in 0..<n {
            let u = rng.nextFloat()
            let v = rng.nextFloat()
            let ang = 2 * Float.pi * u
            let coshR = coshf(maxRho)
            let rho = acoshf(1 + (coshR - 1) * max(v, 1e-7))
            let rad = tanhf(rho / 2)
            poincare[i] = SIMD2(rad * cosf(ang), rad * sinf(ang))
        }

        var neighbors = [Int32](repeating: -1, count: n * k)
        let positionsBox = poincare
        neighbors.withUnsafeMutableBytes { raw in
            let baseAddr = UInt(bitPattern: raw.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: n) { i in
                let nb = UnsafeMutablePointer<Int32>(bitPattern: baseAddr)!
                var bestD = [Float](repeating: .greatestFiniteMagnitude, count: k)
                var bestJ = [Int32](repeating: -1, count: k)
                let pi = positionsBox[i]
                for j in 0..<n where j != i {
                    let d = poincareDistance(pi, positionsBox[j])
                    if d >= bestD[k - 1] { continue }
                    var t = k - 1
                    while t > 0 && d < bestD[t - 1] {
                        bestD[t] = bestD[t - 1]
                        bestJ[t] = bestJ[t - 1]
                        t -= 1
                    }
                    bestD[t] = d
                    bestJ[t] = Int32(j)
                }
                let row = i * k
                for t in 0..<k {
                    nb[row + t] = bestJ[t]
                }
            }
        }
        symmetrize(neighbors: &neighbors, n: n, k: k, positions: poincare)

        let betweenness = approximateBetweenness(
            neighbors: neighbors,
            n: n,
            k: k,
            samples: min(config.betweennessSamples, n),
            rng: &rng
        )

        // Full landmark MDS — greedily navigable enough pre-crash.
        let euclidean = landmarkMDSEmbedding(
            neighbors: neighbors,
            n: n,
            k: k,
            landmarks: min(max(config.landmarkCount, 64), n),
            rng: &rng
        )

        // Crash a vertical cut through the Euclidean plane (high-betweenness nodes
        // in a mid strip). That severs L2 greedy corridors; Poincaré can still
        // arc around in the disk without those Euclidean choke points.
        let crashTargets = selectEuclideanCutTargets(
            betweenness: betweenness,
            euclidean: euclidean,
            count: min(config.crashCount, n)
        )

        return DualEmbeddedGraph(
            nodeCount: n,
            neighborsPerNode: k,
            neighbors: neighbors,
            poincarePositions: poincare,
            euclideanPositions: euclidean,
            betweenness: betweenness,
            crashTargets: crashTargets
        )
    }

    /// Vertical MDS cut: take the full mid-strip (up to `count`), ranked by betweenness.
    private static func selectEuclideanCutTargets(
        betweenness: [Float],
        euclidean: [SIMD2<Float>],
        count: Int
    ) -> [Int] {
        // Widen until we can fill the crash budget — this is a plane cut, not a hub pick.
        for band: Float in [0.12, 0.18, 0.25, 0.32, 0.40] {
            var strip = [(Int, Float)]()
            for i in 0..<euclidean.count {
                if abs(euclidean[i].x) < band {
                    strip.append((i, betweenness[i]))
                }
            }
            if strip.count < count { continue }
            strip.sort { $0.1 > $1.1 }
            return Array(strip.prefix(count).map(\.0))
        }
        var all = (0..<betweenness.count).map { ($0, betweenness[$0]) }
        all.sort { $0.1 > $1.1 }
        return Array(all.prefix(count).map(\.0))
    }

    public static func poincareDistance(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        var aa = a
        var bb = b
        let na = length(aa)
        let nb = length(bb)
        if na >= 0.999 { aa *= 0.999 / na }
        if nb >= 0.999 { bb *= 0.999 / nb }
        let diff = aa - bb
        let num = length(diff)
        let den = max(1 - dot(aa, aa), 1e-6) * max(1 - dot(bb, bb), 1e-6)
        let arg = 1 + 2 * (num * num) / den
        return logf(arg + sqrtf(max(arg * arg - 1, 0)))
    }

    public static func euclideanDistance(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        length(a - b)
    }

    /// Landmark hop-distance embedding → whitened PCA plane, robustly scaled.
    private static func landmarkMDSEmbedding(
        neighbors: [Int32],
        n: Int,
        k: Int,
        landmarks: Int,
        rng: inout SplitMix64
    ) -> [SIMD2<Float>] {
        let L = max(4, min(landmarks, n))
        var pivots = Array(0..<n)
        for i in 0..<L {
            let j = i + Int(rng.next() % UInt64(n - i))
            pivots.swapAt(i, j)
        }
        pivots = Array(pivots.prefix(L))

        var features = [Float](repeating: 0, count: n * L)
        for p in 0..<L {
            let dist = bfsDistances(neighbors: neighbors, n: n, k: k, source: pivots[p])
            for i in 0..<n {
                features[i * L + p] = Float(dist[i])
            }
        }

        // Center + whiten columns so PCA isn't dominated by one pivot.
        var mean = [Float](repeating: 0, count: L)
        var varAcc = [Float](repeating: 0, count: L)
        let invN = 1 / Float(n)
        for i in 0..<n {
            for p in 0..<L { mean[p] += features[i * L + p] }
        }
        for p in 0..<L { mean[p] *= invN }
        for i in 0..<n {
            for p in 0..<L {
                features[i * L + p] -= mean[p]
                let v = features[i * L + p]
                varAcc[p] += v * v
            }
        }
        for p in 0..<L {
            let std = sqrtf(max(varAcc[p] * invN, 1e-8))
            for i in 0..<n {
                features[i * L + p] /= std
            }
        }

        var cov = [Float](repeating: 0, count: L * L)
        for i in 0..<n {
            let row = i * L
            for a in 0..<L {
                let xa = features[row + a]
                for b in a..<L {
                    cov[a * L + b] += xa * features[row + b]
                }
            }
        }
        for a in 0..<L {
            for b in a..<L {
                let v = cov[a * L + b] * invN
                cov[a * L + b] = v
                cov[b * L + a] = v
            }
        }

        let (_, v0) = powerEigen(cov: cov, L: L, exclude: nil)
        let (_, v1) = powerEigen(cov: cov, L: L, exclude: v0)

        var out = [SIMD2<Float>](repeating: .zero, count: n)
        for i in 0..<n {
            let row = i * L
            var x: Float = 0
            var y: Float = 0
            for p in 0..<L {
                let f = features[row + p]
                x += f * v0[p]
                y += f * v1[p]
            }
            out[i] = SIMD2(x, y)
        }
        robustNormalizeAxes(&out)
        return out
    }

    private static func powerEigen(
        cov: [Float],
        L: Int,
        exclude: [Float]?
    ) -> (Float, [Float]) {
        var v = [Float](repeating: 0, count: L)
        for i in 0..<L { v[i] = Float(i + 1) }
        if let exclude {
            let dotE = zip(v, exclude).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            for i in 0..<L { v[i] -= dotE * exclude[i] }
        }
        normalizeVec(&v)
        var lambda: Float = 0
        for _ in 0..<64 {
            var w = [Float](repeating: 0, count: L)
            for i in 0..<L {
                var s: Float = 0
                let row = i * L
                for j in 0..<L { s += cov[row + j] * v[j] }
                w[i] = s
            }
            if let exclude {
                let dotE = zip(w, exclude).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                for i in 0..<L { w[i] -= dotE * exclude[i] }
            }
            lambda = sqrtf(max(zip(w, w).reduce(Float(0)) { $0 + $1.0 * $1.1 }, 1e-12))
            for i in 0..<L { v[i] = w[i] / lambda }
        }
        return (lambda, v)
    }

    private static func normalizeVec(_ v: inout [Float]) {
        let nrm = sqrtf(max(zip(v, v).reduce(Float(0)) { $0 + $1.0 * $1.1 }, 1e-12))
        for i in 0..<v.count { v[i] /= nrm }
    }

    private static func bfsDistances(
        neighbors: [Int32],
        n: Int,
        k: Int,
        source: Int
    ) -> [Int] {
        var dist = [Int](repeating: -1, count: n)
        var queue = [Int]()
        queue.reserveCapacity(n)
        dist[source] = 0
        queue.append(source)
        var qh = 0
        while qh < queue.count {
            let v = queue[qh]
            qh += 1
            let dv = dist[v]
            let base = v * k
            for t in 0..<k {
                let w = Int(neighbors[base + t])
                if w < 0 || dist[w] >= 0 { continue }
                dist[w] = dv + 1
                queue.append(w)
            }
        }
        for i in 0..<n where dist[i] < 0 { dist[i] = n }
        return dist
    }

    /// Scale each axis by the 5th–95th percentile span so outliers don't collapse the cloud.
    private static func robustNormalizeAxes(_ positions: inout [SIMD2<Float>]) {
        let n = positions.count
        guard n > 1 else { return }
        var xs = positions.map(\.x)
        var ys = positions.map(\.y)
        xs.sort()
        ys.sort()
        let lo = max(0, Int(Float(n) * 0.05))
        let hi = min(n - 1, Int(Float(n) * 0.95))
        let midX = (xs[lo] + xs[hi]) * 0.5
        let midY = (ys[lo] + ys[hi]) * 0.5
        let spanX = max(xs[hi] - xs[lo], 1e-4)
        let spanY = max(ys[hi] - ys[lo], 1e-4)
        for i in 0..<n {
            let x = (positions[i].x - midX) / spanX
            let y = (positions[i].y - midY) / spanY
            positions[i] = SIMD2(
                max(-1.1, min(1.1, x * 1.7)),
                max(-1.1, min(1.1, y * 1.7))
            )
        }
    }

    private static func symmetrize(
        neighbors: inout [Int32],
        n: Int,
        k: Int,
        positions: [SIMD2<Float>]
    ) {
        for i in 0..<n {
            for t in 0..<k {
                let j = Int(neighbors[i * k + t])
                if j < 0 { continue }
                var has = false
                var empty = -1
                var farthest = 0
                var farthestD: Float = -1
                for s in 0..<k {
                    let nb = Int(neighbors[j * k + s])
                    if nb == i { has = true }
                    if nb < 0 && empty < 0 { empty = s }
                    if nb >= 0 {
                        let d = poincareDistance(positions[j], positions[nb])
                        if d > farthestD {
                            farthestD = d
                            farthest = s
                        }
                    }
                }
                if has { continue }
                if empty >= 0 {
                    neighbors[j * k + empty] = Int32(i)
                } else if poincareDistance(positions[j], positions[i]) < farthestD {
                    neighbors[j * k + farthest] = Int32(i)
                }
            }
        }
    }

    /// Sampling Brandes: BFS from `samples` random sources, accumulate dependency.
    private static func approximateBetweenness(
        neighbors: [Int32],
        n: Int,
        k: Int,
        samples: Int,
        rng: inout SplitMix64
    ) -> [Float] {
        var score = [Float](repeating: 0, count: n)
        var sources = Array(0..<n)
        let m = min(samples, n)
        for i in 0..<m {
            let j = i + Int(rng.next() % UInt64(n - i))
            sources.swapAt(i, j)
        }

        var stack = [Int]()
        stack.reserveCapacity(n)
        var queue = [Int]()
        queue.reserveCapacity(n)
        var sigma = [Float](repeating: 0, count: n)
        var dist = [Int](repeating: -1, count: n)
        var delta = [Float](repeating: 0, count: n)
        var predHead = [Int](repeating: -1, count: n)
        var predNext = [Int]()
        var predNode = [Int]()
        predNext.reserveCapacity(n * k)
        predNode.reserveCapacity(n * k)

        for sIdx in 0..<m {
            let s = sources[sIdx]
            stack.removeAll(keepingCapacity: true)
            queue.removeAll(keepingCapacity: true)
            predNext.removeAll(keepingCapacity: true)
            predNode.removeAll(keepingCapacity: true)
            for i in 0..<n {
                sigma[i] = 0
                dist[i] = -1
                delta[i] = 0
                predHead[i] = -1
            }
            sigma[s] = 1
            dist[s] = 0
            queue.append(s)
            var qh = 0
            while qh < queue.count {
                let v = queue[qh]
                qh += 1
                stack.append(v)
                let base = v * k
                for t in 0..<k {
                    let w = Int(neighbors[base + t])
                    if w < 0 { continue }
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v]
                        predNext.append(predHead[w])
                        predNode.append(v)
                        predHead[w] = predNext.count - 1
                    }
                }
            }
            while let w = stack.popLast() {
                var e = predHead[w]
                while e >= 0 {
                    let v = predNode[e]
                    if sigma[w] > 0 {
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                    }
                    e = predNext[e]
                }
                if w != s {
                    score[w] += delta[w]
                }
            }
        }
        return score
    }
}

/// Tiny deterministic RNG for reproducible meshes / traffic.
public struct SplitMix64: Sendable {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed == 0 ? 0xDEADBEEF : seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    public mutating func nextFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
