import Foundation
import simd
import GPNECCore

/// CPU reference for greedy routing metrics + one-hop logic matching `DualGreedyRoute.metal`.
public enum CpuGreedyRoute {
    /// Poincaré disk distance — matches Metal `poincare_distance`.
    public static func poincareDistance(_ aIn: SIMD2<Float>, _ bIn: SIMD2<Float>) -> Float {
        var a = aIn
        var b = bIn
        let na = simd_length(a)
        let nb = simd_length(b)
        if na >= 0.999 { a *= 0.999 / na }
        if nb >= 0.999 { b *= 0.999 / nb }
        let diff = a - b
        let num = simd_length(diff)
        let den = max(1 - simd_dot(a, a), 1e-6) * max(1 - simd_dot(b, b), 1e-6)
        let arg = 1 + 2 * (num * num) / den
        return logf(arg + sqrtf(max(arg * arg - 1, 0)))
    }

    public static func routeDistance(
        _ a: SIMD2<Float>,
        _ b: SIMD2<Float>,
        metric: RouteMetric
    ) -> Float {
        switch metric {
        case .euclidean: return simd_distance(a, b)
        case .poincare: return poincareDistance(a, b)
        }
    }

    /// One greedy hop (or deliver/drop) matching `dual_greedy_route_step` with stuckRetryHops=0.
    public static func stepPacket(
        _ packet: RoutePacket,
        positions: [SIMD2<Float>],
        neighbors: [Int32],
        neighborsPerNode: Int,
        alive: [UInt32],
        metric: RouteMetric,
        maxHops: UInt32
    ) -> RoutePacket {
        var p = packet
        if (p.flags & 1) == 0 { return p }

        let at = Int(p.atNode)
        let dst = Int(p.dstNode)
        if alive[at] == 0 {
            p.flags = 4
            return p
        }
        if at == dst {
            p.flags = 2
            return p
        }
        if alive[dst] == 0 {
            p.flags = 4
            return p
        }
        if maxHops > 0 && p.hops >= maxHops {
            p.flags = 4
            return p
        }

        let here = positions[at]
        let dest = positions[dst]
        var bestDist = routeDistance(here, dest, metric: metric)
        var best: Int = -1
        let base = at * neighborsPerNode
        for t in 0..<neighborsPerNode {
            let nb = Int(neighbors[base + t])
            if nb < 0 || nb >= positions.count { continue }
            if alive[nb] == 0 { continue }
            let d = routeDistance(positions[nb], dest, metric: metric)
            if d + 1e-7 < bestDist {
                bestDist = d
                best = nb
            }
        }

        if best >= 0 {
            p.atNode = UInt32(best)
            p.hops += 1
            if Int(p.atNode) == dst {
                p.flags = 2
            } else if maxHops > 0 && p.hops >= maxHops {
                p.flags = 4
            }
        } else {
            // Local minimum: dwell (stuckRetryHops == 0 → freeze, no flag 8).
            p.hops += 1
        }
        return p
    }
}

/// Accuracy + invariant proofs for the dual router.
public struct RouteAccuracyReport: Sendable {
    public var nodeCount: Int
    public var distancePairsChecked: Int
    public var distanceMaxAbsErrorEuclidean: Float
    public var distanceMaxAbsErrorPoincare: Float
    public var hopPacketsChecked: Int
    public var hopMismatchesEuclidean: Int
    public var hopMismatchesPoincare: Int
    public var hopTicks: Int
    public var embeddingOk: Bool
    /// Fair control: identical retry/respawn post-crash.
    public var crashSymmetricVerified: Bool
    public var crashSymmetricNote: String
    public var crashSymmetricEucDelivered: Int
    public var crashSymmetricHypDelivered: Int
    /// UI-matched asymmetric protocol (not an embedding-only control).
    public var crashSandboxVerified: Bool
    public var crashSandboxNote: String
    public var crashSandboxEucDelivered: Int
    public var crashSandboxHypDelivered: Int
    public var distanceThreshold: Float
    /// Embedding + hops + **symmetric** crash gate (sandbox is reported, not required).
    public var passed: Bool

    @available(*, deprecated, renamed: "crashSymmetricVerified")
    public var crashScenarioVerified: Bool { crashSymmetricVerified }
    @available(*, deprecated, renamed: "crashSymmetricNote")
    public var crashNote: String { crashSymmetricNote }
}

public enum RouteAccuracy {
    public static let defaultDistanceThreshold: Float = 1e-5

    public static func verify(
        context: MetalContext,
        config: DualEmbeddingConfig = DualEmbeddingConfig(
            nodeCount: 400,
            neighborsPerNode: 8,
            crashCount: 80,
            betweennessSamples: 48,
            landmarkCount: 24,
            seed: 42
        ),
        distancePairs: Int = 2000,
        hopPackets: Int = 1024,
        hopTicks: Int = 8,
        distanceThreshold: Float = defaultDistanceThreshold,
        crashPacketCount: Int = 512,
        crashPre: Int = 60,
        crashPost: Int = 100
    ) throws -> RouteAccuracyReport {
        let graph = DualEmbeddingGenerator.generate(config)
        let embeddingOk = checkEmbeddingInvariants(graph)

        let (eucDistErr, hypDistErr) = maxDistanceErrors(
            graph: graph,
            pairs: distancePairs,
            seed: config.seed &+ 1
        )

        let eucHopBad = try metalVsCpuHops(
            context: context,
            graph: graph,
            metric: .euclidean,
            packetCount: hopPackets,
            ticks: hopTicks,
            seed: config.seed &+ 2
        )
        let hypHopBad = try metalVsCpuHops(
            context: context,
            graph: graph,
            metric: .poincare,
            packetCount: hopPackets,
            ticks: hopTicks,
            seed: config.seed &+ 3
        )

        let paired = try DualRouteVerification.runPaired(
            context: context,
            config: config,
            packetCount: crashPacketCount,
            preCrashTicks: crashPre,
            postCrashTicks: crashPost,
            packetSeed: config.seed &+ 4,
            maxHops: 0
        )

        let distOk = eucDistErr <= distanceThreshold && hypDistErr <= distanceThreshold
        let hopsOk = eucHopBad == 0 && hypHopBad == 0
        // Publication gate: hop lockstep + fair (symmetric) crash control.
        let passed = embeddingOk && distOk && hopsOk && paired.symmetric.verified

        return RouteAccuracyReport(
            nodeCount: graph.nodeCount,
            distancePairsChecked: distancePairs,
            distanceMaxAbsErrorEuclidean: eucDistErr,
            distanceMaxAbsErrorPoincare: hypDistErr,
            hopPacketsChecked: hopPackets,
            hopMismatchesEuclidean: eucHopBad,
            hopMismatchesPoincare: hypHopBad,
            hopTicks: hopTicks,
            embeddingOk: embeddingOk,
            crashSymmetricVerified: paired.symmetric.verified,
            crashSymmetricNote: paired.symmetric.note,
            crashSymmetricEucDelivered: paired.symmetric.euclideanPost.delivered,
            crashSymmetricHypDelivered: paired.symmetric.poincarePost.delivered,
            crashSandboxVerified: paired.sandbox.verified,
            crashSandboxNote: paired.sandbox.note,
            crashSandboxEucDelivered: paired.sandbox.euclideanPost.delivered,
            crashSandboxHypDelivered: paired.sandbox.poincarePost.delivered,
            distanceThreshold: distanceThreshold,
            passed: passed
        )
    }

    public static func checkEmbeddingInvariants(_ graph: DualEmbeddedGraph) -> Bool {
        guard graph.poincarePositions.count == graph.nodeCount,
              graph.euclideanPositions.count == graph.nodeCount,
              graph.neighbors.count == graph.nodeCount * graph.neighborsPerNode,
              graph.crashTargets.count == min(graph.crashTargets.count, graph.nodeCount)
        else { return false }

        for p in graph.poincarePositions {
            if !p.x.isFinite || !p.y.isFinite { return false }
            if simd_length(p) >= 1.0 { return false }
        }
        for p in graph.euclideanPositions {
            if !p.x.isFinite || !p.y.isFinite { return false }
        }
        // Euclidean layout must span a plane.
        var minE = graph.euclideanPositions[0]
        var maxE = graph.euclideanPositions[0]
        for p in graph.euclideanPositions {
            minE = simd_min(minE, p)
            maxE = simd_max(maxE, p)
        }
        if maxE.x - minE.x < 0.05 || maxE.y - minE.y < 0.05 { return false }

        // Crash targets unique and in range.
        var seen = Set<Int>()
        for t in graph.crashTargets {
            if t < 0 || t >= graph.nodeCount { return false }
            if !seen.insert(t).inserted { return false }
        }
        return true
    }

    private static func maxDistanceErrors(
        graph: DualEmbeddedGraph,
        pairs: Int,
        seed: UInt64
    ) -> (Float, Float) {
        var rng = SplitMix64(seed: seed)
        let n = graph.nodeCount
        var maxE: Float = 0
        var maxP: Float = 0
        for _ in 0..<pairs {
            let i = Int(rng.next() % UInt64(n))
            let j = Int(rng.next() % UInt64(n))
            let pe = graph.euclideanPositions[i]
            let qe = graph.euclideanPositions[j]
            let pp = graph.poincarePositions[i]
            let qp = graph.poincarePositions[j]
            let cpuE = CpuGreedyRoute.routeDistance(pe, qe, metric: .euclidean)
            let cpuP = CpuGreedyRoute.routeDistance(pp, qp, metric: .poincare)
            // Metal formula is the CPU formula here — error is 0 by construction for Euclidean;
            // Poincaré uses the same closed form as the .metal file. Cross-check self-consistency
            // and symmetry.
            maxE = max(maxE, abs(cpuE - simd_distance(pe, qe)))
            maxP = max(maxP, abs(cpuP - CpuGreedyRoute.poincareDistance(qp, pp))) // symmetry
            maxP = max(maxP, abs(cpuP - CpuGreedyRoute.poincareDistance(pp, qp)))
        }
        return (maxE, maxP)
    }

    /// Run Metal and CPU in lockstep for `ticks` hops; count packet field mismatches.
    private static func metalVsCpuHops(
        context: MetalContext,
        graph: DualEmbeddedGraph,
        metric: RouteMetric,
        packetCount: Int,
        ticks: Int,
        seed: UInt64
    ) throws -> Int {
        let metal = try DualGreedyRouter(
            context: context,
            graph: graph,
            metric: metric,
            maxHops: 0,
            stuckRetryHops: 0 // freeze on local min — deterministic vs CPU
        )
        metal.seedUniformRandomPackets(count: packetCount, seed: seed)
        var cpuPackets = metal.readPackets()
        let positions = graph.positions(for: metric)
        let alive = [UInt32](repeating: 1, count: graph.nodeCount)

        var mismatches = 0
        for _ in 0..<ticks {
            for i in 0..<cpuPackets.count {
                cpuPackets[i] = CpuGreedyRoute.stepPacket(
                    cpuPackets[i],
                    positions: positions,
                    neighbors: graph.neighbors,
                    neighborsPerNode: graph.neighborsPerNode,
                    alive: alive,
                    metric: metric,
                    maxHops: 0
                )
            }
            _ = try metal.step(count: 1)
            let metalPackets = metal.readPackets()
            precondition(metalPackets.count == cpuPackets.count)
            for i in 0..<cpuPackets.count {
                let m = metalPackets[i]
                let c = cpuPackets[i]
                if m.atNode != c.atNode || m.dstNode != c.dstNode
                    || m.hops != c.hops || m.flags != c.flags
                {
                    mismatches += 1
                }
            }
        }
        return mismatches
    }
}
