import Foundation
import GPNECCore

/// Post-crash traffic policy for Euclidean vs Poincaré comparison.
public enum DualRoutePostCrashPolicy: String, Sendable, CaseIterable {
    /// Matches the Route UI: Euclidean freezes (no retry/respawn); Poincaré keeps both.
    /// Useful as a demo narrative; **not** a fair embedding-only control.
    case sandbox
    /// Identical mechanics on both metrics: silent retry + respawn after crash.
    /// This is the scientifically fair comparison of embeddings under damage.
    case symmetric
}

/// Euclidean vs Poincaré greedy routing with a betweenness / MDS-strip crash.
public struct DualRouteVerification: Sendable {
    public struct Report: Sendable {
        public var policy: DualRoutePostCrashPolicy
        public var nodeCount: Int
        public var crashCount: Int
        public var packetCount: Int
        public var preTicks: Int
        public var postTicks: Int
        public var euclideanPre: RouteStats
        public var poincarePre: RouteStats
        public var euclideanPost: RouteStats
        public var poincarePost: RouteStats
        public var crashTargets: [Int]
        public var generateMilliseconds: Double
        /// Gate for the configured policy (sandbox gap vs symmetric hyp≥euc).
        public var verified: Bool
        public var note: String
        /// `poincarePost.delivered / max(1, euclideanPost.delivered)`.
        public var postDeliveryRatio: Double {
            Double(poincarePost.delivered) / Double(max(1, euclideanPost.delivered))
        }
    }

    public var config: DualEmbeddingConfig
    public var packetCount: Int
    public var preCrashTicks: Int
    public var postCrashTicks: Int
    public var packetSeed: UInt64
    public var maxHops: UInt32
    public var postCrashPolicy: DualRoutePostCrashPolicy

    public init(
        config: DualEmbeddingConfig = DualEmbeddingConfig(),
        packetCount: Int = 2048,
        preCrashTicks: Int = 40,
        postCrashTicks: Int = 120,
        packetSeed: UInt64 = 0xA11CE,
        maxHops: UInt32 = 0,
        postCrashPolicy: DualRoutePostCrashPolicy = .symmetric
    ) {
        self.config = config
        self.packetCount = packetCount
        self.preCrashTicks = preCrashTicks
        self.postCrashTicks = postCrashTicks
        self.packetSeed = packetSeed
        self.maxHops = maxHops
        self.postCrashPolicy = postCrashPolicy
    }

    public func run(context: MetalContext) throws -> Report {
        let started = ContinuousClock.now
        let graph = DualEmbeddingGenerator.generate(config)
        let genMs = elapsedMs(since: started)

        let euc = try DualGreedyRouter(
            context: context, graph: graph, metric: .euclidean,
            maxHops: maxHops, stuckRetryHops: 40
        )
        let hyp = try DualGreedyRouter(
            context: context, graph: graph, metric: .poincare,
            maxHops: maxHops, stuckRetryHops: 40
        )

        // Pre-crash: identical policy on a healthy mesh.
        euc.seedUniformRandomPackets(count: packetCount, seed: packetSeed)
        hyp.seedUniformRandomPackets(count: packetCount, seed: packetSeed)
        let pre = try runPhase(
            euc: euc, hyp: hyp, ticks: preCrashTicks,
            respawnEuc: true, respawnHyp: true, seed: packetSeed &+ 1
        )

        euc.crash(nodes: graph.crashTargets)
        hyp.crash(nodes: graph.crashTargets)

        let respawnEucPost: Bool
        let respawnHypPost: Bool
        switch postCrashPolicy {
        case .sandbox:
            euc.stuckRetryHops = 0
            respawnEucPost = false
            respawnHypPost = true
        case .symmetric:
            euc.stuckRetryHops = 40
            hyp.stuckRetryHops = 40
            respawnEucPost = true
            respawnHypPost = true
        }

        _ = try euc.step(count: 1)
        _ = try hyp.step(count: 1)

        let postSeed = packetSeed &+ 0xC0A7
        euc.seedUniformRandomPackets(count: packetCount, seed: postSeed)
        hyp.seedUniformRandomPackets(count: packetCount, seed: postSeed)

        let post = try runPhase(
            euc: euc, hyp: hyp, ticks: postCrashTicks,
            respawnEuc: respawnEucPost, respawnHyp: respawnHypPost,
            seed: postSeed &+ 1
        )

        let eucPre = pre.euc
        let hypPre = pre.hyp
        let eucPost = post.euc
        let hypPost = post.hyp

        let preOk = eucPre.delivered >= max(8, packetCount / 64)
            && hypPre.delivered >= max(8, packetCount / 64)

        let verified: Bool
        let note: String
        switch postCrashPolicy {
        case .sandbox:
            // Demo narrative gate: large gap under intentional Euclidean freeze.
            let postGap = hypPost.delivered >= max(eucPost.delivered * 2, eucPost.delivered + 40)
                && hypPost.delivered > eucPost.delivered
                && (eucPost.inFlight + eucPost.dropped) >= max(1, eucPost.delivered / 4)
            verified = preOk && postGap
            if !preOk {
                note =
                    "sandbox: pre-crash too weak (euc=\(eucPre.delivered) hyp=\(hypPre.delivered))"
            } else if verified {
                note =
                    "sandbox (euc freeze): delivered euc=\(eucPost.delivered) hyp=\(hypPost.delivered)"
                    + "; inFlight euc=\(eucPost.inFlight) hyp=\(hypPost.inFlight)"
            } else {
                note =
                    "sandbox: no clear post-crash gap (euc=\(eucPost.delivered) hyp=\(hypPost.delivered))"
            }

        case .symmetric:
            // Fair control: same retry/respawn. Pass if both deliver and hyp ≥ euc.
            // (Does not require a large multiple — that was an artifact of handicapping.)
            let hypAtLeastEuc = hypPost.delivered >= eucPost.delivered
            verified = preOk && hypAtLeastEuc
            let ratio = String(
                format: "%.2f",
                Double(hypPost.delivered) / Double(max(1, eucPost.delivered))
            )
            if !preOk {
                note =
                    "symmetric: pre-crash too weak (euc=\(eucPre.delivered) hyp=\(hypPre.delivered))"
            } else if verified {
                note =
                    "symmetric (identical retry+respawn): delivered euc=\(eucPost.delivered)"
                    + " hyp=\(hypPost.delivered) (ratio \(ratio))"
                    + "; inFlight euc=\(eucPost.inFlight) hyp=\(hypPost.inFlight)"
                    + "; dropped euc=\(eucPost.dropped) hyp=\(hypPost.dropped)"
            } else {
                note =
                    "symmetric: Poincaré did not meet Euclidean (euc=\(eucPost.delivered)"
                    + " hyp=\(hypPost.delivered), ratio \(ratio))"
            }
        }

        return Report(
            policy: postCrashPolicy,
            nodeCount: graph.nodeCount,
            crashCount: graph.crashTargets.count,
            packetCount: packetCount,
            preTicks: preCrashTicks,
            postTicks: postCrashTicks,
            euclideanPre: eucPre,
            poincarePre: hypPre,
            euclideanPost: eucPost,
            poincarePost: hypPost,
            crashTargets: graph.crashTargets,
            generateMilliseconds: genMs,
            verified: verified,
            note: note
        )
    }

    /// Run both policies on the same embedding/seeds for paired comparison.
    public static func runPaired(
        context: MetalContext,
        config: DualEmbeddingConfig,
        packetCount: Int = 512,
        preCrashTicks: Int = 80,
        postCrashTicks: Int = 120,
        packetSeed: UInt64 = 0xA11CE,
        maxHops: UInt32 = 0
    ) throws -> (symmetric: Report, sandbox: Report) {
        let base = DualRouteVerification(
            config: config,
            packetCount: packetCount,
            preCrashTicks: preCrashTicks,
            postCrashTicks: postCrashTicks,
            packetSeed: packetSeed,
            maxHops: maxHops,
            postCrashPolicy: .symmetric
        )
        let sym = try base.run(context: context)
        var sand = base
        sand.postCrashPolicy = .sandbox
        let sandbox = try sand.run(context: context)
        return (sym, sandbox)
    }

    private func runPhase(
        euc: DualGreedyRouter,
        hyp: DualGreedyRouter,
        ticks: Int,
        respawnEuc: Bool,
        respawnHyp: Bool,
        seed: UInt64
    ) throws -> (euc: RouteStats, hyp: RouteStats) {
        var rng = SplitMix64(seed: seed)
        var eucDel = 0, eucDrop = 0, eucHops = 0
        var hypDel = 0, hypDrop = 0, hypHops = 0
        for _ in 0..<ticks {
            _ = try euc.step(count: 1)
            _ = try hyp.step(count: 1)
            let ef = euc.respawnFinishedPackets(using: &rng, respawn: respawnEuc)
            let hf = hyp.respawnFinishedPackets(using: &rng, respawn: respawnHyp)
            eucDel += ef.delivered
            eucDrop += ef.dropped
            eucHops += ef.hopsDelivered
            hypDel += hf.delivered
            hypDrop += hf.dropped
            hypHops += hf.hopsDelivered
        }
        let eucSnap = euc.statsSnapshot()
        let hypSnap = hyp.statsSnapshot()
        if !respawnEuc {
            eucDel += eucSnap.delivered
            eucDrop += eucSnap.dropped
            eucHops += eucSnap.totalHopsDelivered
        }
        if !respawnHyp {
            hypDel += hypSnap.delivered
            hypDrop += hypSnap.dropped
            hypHops += hypSnap.totalHopsDelivered
        }
        return (
            RouteStats(
                delivered: eucDel,
                dropped: eucDrop,
                inFlight: eucSnap.inFlight,
                totalHopsDelivered: eucHops,
                ticks: ticks
            ),
            RouteStats(
                delivered: hypDel,
                dropped: hypDrop,
                inFlight: hypSnap.inFlight,
                totalHopsDelivered: hypHops,
                ticks: ticks
            )
        )
    }

    private func elapsedMs(since start: ContinuousClock.Instant) -> Double {
        let e = ContinuousClock.now - start
        return Double(e.components.seconds) * 1000
            + Double(e.components.attoseconds) / 1e15
    }
}
