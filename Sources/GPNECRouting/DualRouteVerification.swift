import Foundation
import GPNECCore

/// Apples-to-apples Euclidean vs Poincaré greedy routing with a betweenness crash.
public struct DualRouteVerification: Sendable {
    public struct Report: Sendable {
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
        public var verified: Bool
        public var note: String
    }

    public var config: DualEmbeddingConfig
    public var packetCount: Int
    public var preCrashTicks: Int
    public var postCrashTicks: Int
    public var packetSeed: UInt64
    public var maxHops: UInt32

    public init(
        config: DualEmbeddingConfig = DualEmbeddingConfig(),
        packetCount: Int = 2048,
        preCrashTicks: Int = 40,
        postCrashTicks: Int = 120,
        packetSeed: UInt64 = 0xA11CE,
        maxHops: UInt32 = 0
    ) {
        self.config = config
        self.packetCount = packetCount
        self.preCrashTicks = preCrashTicks
        self.postCrashTicks = postCrashTicks
        self.packetSeed = packetSeed
        self.maxHops = maxHops
    }

    public func run(context: MetalContext) throws -> Report {
        let t0 = ContinuousClock.now
        let graph = DualEmbeddingGenerator.generate(config)
        let genMs = elapsedMs(since: t0)

        let euc = try DualGreedyRouter(
            context: context, graph: graph, metric: .euclidean, maxHops: maxHops
        )
        let hyp = try DualGreedyRouter(
            context: context, graph: graph, metric: .poincare, maxHops: maxHops
        )

        // Pre-crash: same SD pairs on a healthy mesh (sanity / baseline).
        euc.seedUniformRandomPackets(count: packetCount, seed: packetSeed)
        hyp.seedUniformRandomPackets(count: packetCount, seed: packetSeed)
        let eucPre = try euc.step(count: preCrashTicks)
        let hypPre = try hyp.step(count: preCrashTicks)

        // Structural backbone crash + hard-drop in-flight on dead nodes.
        euc.crash(nodes: graph.crashTargets)
        hyp.crash(nodes: graph.crashTargets)
        _ = try euc.step(count: 1)
        _ = try hyp.step(count: 1)

        let postSeed = packetSeed &+ 0xC0A7
        euc.seedUniformRandomPackets(count: packetCount, seed: postSeed)
        hyp.seedUniformRandomPackets(count: packetCount, seed: postSeed)

        let eucPost = try euc.step(count: postCrashTicks)
        let hypPost = try hyp.step(count: postCrashTicks)

        // Both must move traffic before the crash.
        let preOk = eucPre.delivered >= max(8, packetCount / 64)
            && hypPre.delivered >= max(8, packetCount / 64)
        let postGap = hypPost.delivered >= max(eucPost.delivered * 3, eucPost.delivered + 100)
            && Float(eucPost.delivered) <= Float(hypPost.delivered) * 0.35
            && eucPost.inFlight > eucPost.delivered
        let verified = preOk && postGap
        let note: String
        if !preOk {
            note =
                "pre-crash too weak (euc=\(eucPre.delivered) hyp=\(hypPre.delivered))"
                + " — raise K / pre ticks"
        } else if verified {
            note =
                "post-crash delivered euc=\(eucPost.delivered) hyp=\(hypPost.delivered)"
                + "; inFlight euc=\(eucPost.inFlight) hyp=\(hypPost.inFlight)"
                + "; dropped euc=\(eucPost.dropped) hyp=\(hypPost.dropped)"
        } else {
            note =
                "no clear post-crash gap (euc=\(eucPost.delivered) hyp=\(hypPost.delivered))"
                + " — raise --crash"
        }

        return Report(
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

    private func elapsedMs(since t0: ContinuousClock.Instant) -> Double {
        let e = ContinuousClock.now - t0
        return Double(e.components.seconds) * 1000
            + Double(e.components.attoseconds) / 1e15
    }
}
