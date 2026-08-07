import Foundation
import Metal
import MetalKit
import GPNECCore
import GPNECRouting

/// Owns dual routers + graph; advances compute then renders zero-copy.
@MainActor
public final class RouteSandbox {
    public let diagnostics: RouteDiagnostics

    private var metalContext: MetalContext?
    private var graph: DualEmbeddedGraph?
    private var euclidean: DualGreedyRouter?
    private var poincare: DualGreedyRouter?
    private var renderer: RouteRenderer?
    private var respawnRng = SplitMix64(seed: 0x9007E)
    private var trafficSeed: UInt64 = 0xA11CE
    private var startTime = CACurrentMediaTime()
    private var crashFlash: Float = 0
    private var lifetimeEucDelivered = 0
    private var lifetimeEucDropped = 0
    private var lifetimeHypDelivered = 0
    private var lifetimeHypDropped = 0
    private var isBootstrapping = false

    public init(diagnostics: RouteDiagnostics) {
        self.diagnostics = diagnostics
    }

    public func attach(device: MTLDevice, pixelFormat: MTLPixelFormat) throws -> MetalContext {
        let ctx = try MetalContext(device: device)
        metalContext = ctx
        renderer = try RouteRenderer(device: device, pixelFormat: pixelFormat)
        diagnostics.deviceName = device.name
        return ctx
    }

    public func bootstrapIfNeeded() {
        guard !isBootstrapping, !diagnostics.isReady, metalContext != nil else { return }
        isBootstrapping = true
        diagnostics.isGenerating = true
        diagnostics.lastError = nil

        let nodeCount = diagnostics.nodeCount
        let crashCount = diagnostics.crashCount
        let packetCount = diagnostics.packetCount
        let neighborsPerNode = diagnostics.neighborsPerNode

        Task.detached(priority: .userInitiated) { [weak self] in
            let t0 = ContinuousClock.now
            let config = DualEmbeddingConfig(
                nodeCount: nodeCount,
                neighborsPerNode: neighborsPerNode,
                crashCount: crashCount,
                betweennessSamples: 256,
                landmarkCount: 64,
                seed: 1
            )
            let graph = DualEmbeddingGenerator.generate(config)
            let elapsed = ContinuousClock.now - t0
            let ms = Double(elapsed.components.seconds) * 1000
                + Double(elapsed.components.attoseconds) / 1e15

            await self?.finishBootstrap(graph: graph, generateMs: ms, packetCount: packetCount)
        }
    }

    private func finishBootstrap(graph: DualEmbeddedGraph, generateMs: Double, packetCount: Int) {
        guard let metalContext else {
            diagnostics.lastError = "Metal context missing"
            isBootstrapping = false
            diagnostics.isGenerating = false
            return
        }
        do {
            let euc = try DualGreedyRouter(
                context: metalContext, graph: graph, metric: .euclidean,
                maxHops: 0, stuckRetryHops: 40
            )
            let hyp = try DualGreedyRouter(
                context: metalContext, graph: graph, metric: .poincare,
                maxHops: 0, stuckRetryHops: 40
            )
            euc.seedUniformRandomPackets(count: packetCount, seed: trafficSeed)
            hyp.seedUniformRandomPackets(count: packetCount, seed: trafficSeed)
            self.graph = graph
            self.euclidean = euc
            self.poincare = hyp
            diagnostics.generateMilliseconds = generateMs
            diagnostics.isReady = true
            diagnostics.isGenerating = false
            isBootstrapping = false
            startTime = CACurrentMediaTime()
        } catch {
            diagnostics.lastError = String(describing: error)
            diagnostics.isGenerating = false
            isBootstrapping = false
        }
    }

    public func requestCrash() {
        guard diagnostics.isReady, !diagnostics.hasCrashed else { return }
        diagnostics.pendingCrash = true
    }

    public func encodeFrame(into view: MTKView, commandBuffer: MTLCommandBuffer) {
        bootstrapIfNeeded()
        guard diagnostics.isReady,
              let euc = euclidean,
              let hyp = poincare,
              let renderer,
              let graph
        else { return }

        if diagnostics.pendingCrash {
            diagnostics.pendingCrash = false
            euc.crash(nodes: graph.crashTargets)
            hyp.crash(nodes: graph.crashTargets)
            // Freeze Euclidean in local minima — no silent retry, no respawn.
            euc.stuckRetryHops = 0
            diagnostics.hasCrashed = true
            crashFlash = 1.0
        }

        // After backbone cut: Euclidean traffic freezes (no respawn) so stuck amber
        // packets bottleneck in-place; Poincaré keeps cycling new SD pairs.
        let allowEucRespawn = !diagnostics.hasCrashed
        var rngA = respawnRng
        let eucFin = euc.respawnFinishedPackets(using: &rngA, respawn: allowEucRespawn)
        var rngB = respawnRng
        let hypFin = hyp.respawnFinishedPackets(using: &rngB, respawn: true)
        respawnRng = rngA
        lifetimeEucDelivered += eucFin.delivered
        lifetimeEucDropped += eucFin.dropped
        lifetimeHypDelivered += hypFin.delivered
        lifetimeHypDropped += hypFin.dropped

        let steps = max(1, diagnostics.stepsPerFrame)
        do {
            for _ in 0..<steps {
                try euc.encodeStep(into: commandBuffer)
                try hyp.encodeStep(into: commandBuffer)
            }
        } catch {
            diagnostics.lastError = String(describing: error)
        }

        crashFlash = max(0, crashFlash - 0.02)
        let time = Float(CACurrentMediaTime() - startTime)
        renderer.encode(
            into: view,
            commandBuffer: commandBuffer,
            euclidean: euc,
            poincare: hyp,
            time: time,
            crashedFlash: max(crashFlash, diagnostics.hasCrashed ? 0.35 : 0)
        )
    }

    /// Call after presenting so the next frame's CPU respawn sees completed hops.
    public func waitForGPU(_ commandBuffer: MTLCommandBuffer) {
        commandBuffer.waitUntilCompleted()
        guard let euc = euclidean, let hyp = poincare else { return }
        let eucSnap = euc.statsSnapshot()
        let hypSnap = hyp.statsSnapshot()
        diagnostics.eucInFlight = eucSnap.inFlight
        diagnostics.hypInFlight = hypSnap.inFlight
        diagnostics.eucDelivered = lifetimeEucDelivered
        diagnostics.eucDropped = lifetimeEucDropped
        diagnostics.hypDelivered = lifetimeHypDelivered
        diagnostics.hypDropped = lifetimeHypDropped
        diagnostics.ticks = max(euc.tickCount, hyp.tickCount)
    }

    public func resetTraffic() {
        guard let euc = euclidean, let hyp = poincare else { return }
        trafficSeed &+= 1
        euc.seedUniformRandomPackets(count: diagnostics.packetCount, seed: trafficSeed)
        hyp.seedUniformRandomPackets(count: diagnostics.packetCount, seed: trafficSeed)
        lifetimeEucDelivered = 0
        lifetimeEucDropped = 0
        lifetimeHypDelivered = 0
        lifetimeHypDropped = 0
        respawnRng = SplitMix64(seed: trafficSeed &+ 0x99)
    }
}
