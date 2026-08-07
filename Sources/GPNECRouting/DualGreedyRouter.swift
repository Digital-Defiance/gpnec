import Foundation
import Metal
import simd
import GPNECCore

public enum RouteMetric: UInt32, Sendable {
    case euclidean = 0
    case poincare = 1
}

public struct RoutePacket {
    public var atNode: UInt32
    public var dstNode: UInt32
    public var hops: UInt32
    public var flags: UInt32

    public var isAlive: Bool { flags & 1 != 0 }
    public var isDelivered: Bool { flags & 2 != 0 }
    public var isDropped: Bool { flags & 4 != 0 }
    /// Local-minimum abandon → silent retry (not a hard drop).
    public var needsRetry: Bool { flags & 8 != 0 }
}

public struct RouteStats: Sendable {
    public var delivered: Int
    public var dropped: Int
    public var inFlight: Int
    public var totalHopsDelivered: Int
    public var ticks: Int

    public var deliveryRatio: Double {
        guard delivered + dropped > 0 else {
            return delivered > 0 ? 1 : 0
        }
        return Double(delivered) / Double(delivered + dropped)
    }

    public var meanHopsDelivered: Double {
        guard delivered > 0 else { return 0 }
        return Double(totalHopsDelivered) / Double(delivered)
    }
}

private struct RouteUniforms {
    var nodeCount: UInt32
    var neighborsPerNode: UInt32
    var packetCount: UInt32
    var metric: UInt32
    var maxHops: UInt32
    /// If >0, local-min dwell reaching this hop count → silent retry (flag 8).
    var stuckRetryHops: UInt32
}

/// GPU dual-greedy router over a DualEmbeddedGraph (fixed-K neighbor list).
public final class DualGreedyRouter: @unchecked Sendable {
    public let graph: DualEmbeddedGraph
    public let metric: RouteMetric
    /// Hop TTL; 0 = off. Packets that exceed this without delivery are hard-dropped.
    public var maxHops: UInt32
    /// Local-min dwell before silent retry; 0 = freeze in place (post-crash bottleneck).
    public var stuckRetryHops: UInt32

    private let context: MetalContext
    private let positionsBuffer: MTLBuffer
    private let neighborsBuffer: MTLBuffer
    private let aliveBuffer: MTLBuffer
    private var packetsBuffer: MTLBuffer?
    private var packetCount: Int = 0
    private var ticks: Int = 0

    public init(
        context: MetalContext,
        graph: DualEmbeddedGraph,
        metric: RouteMetric,
        maxHops: UInt32 = 0,
        stuckRetryHops: UInt32 = 40
    ) throws {
        self.context = context
        self.graph = graph
        self.metric = metric
        self.maxHops = maxHops
        self.stuckRetryHops = stuckRetryHops
        _ = try RoutingShaderLibrary.sharedPipeline(device: context.device)

        let pos = graph.positions(for: metric)
        let posBytes = pos.count * MemoryLayout<SIMD2<Float>>.stride
        guard
            let pb = context.device.makeBuffer(
                bytes: pos,
                length: posBytes,
                options: .storageModeShared
            ),
            let nb = context.device.makeBuffer(
                bytes: graph.neighbors,
                length: graph.neighbors.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
        else {
            throw EngineError.bufferAllocationFailed
        }
        self.positionsBuffer = pb
        self.neighborsBuffer = nb

        var alive = [UInt32](repeating: 1, count: graph.nodeCount)
        guard let ab = context.device.makeBuffer(
            bytes: &alive,
            length: alive.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw EngineError.bufferAllocationFailed
        }
        self.aliveBuffer = ab
    }

    public var positionsMetalBuffer: MTLBuffer { positionsBuffer }
    public var neighborsMetalBuffer: MTLBuffer { neighborsBuffer }
    public var aliveMetalBuffer: MTLBuffer { aliveBuffer }
    public var packetsMetalBuffer: MTLBuffer? { packetsBuffer }
    public var activePacketCount: Int { packetCount }
    public var tickCount: Int { ticks }

    public func resetAlive() {
        let ptr = aliveBuffer.contents().bindMemory(to: UInt32.self, capacity: graph.nodeCount)
        for i in 0..<graph.nodeCount { ptr[i] = 1 }
    }

    /// Mark structural backbone nodes dead (hard drop on next hop).
    public func crash(nodes: [Int]) {
        let ptr = aliveBuffer.contents().bindMemory(to: UInt32.self, capacity: graph.nodeCount)
        for i in nodes where i >= 0 && i < graph.nodeCount {
            ptr[i] = 0
        }
    }

    public func seedUniformRandomPackets(count: Int, seed: UInt64) {
        var rng = SplitMix64(seed: seed)
        var packets = [RoutePacket]()
        packets.reserveCapacity(count)
        let n = graph.nodeCount
        let alivePtr = aliveBuffer.contents().bindMemory(to: UInt32.self, capacity: n)

        var attempts = 0
        while packets.count < count && attempts < count * 40 {
            attempts += 1
            let s = Int(rng.next() % UInt64(n))
            let d = Int(rng.next() % UInt64(n))
            if s == d { continue }
            if alivePtr[s] == 0 || alivePtr[d] == 0 { continue }
            packets.append(RoutePacket(atNode: UInt32(s), dstNode: UInt32(d), hops: 0, flags: 1))
        }

        packetCount = packets.count
        ticks = 0
        let byteCount = max(packets.count, 1) * MemoryLayout<RoutePacket>.stride
        packetsBuffer = context.device.makeBuffer(
            bytes: packets,
            length: byteCount,
            options: .storageModeShared
        )
    }

    /// Respawn delivered / dropped / abandoned-retry slots.
    /// If `respawn` is false, finished slots are retired after tallying once.
    public func respawnFinishedPackets(
        using rng: inout SplitMix64,
        respawn: Bool = true
    ) -> (delivered: Int, dropped: Int, hopsDelivered: Int) {
        guard let packetsBuffer, packetCount > 0 else { return (0, 0, 0) }
        let ptr = packetsBuffer.contents().bindMemory(to: RoutePacket.self, capacity: packetCount)
        let n = graph.nodeCount
        let alivePtr = aliveBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        var delivered = 0
        var dropped = 0
        var hopsDelivered = 0
        for i in 0..<packetCount {
            let flags = ptr[i].flags
            let deliveredBit = flags & 2 != 0
            let droppedBit = flags & 4 != 0
            let retryBit = flags & 8 != 0
            if deliveredBit {
                delivered += 1
                hopsDelivered += Int(ptr[i].hops)
            } else if droppedBit {
                dropped += 1
            } else if !retryBit {
                continue
            }
            // retryBit: recycle without counting as drop/deliver
            if !respawn {
                ptr[i] = RoutePacket(atNode: 0, dstNode: 0, hops: 0, flags: 0)
                continue
            }
            var placed = false
            for _ in 0..<64 {
                let s = Int(rng.next() % UInt64(n))
                let d = Int(rng.next() % UInt64(n))
                if s == d { continue }
                if alivePtr[s] == 0 || alivePtr[d] == 0 { continue }
                ptr[i] = RoutePacket(atNode: UInt32(s), dstNode: UInt32(d), hops: 0, flags: 1)
                placed = true
                break
            }
            if !placed {
                ptr[i].flags = 4
            }
        }
        return (delivered, dropped, hopsDelivered)
    }

    /// Encode one greedy hop into an existing command buffer (no wait).
    public func encodeStep(into commandBuffer: MTLCommandBuffer) throws {
        guard let packetsBuffer, packetCount > 0 else { return }
        let pipe = try RoutingShaderLibrary.sharedPipeline(device: context.device)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw EngineError.invalidState
        }
        var u = RouteUniforms(
            nodeCount: UInt32(graph.nodeCount),
            neighborsPerNode: UInt32(graph.neighborsPerNode),
            packetCount: UInt32(packetCount),
            metric: metric.rawValue,
            maxHops: maxHops,
            stuckRetryHops: stuckRetryHops
        )
        enc.setComputePipelineState(pipe)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(neighborsBuffer, offset: 0, index: 1)
        enc.setBuffer(aliveBuffer, offset: 0, index: 2)
        enc.setBuffer(packetsBuffer, offset: 0, index: 3)
        enc.setBytes(&u, length: MemoryLayout<RouteUniforms>.stride, index: 4)
        let width = pipe.threadExecutionWidth
        enc.dispatchThreads(
            MTLSize(width: packetCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        enc.endEncoding()
        ticks += 1
    }

    @discardableResult
    public func step(count: Int = 1) throws -> RouteStats {
        guard packetsBuffer != nil, packetCount > 0 else {
            return statsSnapshot()
        }
        for _ in 0..<count {
            guard let cb = context.commandQueue.makeCommandBuffer() else {
                throw EngineError.invalidState
            }
            try encodeStep(into: cb)
            cb.commit()
            cb.waitUntilCompleted()
        }
        return statsSnapshot()
    }

    public func statsSnapshot() -> RouteStats {
        guard let packetsBuffer, packetCount > 0 else {
            return RouteStats(delivered: 0, dropped: 0, inFlight: 0, totalHopsDelivered: 0, ticks: ticks)
        }
        let ptr = packetsBuffer.contents().bindMemory(to: RoutePacket.self, capacity: packetCount)
        var delivered = 0, dropped = 0, inFlight = 0, hops = 0
        for i in 0..<packetCount {
            let p = ptr[i]
            if p.isDelivered {
                delivered += 1
                hops += Int(p.hops)
            } else if p.isDropped {
                dropped += 1
            } else if p.isAlive {
                inFlight += 1
            }
        }
        return RouteStats(
            delivered: delivered,
            dropped: dropped,
            inFlight: inFlight,
            totalHopsDelivered: hops,
            ticks: ticks
        )
    }

    /// Snapshot packet table (for Metal ≡ CPU hop verification).
    public func readPackets() -> [RoutePacket] {
        guard let packetsBuffer, packetCount > 0 else { return [] }
        let ptr = packetsBuffer.contents().bindMemory(to: RoutePacket.self, capacity: packetCount)
        return (0..<packetCount).map { ptr[$0] }
    }
}

enum RoutingShaderLibrary {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pipeline: MTLComputePipelineState?
    nonisolated(unsafe) private static var sourceFingerprint: Int = 0

    static func sharedPipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }
        let source = try loadSource()
        let fp = source.hashValue
        if let pipeline, sourceFingerprint == fp { return pipeline }
        let lib = try device.makeLibrary(source: source, options: nil)
        guard let fn = lib.makeFunction(name: "dual_greedy_route_step") else {
            throw EngineError.kernelNotFound("dual_greedy_route_step")
        }
        let p = try device.makeComputePipelineState(function: fn)
        pipeline = p
        sourceFingerprint = fp
        return p
    }

    private static func loadSource() throws -> String {
        if let url = Bundle.module.url(
            forResource: "DualGreedyRoute",
            withExtension: "metal",
            subdirectory: "Shaders"
        ) ?? Bundle.module.url(forResource: "DualGreedyRoute", withExtension: "metal") {
            return try String(contentsOf: url, encoding: .utf8)
        }
        let fallback = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Shaders/DualGreedyRoute.metal")
        return try String(contentsOf: fallback, encoding: .utf8)
    }
}
