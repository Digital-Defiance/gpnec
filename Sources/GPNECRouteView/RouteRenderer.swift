import Foundation
import Metal
import MetalKit
import GPNECCore
import GPNECRouting

struct RouteVizUniforms {
    var scale: SIMD2<Float>
    var origin: SIMD2<Float>
    var nodeCount: UInt32
    var packetCount: UInt32
    var panelMode: UInt32
    var nodePointSize: Float
    var packetPointSize: Float
    var time: Float
    var crashedFlash: Float
    var _pad: Float = 0
}

/// Zero-copy point instancing of nodes + packets from router MTLBuffers.
public final class RouteRenderer {
    public let device: MTLDevice
    private let bgPipeline: MTLRenderPipelineState
    private let nodePipeline: MTLRenderPipelineState
    private let packetPipeline: MTLRenderPipelineState

    public init(device: MTLDevice, pixelFormat: MTLPixelFormat) throws {
        self.device = device
        let source = try Self.loadSource()
        let lib = try device.makeLibrary(source: source, options: nil)

        guard
            let bgV = lib.makeFunction(name: "route_bg_vertex"),
            let bgF = lib.makeFunction(name: "route_bg_fragment"),
            let nV = lib.makeFunction(name: "route_node_vertex"),
            let nF = lib.makeFunction(name: "route_node_fragment"),
            let pV = lib.makeFunction(name: "route_packet_vertex"),
            let pF = lib.makeFunction(name: "route_packet_fragment")
        else {
            throw EngineError.kernelNotFound("route_* visualize")
        }

        let bgDesc = MTLRenderPipelineDescriptor()
        bgDesc.vertexFunction = bgV
        bgDesc.fragmentFunction = bgF
        bgDesc.colorAttachments[0].pixelFormat = pixelFormat
        bgPipeline = try device.makeRenderPipelineState(descriptor: bgDesc)

        func pointsPipeline(v: MTLFunction, f: MTLFunction) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = v
            d.fragmentFunction = f
            d.colorAttachments[0].pixelFormat = pixelFormat
            d.colorAttachments[0].isBlendingEnabled = true
            d.colorAttachments[0].rgbBlendOperation = .add
            d.colorAttachments[0].alphaBlendOperation = .add
            d.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            d.colorAttachments[0].sourceAlphaBlendFactor = .one
            d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            return try device.makeRenderPipelineState(descriptor: d)
        }

        nodePipeline = try pointsPipeline(v: nV, f: nF)
        packetPipeline = try pointsPipeline(v: pV, f: pF)
    }

    @MainActor
    public func encode(
        into view: MTKView,
        commandBuffer: MTLCommandBuffer,
        euclidean: DualGreedyRouter,
        poincare: DualGreedyRouter,
        time: Float,
        crashedFlash: Float
    ) {
        guard let rpd = view.currentRenderPassDescriptor,
              let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
        else { return }

        let drawable = view.drawableSize
        let aspect = Float(drawable.width / max(drawable.height, 1))

        // Background (full frame)
        var bgU = RouteVizUniforms(
            scale: .zero,
            origin: .zero,
            nodeCount: 0,
            packetCount: 0,
            panelMode: 0,
            nodePointSize: 1,
            packetPointSize: 1,
            time: time,
            crashedFlash: crashedFlash
        )
        enc.setRenderPipelineState(bgPipeline)
        enc.setFragmentBytes(&bgU, length: MemoryLayout<RouteVizUniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        let nodeSize: Float = max(1.2, min(2.8, 1800 / Float(euclidean.graph.nodeCount)))
        let packetSize: Float = nodeSize * 2.4

        // Left: Euclidean — slightly larger points so the cloud reads clearly.
        encodePanel(
            encoder: enc,
            router: euclidean,
            origin: SIMD2(-0.50, 0),
            scale: SIMD2(0.42 / max(aspect, 1), 0.80),
            panelMode: 0,
            nodeSize: nodeSize * 1.35,
            packetSize: packetSize * 1.15,
            time: time,
            crashedFlash: crashedFlash
        )

        // Right: Poincaré
        encodePanel(
            encoder: enc,
            router: poincare,
            origin: SIMD2(0.50, 0),
            scale: SIMD2(0.40 / max(aspect, 1), 0.82),
            panelMode: 1,
            nodeSize: nodeSize,
            packetSize: packetSize,
            time: time,
            crashedFlash: crashedFlash
        )

        enc.endEncoding()
    }

    private func encodePanel(
        encoder: MTLRenderCommandEncoder,
        router: DualGreedyRouter,
        origin: SIMD2<Float>,
        scale: SIMD2<Float>,
        panelMode: UInt32,
        nodeSize: Float,
        packetSize: Float,
        time: Float,
        crashedFlash: Float
    ) {
        var u = RouteVizUniforms(
            scale: scale,
            origin: origin,
            nodeCount: UInt32(router.graph.nodeCount),
            packetCount: UInt32(router.activePacketCount),
            panelMode: panelMode,
            nodePointSize: nodeSize,
            packetPointSize: packetSize,
            time: time,
            crashedFlash: crashedFlash
        )

        encoder.setRenderPipelineState(nodePipeline)
        encoder.setVertexBuffer(router.positionsMetalBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(router.aliveMetalBuffer, offset: 0, index: 1)
        encoder.setVertexBytes(&u, length: MemoryLayout<RouteVizUniforms>.stride, index: 2)
        encoder.drawPrimitives(
            type: .point,
            vertexStart: 0,
            vertexCount: 1,
            instanceCount: router.graph.nodeCount
        )

        guard let packets = router.packetsMetalBuffer, router.activePacketCount > 0 else { return }
        encoder.setRenderPipelineState(packetPipeline)
        encoder.setVertexBuffer(router.positionsMetalBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(packets, offset: 0, index: 1)
        encoder.setVertexBytes(&u, length: MemoryLayout<RouteVizUniforms>.stride, index: 2)
        encoder.drawPrimitives(
            type: .point,
            vertexStart: 0,
            vertexCount: 1,
            instanceCount: router.activePacketCount
        )
    }

    private static func loadSource() throws -> String {
        if let url = Bundle.module.url(
            forResource: "RouteVisualize",
            withExtension: "metal",
            subdirectory: "Shaders"
        ) ?? Bundle.module.url(forResource: "RouteVisualize", withExtension: "metal") {
            return try String(contentsOf: url, encoding: .utf8)
        }
        let fallback = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Shaders/RouteVisualize.metal")
        return try String(contentsOf: fallback, encoding: .utf8)
    }
}
