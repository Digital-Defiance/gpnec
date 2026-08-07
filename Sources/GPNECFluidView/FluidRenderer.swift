import Foundation
import Metal
import MetalKit
import GPNECCore

public struct FluidRenderUniforms {
    public var width: UInt32
    public var height: UInt32
    public var channels: UInt32
    public var batchIndex: UInt32
    public var densityBias: Float
    public var densityScale: Float
    public var velocityScale: Float
    public var vorticityScale: Float

    public init(
        width: Int,
        height: Int,
        channels: Int,
        batchIndex: Int = 0,
        densityBias: Float = 1.0,
        densityScale: Float = 4.0,
        velocityScale: Float = 12.0,
        vorticityScale: Float = 25.0
    ) {
        self.width = UInt32(width)
        self.height = UInt32(height)
        self.channels = UInt32(channels)
        self.batchIndex = UInt32(batchIndex)
        self.densityBias = densityBias
        self.densityScale = densityScale
        self.velocityScale = velocityScale
        self.vorticityScale = vorticityScale
    }
}

/// Binds the engine `stateBuffer` directly into a fragment shader (Option A).
public final class FluidRenderer: NSObject {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var uniforms: FluidRenderUniforms

    public init(device: MTLDevice, pixelFormat: MTLPixelFormat, uniforms: FluidRenderUniforms) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw EngineError.commandQueueFailed
        }
        self.commandQueue = queue
        self.uniforms = uniforms

        let source = try Self.loadShaderSource()
        let library = try device.makeLibrary(source: source, options: nil)
        guard
            let v = library.makeFunction(name: "fluid_fullscreen_vertex"),
            let f = library.makeFunction(name: "fluid_density_velocity_fragment")
        else {
            throw EngineError.kernelNotFound("fluid_density_velocity_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = v
        desc.fragmentFunction = f
        desc.colorAttachments[0].pixelFormat = pixelFormat
        self.pipeline = try device.makeRenderPipelineState(descriptor: desc)
        super.init()
    }

    public func updateUniforms(_ uniforms: FluidRenderUniforms) {
        self.uniforms = uniforms
    }

    /// Encode a fullscreen pass sampling `stateBuffer` — no CPU fluid readback.
    @MainActor
    public func encode(
        into view: MTKView,
        stateBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)
        else {
            return
        }

        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBuffer(stateBuffer, offset: 0, index: 0)
        var u = uniforms
        encoder.setFragmentBytes(&u, length: MemoryLayout<FluidRenderUniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commandBuffer.present(drawable)
    }

    private static func loadShaderSource() throws -> String {
        let name = "FluidVisualize"
        if let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Shaders")
            ?? Bundle.module.url(forResource: name, withExtension: "metal")
        {
            return try String(contentsOf: url, encoding: .utf8)
        }
        let fallback = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Shaders/FluidVisualize.metal")
        return try String(contentsOf: fallback, encoding: .utf8)
    }
}
