import Metal
import MetalPerformanceShaders

/// Custom transition kernel \(\Phi\) encoded onto the same command buffer as the topology graph.
public protocol PhiKernel: AnyObject {
    /// Encode \(\Phi\) reading `scratch` (post-topology) and writing `output`.
    func encode(
        commandBuffer: MTLCommandBuffer,
        scratch: MTLBuffer,
        output: MTLBuffer,
        shape: TensorShape
    ) throws
}

/// Shared uniforms for lattice / mesh sized kernels.
public struct PhiUniforms {
    public var batch: UInt32
    public var nodes: UInt32
    public var channels: UInt32
    public var param0: Float  // tau / temperature / curvature scale
    public var param1: Float
    public var param2: Float
    public var param3: Float

    public init(
        shape: TensorShape,
        param0: Float = 0,
        param1: Float = 0,
        param2: Float = 0,
        param3: Float = 0
    ) {
        self.batch = UInt32(shape.batch)
        self.nodes = UInt32(shape.nodes)
        self.channels = UInt32(shape.channels)
        self.param0 = param0
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
    }
}

/// Generic Metal compute Φ backed by a named kernel in a `.metal` source file.
public final class MetalPhiKernel: PhiKernel {
    private let pipeline: MTLComputePipelineState
    private let uniforms: PhiUniforms
    private let auxBuffer: MTLBuffer?

    public init(
        context: MetalContext,
        kernelName: String,
        sourceFile: String,
        uniforms: PhiUniforms,
        auxData: [Float]? = nil
    ) throws {
        self.pipeline = try context.pipeline(kernelName: kernelName, sourceFile: sourceFile)
        self.uniforms = uniforms
        if let auxData, !auxData.isEmpty {
            self.auxBuffer = context.device.makeBuffer(
                bytes: auxData,
                length: auxData.count * MemoryLayout<Float>.stride,
                options: [.storageModeShared]
            )
        } else {
            self.auxBuffer = nil
        }
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        scratch: MTLBuffer,
        output: MTLBuffer,
        shape: TensorShape
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw EngineError.invalidState
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var u = uniforms
        encoder.setBytes(&u, length: MemoryLayout<PhiUniforms>.stride, index: 2)
        if let auxBuffer {
            encoder.setBuffer(auxBuffer, offset: 0, index: 3)
        }

        let total = shape.batch * shape.nodes
        let w = pipeline.threadExecutionWidth
        let tg = MTLSize(width: w, height: 1, depth: 1)
        let grid = MTLSize(width: total, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }
}
