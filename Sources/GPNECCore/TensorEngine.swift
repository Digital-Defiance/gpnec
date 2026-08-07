import Foundation
import Metal
import MetalPerformanceShaders

/// How one automaton step is encoded onto the GPU.
public enum StepBackend: Sendable {
    /// \(Y = W X\) via MPSGraph, then custom \(\Phi\).
    case graphTopology
    /// Custom collide then stream Metal kernels (true LBM; no dense \(W\)).
    case metalCollideStream
}

/// Recurrent non-Euclidean state automaton with ping-pong `MTLBuffer`s.
public final class TensorEngine: @unchecked Sendable {
    public let context: MetalContext
    public let shape: TensorShape
    public let domain: AdapterDomain
    public let backend: StepBackend

    private let buffers: PingPongBuffers
    private let scratch: MTLBuffer
    private let topology: TopologyGraph?
    private let phi: PhiKernel
    private let stream: PhiKernel?
    private var stepCount: UInt64 = 0

    /// MPSGraph topology path (subspace / hyperbolic / legacy).
    public init(
        context: MetalContext,
        shape: TensorShape,
        domain: AdapterDomain,
        topology W: [Float],
        phi: PhiKernel,
        initialState: [Float]? = nil
    ) throws {
        self.context = context
        self.shape = shape
        self.domain = domain
        self.backend = .graphTopology
        self.buffers = try PingPongBuffers(device: context.device, shape: shape, initial: initialState)
        guard let scratch = context.device.makeBuffer(
            length: shape.byteCount,
            options: [.storageModeShared]
        ) else {
            throw EngineError.bufferAllocationFailed
        }
        self.scratch = scratch
        self.topology = try TopologyGraph(device: context.device, shape: shape, topology: W)
        self.phi = phi
        self.stream = nil
    }

    /// Metal collide→stream path (D2Q9 LBM). `phi` = collide, `stream` = stream/bounce.
    public init(
        context: MetalContext,
        shape: TensorShape,
        domain: AdapterDomain,
        collide: PhiKernel,
        stream: PhiKernel,
        initialState: [Float]? = nil
    ) throws {
        self.context = context
        self.shape = shape
        self.domain = domain
        self.backend = .metalCollideStream
        self.buffers = try PingPongBuffers(device: context.device, shape: shape, initial: initialState)
        guard let scratch = context.device.makeBuffer(
            length: shape.byteCount,
            options: [.storageModeShared]
        ) else {
            throw EngineError.bufferAllocationFailed
        }
        self.scratch = scratch
        self.topology = nil
        self.phi = collide
        self.stream = stream
    }

    public var stepsExecuted: UInt64 { stepCount }

    /// Current state tensor in unified memory — bind directly to render/compute.
    public var stateBuffer: MTLBuffer { buffers.input }

    /// Advance without CPU readback. LBM batches encodes; one GPU wait at the end.
    public func advance(steps: Int = 1) throws {
        precondition(steps > 0)
        switch backend {
        case .graphTopology:
            for _ in 0..<steps {
                try encodeGraphStep()
                stepCount += 1
            }
        case .metalCollideStream:
            guard let raw = context.commandQueue.makeCommandBuffer() else {
                throw EngineError.invalidState
            }
            for _ in 0..<steps {
                try encodeMetalLBMStep(into: raw)
                stepCount += 1
            }
            raw.commit()
            raw.waitUntilCompleted()
        }
    }

    @discardableResult
    public func step(count: Int = 1) throws -> [Float] {
        try advance(steps: count)
        return buffers.readState()
    }

    public func currentState() -> [Float] {
        buffers.readState()
    }

    public func setState(_ values: [Float]) {
        buffers.writeState(values)
    }

    private func encodeGraphStep() throws {
        guard let topology else { throw EngineError.invalidState }
        guard let raw = context.commandQueue.makeCommandBuffer() else {
            throw EngineError.invalidState
        }
        let mpsCB = MPSCommandBuffer(commandBuffer: raw)
        topology.encode(
            mpsCommandBuffer: mpsCB,
            input: buffers.input,
            scratch: scratch
        )
        try phi.encode(
            commandBuffer: mpsCB.commandBuffer,
            scratch: scratch,
            output: buffers.output,
            shape: shape
        )
        mpsCB.commit()
        mpsCB.waitUntilCompleted()
        buffers.swap()
    }

    private func encodeMetalLBMStep(into commandBuffer: MTLCommandBuffer) throws {
        guard let stream else { throw EngineError.invalidState }
        try phi.encode(
            commandBuffer: commandBuffer,
            scratch: buffers.input,
            output: scratch,
            shape: shape
        )
        try stream.encode(
            commandBuffer: commandBuffer,
            scratch: scratch,
            output: buffers.output,
            shape: shape
        )
        buffers.swap()
    }
}
