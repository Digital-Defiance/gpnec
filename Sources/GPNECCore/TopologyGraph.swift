import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Applies topology \(W\) via MPSGraph batched matmul.
/// For each batch and channel: `Y[b,:,c] = W @ X[b,:,c]` with \(W \in \mathbb{R}^{N\times N}\).
public final class TopologyGraph: @unchecked Sendable {
    public let shape: TensorShape
    public let nodeCount: Int

    private let graph: MPSGraph
    private let xPlaceholder: MPSGraphTensor
    private let wPlaceholder: MPSGraphTensor
    private let yTensor: MPSGraphTensor
    private let wBuffer: MTLBuffer

    public init(device: MTLDevice, shape: TensorShape, topology W: [Float]) throws {
        precondition(W.count == shape.nodes * shape.nodes, "W must be [N×N]")
        self.shape = shape
        self.nodeCount = shape.nodes

        guard let wBuf = device.makeBuffer(
            bytes: W,
            length: W.count * MemoryLayout<Float>.stride,
            options: [.storageModeShared]
        ) else {
            throw EngineError.bufferAllocationFailed
        }
        self.wBuffer = wBuf

        let graph = MPSGraph()
        self.graph = graph

        let n = shape.nodes
        let c = shape.channels
        let b = shape.batch

        let xPlaceholder = graph.placeholder(
            shape: [NSNumber(value: b), NSNumber(value: n), NSNumber(value: c)],
            dataType: .float32,
            name: "X"
        )
        let wPlaceholder = graph.placeholder(
            shape: [NSNumber(value: n), NSNumber(value: n)],
            dataType: .float32,
            name: "W"
        )
        self.xPlaceholder = xPlaceholder
        self.wPlaceholder = wPlaceholder

        // X: [B,N,C] → [B,C,N] → [B*C, N]
        let xPerm = graph.transpose(xPlaceholder, permutation: [0, 2, 1] as [NSNumber], name: nil)
        let xFlat = graph.reshape(
            xPerm,
            shape: [NSNumber(value: b * c), NSNumber(value: n)],
            name: nil
        )
        // Column-vector convention: W @ x  ⇒  (W @ X^T)^T
        let xT = graph.transpose(xFlat, permutation: [1, 0] as [NSNumber], name: nil)
        let wXT = graph.matrixMultiplication(primary: wPlaceholder, secondary: xT, name: "Wx")
        let yFlat = graph.transpose(wXT, permutation: [1, 0] as [NSNumber], name: nil)
        let yPerm = graph.reshape(
            yFlat,
            shape: [NSNumber(value: b), NSNumber(value: c), NSNumber(value: n)],
            name: nil
        )
        self.yTensor = graph.transpose(yPerm, permutation: [0, 2, 1] as [NSNumber], name: "Y")
    }

    /// Encode \(Y = W X\) into `scratch`, reading state from `input`.
    public func encode(
        mpsCommandBuffer: MPSCommandBuffer,
        input: MTLBuffer,
        scratch: MTLBuffer
    ) {
        let xTD = MPSGraphTensorData(input, shape: shape.mpsDims, dataType: .float32)
        let wTD = MPSGraphTensorData(
            wBuffer,
            shape: [NSNumber(value: nodeCount), NSNumber(value: nodeCount)],
            dataType: .float32
        )
        let yTD = MPSGraphTensorData(scratch, shape: shape.mpsDims, dataType: .float32)

        graph.encode(
            to: mpsCommandBuffer,
            feeds: [xPlaceholder: xTD, wPlaceholder: wTD],
            targetOperations: nil,
            resultsDictionary: [yTensor: yTD],
            executionDescriptor: nil
        )
    }
}
