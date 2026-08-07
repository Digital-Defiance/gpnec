import Foundation

/// State tensor layout: `[Batch, Nodes/Cells, Channels]`.
public struct TensorShape: Sendable, Equatable, Hashable {
    public var batch: Int
    public var nodes: Int
    public var channels: Int

    public init(batch: Int, nodes: Int, channels: Int) {
        precondition(batch > 0 && nodes > 0 && channels > 0)
        self.batch = batch
        self.nodes = nodes
        self.channels = channels
    }

    public var elementCount: Int { batch * nodes * channels }

    public var byteCount: Int { elementCount * MemoryLayout<Float>.stride }

    public var mpsDims: [NSNumber] {
        [NSNumber(value: batch), NSNumber(value: nodes), NSNumber(value: channels)]
    }
}
