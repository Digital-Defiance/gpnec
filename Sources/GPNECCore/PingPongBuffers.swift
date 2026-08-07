import Darwin
import Metal

/// Zero-copy double-buffered state storage. After each step, roles swap so
/// the previous output becomes the next input without host round-trips.
public final class PingPongBuffers: @unchecked Sendable {
    public private(set) var input: MTLBuffer
    public private(set) var output: MTLBuffer
    public let shape: TensorShape
    public let byteCount: Int

    public init(device: MTLDevice, shape: TensorShape, initial: [Float]? = nil) throws {
        self.shape = shape
        let bytes = shape.byteCount
        self.byteCount = bytes
        let opts: MTLResourceOptions = [.storageModeShared, .cpuCacheModeWriteCombined]

        guard
            let a = device.makeBuffer(length: bytes, options: opts),
            let b = device.makeBuffer(length: bytes, options: opts)
        else {
            throw EngineError.bufferAllocationFailed
        }

        self.input = a
        self.output = b

        if let initial {
            precondition(initial.count == shape.elementCount)
            initial.withUnsafeBytes { raw in
                a.contents().copyMemory(from: raw.baseAddress!, byteCount: bytes)
            }
        } else {
            memset(a.contents(), 0, bytes)
        }
        memset(b.contents(), 0, bytes)
    }

    /// Swap input/output roles (Frame N output → Frame N+1 input).
    public func swap() {
        Swift.swap(&input, &output)
    }

    public func readState() -> [Float] {
        let count = shape.elementCount
        let ptr = input.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    public func writeState(_ values: [Float]) {
        precondition(values.count == shape.elementCount)
        values.withUnsafeBytes { raw in
            input.contents().copyMemory(from: raw.baseAddress!, byteCount: byteCount)
        }
    }
}
