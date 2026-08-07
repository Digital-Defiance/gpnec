import Foundation
import GPNECCore
import GPNECAdapters

/// Stable C ABI for the Rust/C++ host.
///
/// Lifecycle: `gpnec_create` → `gpnec_step`* → `gpnec_destroy`.
nonisolated(unsafe) private var engines: [UInt64: TensorEngine] = [:]
nonisolated(unsafe) private var nextHandle: UInt64 = 1
nonisolated(unsafe) private var sharedContext: MetalContext?

private func context() throws -> MetalContext {
    if let sharedContext { return sharedContext }
    let ctx = try MetalContext()
    sharedContext = ctx
    return ctx
}

@_cdecl("gpnec_create")
public func gpnec_create(_ domainRaw: UnsafePointer<CChar>?, _ stepsHint: Int32) -> UInt64 {
    let raw = domainRaw.map { String(cString: $0) } ?? "lbm"
    let domain = AdapterDomain(rawValue: raw) ?? .latticeBoltzmann
    do {
        let ctx = try context()
        let adapter: any EngineAdapter = switch domain {
        case .latticeBoltzmann:
            FluidSimulatorAdapter()
        case .subspaceLattice:
            SubspaceLatticeAdapter()
        case .hyperbolicRouter:
            HyperbolicRouterAdapter()
        }
        let engine = try adapter.makeEngine(context: ctx)
        let handle = nextHandle
        nextHandle += 1
        engines[handle] = engine
        _ = stepsHint
        return handle
    } catch {
        return 0
    }
}

/// Create a subspace lattice engine sized to the live board (e.g. 11×11).
@_cdecl("gpnec_create_subspace")
public func gpnec_create_subspace(_ width: Int32, _ height: Int32) -> UInt64 {
    let boardWidth = Int(width)
    let boardHeight = Int(height)
    guard boardWidth >= 2, boardHeight >= 2, boardWidth <= 64, boardHeight <= 64 else { return 0 }
    do {
        let ctx = try context()
        let adapter = SubspaceLatticeAdapter(width: boardWidth, height: boardHeight)
        let engine = try adapter.makeEngine(context: ctx)
        let handle = nextHandle
        nextHandle += 1
        engines[handle] = engine
        return handle
    } catch {
        return 0
    }
}

@_cdecl("gpnec_step")
public func gpnec_step(_ handle: UInt64, _ count: Int32) -> Int32 {
    guard let engine = engines[handle] else { return -1 }
    do {
        _ = try engine.step(count: Int(max(count, 1)))
        return 0
    } catch {
        return -2
    }
}

@_cdecl("gpnec_steps_executed")
public func gpnec_steps_executed(_ handle: UInt64) -> UInt64 {
    engines[handle]?.stepsExecuted ?? 0
}

@_cdecl("gpnec_shape")
public func gpnec_shape(
    _ handle: UInt64,
    _ batch: UnsafeMutablePointer<Int32>?,
    _ nodes: UnsafeMutablePointer<Int32>?,
    _ channels: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let engine = engines[handle] else { return -1 }
    batch?.pointee = Int32(engine.shape.batch)
    nodes?.pointee = Int32(engine.shape.nodes)
    channels?.pointee = Int32(engine.shape.channels)
    return 0
}

@_cdecl("gpnec_read_state")
public func gpnec_read_state(
    _ handle: UInt64,
    _ out: UnsafeMutablePointer<Float>?,
    _ capacity: Int32
) -> Int32 {
    guard let engine = engines[handle], let out else { return -1 }
    let state = engine.currentState()
    let n = min(state.count, Int(capacity))
    for i in 0..<n { out[i] = state[i] }
    return Int32(n)
}

/// Read one channel strip (length = batch * nodes) into `out`.
/// Channel 1 = Sensor Net / control field for the subspace adapter.
@_cdecl("gpnec_read_channel")
public func gpnec_read_channel(
    _ handle: UInt64,
    _ channel: Int32,
    _ out: UnsafeMutablePointer<Float>?,
    _ capacity: Int32
) -> Int32 {
    guard let engine = engines[handle], let out else { return -1 }
    let ch = Int(channel)
    guard ch >= 0, ch < engine.shape.channels else { return -3 }
    let state = engine.currentState()
    let nodes = engine.shape.batch * engine.shape.nodes
    let channels = engine.shape.channels
    let n = min(nodes, Int(capacity))
    for i in 0..<n {
        out[i] = state[i * channels + ch]
    }
    return Int32(n)
}

/// Seed occupancy (ch0) and Sensor Net control (ch1) from host arrays (length = nodes).
@_cdecl("gpnec_seed_sensor_net")
public func gpnec_seed_sensor_net(
    _ handle: UInt64,
    _ occupancy: UnsafePointer<Float>?,
    _ control: UnsafePointer<Float>?,
    _ nodeCount: Int32
) -> Int32 {
    guard let engine = engines[handle] else { return -1 }
    let nodes = engine.shape.nodes
    guard Int(nodeCount) == nodes else { return -3 }
    var state = engine.currentState()
    let channels = engine.shape.channels
    for n in 0..<nodes {
        let base = n * channels
        if let occupancy { state[base] = occupancy[n] }
        if channels > 1, let control { state[base + 1] = control[n] }
    }
    engine.setState(state)
    return 0
}

/// Σ|control| over the board (channel 1).
@_cdecl("gpnec_control_l1")
public func gpnec_control_l1(_ handle: UInt64) -> Float {
    guard let engine = engines[handle], engine.shape.channels > 1 else { return 0 }
    let state = engine.currentState()
    let channels = engine.shape.channels
    let nodes = engine.shape.batch * engine.shape.nodes
    var sum: Float = 0
    for n in 0..<nodes {
        sum += abs(state[n * channels + 1])
    }
    return sum
}

@_cdecl("gpnec_destroy")
public func gpnec_destroy(_ handle: UInt64) {
    engines.removeValue(forKey: handle)
}

@_cdecl("gpnec_version")
public func gpnec_version() -> UnsafePointer<CChar> {
    let bytes: [CChar] = Array("0.1.0".utf8CString)
    let ptr = UnsafeMutablePointer<CChar>.allocate(capacity: bytes.count)
    ptr.initialize(from: bytes, count: bytes.count)
    return UnsafePointer(ptr)
}
