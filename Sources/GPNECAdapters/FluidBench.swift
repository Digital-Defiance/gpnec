import Foundation
import GPNECCore
import Metal

/// Backends compared by `gpnec bench` (same lattice size / step count / protocol).
public enum FluidBenchBackend: String, CaseIterable, Sendable {
    /// True D2Q9 collide → stream Metal kernels (production fluid path).
    case metal
    /// External baseline: single-thread CPU D2Q9 matching Metal physics (accuracy gold).
    case cpu
    /// External baseline: multi-thread CPU (`concurrentPerform` over rows).
    case cpuMt = "cpu-mt"
    /// Legacy: dense \(N\times N\) MPSGraph streaming + BGK Φ (internal cost-model check).
    case dense

    public var label: String {
        switch self {
        case .metal: return "metal-collide-stream"
        case .cpu: return "cpu-d2q9"
        case .cpuMt: return "cpu-d2q9-mt"
        case .dense: return "dense-Wx-mpsgraph"
        }
    }

    public var needsMetal: Bool {
        switch self {
        case .metal, .dense: return true
        case .cpu, .cpuMt: return false
        }
    }
}

public struct FluidBenchConfig: Sendable {
    public var width: Int
    public var height: Int
    public var tau: Float
    public var warmupSteps: Int
    public var timedSteps: Int

    public init(
        width: Int,
        height: Int,
        tau: Float = 0.56,
        warmupSteps: Int = 64,
        timedSteps: Int = 256
    ) {
        self.width = width
        self.height = height
        self.tau = tau
        self.warmupSteps = warmupSteps
        self.timedSteps = timedSteps
    }

    public var nodes: Int { width * height }

    /// Bytes for a dense \(N\times N\) float32 topology matrix.
    public var denseTopologyBytes: Int { nodes * nodes * MemoryLayout<Float>.stride }

    public var stateBytes: Int {
        // Metal/CPU bench use 10 channels (D2Q9 + dye); dense uses 9.
        nodes * 10 * MemoryLayout<Float>.stride
    }
}

public struct FluidBenchResult: Sendable {
    public var backend: FluidBenchBackend
    public var width: Int
    public var height: Int
    public var timedSteps: Int
    public var warmupSteps: Int
    public var milliseconds: Double
    public var topologyBytes: Int
    public var stateBytes: Int
    public var skipped: Bool
    public var skipReason: String?

    public var microsecondsPerStep: Double {
        guard timedSteps > 0 else { return 0 }
        return (milliseconds * 1000.0) / Double(timedSteps)
    }

    /// Million lattice-updates per second.
    public var mlups: Double {
        let seconds = milliseconds / 1000.0
        guard seconds > 0 else { return 0 }
        return Double(width * height * timedSteps) / (seconds * 1_000_000.0)
    }
}

/// Builds matched LBM engines for throughput comparison (periodic-Y, open-X, no obstacle).
public enum FluidBench {
    public static let maxDenseTopologyBytes = 256 * 1024 * 1024

    public static func makeEngine(
        backend: FluidBenchBackend,
        config: FluidBenchConfig,
        context: MetalContext
    ) throws -> TensorEngine {
        let nodes = config.nodes
        let channels = 9
        let shape = TensorShape(batch: 1, nodes: nodes, channels: channels)
        let state = makeInitialState(width: config.width, height: config.height, channels: channels)

        switch backend {
        case .cpu, .cpuMt:
            preconditionFailure("cpu backends use CpuD2Q9.Engine, not TensorEngine")

        case .metal:
            let solid = [Float](repeating: 0, count: nodes)
            let uniforms = PhiUniforms(
                shape: TensorShape(batch: 1, nodes: nodes, channels: 10),
                param0: config.tau,
                param1: Float(config.width),
                param2: Float(config.height),
                param3: 0
            )
            let paddedShape = TensorShape(batch: 1, nodes: nodes, channels: 10)
            var padded = [Float](repeating: 0, count: paddedShape.elementCount)
            for n in 0..<nodes {
                for q in 0..<9 {
                    padded[n * 10 + q] = state[n * 9 + q]
                }
            }
            let collide = try MetalPhiKernel(
                context: context,
                kernelName: "lbm_bgk_collide",
                sourceFile: "PhiLBM.metal",
                uniforms: uniforms,
                auxData: solid
            )
            let stream = try MetalPhiKernel(
                context: context,
                kernelName: "lbm_stream_bounce",
                sourceFile: "PhiLBM.metal",
                uniforms: uniforms,
                auxData: solid
            )
            return try TensorEngine(
                context: context,
                shape: paddedShape,
                domain: .latticeBoltzmann,
                collide: collide,
                stream: stream,
                initialState: padded
            )

        case .dense:
            if config.denseTopologyBytes > maxDenseTopologyBytes {
                throw BenchSkipError.denseTooLarge(
                    bytes: config.denseTopologyBytes,
                    limit: maxDenseTopologyBytes
                )
            }
            let W = TopologyBuilder.latticeBoltzmannStreaming(
                width: config.width,
                height: config.height
            )
            let uniforms = PhiUniforms(shape: shape, param0: config.tau)
            let phi = try MetalPhiKernel(
                context: context,
                kernelName: "phi_bgk_collision",
                sourceFile: "PhiBGK.metal",
                uniforms: uniforms
            )
            return try TensorEngine(
                context: context,
                shape: shape,
                domain: .latticeBoltzmann,
                topology: W,
                phi: phi,
                initialState: state
            )
        }
    }

    public static func run(
        backend: FluidBenchBackend,
        config: FluidBenchConfig,
        context: MetalContext?
    ) -> FluidBenchResult {
        switch backend {
        case .cpu:
            return runCPU(config: config, parallel: false)
        case .cpuMt:
            return runCPU(config: config, parallel: true)
        case .metal, .dense:
            guard let context else {
                return skippedResult(
                    backend: backend,
                    config: config,
                    reason: "Metal context required"
                )
            }
            return runMetal(backend: backend, config: config, context: context)
        }
    }

    private static func runCPU(config: FluidBenchConfig, parallel: Bool) -> FluidBenchResult {
        let state = makeInitialState(width: config.width, height: config.height, channels: 9)
        var engine = CpuD2Q9.Engine(
            width: config.width,
            height: config.height,
            tau: config.tau,
            inletUx: 0,
            initial9: state,
            parallel: parallel
        )
        if config.warmupSteps > 0 {
            engine.advance(steps: config.warmupSteps)
        }
        let t0 = ContinuousClock.now
        engine.advance(steps: config.timedSteps)
        let elapsed = ContinuousClock.now - t0
        let ms = Double(elapsed.components.seconds) * 1000
            + Double(elapsed.components.attoseconds) / 1e15

        return FluidBenchResult(
            backend: parallel ? .cpuMt : .cpu,
            width: config.width,
            height: config.height,
            timedSteps: config.timedSteps,
            warmupSteps: config.warmupSteps,
            milliseconds: ms,
            topologyBytes: 0,
            stateBytes: engine.stateBytes,
            skipped: false,
            skipReason: nil
        )
    }

    private static func runMetal(
        backend: FluidBenchBackend,
        config: FluidBenchConfig,
        context: MetalContext
    ) -> FluidBenchResult {
        do {
            let engine = try makeEngine(backend: backend, config: config, context: context)
            if config.warmupSteps > 0 {
                try engine.advance(steps: config.warmupSteps)
            }
            let t0 = ContinuousClock.now
            try engine.advance(steps: config.timedSteps)
            let elapsed = ContinuousClock.now - t0
            let ms = Double(elapsed.components.seconds) * 1000
                + Double(elapsed.components.attoseconds) / 1e15

            return FluidBenchResult(
                backend: backend,
                width: config.width,
                height: config.height,
                timedSteps: config.timedSteps,
                warmupSteps: config.warmupSteps,
                milliseconds: ms,
                topologyBytes: backend == .dense ? config.denseTopologyBytes : 0,
                stateBytes: engine.shape.byteCount,
                skipped: false,
                skipReason: nil
            )
        } catch let skip as BenchSkipError {
            return skippedResult(backend: backend, config: config, reason: skip.description)
        } catch {
            return skippedResult(
                backend: backend,
                config: config,
                reason: String(describing: error)
            )
        }
    }

    private static func skippedResult(
        backend: FluidBenchBackend,
        config: FluidBenchConfig,
        reason: String
    ) -> FluidBenchResult {
        FluidBenchResult(
            backend: backend,
            width: config.width,
            height: config.height,
            timedSteps: config.timedSteps,
            warmupSteps: config.warmupSteps,
            milliseconds: 0,
            topologyBytes: backend == .dense ? config.denseTopologyBytes : 0,
            stateBytes: config.stateBytes,
            skipped: true,
            skipReason: reason
        )
    }

    /// Shared IC for bench + accuracy (public for `LBMAccuracy`).
    public static func makeInitialStatePublic(width: Int, height: Int, channels: Int) -> [Float] {
        makeInitialState(width: width, height: height, channels: channels)
    }

    private static func makeInitialState(width: Int, height: Int, channels: Int) -> [Float] {
        let nodes = width * height
        let w9: [Float] = [
            4.0 / 9.0,
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        ]
        var state = [Float](repeating: 0, count: nodes * channels)
        let cx = width / 2
        let cy = height / 2
        for y in 0..<height {
            for x in 0..<width {
                let n = y * width + x
                let dx = Float(x - cx), dy = Float(y - cy)
                let bump = expf(-(dx * dx + dy * dy) / 40.0)
                let rho: Float = 1.0 + 0.05 * bump
                let base = n * channels
                for q in 0..<min(9, channels) {
                    state[base + q] = w9[q] * rho
                }
                if channels > 1 {
                    state[base + 1] += 0.01 * bump
                }
            }
        }
        return state
    }
}

public enum BenchSkipError: Error, CustomStringConvertible {
    case denseTooLarge(bytes: Int, limit: Int)

    public var description: String {
        switch self {
        case .denseTooLarge(let bytes, let limit):
            let mb = Double(bytes) / (1024 * 1024)
            let lim = Double(limit) / (1024 * 1024)
            return String(format: "dense W would be %.0f MiB (limit %.0f MiB)", mb, lim)
        }
    }
}
