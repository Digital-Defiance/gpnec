import Foundation
import GPNECCore

/// Accuracy protocol: prove Metal LBM matches the CPU reference (and stays physical).
public struct LBMAccuracyReport: Sendable {
    public var width: Int
    public var height: Int
    public var steps: Int
    public var tau: Float
    public var l2Relative: Double
    public var maxAbsError: Float
    public var metalMass0: Float
    public var metalMassN: Float
    public var cpuMass0: Float
    public var cpuMassN: Float
    public var metalFinite: Bool
    public var cpuFinite: Bool
    /// Pass if L2 below threshold and all values finite.
    public var passed: Bool
    public var thresholdL2: Double

    public var metalMassRelChange: Float {
        guard abs(metalMass0) > 1e-12 else { return 0 }
        return abs(metalMassN - metalMass0) / abs(metalMass0)
    }

    public var cpuMassRelChange: Float {
        guard abs(cpuMass0) > 1e-12 else { return 0 }
        return abs(cpuMassN - cpuMass0) / abs(cpuMass0)
    }
}

public enum LBMAccuracy {
    /// Default L2 relative threshold (float32 GPU vs CPU; not bit-exact).
    public static let defaultL2Threshold: Double = 1e-4

    /// Same IC / τ / boundaries as `FluidBench` metal path; compare full 10-channel state.
    public static func compareMetalToCPU(
        context: MetalContext,
        width: Int = 64,
        height: Int = 64,
        steps: Int = 32,
        tau: Float = 0.56,
        l2Threshold: Double = defaultL2Threshold
    ) throws -> LBMAccuracyReport {
        let config = FluidBenchConfig(
            width: width,
            height: height,
            tau: tau,
            warmupSteps: 0,
            timedSteps: steps
        )
        let state9 = FluidBench.makeInitialStatePublic(
            width: width,
            height: height,
            channels: 9
        )

        let metalEngine = try FluidBench.makeEngine(
            backend: .metal,
            config: config,
            context: context
        )
        let metal0 = metalEngine.currentState()
        let metalMass0 = fluidMass(metal0, channels: 10)
        try metalEngine.advance(steps: steps)
        let metalN = metalEngine.currentState()
        let metalMassN = fluidMass(metalN, channels: 10)

        var cpu = CpuD2Q9.Engine(
            width: width,
            height: height,
            tau: tau,
            inletUx: 0,
            initial9: state9,
            parallel: false
        )
        let cpuMass0 = cpu.fluidMass()
        cpu.advance(steps: steps)
        let cpuN = cpu.state
        let cpuMassN = cpu.fluidMass()

        let (l2, maxAbs) = relativeL2AndMaxAbs(metal: metalN, cpu: cpuN)
        let metalFinite = metalN.allSatisfy(\.isFinite)
        let cpuFinite = cpuN.allSatisfy(\.isFinite)
        let passed = metalFinite && cpuFinite && l2 <= l2Threshold

        return LBMAccuracyReport(
            width: width,
            height: height,
            steps: steps,
            tau: tau,
            l2Relative: l2,
            maxAbsError: maxAbs,
            metalMass0: metalMass0,
            metalMassN: metalMassN,
            cpuMass0: cpuMass0,
            cpuMassN: cpuMassN,
            metalFinite: metalFinite,
            cpuFinite: cpuFinite,
            passed: passed,
            thresholdL2: l2Threshold
        )
    }

    /// Rest fluid: BGK at equilibrium must stay put (machine epsilon).
    public static func restEquilibriumStable(
        width: Int = 32,
        height: Int = 32,
        steps: Int = 64,
        tau: Float = 0.8
    ) -> (maxAbsDrift: Float, passed: Bool) {
        let nodes = width * height
        var state9 = [Float](repeating: 0, count: nodes * 9)
        for n in 0..<nodes {
            for q in 0..<9 {
                state9[n * 9 + q] = CpuD2Q9.w9[q]
            }
        }
        var cpu = CpuD2Q9.Engine(
            width: width,
            height: height,
            tau: tau,
            inletUx: 0,
            initial9: state9,
            parallel: false
        )
        // Disable left-column inlet override effect: with ux=0 equilibrium equals rest,
        // but stream still rewrites x==0 — still feq(rest). Drift should be tiny.
        cpu.advance(steps: steps)
        var maxAbs: Float = 0
        for n in 0..<nodes {
            for q in 0..<9 {
                let d = abs(cpu.state[n * 10 + q] - CpuD2Q9.w9[q])
                if d > maxAbs { maxAbs = d }
            }
        }
        return (maxAbs, maxAbs < 1e-5)
    }

    private static func fluidMass(_ state: [Float], channels: Int) -> Float {
        let nodes = state.count / channels
        var sum: Float = 0
        for n in 0..<nodes {
            let base = n * channels
            for q in 0..<9 {
                sum += state[base + q]
            }
        }
        return sum
    }

    private static func relativeL2AndMaxAbs(metal: [Float], cpu: [Float]) -> (Double, Float) {
        precondition(metal.count == cpu.count)
        var num: Double = 0
        var den: Double = 0
        var maxAbs: Float = 0
        for i in 0..<metal.count {
            let d = metal[i] - cpu[i]
            num += Double(d) * Double(d)
            den += Double(cpu[i]) * Double(cpu[i])
            let a = abs(d)
            if a > maxAbs { maxAbs = a }
        }
        let l2 = den > 1e-30 ? sqrt(num / den) : sqrt(num)
        return (l2, maxAbs)
    }
}
