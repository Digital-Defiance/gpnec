import XCTest
@testable import GPNECCore
@testable import GPNECAdapters

final class LBMAccuracyTests: XCTestCase {
    func testMetalMatchesCPUReference() throws {
        let ctx = try MetalContext()
        let report = try LBMAccuracy.compareMetalToCPU(
            context: ctx,
            width: 48,
            height: 48,
            steps: 24,
            tau: 0.56,
            l2Threshold: 1e-4
        )
        XCTAssertTrue(report.metalFinite)
        XCTAssertTrue(report.cpuFinite)
        XCTAssertLessThanOrEqual(
            report.l2Relative,
            report.thresholdL2,
            "Metal diverged from CPU reference: L2=\(report.l2Relative) maxAbs=\(report.maxAbsError)"
        )
    }

    func testRestEquilibriumStableOnCPU() {
        let result = LBMAccuracy.restEquilibriumStable(width: 24, height: 24, steps: 48, tau: 0.8)
        XCTAssertTrue(result.passed, "rest drift \(result.maxAbsDrift)")
    }

    func testCpuMtAgreesWithCpu() {
        let w = 32, h = 32, steps = 16
        let state9 = FluidBench.makeInitialStatePublic(width: w, height: h, channels: 9)
        var single = CpuD2Q9.Engine(
            width: w, height: h, tau: 0.56, inletUx: 0, initial9: state9, parallel: false
        )
        var multi = CpuD2Q9.Engine(
            width: w, height: h, tau: 0.56, inletUx: 0, initial9: state9, parallel: true
        )
        single.advance(steps: steps)
        multi.advance(steps: steps)
        var maxAbs: Float = 0
        for i in 0..<single.state.count {
            maxAbs = max(maxAbs, abs(single.state[i] - multi.state[i]))
        }
        XCTAssertLessThan(maxAbs, 1e-5, "cpu-mt diverged from cpu: maxAbs=\(maxAbs)")
    }
}
