import XCTest
@testable import GPNECCore
import GPNECAdapters

final class EngineSmokeTests: XCTestCase {
    func testFluidLBMRunsAndStaysFinite() throws {
        let ctx = try MetalContext()
        let engine = try FluidSimulatorAdapter(width: 64, height: 64, tau: 0.56, inletUx: 0.05)
            .makeEngine(context: ctx)
        XCTAssertEqual(engine.backend, .metalCollideStream)
        let after = try engine.step(count: 16)
        XCTAssertEqual(after.count, 64 * 64 * 10)
        let mass = after.reduce(0, +)
        XCTAssertTrue(mass.isFinite && mass > 0)
        XCTAssertTrue(after.allSatisfy { $0.isFinite })
    }

    func testSubspaceRuns() throws {
        let ctx = try MetalContext()
        let engine = try SubspaceLatticeAdapter(width: 8, height: 8)
            .makeEngine(context: ctx)
        let state = try engine.step(count: 4)
        XCTAssertEqual(state.count, 8 * 8 * 4)
        XCTAssertEqual(engine.backend, .graphTopology)
    }

    func testHyperbolicRuns() throws {
        let ctx = try MetalContext()
        let engine = try HyperbolicRouterAdapter(nodeCount: 16)
            .makeEngine(context: ctx)
        let state = try engine.step(count: 4)
        XCTAssertEqual(state.count, 16 * 6)
    }
}
