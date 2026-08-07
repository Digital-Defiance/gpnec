import XCTest
@testable import GPNECCore
import GPNECRouting
import simd

final class DualRouteTests: XCTestCase {
    func testDualEmbeddingSmall() {
        let g = DualEmbeddingGenerator.generate(
            DualEmbeddingConfig(nodeCount: 64, neighborsPerNode: 4, crashCount: 3, betweennessSamples: 16, seed: 7)
        )
        XCTAssertEqual(g.nodeCount, 64)
        XCTAssertEqual(g.crashTargets.count, 3)
        XCTAssertEqual(g.neighbors.count, 64 * 4)
        for p in g.poincarePositions {
            XCTAssertLessThan(simd_length(p), 1.0)
        }
        XCTAssertEqual(g.euclideanPositions.count, 64)
        // Euclidean layout must span a usable plane (not collapsed to a point).
        var minE = g.euclideanPositions[0]
        var maxE = g.euclideanPositions[0]
        for p in g.euclideanPositions {
            minE = simd_min(minE, p)
            maxE = simd_max(maxE, p)
        }
        XCTAssertGreaterThan(maxE.x - minE.x, 0.2)
        XCTAssertGreaterThan(maxE.y - minE.y, 0.2)
    }

    func testMetalRouteVerifyAdvantage() throws {
        let ctx = try MetalContext()
        let report = try DualRouteVerification(
            config: DualEmbeddingConfig(
                nodeCount: 800,
                neighborsPerNode: 10,
                crashCount: 250,
                betweennessSamples: 64,
                landmarkCount: 32,
                seed: 42
            ),
            packetCount: 512,
            preCrashTicks: 80,
            postCrashTicks: 120,
            packetSeed: 99,
            maxHops: 0
        ).run(context: ctx)
        XCTAssertGreaterThan(report.euclideanPre.delivered, 0)
        XCTAssertGreaterThan(report.poincarePre.delivered, 0)
        // Soft check: after cut, hyp should out-deliver euc (allow small graphs slack).
        XCTAssertGreaterThanOrEqual(
            report.poincarePost.delivered,
            report.euclideanPost.delivered,
            report.note
        )
    }
}
