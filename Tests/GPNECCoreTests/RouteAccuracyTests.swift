import XCTest
@testable import GPNECCore
@testable import GPNECRouting
import simd

final class RouteAccuracyTests: XCTestCase {
    func testEmbeddingInvariants() {
        let g = DualEmbeddingGenerator.generate(
            DualEmbeddingConfig(
                nodeCount: 128,
                neighborsPerNode: 6,
                crashCount: 12,
                betweennessSamples: 24,
                landmarkCount: 16,
                seed: 11
            )
        )
        XCTAssertTrue(RouteAccuracy.checkEmbeddingInvariants(g))
    }

    func testCpuPoincareSymmetricAndFinite() {
        var rng = SplitMix64(seed: 99)
        for _ in 0..<200 {
            let a = SIMD2<Float>(Float(rng.next()) / Float(UInt64.max) * 1.8 - 0.9,
                                 Float(rng.next()) / Float(UInt64.max) * 1.8 - 0.9)
            let b = SIMD2<Float>(Float(rng.next()) / Float(UInt64.max) * 1.8 - 0.9,
                                 Float(rng.next()) / Float(UInt64.max) * 1.8 - 0.9)
            let d1 = CpuGreedyRoute.poincareDistance(a, b)
            let d2 = CpuGreedyRoute.poincareDistance(b, a)
            XCTAssertTrue(d1.isFinite && d2.isFinite)
            XCTAssertGreaterThanOrEqual(d1, 0)
            XCTAssertEqual(d1, d2, accuracy: 1e-5)
        }
    }

    func testMetalMatchesCpuGreedyHops() throws {
        let ctx = try MetalContext()
        let report = try RouteAccuracy.verify(
            context: ctx,
            config: DualEmbeddingConfig(
                nodeCount: 500,
                neighborsPerNode: 10,
                crashCount: 180,
                betweennessSamples: 64,
                landmarkCount: 24,
                seed: 42
            ),
            distancePairs: 500,
            hopPackets: 400,
            hopTicks: 6,
            crashPacketCount: 400,
            crashPre: 50,
            crashPost: 90
        )
        XCTAssertTrue(report.embeddingOk)
        XCTAssertEqual(report.hopMismatchesEuclidean, 0, "euclidean hop mismatches")
        XCTAssertEqual(report.hopMismatchesPoincare, 0, "poincaré hop mismatches")
        XCTAssertTrue(
            report.crashSymmetricVerified,
            "symmetric crash: \(report.crashSymmetricNote)"
        )
        XCTAssertGreaterThanOrEqual(
            report.crashSymmetricHypDelivered,
            report.crashSymmetricEucDelivered
        )
        // Sandbox is a UI narrative, not required for publication gate.
        XCTAssertTrue(report.passed, report.crashSymmetricNote)
    }
}
