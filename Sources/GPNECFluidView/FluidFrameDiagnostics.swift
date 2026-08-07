import Foundation
import Observation

/// Display / compute timing for the SwiftUI HUD (not sourced from the fluid buffer).
@Observable
public final class FluidFrameDiagnostics: @unchecked Sendable {
    public var stepsPerFrame: Int
    public var stepsExecuted: UInt64 = 0
    public var computeMilliseconds: Double = 0
    public var frameMilliseconds: Double = 0
    public var framesPerSecond: Double = 0
    public var latticeWidth: Int
    public var latticeHeight: Int
    public var deviceName: String = ""
    public var lastError: String?
    public var dropletCount: Int = 0

    public init(stepsPerFrame: Int = 4, latticeWidth: Int, latticeHeight: Int) {
        self.stepsPerFrame = stepsPerFrame
        self.latticeWidth = latticeWidth
        self.latticeHeight = latticeHeight
    }

    /// Fraction of the display frame spent in deterministic compute (0…1+).
    public var computeFrameRatio: Double {
        guard frameMilliseconds > 1e-9 else { return 0 }
        return computeMilliseconds / frameMilliseconds
    }

    public var headroomMilliseconds: Double {
        max(0, frameMilliseconds - computeMilliseconds)
    }
}
