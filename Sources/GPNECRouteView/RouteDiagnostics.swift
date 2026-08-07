import Foundation
import Observation

@Observable
public final class RouteDiagnostics: @unchecked Sendable {
    public var crashCount: Int = 800
    public var neighborsPerNode: Int = 12
    public var packetCount: Int = 2048
    public var nodeCount: Int = 10_000
    public var stepsPerFrame: Int = 4
    public var isReady: Bool = false
    public var isGenerating: Bool = false
    public var hasCrashed: Bool = false
    public var generateMilliseconds: Double = 0
    public var frameMilliseconds: Double = 0
    public var framesPerSecond: Double = 0
    public var lastError: String?

    public var eucDelivered: Int = 0
    public var eucDropped: Int = 0
    public var eucInFlight: Int = 0
    public var hypDelivered: Int = 0
    public var hypDropped: Int = 0
    public var hypInFlight: Int = 0
    public var ticks: Int = 0

    public var pendingCrash: Bool = false

    public var deviceName: String = ""

    public init() {}
}
