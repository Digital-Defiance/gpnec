import Foundation
import Metal
import GPNECCore

/// Host-side LBM input forcing (does not change BGK / streaming math).
public enum LatticeBoltzmannInput {
    private static let weightsD2Q9: [Float] = [
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    ]

    /// Gaussian density + dye pulse with mild outward splash.
    public static func injectDroplet(
        into engine: TensorEngine,
        width: Int,
        height: Int,
        cellX: Int,
        cellY: Int,
        radius: Float = 4.0,
        amplitude: Float = 0.45,
        batch: Int = 0
    ) {
        precondition(engine.shape.channels >= 9)
        precondition(engine.shape.nodes == width * height)

        let channels = engine.shape.channels
        let nodes = engine.shape.nodes
        let count = engine.shape.elementCount
        let ptr = engine.stateBuffer.contents().bindMemory(to: Float.self, capacity: count)

        let radiusSq = max(radius * radius, 0.25)
        let rad = Int(ceil(radius)) + 1
        let centerX = cellX
        let centerY = cellY

        for offsetY in -rad...rad {
            for offsetX in -rad...rad {
                let x = centerX + offsetX
                let y = centerY + offsetY
                guard x >= 0, x < width, y >= 0, y < height else { continue }
                let dist2 = Float(offsetX * offsetX + offsetY * offsetY)
                let falloff = expf(-dist2 / radiusSq)
                if falloff < 1e-4 { continue }

                let node = y * width + x
                let base = (batch * nodes + node) * channels
                let bump = amplitude * falloff

                for q in 0..<9 {
                    ptr[base + q] += weightsD2Q9[q] * bump
                }
                if channels > 9 {
                    ptr[base + 9] = min(1.0, ptr[base + 9] + bump * 1.8)
                }
                if dist2 > 0.25 {
                    let inv = 1.0 / sqrtf(dist2)
                    let velX = Float(offsetX) * inv * bump * 0.2
                    let velY = Float(offsetY) * inv * bump * 0.2
                    ptr[base + 1] += 0.5 * velX
                    ptr[base + 3] -= 0.5 * velX
                    ptr[base + 2] += 0.5 * velY
                    ptr[base + 4] -= 0.5 * velY
                }
            }
        }
    }
}
