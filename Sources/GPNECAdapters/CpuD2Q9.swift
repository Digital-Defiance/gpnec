import Foundation

/// CPU D2Q9 BGK + pull-stream matching `PhiLBM.metal`
/// (bench path: no solids, inlet ux from config, dye channel included).
///
/// External baselines for `gpnec bench`:
/// - `cpu` — single-thread reference (accuracy gold for Metal)
/// - `cpu-mt` — multi-thread over rows (`DispatchQueue.concurrentPerform`)
enum CpuD2Q9 {
    static let w9: [Float] = [
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    ]
    static let ex: [Float] = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    static let ey: [Float] = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    static let eix: [Int] = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    static let eiy: [Int] = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    static let opp: [Int] = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    static let channels = 10 // D2Q9 + dye (matches Metal bench padding)

    struct Engine {
        let width: Int
        let height: Int
        let tau: Float
        let inletUx: Float
        let parallel: Bool
        var a: [Float]
        var b: [Float]
        private(set) var stepsExecuted: UInt64 = 0

        var nodes: Int { width * height }
        var stateBytes: Int { a.count * MemoryLayout<Float>.stride }

        /// Current post-step state (distributions + dye), length = nodes * 10.
        var state: [Float] { a }

        init(
            width: Int,
            height: Int,
            tau: Float,
            inletUx: Float,
            initial9: [Float],
            parallel: Bool = false
        ) {
            self.width = width
            self.height = height
            self.tau = max(tau, 0.51)
            self.inletUx = inletUx
            self.parallel = parallel
            let n = width * height
            var padded = [Float](repeating: 0, count: n * channels)
            for i in 0..<n {
                for q in 0..<9 {
                    padded[i * channels + q] = initial9[i * 9 + q]
                }
            }
            self.a = padded
            self.b = padded
        }

        /// Pad a 9-channel-per-node buffer to 10 channels (dye = 0).
        static func pad9to10(_ state9: [Float], nodes: Int) -> [Float] {
            var out = [Float](repeating: 0, count: nodes * channels)
            for i in 0..<nodes {
                for q in 0..<9 {
                    out[i * channels + q] = state9[i * 9 + q]
                }
            }
            return out
        }

        mutating func advance(steps: Int) {
            for _ in 0..<steps {
                collide(from: a, into: &b)
                stream(from: b, into: &a)
                stepsExecuted &+= 1
            }
        }

        /// Σ of distribution channels 0…8 (excludes dye).
        func fluidMass() -> Float {
            var sum: Float = 0
            let n = nodes
            let ch = channels
            for i in 0..<n {
                let base = i * ch
                for q in 0..<9 {
                    sum += a[base + q]
                }
            }
            return sum
        }

        private func collide(from input: [Float], into out: inout [Float]) {
            let w = width
            let h = height
            let ch = channels
            let tau = self.tau

            func collideRow(_ y: Int, _ outBase: UnsafeMutablePointer<Float>) {
                for x in 0..<w {
                    let i = y * w + x
                    let base = i * ch
                    let f0 = input[base], f1 = input[base + 1], f2 = input[base + 2]
                    let f3 = input[base + 3], f4 = input[base + 4], f5 = input[base + 5]
                    let f6 = input[base + 6], f7 = input[base + 7], f8 = input[base + 8]
                    var rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
                    var ux = f1 - f3 + f5 - f6 - f7 + f8
                    var uy = f2 - f4 + f5 + f6 - f7 - f8
                    rho = max(rho, 1e-8)
                    ux /= rho
                    uy /= rho
                    let usq = ux * ux + uy * uy
                    let f = [f0, f1, f2, f3, f4, f5, f6, f7, f8]
                    for q in 0..<9 {
                        let eu = ex[q] * ux + ey[q] * uy
                        let feq = w9[q] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usq)
                        outBase[base + q] = f[q] - (f[q] - feq) / tau
                    }
                    outBase[base + 9] = input[base + 9]
                }
            }

            if parallel {
                out.withUnsafeMutableBufferPointer { buf in
                    let p = buf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: h) { y in
                        collideRow(y, p)
                    }
                }
            } else {
                out.withUnsafeMutableBufferPointer { buf in
                    let p = buf.baseAddress!
                    for y in 0..<h { collideRow(y, p) }
                }
            }
        }

        private func stream(from collided: [Float], into out: inout [Float]) {
            let w = width
            let h = height
            let ch = channels
            let inletUx = self.inletUx

            func streamRow(_ y: Int, _ outBase: UnsafeMutablePointer<Float>) {
                for x in 0..<w {
                    let n = y * w + x
                    let base = n * ch
                    var f0: Float = 0, f1: Float = 0, f2: Float = 0, f3: Float = 0, f4: Float = 0
                    var f5: Float = 0, f6: Float = 0, f7: Float = 0, f8: Float = 0

                    for q in 0..<9 {
                        let sx = x - eix[q]
                        var sy = y - eiy[q]
                        if sy < 0 { sy += h }
                        if sy >= h { sy -= h }
                        let v: Float
                        if sx < 0 || sx >= w {
                            v = collided[base + opp[q]]
                        } else {
                            v = collided[(sy * w + sx) * ch + q]
                        }
                        switch q {
                        case 0: f0 = v
                        case 1: f1 = v
                        case 2: f2 = v
                        case 3: f3 = v
                        case 4: f4 = v
                        case 5: f5 = v
                        case 6: f6 = v
                        case 7: f7 = v
                        default: f8 = v
                        }
                    }

                    if x == 0 {
                        let rho: Float = 1
                        let ux = inletUx
                        let uy: Float = 0
                        let usq = ux * ux
                        for q in 0..<9 {
                            let eu = ex[q] * ux + ey[q] * uy
                            let feq = w9[q] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usq)
                            switch q {
                            case 0: f0 = feq
                            case 1: f1 = feq
                            case 2: f2 = feq
                            case 3: f3 = feq
                            case 4: f4 = feq
                            case 5: f5 = feq
                            case 6: f6 = feq
                            case 7: f7 = feq
                            default: f8 = feq
                            }
                        }
                    }

                    outBase[base] = f0; outBase[base + 1] = f1; outBase[base + 2] = f2
                    outBase[base + 3] = f3; outBase[base + 4] = f4; outBase[base + 5] = f5
                    outBase[base + 6] = f6; outBase[base + 7] = f7; outBase[base + 8] = f8

                    var rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
                    var ux = f1 - f3 + f5 - f6 - f7 + f8
                    var uy = f2 - f4 + f5 + f6 - f7 - f8
                    rho = max(rho, 1e-8)
                    ux /= rho
                    uy /= rho

                    var dye = collided[base + 9]
                    let dx = ux > 0.08 ? 1 : (ux < -0.08 ? -1 : 0)
                    let dy = uy > 0.08 ? 1 : (uy < -0.08 ? -1 : 0)
                    let sx = x - dx
                    var sy = y - dy
                    if sy < 0 { sy += h }
                    if sy >= h { sy -= h }
                    if sx >= 0 && sx < w {
                        dye = collided[(sy * w + sx) * ch + 9]
                    }
                    dye *= 0.999
                    if x == 1 && (y % 32) < 6 {
                        dye = max(dye, 0.85)
                    }
                    outBase[base + 9] = min(max(dye, 0), 1)
                }
            }

            if parallel {
                out.withUnsafeMutableBufferPointer { buf in
                    let p = buf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: h) { y in
                        streamRow(y, p)
                    }
                }
            } else {
                out.withUnsafeMutableBufferPointer { buf in
                    let p = buf.baseAddress!
                    for y in 0..<h { streamRow(y, p) }
                }
            }
        }
    }
}
