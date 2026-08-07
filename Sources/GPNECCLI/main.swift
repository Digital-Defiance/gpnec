import Foundation
import GPNECCore
import GPNECAdapters
import GPNECRouting

@main
struct GPNECCLI {
    static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            if args.first == "bench" {
                try runBench(Array(args.dropFirst()))
            } else if args.first == "route-verify" {
                try runRouteVerify(Array(args.dropFirst()))
            } else if args.first == "help" || args.contains("-h") || args.contains("--help") {
                printUsage()
            } else {
                try runDemo(args)
            }
        } catch {
            fputs("gpnec error: \(error)\n", stderr)
            exit(1)
        }
    }

    static func printUsage() {
        print(
            """
            GPNEC — General Purpose Non-Euclidean Computer

            Usage:
              gpnec [--adapter lbm|subspace|hyperbolic] [--steps N]
              gpnec bench [--sizes 32,48,64,96,128,256] [--steps 256] [--warmup 64]
                          [--backends metal,dense] [--csv]
              gpnec route-verify [--n 10000] [--k 12] [--crash 800] [--packets 2048]
                                 [--pre 80] [--post 120] [--seed 1]
                                 [--betweenness-samples 256] [--max-hops 0]

            route-verify: dual-embedding mesh + Metal Euclidean vs Poincaré greedy.
              Crash = Euclidean MDS mid-strip (high betweenness); dead nodes = hard drop.
              Stuck local-minima can silent-retry when stuckRetry is enabled in the UI path.
              No UI — prints pre/post crash stats and verified=true/false.

            Visual apps (from repo root):
              ./demo-fluid.sh
              ./demo-network.sh
            """
        )
    }

    // MARK: - Demo

    static func runDemo(_ args: [String]) throws {
        let domain = parseDomain(args)
        let steps = parseFlag(args, name: "--steps").flatMap(Int.init) ?? 32

        print("GPNEC — General Purpose Non-Euclidean Computer")
        print("Metal · MPSGraph topology / Metal collide+stream · custom Φ")
        print("adapter: \(domain.rawValue)  steps: \(steps)")

        let context = try MetalContext()
        print("device: \(context.device.name)")

        let adapter: any EngineAdapter = switch domain {
        case .latticeBoltzmann:
            FluidSimulatorAdapter(width: 256, height: 256, tau: 0.56, inletUx: 0.08)
        case .subspaceLattice:
            SubspaceLatticeAdapter(width: 16, height: 16)
        case .hyperbolicRouter:
            HyperbolicRouterAdapter(nodeCount: 32)
        }

        let engine = try adapter.makeEngine(context: context)
        print("backend: \(engine.backend)")
        let t0 = ContinuousClock.now
        let state = try engine.step(count: steps)
        let elapsed = ContinuousClock.now - t0
        reportDemo(domain: domain, engine: engine, state: state, elapsed: elapsed)
    }

    static func reportDemo(
        domain: AdapterDomain,
        engine: TensorEngine,
        state: [Float],
        elapsed: ContinuousClock.Duration
    ) {
        let ms = Double(elapsed.components.seconds) * 1000
            + Double(elapsed.components.attoseconds) / 1e15
        let shape = engine.shape
        print(String(format: "shape: [%d, %d, %d]", shape.batch, shape.nodes, shape.channels))
        print(String(format: "wall: %.3f ms  (%.3f µs/step)", ms, ms * 1000 / Double(max(engine.stepsExecuted, 1))))

        switch domain {
        case .latticeBoltzmann:
            let mass = state.reduce(0, +)
            print(String(format: "fluid mass Σf: %.6f", mass))
        case .subspaceLattice:
            let nodes = shape.nodes
            let ch = shape.channels
            var control: Float = 0
            for n in 0..<nodes { control += abs(state[n * ch + 1]) }
            print(String(format: "control field L1: %.6f", control))
        case .hyperbolicRouter:
            let nodes = shape.nodes
            let ch = shape.channels
            var packetNode = -1
            var bestDist = Float.greatestFiniteMagnitude
            for n in 0..<nodes {
                let base = n * ch
                if state[base + 4] > 0.5 { packetNode = n }
                bestDist = min(bestDist, state[base + 5])
            }
            print("packet node: \(packetNode)  min hyperbolic dist: \(bestDist)")
        }
    }

    // MARK: - Bench

    static func runBench(_ args: [String]) throws {
        let sizes = parseSizes(args)
        let timedSteps = parseFlag(args, name: "--steps").flatMap(Int.init) ?? 256
        let warmup = parseFlag(args, name: "--warmup").flatMap(Int.init) ?? 64
        let backends = parseBackends(args)
        let csv = args.contains("--csv")

        let context = try MetalContext()
        print("GPNEC bench — apples-to-apples LBM throughput")
        print("device: \(context.device.name)")
        print("warmup: \(warmup)  timed steps: \(timedSteps)")
        print("protocol: same lattice/tau/IC; time only advance() after warmup; MLups = WH·steps / s / 1e6")
        print("note: dense ≠ identical physics (averaged W); compares topology cost model O(N^2) vs O(N)")
        print("")

        var results: [FluidBenchResult] = []
        for size in sizes {
            for backend in backends {
                let config = FluidBenchConfig(
                    width: size,
                    height: size,
                    warmupSteps: warmup,
                    timedSteps: timedSteps
                )
                fputs("  running \(backend.label) @ \(size)x\(size) …\n", stderr)
                fflush(stderr)
                let result = FluidBench.run(backend: backend, config: config, context: context)
                results.append(result)
            }
        }

        print("")
        if csv {
            printCSV(results)
        } else {
            printTable(results)
            printSpeedupSummary(results)
        }
    }

    static func printTable(_ results: [FluidBenchResult]) {
        print(
            pad("backend", 24)
                + pad("size", 8)
                + pad("us/step", 12)
                + pad("MLups", 12)
                + pad("W_MiB", 10)
                + pad("state_MiB", 12)
        )
        print(String(repeating: "-", count: 78))
        for r in results {
            let size = "\(r.width)^2"
            if r.skipped {
                print(
                    pad(r.backend.label, 24)
                        + pad(size, 8)
                        + "SKIP  \(r.skipReason ?? "")"
                )
                continue
            }
            print(
                pad(r.backend.label, 24)
                    + pad(size, 8)
                    + pad(String(format: "%.2f", r.microsecondsPerStep), 12)
                    + pad(String(format: "%.2f", r.mlups), 12)
                    + pad(String(format: "%.2f", Double(r.topologyBytes) / (1024 * 1024)), 10)
                    + pad(String(format: "%.2f", Double(r.stateBytes) / (1024 * 1024)), 12)
            )
        }
        print("")
    }

    static func pad(_ text: String, _ width: Int) -> String {
        if text.count >= width { return String(text.prefix(width - 1)) + " " }
        return text + String(repeating: " ", count: width - text.count)
    }

    static func printCSV(_ results: [FluidBenchResult]) {
        print("backend,width,height,timed_steps,warmup,us_per_step,mlups,topology_mib,state_mib,skipped,skip_reason")
        for r in results {
            let reason = (r.skipReason ?? "").replacingOccurrences(of: ",", with: ";")
            let us = r.skipped ? 0.0 : r.microsecondsPerStep
            let ml = r.skipped ? 0.0 : r.mlups
            print(
                "\(r.backend.rawValue),\(r.width),\(r.height),\(r.timedSteps),\(r.warmupSteps),"
                    + "\(us),\(ml),"
                    + "\(Double(r.topologyBytes) / (1024 * 1024)),"
                    + "\(Double(r.stateBytes) / (1024 * 1024)),"
                    + "\(r.skipped),\(reason)"
            )
        }
    }

    static func printSpeedupSummary(_ results: [FluidBenchResult]) {
        let metal = results.filter { $0.backend == .metal && !$0.skipped }
        let dense = results.filter { $0.backend == .dense && !$0.skipped }
        print("Speedup (dense us/step / metal us/step) at matched sizes:")
        var any = false
        for m in metal {
            if let d = dense.first(where: { $0.width == m.width }) {
                let speedup = d.microsecondsPerStep / max(m.microsecondsPerStep, 1e-9)
                print(
                    "  \(m.width)^2: metal \(String(format: "%.2f", m.microsecondsPerStep)) us/step"
                        + " · dense \(String(format: "%.2f", d.microsecondsPerStep)) us/step"
                        + " · dense/metal = \(String(format: "%.2f", speedup))x"
                )
                any = true
            }
        }
        if !any {
            print("  (no overlapping completed sizes)")
        }
        print("")
        print("Memory: dense W is N^2 floats; metal topology aux is O(N). Past ~64-96^2 dense is skipped by default.")
    }

    // MARK: - Route verify (Directive 4 math)

    static func runRouteVerify(_ args: [String]) throws {
        let n = parseFlag(args, name: "--n").flatMap(Int.init) ?? 10_000
        let k = parseFlag(args, name: "--k").flatMap(Int.init) ?? 12
        let crash = parseFlag(args, name: "--crash").flatMap(Int.init) ?? 800
        let packets = parseFlag(args, name: "--packets").flatMap(Int.init) ?? 2048
        let pre = parseFlag(args, name: "--pre").flatMap(Int.init) ?? 80
        let post = parseFlag(args, name: "--post").flatMap(Int.init) ?? 120
        let seed = parseFlag(args, name: "--seed").flatMap(UInt64.init) ?? 1
        let samples = parseFlag(args, name: "--betweenness-samples").flatMap(Int.init) ?? 256
        let maxHops = parseFlag(args, name: "--max-hops").flatMap(UInt32.init) ?? 0

        print("GPNEC route-verify — Euclidean vs Poincaré greedy")
        print("N=\(n) K=\(k) crashTopC=\(crash) packets=\(packets) pre=\(pre) post=\(post) maxHops=\(maxHops)")
        print("crash=Euclidean MDS cut; dead-node hard drop; stuck=in-flight")
        fflush(stdout)
        fputs("  generating dual embedding (Poincaré K-NN + betweenness)…\n", stderr)
        fflush(stderr)

        let context = try MetalContext()
        print("device: \(context.device.name)")

        let config = DualEmbeddingConfig(
            nodeCount: n,
            neighborsPerNode: k,
            crashCount: crash,
            betweennessSamples: samples,
            seed: seed
        )
        let verification = DualRouteVerification(
            config: config,
            packetCount: packets,
            preCrashTicks: pre,
            postCrashTicks: post,
            packetSeed: seed &+ 0xBEEF,
            maxHops: maxHops
        )
        let report = try verification.run(context: context)
        fputs("  embedding + dual Metal routes done\n", stderr)
        fflush(stderr)

        func line(_ label: String, _ s: RouteStats) {
            let ratio = String(format: "%.3f", s.deliveryRatio)
            let hops = String(format: "%.2f", s.meanHopsDelivered)
            print(
                "  \(label.padding(toLength: 18, withPad: " ", startingAt: 0))"
                    + " delivered=\(s.delivered) dropped=\(s.dropped) inFlight=\(s.inFlight)"
                    + " ratio=\(ratio) meanHops=\(hops) ticks=\(s.ticks)"
            )
        }

        print(String(format: "generate: %.1f ms", report.generateMilliseconds))
        print(
            "crashTargets[\(report.crashCount)]: "
                + report.crashTargets.prefix(8).map(String.init).joined(separator: ",")
                + (report.crashCount > 8 ? ",…" : "")
        )
        print("pre-crash:")
        line("euclidean", report.euclideanPre)
        line("poincare", report.poincarePre)
        print("post-crash (fresh SD traffic on damaged mesh):")
        line("euclidean", report.euclideanPost)
        line("poincare", report.poincarePost)
        print("note: \(report.note)")
        print("verified: \(report.verified)")
        if !report.verified {
            exit(2)
        }
    }

    // MARK: - Parsing

    static func parseDomain(_ args: [String]) -> AdapterDomain {
        if let raw = parseFlag(args, name: "--adapter")
            ?? args.first(where: {
                !$0.hasPrefix("-") && $0 != "bench" && $0 != "route-verify"
            })
        {
            return AdapterDomain(rawValue: raw) ?? .latticeBoltzmann
        }
        return .latticeBoltzmann
    }

    static func parseFlag(_ args: [String], name: String) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func parseSizes(_ args: [String]) -> [Int] {
        let raw = parseFlag(args, name: "--sizes") ?? "32,48,64,96,128,256"
        return raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            .filter { $0 > 0 }
    }

    static func parseBackends(_ args: [String]) -> [FluidBenchBackend] {
        let raw = parseFlag(args, name: "--backends") ?? "metal,dense"
        let parts = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let parsed = parts.compactMap { FluidBenchBackend(rawValue: $0) }
        return parsed.isEmpty ? [.metal, .dense] : parsed
    }
}
