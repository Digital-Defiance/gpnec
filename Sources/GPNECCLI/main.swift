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
            } else if args.first == "verify-lbm" {
                try runVerifyLBM(Array(args.dropFirst()))
            } else if args.first == "verify-route" {
                try runVerifyRoute(Array(args.dropFirst()))
            } else if args.first == "verify" || args.first == "verify-all" {
                try runVerifyAll(Array(args.dropFirst()))
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
                          [--backends metal,cpu,cpu-mt,dense] [--csv]
              gpnec verify-lbm [--size 64] [--steps 32] [--tau 0.56] [--l2 1e-4]
              gpnec verify-route [--n 800] [--k 10] [--crash 250] [--hops 8] [--packets 1024]
              gpnec verify | verify-all
              gpnec route-verify [--n 10000] [--k 12] [--crash 800] [--packets 2048]
                                 [--pre 80] [--post 120] [--seed 1]
                                 [--betweenness-samples 256] [--max-hops 0]
                                 [--policy symmetric|sandbox|both]

            bench: same lattice/τ/IC. Default backends metal,cpu.
              cpu = single-thread D2Q9 matching PhiLBM.metal (accuracy gold; not a tuned CFD baseline).
              cpu-mt = multi-thread rows (optional; often slower — memory-bound).
              dense = optional internal O(N²) MPSGraph cost-model check.
            verify-lbm: Metal ≡ CPU LBM (relative L2 + mass + rest equilibrium).
            verify-route: embedding + Metal≡CPU hops + symmetric crash (sandbox reported).
            verify / verify-all: run verify-lbm then verify-route (exit ≠0 if either fails).
            route-verify: dual-embedding crash stats. Default --policy symmetric (fair control).
              sandbox = UI-matched euc freeze (demo narrative, not embedding-only proof).
              both = print symmetric then sandbox.
              Crash = Euclidean MDS mid-strip; dead nodes = hard drop.

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
        let needsMetal = backends.contains(where: \.needsMetal)
        let context: MetalContext? = needsMetal ? try MetalContext() : nil

        print("GPNEC bench — apples-to-apples LBM throughput")
        if let context {
            print("device: \(context.device.name)")
        } else {
            print("device: CPU only")
        }
        print("warmup: \(warmup)  timed steps: \(timedSteps)")
        print("protocol: same lattice/tau/IC; time only advance() after warmup; MLups = WH·steps / s / 1e6")
        print("baseline: cpu-d2q9 = single-thread Swift matching PhiLBM.metal (external to Metal)")
        print("note: dense ≠ identical physics (averaged W); optional O(N^2) cost-model check")
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
        let cpu = results.filter { $0.backend == .cpu && !$0.skipped }
        let cpuMt = results.filter { $0.backend == .cpuMt && !$0.skipped }
        let dense = results.filter { $0.backend == .dense && !$0.skipped }

        print("Speedup vs external CPU baselines (us/step ratios → metal):")
        var anyCPU = false
        for m in metal {
            if let c = cpu.first(where: { $0.width == m.width }) {
                let speedup = c.microsecondsPerStep / max(m.microsecondsPerStep, 1e-9)
                print(
                    "  \(m.width)^2 cpu/metal = \(String(format: "%.1f", speedup))x"
                        + "  (metal \(String(format: "%.2f", m.microsecondsPerStep)) · cpu \(String(format: "%.2f", c.microsecondsPerStep)) us/step"
                        + " · \(String(format: "%.1f", m.mlups)) / \(String(format: "%.2f", c.mlups)) MLups)"
                )
                anyCPU = true
            }
            if let c = cpuMt.first(where: { $0.width == m.width }) {
                let speedup = c.microsecondsPerStep / max(m.microsecondsPerStep, 1e-9)
                print(
                    "  \(m.width)^2 cpu-mt/metal = \(String(format: "%.1f", speedup))x"
                        + "  (metal \(String(format: "%.2f", m.microsecondsPerStep)) · cpu-mt \(String(format: "%.2f", c.microsecondsPerStep)) us/step"
                        + " · \(String(format: "%.1f", m.mlups)) / \(String(format: "%.2f", c.mlups)) MLups)"
                )
                anyCPU = true
            }
        }
        if !anyCPU {
            print("  (no overlapping metal+cpu sizes — pass --backends metal,cpu,cpu-mt)")
        }

        if !dense.isEmpty {
            print("")
            print("Internal cost-model check (dense us/step / metal us/step):")
            for m in metal {
                if let d = dense.first(where: { $0.width == m.width }) {
                    let speedup = d.microsecondsPerStep / max(m.microsecondsPerStep, 1e-9)
                    print(
                        "  \(m.width)^2: dense/metal = \(String(format: "%.2f", speedup))x"
                            + " (dense W \(String(format: "%.1f", Double(d.topologyBytes) / (1024 * 1024))) MiB)"
                    )
                }
            }
            print("Memory: dense W is N^2 floats; skipped past ~64-96^2 by default.")
        }
        print("")
    }

    // MARK: - LBM accuracy (Metal ≡ CPU)

    static func runVerifyLBM(_ args: [String]) throws {
        let size = parseFlag(args, name: "--size").flatMap(Int.init) ?? 64
        let steps = parseFlag(args, name: "--steps").flatMap(Int.init) ?? 32
        let tau = parseFlag(args, name: "--tau").flatMap(Float.init) ?? 0.56
        let l2Max = parseFlag(args, name: "--l2").flatMap(Double.init) ?? LBMAccuracy.defaultL2Threshold

        let context = try MetalContext()
        print("GPNEC verify-lbm — Metal vs CPU accuracy")
        print("device: \(context.device.name)")
        print("lattice: \(size)×\(size)  steps: \(steps)  τ: \(tau)  L2 threshold: \(l2Max)")
        print("protocol: identical IC/boundaries to bench; compare full 10-channel state after N steps")
        print("")

        let report = try LBMAccuracy.compareMetalToCPU(
            context: context,
            width: size,
            height: size,
            steps: steps,
            tau: tau,
            l2Threshold: l2Max
        )

        print(String(format: "relative L2 (metal vs cpu): %.6e", report.l2Relative))
        print(String(format: "max |metal−cpu|:            %.6e", report.maxAbsError))
        print(String(format: "metal mass Σf:  %.6f → %.6f  (rel Δ %.3e)",
                     report.metalMass0, report.metalMassN, report.metalMassRelChange))
        print(String(format: "cpu mass Σf:    %.6f → %.6f  (rel Δ %.3e)",
                     report.cpuMass0, report.cpuMassN, report.cpuMassRelChange))
        print("finite: metal=\(report.metalFinite) cpu=\(report.cpuFinite)")

        let rest = LBMAccuracy.restEquilibriumStable()
        print(String(format: "rest equilibrium max |drift| (cpu): %.3e  pass=%@",
                     rest.maxAbsDrift, String(describing: rest.passed)))

        print("")
        if report.passed && rest.passed {
            print("verified: true")
        } else {
            print("verified: false")
            exit(2)
        }
    }

    // MARK: - Route accuracy (Metal ≡ CPU + crash scenario)

    static func runVerifyRoute(_ args: [String]) throws {
        let n = parseFlag(args, name: "--n").flatMap(Int.init) ?? 800
        let k = parseFlag(args, name: "--k").flatMap(Int.init) ?? 10
        let crash = parseFlag(args, name: "--crash").flatMap(Int.init) ?? 250
        let hops = parseFlag(args, name: "--hops").flatMap(Int.init) ?? 8
        let packets = parseFlag(args, name: "--packets").flatMap(Int.init) ?? 1024
        let pairs = parseFlag(args, name: "--pairs").flatMap(Int.init) ?? 2000
        let seed = parseFlag(args, name: "--seed").flatMap(UInt64.init) ?? 42

        let context = try MetalContext()
        print("GPNEC verify-route — embedding + Metal≡CPU hops + crash scenario")
        print("device: \(context.device.name)")
        print("N=\(n) K=\(k) crash=\(crash) hopPackets=\(packets) hopTicks=\(hops) seed=\(seed)")
        print("")

        let report = try RouteAccuracy.verify(
            context: context,
            config: DualEmbeddingConfig(
                nodeCount: n,
                neighborsPerNode: k,
                crashCount: crash,
                betweennessSamples: min(128, max(32, n / 8)),
                landmarkCount: min(48, max(16, n / 20)),
                seed: seed
            ),
            distancePairs: pairs,
            hopPackets: packets,
            hopTicks: hops,
            crashPacketCount: min(packets, 512),
            crashPre: 60,
            crashPost: 100
        )

        print("embedding invariants: \(report.embeddingOk ? "pass" : "FAIL")")
        print(String(
            format: "distance self-check max|Δ|  euc=%.3e  poincaré(sym)=%.3e  (thr %.1e)",
            report.distanceMaxAbsErrorEuclidean,
            report.distanceMaxAbsErrorPoincare,
            report.distanceThreshold
        ))
        print(
            "Metal≡CPU greedy hops (\(report.hopTicks) ticks × \(report.hopPacketsChecked) pkts):"
                + " euc mismatches=\(report.hopMismatchesEuclidean)"
                + " poincaré mismatches=\(report.hopMismatchesPoincare)"
        )
        print(
            "crash SYMMETRIC (identical retry+respawn): \(report.crashSymmetricVerified)"
                + " — euc=\(report.crashSymmetricEucDelivered)"
                + " hyp=\(report.crashSymmetricHypDelivered)"
                + " — \(report.crashSymmetricNote)"
        )
        print(
            "crash SANDBOX (euc freeze; UI narrative): \(report.crashSandboxVerified)"
                + " — euc=\(report.crashSandboxEucDelivered)"
                + " hyp=\(report.crashSandboxHypDelivered)"
                + " — \(report.crashSandboxNote)"
        )
        print("publication gate uses symmetric crash (sandbox is reported only)")
        print("")
        if report.passed {
            print("verified: true")
        } else {
            print("verified: false")
            exit(2)
        }
    }

    static func runVerifyAll(_ args: [String]) throws {
        print("=== GPNEC verify-all ===\n")
        try runVerifyLBM(args)
        print("")
        try runVerifyRoute(args)
        print("\n=== all verified ===")
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
        let policyRaw = parseFlag(args, name: "--policy") ?? "symmetric"

        print("GPNEC route-verify — Euclidean vs Poincaré greedy")
        print("N=\(n) K=\(k) crashTopC=\(crash) packets=\(packets) pre=\(pre) post=\(post) maxHops=\(maxHops)")
        print("policy=\(policyRaw)  (symmetric = fair control; sandbox = UI euc-freeze narrative)")
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

        func printReport(_ report: DualRouteVerification.Report) {
            func line(_ label: String, _ s: RouteStats) {
                let ratio = String(format: "%.3f", s.deliveryRatio)
                let hops = String(format: "%.2f", s.meanHopsDelivered)
                print(
                    "  \(label.padding(toLength: 18, withPad: " ", startingAt: 0))"
                        + " delivered=\(s.delivered) dropped=\(s.dropped) inFlight=\(s.inFlight)"
                        + " ratio=\(ratio) meanHops=\(hops) ticks=\(s.ticks)"
                )
            }
            print("--- policy: \(report.policy.rawValue) ---")
            print(String(format: "generate: %.1f ms", report.generateMilliseconds))
            print(
                "crashTargets[\(report.crashCount)]: "
                    + report.crashTargets.prefix(8).map(String.init).joined(separator: ",")
                    + (report.crashCount > 8 ? ",…" : "")
            )
            print("pre-crash:")
            line("euclidean", report.euclideanPre)
            line("poincare", report.poincarePre)
            print("post-crash:")
            line("euclidean", report.euclideanPost)
            line("poincare", report.poincarePost)
            print(String(format: "post hyp/euc delivery ratio: %.2f", report.postDeliveryRatio))
            print("note: \(report.note)")
            print("verified: \(report.verified)")
        }

        var anyFail = false
        switch policyRaw {
        case "both":
            let paired = try DualRouteVerification.runPaired(
                context: context,
                config: config,
                packetCount: packets,
                preCrashTicks: pre,
                postCrashTicks: post,
                packetSeed: seed &+ 0xBEEF,
                maxHops: maxHops
            )
            fputs("  embedding + dual Metal routes done\n", stderr)
            printReport(paired.symmetric)
            print("")
            printReport(paired.sandbox)
            anyFail = !paired.symmetric.verified
        case "sandbox":
            let report = try DualRouteVerification(
                config: config,
                packetCount: packets,
                preCrashTicks: pre,
                postCrashTicks: post,
                packetSeed: seed &+ 0xBEEF,
                maxHops: maxHops,
                postCrashPolicy: .sandbox
            ).run(context: context)
            fputs("  embedding + dual Metal routes done\n", stderr)
            printReport(report)
            anyFail = !report.verified
        default: // symmetric
            let report = try DualRouteVerification(
                config: config,
                packetCount: packets,
                preCrashTicks: pre,
                postCrashTicks: post,
                packetSeed: seed &+ 0xBEEF,
                maxHops: maxHops,
                postCrashPolicy: .symmetric
            ).run(context: context)
            fputs("  embedding + dual Metal routes done\n", stderr)
            printReport(report)
            anyFail = !report.verified
        }
        if anyFail {
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
        let raw = parseFlag(args, name: "--backends") ?? "metal,cpu"
        let parts = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let parsed = parts.compactMap { FluidBenchBackend(rawValue: $0) }
        return parsed.isEmpty ? [.metal, .cpu] : parsed
    }
}
