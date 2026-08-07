import AppKit
import Metal
import MetalKit
import SwiftUI
import GPNECCore
import GPNECAdapters

/// Hosts an `MTKView` that advances LBM with a fixed steps-per-frame budget
/// and renders via zero-copy buffer→fragment binding.
public struct FluidMetalView: NSViewRepresentable {
    public var diagnostics: FluidFrameDiagnostics

    public init(diagnostics: FluidFrameDiagnostics) {
        self.diagnostics = diagnostics
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(diagnostics: diagnostics)
    }

    public func makeNSView(context: Context) -> MTKView {
        context.coordinator.makeView()
    }

    public func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.diagnostics = diagnostics
        context.coordinator.stepsPerFrame = diagnostics.stepsPerFrame
        if let view = nsView as? InteractiveMTKView {
            view.coordinator = context.coordinator
        }
    }

    /// Click / drag → lattice droplet injection.
    final class InteractiveMTKView: MTKView {
        weak var coordinator: Coordinator?

        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            window?.makeFirstResponder(self)
        }

        override func mouseDown(with event: NSEvent) {
            window?.makeFirstResponder(self)
            coordinator?.handlePointer(at: event, in: self, primary: true)
        }

        override func mouseDragged(with event: NSEvent) {
            coordinator?.handlePointer(at: event, in: self, primary: false)
        }
    }

    @MainActor
    public final class Coordinator: NSObject, MTKViewDelegate {
        var diagnostics: FluidFrameDiagnostics
        var stepsPerFrame: Int

        private var metalContext: MetalContext!
        private var engine: TensorEngine?
        private var renderer: FluidRenderer?
        private var lastFrameTime: CFTimeInterval = CACurrentMediaTime()
        private let latticeWidth: Int
        private let latticeHeight: Int
        private var isBootstrapping = false
        private var didStartBootstrap = false
        private var pendingDroplets: [(Int, Int)] = []
        private var lastDropCell: (Int, Int)?
        private var dropletCount: Int = 0
        private var lastRenderBuffer: MTLCommandBuffer?

        init(diagnostics: FluidFrameDiagnostics) {
            self.diagnostics = diagnostics
            self.stepsPerFrame = diagnostics.stepsPerFrame
            self.latticeWidth = diagnostics.latticeWidth
            self.latticeHeight = diagnostics.latticeHeight
            super.init()
        }

        func makeView() -> MTKView {
            do {
                let ctx = try MetalContext()
                self.metalContext = ctx
                diagnostics.deviceName = ctx.device.name

                let view = InteractiveMTKView(frame: .zero, device: ctx.device)
                view.coordinator = self
                view.delegate = self
                view.framebufferOnly = true
                view.colorPixelFormat = .bgra8Unorm
                view.clearColor = MTLClearColor(red: 0.02, green: 0.03, blue: 0.05, alpha: 1)
                view.isPaused = false
                view.enableSetNeedsDisplay = false
                view.preferredFramesPerSecond = 60
                return view
            } catch {
                diagnostics.lastError = String(describing: error)
                let fallback = MTKView(frame: .zero)
                fallback.device = MTLCreateSystemDefaultDevice()
                return fallback
            }
        }

        func handlePointer(at event: NSEvent, in view: MTKView, primary: Bool) {
            guard let cell = latticeCell(for: event, in: view) else { return }
            if !primary, let last = lastDropCell, last == cell { return }
            lastDropCell = cell
            pendingDroplets.append(cell)
            if primary {
                dropletCount += 1
                diagnostics.dropletCount = dropletCount
            }
        }

        private func latticeCell(for event: NSEvent, in view: MTKView) -> (Int, Int)? {
            let p = view.convert(event.locationInWindow, from: nil)
            guard view.bounds.width > 0, view.bounds.height > 0 else { return nil }
            // AppKit: origin bottom-left. Fragment maps screen-top → lattice y = 0.
            let nx = p.x / view.bounds.width
            let ny = p.y / view.bounds.height
            let x = Int(floor(nx * CGFloat(latticeWidth)))
            let y = Int(floor((1.0 - ny) * CGFloat(latticeHeight)))
            guard x >= 0, x < latticeWidth, y >= 0, y < latticeHeight else { return nil }
            return (x, y)
        }

        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        public func draw(in view: MTKView) {
            if engine == nil {
                bootstrapIfNeeded(view: view)
                return
            }
            guard let engine, let renderer else { return }

            // Ensure prior frame finished before CPU droplet writes into stateBuffer.
            lastRenderBuffer?.waitUntilCompleted()
            lastRenderBuffer = nil

            // Apply clicks before the deterministic step budget (stable forcing).
            if !pendingDroplets.isEmpty {
                let drops = pendingDroplets
                pendingDroplets.removeAll(keepingCapacity: true)
                for (x, y) in drops {
                    LatticeBoltzmannInput.injectDroplet(
                        into: engine,
                        width: latticeWidth,
                        height: latticeHeight,
                        cellX: x,
                        cellY: y
                    )
                }
            }

            let frameStart = CACurrentMediaTime()
            let frameDelta = frameStart - lastFrameTime
            lastFrameTime = frameStart
            if frameDelta > 1e-6 {
                diagnostics.frameMilliseconds = frameDelta * 1000
                diagnostics.framesPerSecond = 1.0 / frameDelta
            }

            let budget = max(1, stepsPerFrame)
            let computeStart = CACurrentMediaTime()
            do {
                try engine.advance(steps: budget)
            } catch {
                diagnostics.lastError = String(describing: error)
                return
            }
            let computeEnd = CACurrentMediaTime()
            diagnostics.computeMilliseconds = (computeEnd - computeStart) * 1000
            diagnostics.stepsExecuted = engine.stepsExecuted

            guard let commandBuffer = renderer.commandQueue.makeCommandBuffer() else { return }
            renderer.encode(
                into: view,
                stateBuffer: engine.stateBuffer,
                commandBuffer: commandBuffer
            )
            commandBuffer.commit()
            lastRenderBuffer = commandBuffer
        }

        private func bootstrapIfNeeded(view: MTKView) {
            guard !didStartBootstrap, !isBootstrapping else { return }
            didStartBootstrap = true
            isBootstrapping = true
            diagnostics.lastError = nil

            let ctx = metalContext!
            let width = latticeWidth
            let height = latticeHeight
            let pixelFormat = view.colorPixelFormat

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                do {
                    let adapter = FluidSimulatorAdapter(
                        width: width,
                        height: height,
                        tau: 0.56,
                        inletUx: 0.08
                    )
                    let built = try adapter.makeEngine(context: ctx)
                    let uniforms = FluidRenderUniforms(
                        width: width,
                        height: height,
                        channels: built.shape.channels
                    )
                    let builtRenderer = try FluidRenderer(
                        device: ctx.device,
                        pixelFormat: pixelFormat,
                        uniforms: uniforms
                    )
                    try built.advance(steps: 1)

                    DispatchQueue.main.async {
                        self.engine = built
                        self.renderer = builtRenderer
                        self.diagnostics.stepsExecuted = built.stepsExecuted
                        self.isBootstrapping = false
                        fputs("gpnec-fluid: engine ready (\(width)×\(height)) — click to drop\n", stderr)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.diagnostics.lastError = String(describing: error)
                        self.isBootstrapping = false
                    }
                }
            }
        }
    }
}
