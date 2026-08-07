import AppKit
import Metal
import MetalKit
import SwiftUI
import GPNECCore

/// Full-window MTKView: split Euclidean | Poincaré, zero-copy from router buffers.
public struct RouteMetalView: NSViewRepresentable {
    public var diagnostics: RouteDiagnostics

    public init(diagnostics: RouteDiagnostics) {
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
        context.coordinator.sandbox.diagnostics.stepsPerFrame = diagnostics.stepsPerFrame
    }

    @MainActor
    public final class Coordinator: NSObject, MTKViewDelegate {
        var diagnostics: RouteDiagnostics
        let sandbox: RouteSandbox
        private var lastFrameTime: CFTimeInterval = CACurrentMediaTime()
        private var commandQueue: MTLCommandQueue?

        init(diagnostics: RouteDiagnostics) {
            self.diagnostics = diagnostics
            self.sandbox = RouteSandbox(diagnostics: diagnostics)
            super.init()
        }

        func makeView() -> MTKView {
            do {
                guard let device = MTLCreateSystemDefaultDevice() else {
                    throw EngineError.noMetalDevice
                }
                let ctx = try sandbox.attach(device: device, pixelFormat: .bgra8Unorm)
                commandQueue = ctx.commandQueue

                let view = MTKView(frame: .zero, device: device)
                view.delegate = self
                view.framebufferOnly = true
                view.colorPixelFormat = .bgra8Unorm
                view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.045, alpha: 1)
                view.isPaused = false
                view.enableSetNeedsDisplay = false
                view.preferredFramesPerSecond = 60
                sandbox.bootstrapIfNeeded()
                return view
            } catch {
                diagnostics.lastError = String(describing: error)
                let fallback = MTKView(frame: .zero)
                fallback.device = MTLCreateSystemDefaultDevice()
                return fallback
            }
        }

        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        public func draw(in view: MTKView) {
            let t0 = CACurrentMediaTime()
            guard let queue = commandQueue,
                  let cb = queue.makeCommandBuffer(),
                  let drawable = view.currentDrawable
            else { return }

            sandbox.encodeFrame(into: view, commandBuffer: cb)
            cb.present(drawable)
            cb.commit()
            sandbox.waitForGPU(cb)

            let t1 = CACurrentMediaTime()
            let dt = t1 - lastFrameTime
            lastFrameTime = t1
            diagnostics.frameMilliseconds = (t1 - t0) * 1000
            if dt > 0 {
                diagnostics.framesPerSecond = 1 / dt
            }
        }

        func triggerCrash() {
            sandbox.requestCrash()
        }

        func resetTraffic() {
            sandbox.resetTraffic()
        }
    }
}
