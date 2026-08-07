import AppKit
import SwiftUI
import GPNECFluidView

@main
struct GPNECFluidApp: App {
    @NSApplicationDelegateAdaptor(FluidAppDelegate.self) private var appDelegate
    @State private var diagnostics = FluidFrameDiagnostics(
        stepsPerFrame: 16,
        latticeWidth: 256,
        latticeHeight: 256
    )

    var body: some Scene {
        WindowGroup("GPNEC Fluid") {
            ZStack(alignment: .topLeading) {
                FluidMetalView(diagnostics: diagnostics)
                    .ignoresSafeArea()

                DiagnosticHUD(diagnostics: diagnostics)
                    .padding(12)
            }
            .frame(minWidth: 720, minHeight: 520)
        }
        .defaultSize(width: 960, height: 720)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

/// SPM CLI executables default to `.accessory` and never surface a window/Dock icon.
final class FluidAppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        // Must run before scenes materialize.
        NSApplication.shared.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        fputs("gpnec-fluid: NSApp activationPolicy=regular — activating\n", stderr)
        NSApp.activate(ignoringOtherApps: true)
        for window in NSApp.windows {
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
        }
        // Retry once after SwiftUI creates the WindowGroup content.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            NSApp.activate(ignoringOtherApps: true)
            if let window = NSApp.windows.first {
                window.makeKeyAndOrderFront(nil)
                fputs("gpnec-fluid: window count=\(NSApp.windows.count) title=\(window.title)\n", stderr)
            } else {
                fputs("gpnec-fluid: WARNING — no NSWindow after launch\n", stderr)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

/// SwiftUI overlay only — Metal stays dedicated to simulate + fluid shade.
struct DiagnosticHUD: View {
    @Bindable var diagnostics: FluidFrameDiagnostics

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("GPNEC · D2Q9 LBM")
                .font(.system(.headline, design: .monospaced))
            Text(diagnostics.deviceName.isEmpty ? "device: —" : diagnostics.deviceName)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)

            if diagnostics.stepsExecuted == 0 && diagnostics.lastError == nil {
                Text("booting engine…")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.orange)
            }

            Divider().frame(width: 220)

            metric("lattice", "\(diagnostics.latticeWidth)×\(diagnostics.latticeHeight)")
            metric("backend", "collide+stream")
            metric("droplets", "\(diagnostics.dropletCount)")
            metric("steps/frame", "\(diagnostics.stepsPerFrame)")
            metric("steps total", "\(diagnostics.stepsExecuted)")
            metric("compute", String(format: "%.3f ms", diagnostics.computeMilliseconds))
            metric("frame", String(format: "%.3f ms", diagnostics.frameMilliseconds))
            metric("display", String(format: "%.1f fps", diagnostics.framesPerSecond))
            metric("compute/frame", String(format: "%.1f%%", diagnostics.computeFrameRatio * 100))
            metric("headroom", String(format: "%.3f ms", diagnostics.headroomMilliseconds))

            Text("cylinder wake · click injects dye")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)

            Stepper("Budget \(diagnostics.stepsPerFrame)", value: $diagnostics.stepsPerFrame, in: 1...64)
                .font(.system(.caption, design: .monospaced))
                .frame(width: 220)

            if let err = diagnostics.lastError {
                Text(err)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.red)
                    .frame(maxWidth: 280, alignment: .leading)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .padding(4)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value)
        }
        .font(.system(.caption, design: .monospaced))
        .frame(width: 220)
    }
}
