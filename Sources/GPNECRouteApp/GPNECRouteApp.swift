import AppKit
import SwiftUI
import GPNECRouteView

@main
struct GPNECRouteApp: App {
    @NSApplicationDelegateAdaptor(RouteAppDelegate.self) private var appDelegate
    @State private var diagnostics = RouteDiagnostics()

    var body: some Scene {
        WindowGroup("GPNEC Route") {
            RouteSandboxScreen(diagnostics: diagnostics)
                .frame(minWidth: 960, minHeight: 640)
        }
        .defaultSize(width: 1280, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

final class RouteAppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.activate(ignoringOtherApps: true)
        for window in NSApp.windows {
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

struct RouteSandboxScreen: View {
    @Bindable var diagnostics: RouteDiagnostics

    var body: some View {
        ZStack {
            RouteMetalView(diagnostics: diagnostics)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                    .padding(.horizontal, 28)
                    .padding(.top, 20)

                Spacer()

                HStack(alignment: .bottom, spacing: 24) {
                    panelStats(
                        title: "Euclidean",
                        subtitle: "landmark MDS plane",
                        delivered: diagnostics.eucDelivered,
                        dropped: diagnostics.eucDropped,
                        inFlight: diagnostics.eucInFlight,
                        accent: Color(red: 0.95, green: 0.55, blue: 0.15)
                    )
                    Spacer()
                    controls
                    Spacer()
                    panelStats(
                        title: "Poincaré",
                        subtitle: "hyperbolic disk",
                        delivered: diagnostics.hypDelivered,
                        dropped: diagnostics.hypDropped,
                        inFlight: diagnostics.hypInFlight,
                        accent: Color(red: 0.25, green: 0.85, blue: 0.82)
                    )
                }
                .padding(.horizontal, 28)
                .padding(.bottom, 24)
            }

            if diagnostics.isGenerating {
                loadingOverlay
            }
        }
        .background(Color(red: 0.05, green: 0.05, blue: 0.045))
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("GPNEC")
                    .font(.custom("Futura-Bold", size: 42))
                    .tracking(4)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [
                                Color(red: 0.92, green: 0.88, blue: 0.78),
                                Color(red: 0.55, green: 0.78, blue: 0.76),
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                Text("dual-greedy routing · \(diagnostics.nodeCount) nodes")
                    .font(.custom("Futura-Medium", size: 13))
                    .tracking(1.2)
                    .foregroundStyle(Color(red: 0.65, green: 0.62, blue: 0.55))
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(diagnostics.deviceName.isEmpty ? "—" : diagnostics.deviceName)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                if diagnostics.isReady {
                    Text(String(format: "%.0f fps · %d ticks", diagnostics.framesPerSecond, diagnostics.ticks))
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var controls: some View {
        VStack(spacing: 10) {
            Button {
                diagnostics.pendingCrash = true
            } label: {
                Text(diagnostics.hasCrashed ? "BACKBONE DOWN" : "CRASH BACKBONE")
                    .font(.custom("Futura-Bold", size: 14))
                    .tracking(2)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 12)
                    .background(
                        diagnostics.hasCrashed
                            ? Color(red: 0.45, green: 0.12, blue: 0.1)
                            : Color(red: 0.85, green: 0.2, blue: 0.16)
                    )
                    .foregroundStyle(Color(red: 0.98, green: 0.94, blue: 0.9))
            }
            .buttonStyle(.plain)
            .disabled(!diagnostics.isReady || diagnostics.hasCrashed)
            .opacity(diagnostics.isReady ? 1 : 0.4)

            Stepper(
                "steps/frame \(diagnostics.stepsPerFrame)",
                value: $diagnostics.stepsPerFrame,
                in: 1...16
            )
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(Color(red: 0.7, green: 0.68, blue: 0.6))
            .frame(width: 180)

            if let err = diagnostics.lastError {
                Text(err)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.red)
                    .frame(maxWidth: 260)
            }
        }
    }

    private func panelStats(
        title: String,
        subtitle: String,
        delivered: Int,
        dropped: Int,
        inFlight: Int,
        accent: Color
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.custom("Futura-Bold", size: 18))
                .foregroundStyle(accent)
            Text(subtitle)
                .font(.custom("Futura-Medium", size: 11))
                .foregroundStyle(Color(red: 0.55, green: 0.52, blue: 0.46))
            HStack(spacing: 16) {
                metric("delivered", "\(delivered)")
                metric("dropped", "\(dropped)")
                metric("in-flight", "\(inFlight)")
            }
            .font(.system(.caption, design: .monospaced))
        }
        .padding(14)
        .background(Color.black.opacity(0.28))
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .foregroundStyle(Color(red: 0.5, green: 0.48, blue: 0.42))
            Text(value)
                .foregroundStyle(Color(red: 0.9, green: 0.88, blue: 0.82))
        }
    }

    private var loadingOverlay: some View {
        VStack(spacing: 14) {
            Text("GPNEC")
                .font(.custom("Futura-Bold", size: 56))
                .tracking(6)
                .foregroundStyle(Color(red: 0.9, green: 0.86, blue: 0.76))
            Text("embedding \(diagnostics.nodeCount) nodes…")
                .font(.custom("Futura-Medium", size: 15))
                .foregroundStyle(Color(red: 0.6, green: 0.75, blue: 0.74))
            ProgressView()
                .controlSize(.small)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.05, green: 0.05, blue: 0.045).opacity(0.92))
    }
}
