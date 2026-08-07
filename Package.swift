// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GPNEC",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "GPNECCore", targets: ["GPNECCore"]),
        .library(name: "GPNECAdapters", targets: ["GPNECAdapters"]),
        .library(name: "GPNECFluidView", targets: ["GPNECFluidView"]),
        .library(name: "GPNECRouting", targets: ["GPNECRouting"]),
        .library(name: "GPNECRouteView", targets: ["GPNECRouteView"]),
        .library(name: "GPNECCBridge", type: .dynamic, targets: ["GPNECCBridge"]),
        .executable(name: "gpnec", targets: ["GPNECCLI"]),
        .executable(name: "gpnec-fluid", targets: ["GPNECFluidApp"]),
        .executable(name: "gpnec-route", targets: ["GPNECRouteApp"]),
    ],
    targets: [
        .target(
            name: "GPNECCore",
            resources: [
                .copy("Shaders"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),
        .target(
            name: "GPNECAdapters",
            dependencies: ["GPNECCore"]
        ),
        .target(
            name: "GPNECRouting",
            dependencies: ["GPNECCore"],
            resources: [
                .copy("Shaders"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),
        .target(
            name: "GPNECRouteView",
            dependencies: ["GPNECCore", "GPNECRouting"],
            resources: [
                .copy("Shaders"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
            ]
        ),
        .target(
            name: "GPNECFluidView",
            dependencies: ["GPNECCore", "GPNECAdapters"],
            resources: [
                .copy("Shaders"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
            ]
        ),
        .target(
            name: "GPNECCBridge",
            dependencies: ["GPNECCore", "GPNECAdapters"]
        ),
        .executableTarget(
            name: "GPNECCLI",
            dependencies: ["GPNECCore", "GPNECAdapters", "GPNECRouting"]
        ),
        .executableTarget(
            name: "GPNECFluidApp",
            dependencies: ["GPNECFluidView", "GPNECCore", "GPNECAdapters"]
        ),
        .executableTarget(
            name: "GPNECRouteApp",
            dependencies: ["GPNECRouteView", "GPNECCore", "GPNECRouting"]
        ),
        .testTarget(
            name: "GPNECCoreTests",
            dependencies: ["GPNECCore", "GPNECAdapters", "GPNECRouting"]
        ),
    ]
)
