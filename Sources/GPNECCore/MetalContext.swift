import Metal

/// Owns the MTLDevice and compiled Φ compute libraries.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private var libraries: [String: MTLLibrary] = [:]
    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw EngineError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw EngineError.commandQueueFailed
        }
        self.device = device
        self.commandQueue = queue
    }

    /// Compile a named Metal source file from the bundled Shaders directory.
    public func pipeline(kernelName: String, sourceFile: String) throws -> MTLComputePipelineState {
        if let existing = pipelines[kernelName] {
            return existing
        }
        let library = try loadLibrary(sourceFile: sourceFile)
        guard let function = library.makeFunction(name: kernelName) else {
            throw EngineError.kernelNotFound(kernelName)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[kernelName] = pipeline
        return pipeline
    }

    private func loadLibrary(sourceFile: String) throws -> MTLLibrary {
        if let cached = libraries[sourceFile] {
            return cached
        }
        let source = try Self.loadShaderSource(named: sourceFile)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(source: source, options: options)
        libraries[sourceFile] = library
        return library
    }

    private static func loadShaderSource(named file: String) throws -> String {
        let name = (file as NSString).deletingPathExtension
        let ext = (file as NSString).pathExtension.isEmpty ? "metal" : (file as NSString).pathExtension

        if let url = Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Shaders")
            ?? Bundle.module.url(forResource: name, withExtension: ext)
        {
            return try String(contentsOf: url, encoding: .utf8)
        }

        // Fallback for tests / direct execution without resource bundle.
        let candidates = [
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .appendingPathComponent("Shaders")
                .appendingPathComponent("\(name).\(ext)"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Sources/GPNECCore/Shaders/\(name).\(ext)"),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            return try String(contentsOf: url, encoding: .utf8)
        }
        throw EngineError.shaderNotFound(file)
    }
}

public enum EngineError: Error, CustomStringConvertible {
    case noMetalDevice
    case commandQueueFailed
    case bufferAllocationFailed
    case kernelNotFound(String)
    case shaderNotFound(String)
    case graphCompilationFailed(String)
    case invalidState

    public var description: String {
        switch self {
        case .noMetalDevice: return "No Metal device available"
        case .commandQueueFailed: return "Failed to create MTLCommandQueue"
        case .bufferAllocationFailed: return "Failed to allocate MTLBuffer"
        case .kernelNotFound(let n): return "Metal kernel not found: \(n)"
        case .shaderNotFound(let n): return "Shader source not found: \(n)"
        case .graphCompilationFailed(let m): return "MPSGraph compilation failed: \(m)"
        case .invalidState: return "Engine in invalid state"
        }
    }
}
