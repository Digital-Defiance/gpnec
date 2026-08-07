import Foundation

/// Identifies which Φ kernel and topology construction an adapter uses.
public enum AdapterDomain: String, Sendable, CaseIterable {
    case latticeBoltzmann = "lbm"
    case subspaceLattice = "subspace"
    case hyperbolicRouter = "hyperbolic"
}
