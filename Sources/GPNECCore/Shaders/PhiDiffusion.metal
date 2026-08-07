#include <metal_stdlib>
using namespace metal;

struct PhiUniforms {
    uint batch;
    uint nodes;
    uint channels;
    float param0; // diffusion / temperature mix
    float param1; // control gain
    float param2;
    float param3;
};

/// Positional control diffusion on a non-Euclidean lattice.
/// Channel 0: board occupancy / piece identity
/// Channel 1: control field (Sensor Net) — diffused by topology W, then Φ
/// Channel 2+: local curvature / manifold features (pass-through with light damp)
///
/// `param0` = retain mix in [0,1]: how much of the W-diffused control to keep
/// versus pulling toward occupancy. High retain (≈0.85–0.95) preserves Sensor
/// Net bloom seeded from the host; low retain collapses toward pieces only.
kernel void phi_subspace_diffusion(
    device const float *scratch [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant PhiUniforms &u [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint totalNodes = u.batch * u.nodes;
    if (gid >= totalNodes) return;
    uint b = gid / u.nodes;
    uint n = gid % u.nodes;
    uint base = (b * u.nodes + n) * u.channels;

    float retain = clamp(u.param0, 0.0f, 1.0f);
    float gain = u.param1;

    float occ = scratch[base + 0];
    float ctrl = (u.channels > 1) ? scratch[base + 1] : 0.0f;

    // W already mixed neighbors into ctrl. Keep that bloom; lightly reinforce
    // from pieces without wiping empty coverage cells toward zero.
    float piecePull = tanh(gain * occ);
    float diffused = retain * ctrl + (1.0f - retain) * piecePull;
    diffused = clamp(diffused, -1.0f, 1.0f);

    output[base + 0] = occ; // pieces persist (movement is via W)
    if (u.channels > 1) {
        output[base + 1] = diffused;
    }
    for (uint c = 2; c < u.channels; ++c) {
        float curv = scratch[base + c];
        output[base + c] = curv * (0.98f + 0.02f * diffused);
    }
}
