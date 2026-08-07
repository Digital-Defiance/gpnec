#include <metal_stdlib>
using namespace metal;

struct PhiUniforms {
    uint batch;
    uint nodes;
    uint channels;
    float param0; // tau (relaxation time)
    float param1; // unused
    float param2;
    float param3;
};

// D2Q9 lattice weights and discrete velocities (channels 0..8).
constant float w9[9] = {
    4.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};
constant float2 e9[9] = {
    float2( 0,  0),
    float2( 1,  0), float2( 0,  1), float2(-1,  0), float2( 0, -1),
    float2( 1,  1), float2(-1,  1), float2(-1, -1), float2( 1, -1)
};

/// BGK collision: f* = f - (f - f_eq) / tau, conserving mass & momentum per node.
kernel void phi_bgk_collision(
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

    float tau = max(u.param0, 0.5f);
    uint Q = min(u.channels, 9u);

    float rho = 0.0f;
    float2 vel = float2(0.0f, 0.0f);
    float f[9];
    for (uint i = 0; i < 9; ++i) {
        f[i] = (i < Q) ? scratch[base + i] : 0.0f;
        rho += f[i];
        vel += f[i] * e9[i];
    }
    rho = max(rho, 1e-8f);
    vel /= rho;

    float usq = dot(vel, vel);
    for (uint i = 0; i < Q; ++i) {
        float eu = dot(e9[i], vel);
        float feq = w9[i] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
        output[base + i] = f[i] - (f[i] - feq) / tau;
    }
    for (uint i = Q; i < u.channels; ++i) {
        output[base + i] = scratch[base + i];
    }
}
