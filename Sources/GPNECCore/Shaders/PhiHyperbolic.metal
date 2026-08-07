#include <metal_stdlib>
using namespace metal;

struct PhiUniforms {
    uint batch;
    uint nodes;
    uint channels;
    float param0; // curvature scale
    float param1; // neighbors per node (stride in aux)
    float param2;
    float param3;
};

inline float poincare_distance(float2 a, float2 d, float scale) {
    float na = length(a);
    float nd = length(d);
    if (na >= 0.999f) a *= 0.999f / na;
    if (nd >= 0.999f) d *= 0.999f / nd;
    float num = length(a - d);
    float den = max(1.0f - dot(a, a), 1e-6f) * max(1.0f - dot(d, d), 1e-6f);
    float arg = 1.0f + 2.0f * (num * num) / den;
    return log(arg + sqrt(max(arg * arg - 1.0f, 0.0f))) * scale;
}

/// Greedy hyperbolic router. Aux layout: for each node, `K` neighbor indices (int-as-float).
/// Thread 0 per batch performs the hop so a single in-flight packet stays coherent.
kernel void phi_hyperbolic_greedy(
    device const float *scratch [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant PhiUniforms &u [[buffer(2)]],
    device const float *aux [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint totalNodes = u.batch * u.nodes;
    if (gid >= totalNodes) return;

    uint b = gid / u.nodes;
    uint n = gid % u.nodes;
    uint base = (b * u.nodes + n) * u.channels;
    float scale = max(u.param0, 1e-6f);
    uint K = max(uint(u.param1), 1u);

    // Default: copy geometry through.
    for (uint c = 0; c < u.channels; ++c) {
        output[base + c] = scratch[base + c];
    }

    float2 pos = float2(scratch[base + 0], scratch[base + 1]);
    float2 dest = float2(scratch[base + 2], scratch[base + 3]);
    float dist = poincare_distance(pos, dest, scale);
    output[base + 5] = dist;

    // One thread per batch owns packet flags to avoid races.
    if (n != 0) return;

    for (uint i = 0; i < u.nodes; ++i) {
        uint ib = (b * u.nodes + i) * u.channels;
        output[ib + 4] = 0.0f;
    }

    int owner = -1;
    for (uint i = 0; i < u.nodes; ++i) {
        uint ib = (b * u.nodes + i) * u.channels;
        if (scratch[ib + 4] > 0.5f) {
            owner = int(i);
            break;
        }
    }
    if (owner < 0) return;

    uint ob = (b * u.nodes + uint(owner)) * u.channels;
    float2 opos = float2(scratch[ob + 0], scratch[ob + 1]);
    float2 odst = float2(scratch[ob + 2], scratch[ob + 3]);
    float bestDist = poincare_distance(opos, odst, scale);
    int best = owner;

    for (uint k = 0; k < K; ++k) {
        int nb = int(aux[uint(owner) * K + k]);
        if (nb < 0 || uint(nb) >= u.nodes) continue;
        uint nbBase = (b * u.nodes + uint(nb)) * u.channels;
        float2 npos = float2(scratch[nbBase + 0], scratch[nbBase + 1]);
        float nd = poincare_distance(npos, odst, scale);
        if (nd + 1e-6f < bestDist) {
            bestDist = nd;
            best = nb;
        }
    }

    uint bestBase = (b * u.nodes + uint(best)) * u.channels;
    output[bestBase + 4] = 1.0f;
}
