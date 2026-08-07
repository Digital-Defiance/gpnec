#include <metal_stdlib>
using namespace metal;

struct PhiUniforms {
    uint batch;
    uint nodes;
    uint channels; // 10 = D2Q9 + dye
    float param0;  // tau
    float param1;  // width
    float param2;  // height
    float param3;  // inlet ux
};

constant float w9[9] = {
    4.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};
constant int2 e9i[9] = {
    int2( 0,  0),
    int2( 1,  0), int2( 0,  1), int2(-1,  0), int2( 0, -1),
    int2( 1,  1), int2(-1,  1), int2(-1, -1), int2( 1, -1)
};
constant float2 e9[9] = {
    float2( 0,  0),
    float2( 1,  0), float2( 0,  1), float2(-1,  0), float2( 0, -1),
    float2( 1,  1), float2(-1,  1), float2(-1, -1), float2( 1, -1)
};
// Opposite discrete velocities for bounce-back.
constant uint opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

inline uint idx(uint b, uint x, uint y, uint w, uint h, uint ch, uint q) {
    uint n = y * w + x;
    uint nodes = w * h;
    return (b * nodes + n) * ch + q;
}

/// Pass A: BGK collision. Solids copy through. Dye (ch 9) copied unchanged.
kernel void lbm_bgk_collide(
    device const float *input [[buffer(0)]],
    device float *collided [[buffer(1)]],
    constant PhiUniforms &u [[buffer(2)]],
    device const float *solid [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = u.batch * u.nodes;
    if (gid >= total) return;

    uint width = max(uint(u.param1), 1u);
    uint height = max(uint(u.param2), 1u);
    uint ch = max(u.channels, 10u);
    uint b = gid / u.nodes;
    uint n = gid % u.nodes;
    uint x = n % width;
    uint y = n / width;
    uint base = (b * u.nodes + n) * ch;

    if (solid[n] > 0.5f) {
        for (uint q = 0; q < ch; ++q) {
            collided[base + q] = input[base + q];
        }
        return;
    }

    float tau = max(u.param0, 0.51f);
    float f[9];
    float rho = 0.0f;
    float2 vel = float2(0.0f);
    for (uint q = 0; q < 9u; ++q) {
        f[q] = input[base + q];
        rho += f[q];
        vel += f[q] * e9[q];
    }
    rho = max(rho, 1e-8f);
    vel /= rho;
    float usq = dot(vel, vel);

    for (uint q = 0; q < 9u; ++q) {
        float eu = dot(e9[q], vel);
        float feq = w9[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
        collided[base + q] = f[q] - (f[q] - feq) / tau;
    }
    // Dye pass-through (advection happens in stream kernel).
    collided[base + 9] = (ch > 9u) ? input[base + 9] : 0.0f;
}

/// Pass B: pull streaming + bounce-back + inlet + upwind dye.
kernel void lbm_stream_bounce(
    device const float *collided [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant PhiUniforms &u [[buffer(2)]],
    device const float *solid [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = u.batch * u.nodes;
    if (gid >= total) return;

    uint width = max(uint(u.param1), 1u);
    uint height = max(uint(u.param2), 1u);
    uint ch = max(u.channels, 10u);
    float inletUx = u.param3;
    uint b = gid / u.nodes;
    uint n = gid % u.nodes;
    uint x = n % width;
    uint y = n / width;
    uint base = (b * u.nodes + n) * ch;

    if (solid[n] > 0.5f) {
        // Solid: rest equilibrium, no dye.
        for (uint q = 0; q < 9u; ++q) {
            output[base + q] = w9[q];
        }
        if (ch > 9u) output[base + 9] = 0.0f;
        return;
    }

    float f[9];
    for (uint q = 0; q < 9u; ++q) {
        int sx = int(x) - e9i[q].x;
        int sy = int(y) - e9i[q].y;
        // Periodic in y; bounce / inlet handled in x.
        if (sy < 0) sy += int(height);
        if (sy >= int(height)) sy -= int(height);

        bool outX = (sx < 0 || sx >= int(width));
        if (outX) {
            // Horizontal open boundaries: bounce from self (approx).
            f[q] = collided[base + opp[q]];
            continue;
        }

        uint sn = uint(sy) * width + uint(sx);
        if (solid[sn] > 0.5f) {
            // Mid-grid bounce-back from obstacle.
            f[q] = collided[base + opp[q]];
        } else {
            f[q] = collided[idx(b, uint(sx), uint(sy), width, height, ch, q)];
        }
    }

    // Simple velocity inlet on left column.
    if (x == 0u) {
        float rho = 1.0f;
        float2 vel = float2(inletUx, 0.0f);
        float usq = dot(vel, vel);
        for (uint q = 0; q < 9u; ++q) {
            float eu = dot(e9[q], vel);
            f[q] = w9[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
        }
    }

    for (uint q = 0; q < 9u; ++q) {
        output[base + q] = f[q];
    }

    // Semi-Lagrangian dye pull with bounce (no flux into solid).
    float rho = 0.0f;
    float2 vel = float2(0.0f);
    for (uint q = 0; q < 9u; ++q) {
        rho += f[q];
        vel += f[q] * e9[q];
    }
    rho = max(rho, 1e-8f);
    vel /= rho;

    float dye = collided[base + 9];
    int dx = (vel.x > 0.08f) ? 1 : ((vel.x < -0.08f) ? -1 : 0);
    int dy = (vel.y > 0.08f) ? 1 : ((vel.y < -0.08f) ? -1 : 0);
    int sx = int(x) - dx;
    int sy = int(y) - dy;
    if (sy < 0) sy += int(height);
    if (sy >= int(height)) sy -= int(height);
    if (sx >= 0 && sx < int(width)) {
        uint sn = uint(sy) * width + uint(sx);
        if (solid[sn] < 0.5f) {
            dye = collided[idx(b, uint(sx), uint(sy), width, height, ch, 9)];
        }
    }
    // Mild dissipation + inlet dye seed strips (wake tracers).
    dye *= 0.999f;
    if (x == 1u && (y % 32u) < 6u) {
        dye = max(dye, 0.85f);
    }
    output[base + 9] = saturate(dye);
}
