#include <metal_stdlib>
using namespace metal;

struct FluidRenderUniforms {
    uint width;
    uint height;
    uint channels;
    uint batchIndex;
    float densityBias;
    float densityScale;
    float velocityScale;
    float vorticityScale;
};

struct FluidVertexOut {
    float4 position [[position]];
    float2 uv;
};

constant float2 e9[9] = {
    float2( 0,  0),
    float2( 1,  0), float2( 0,  1), float2(-1,  0), float2( 0, -1),
    float2( 1,  1), float2(-1,  1), float2(-1, -1), float2( 1, -1)
};

vertex FluidVertexOut fluid_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 pos;
    switch (vid) {
        case 0:  pos = float2(-1.0, -1.0); break;
        case 1:  pos = float2( 3.0, -1.0); break;
        default: pos = float2(-1.0,  3.0); break;
    }
    FluidVertexOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    return out;
}

inline void sample_ru(
    device const float *state,
    int x, int y, uint w, uint h, uint ch, uint batch,
    thread float &rho, thread float2 &vel, thread float &dye
) {
    x = clamp(x, 0, int(w) - 1);
    y = clamp(y, 0, int(h) - 1);
    uint node = uint(y) * w + uint(x);
    uint nodes = w * h;
    uint base = (batch * nodes + node) * ch;
    rho = 0.0;
    vel = float2(0.0);
    for (uint q = 0; q < 9u; ++q) {
        float f = state[base + q];
        rho += f;
        vel += f * e9[q];
    }
    rho = max(rho, 1e-8);
    vel /= rho;
    dye = (ch > 9u) ? state[base + 9] : 0.0;
}

/// Bilinear filter over lattice cells (smooth upscale; sim stays 256²).
inline void sample_ru_bilinear(
    device const float *state,
    float2 latticeUV, // continuous cell coords, y down
    uint w, uint h, uint ch, uint batch,
    thread float &rho, thread float2 &vel, thread float &dye
) {
    float fx = latticeUV.x - 0.5;
    float fy = latticeUV.y - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float tx = fx - float(x0);
    float ty = fy - float(y0);

    float r00, r10, r01, r11;
    float2 v00, v10, v01, v11;
    float d00, d10, d01, d11;
    sample_ru(state, x0,     y0,     w, h, ch, batch, r00, v00, d00);
    sample_ru(state, x0 + 1, y0,     w, h, ch, batch, r10, v10, d10);
    sample_ru(state, x0,     y0 + 1, w, h, ch, batch, r01, v01, d01);
    sample_ru(state, x0 + 1, y0 + 1, w, h, ch, batch, r11, v11, d11);

    float r0 = mix(r00, r10, tx);
    float r1 = mix(r01, r11, tx);
    rho = mix(r0, r1, ty);

    float2 v0 = mix(v00, v10, tx);
    float2 v1 = mix(v01, v11, tx);
    vel = mix(v0, v1, ty);

    float d0 = mix(d00, d10, tx);
    float d1 = mix(d01, d11, tx);
    dye = mix(d0, d1, ty);
}

inline float3 palette(float speed, float vort, float dye) {
    float t = saturate(speed);
    float3 cold = float3(0.02, 0.06, 0.14);
    float3 mid  = float3(0.05, 0.42, 0.58);
    float3 hot  = float3(0.95, 0.55, 0.12);
    float3 rgb = (t < 0.5) ? mix(cold, mid, t * 2.0) : mix(mid, hot, (t - 0.5) * 2.0);
    rgb += float3(-0.2, 0.04, 0.28) * vort;
    rgb += float3(0.28, -0.12, -0.2) * (-vort);
    // Soft dye glow (sqrt softens hard stair-steps after bilinear).
    float dyeSoft = pow(saturate(dye), 0.65);
    rgb = mix(rgb, float3(0.92, 0.95, 1.0), dyeSoft * 0.9);
    return saturate(rgb);
}

fragment float4 fluid_density_velocity_fragment(
    FluidVertexOut in [[stage_in]],
    device const float *state [[buffer(0)]],
    constant FluidRenderUniforms &u [[buffer(1)]]
) {
    uint w = max(u.width, 1u);
    uint h = max(u.height, 1u);
    uint ch = max(u.channels, 9u);

    float2 uv = saturate(in.uv);
    // Continuous lattice coords (match sim: y=0 at top of lattice / top of screen).
    float2 lattice = float2(uv.x * float(w), (1.0 - uv.y) * float(h));

    float rho;
    float2 vel;
    float dye;
    sample_ru_bilinear(state, lattice, w, h, ch, u.batchIndex, rho, vel, dye);

    // Vorticity from bilinear velocity at ±½ cell (smooth curl, less grid shimmer).
    float rhoT;
    float2 vL, vR, vD, vU;
    float dyeT;
    sample_ru_bilinear(state, lattice + float2(-0.5, 0.0), w, h, ch, u.batchIndex, rhoT, vL, dyeT);
    sample_ru_bilinear(state, lattice + float2( 0.5, 0.0), w, h, ch, u.batchIndex, rhoT, vR, dyeT);
    sample_ru_bilinear(state, lattice + float2( 0.0,-0.5), w, h, ch, u.batchIndex, rhoT, vD, dyeT);
    sample_ru_bilinear(state, lattice + float2( 0.0, 0.5), w, h, ch, u.batchIndex, rhoT, vU, dyeT);

    float vort = ((vR.y - vL.y) - (vU.x - vD.x)) * u.vorticityScale;
    vort = clamp(vort, -1.0, 1.0);

    float speed = length(vel) * u.velocityScale;
    float dens = (rho - u.densityBias) * u.densityScale;
    float3 rgb = palette(speed, vort, dye);
    rgb *= (0.75 + 0.25 * saturate(0.5 + dens));

    // Soft cylinder shading (no hard per-cell black blocks).
    float stillness = 1.0 - smoothstep(0.0, 0.03, length(vel));
    float undyed = 1.0 - smoothstep(0.0, 0.08, dye);
    rgb *= mix(1.0, 0.4, stillness * undyed * 0.85);

    return float4(rgb, 1.0);
}
