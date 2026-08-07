#include <metal_stdlib>
using namespace metal;

struct VizUniforms {
    float2 scale;   // position → NDC scale
    float2 origin;  // NDC center of panel
    uint nodeCount;
    uint packetCount;
    uint panelMode; // 0 = Euclidean plane, 1 = Poincaré disk
    float nodePointSize;
    float packetPointSize;
    float time;
    float crashedFlash;
    float _pad;
};

struct Packet {
    uint atNode;
    uint dstNode;
    uint hops;
    uint flags;
};

struct NodeVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
};

struct PacketVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
};

inline float2 map_to_ndc(float2 p, constant VizUniforms &u) {
    return u.origin + p * u.scale;
}

vertex NodeVertexOut route_node_vertex(
    uint iid [[instance_id]],
    device const float2 *positions [[buffer(0)]],
    device const uint *alive [[buffer(1)]],
    constant VizUniforms &u [[buffer(2)]]
) {
    NodeVertexOut out;
    float2 p = positions[iid];
    float2 ndc = map_to_ndc(p, u);
    out.position = float4(ndc, 0.0, 1.0);

    bool dead = alive[iid] == 0u;
    if (dead) {
        out.pointSize = u.nodePointSize * 1.8;
        float pulse = 0.55 + 0.45 * sin(u.time * 6.0 + float(iid) * 0.01);
        out.color = float4(0.92, 0.22, 0.18, 0.55 + 0.35 * pulse * u.crashedFlash);
    } else {
        out.pointSize = u.nodePointSize;
        if (u.panelMode == 1u) {
            out.color = float4(0.35, 0.55, 0.62, 0.22);
        } else {
            out.color = float4(0.62, 0.55, 0.42, 0.38);
        }
    }
    return out;
}

fragment float4 route_node_fragment(NodeVertexOut in [[stage_in]],
                                    float2 pc [[point_coord]]) {
    float2 d = pc - float2(0.5);
    float r = length(d);
    if (r > 0.5) discard_fragment();
    float alpha = smoothstep(0.5, 0.2, r) * in.color.a;
    return float4(in.color.rgb, alpha);
}

vertex PacketVertexOut route_packet_vertex(
    uint iid [[instance_id]],
    device const float2 *positions [[buffer(0)]],
    device const Packet *packets [[buffer(1)]],
    constant VizUniforms &u [[buffer(2)]]
) {
    PacketVertexOut out;
    Packet pkt = packets[iid];
    float2 p = positions[pkt.atNode];
    float2 ndc = map_to_ndc(p, u);
    out.position = float4(ndc, 0.0, 1.0);

    bool alive = (pkt.flags & 1u) != 0u;
    bool delivered = (pkt.flags & 2u) != 0u;
    bool dropped = (pkt.flags & 4u) != 0u;

    if (dropped) {
        out.pointSize = 1.0;
        out.color = float4(0.0);
    } else if (delivered) {
        out.pointSize = u.packetPointSize * 0.6;
        out.color = float4(0.55, 0.85, 0.45, 0.35);
    } else if (alive) {
        out.pointSize = u.packetPointSize;
        float glow = 0.75 + 0.25 * sin(u.time * 8.0 + float(iid) * 0.2);
        if (u.panelMode == 1u) {
            // Cyan / sea-glass packets (Poincaré)
            out.color = float4(0.25, 0.92, 0.88, 0.92 * glow);
        } else {
            // Amber packets (Euclidean) — bottleneck heat
            out.color = float4(0.98, 0.62, 0.18, 0.95 * glow);
        }
    } else {
        out.pointSize = 1.0;
        out.color = float4(0.0);
    }
    return out;
}

fragment float4 route_packet_fragment(PacketVertexOut in [[stage_in]],
                                      float2 pc [[point_coord]]) {
    if (in.color.a < 0.01) discard_fragment();
    float2 d = pc - float2(0.5);
    float r = length(d);
    if (r > 0.5) discard_fragment();
    float core = smoothstep(0.5, 0.05, r);
    return float4(in.color.rgb, in.color.a * core);
}

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenOut route_bg_vertex(uint vid [[vertex_id]]) {
    float2 pos[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    FullscreenOut out;
    out.position = float4(pos[vid], 0, 1);
    out.uv = pos[vid] * 0.5 + 0.5;
    return out;
}

fragment float4 route_bg_fragment(FullscreenOut in [[stage_in]],
                                  constant VizUniforms &u [[buffer(0)]]) {
    // Split-screen atmosphere: warm ash left / cool teal right.
    float x = in.uv.x;
    float3 left = float3(0.07, 0.065, 0.055);
    float3 right = float3(0.04, 0.07, 0.08);
    float3 base = mix(left, right, smoothstep(0.48, 0.52, x));

    // Soft vignette
    float2 c = in.uv - 0.5;
    float vig = 1.0 - dot(c, c) * 0.55;
    base *= vig;

    // Poincaré disk rim on right half
    if (x > 0.5) {
        float2 local = float2((x - 0.75) / 0.25, (in.uv.y - 0.5) / 0.48);
        float r = length(local);
        float rim = smoothstep(1.02, 0.98, r) * smoothstep(0.92, 1.0, r);
        base += float3(0.15, 0.45, 0.5) * rim * 0.55;
        // Interior disk wash
        float inside = 1.0 - smoothstep(0.98, 1.05, r);
        base = mix(base, base + float3(0.02, 0.05, 0.06), inside * 0.35);
    } else {
        // Faint grid suggestion on Euclidean side
        float2 g = fract(in.uv * float2(18.0, 12.0));
        float line = step(0.97, g.x) + step(0.97, g.y);
        base += float3(0.08, 0.07, 0.05) * line * 0.25;
    }

    return float4(base, 1.0);
}
