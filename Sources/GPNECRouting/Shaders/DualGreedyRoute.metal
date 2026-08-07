#include <metal_stdlib>
using namespace metal;

struct RouteUniforms {
    uint nodeCount;
    uint neighborsPerNode;
    uint packetCount;
    uint metric; // 0 = Euclidean, 1 = Poincaré
    uint maxHops; // TTL; 0 = unlimited
    uint stuckRetryHops; // local-min dwell → silent retry; 0 = freeze
};

struct Packet {
    uint atNode;
    uint dstNode;
    uint hops;
    uint flags; // bit0 alive, bit1 delivered, bit2 dropped
};

inline float poincare_distance(float2 a, float2 b) {
    float na = length(a);
    float nb = length(b);
    if (na >= 0.999f) a *= 0.999f / na;
    if (nb >= 0.999f) b *= 0.999f / nb;
    float2 diff = a - b;
    float num = length(diff);
    float den = max(1.0f - dot(a, a), 1e-6f) * max(1.0f - dot(b, b), 1e-6f);
    float arg = 1.0f + 2.0f * (num * num) / den;
    return log(arg + sqrt(max(arg * arg - 1.0f, 0.0f)));
}

inline float route_distance(float2 a, float2 b, uint metric) {
    if (metric == 0u) {
        return distance(a, b);
    }
    return poincare_distance(a, b);
}

/// One greedy hop for every in-flight packet.
/// Hard-drop if current node is dead. Deliver if at == dst.
/// Local minimum or TTL exhaustion → hard drop (populates dropped metrics).
kernel void dual_greedy_route_step(
    device const float2 *positions [[buffer(0)]],
    device const int *neighbors [[buffer(1)]],
    device const uint *nodeAlive [[buffer(2)]],
    device Packet *packets [[buffer(3)]],
    constant RouteUniforms &u [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= u.packetCount) return;
    Packet p = packets[gid];
    if ((p.flags & 1u) == 0u) return; // not alive

    // Hard drop on dead router.
    if (nodeAlive[p.atNode] == 0u) {
        p.flags = 4u; // dropped
        packets[gid] = p;
        return;
    }

    if (p.atNode == p.dstNode) {
        p.flags = 2u; // delivered
        packets[gid] = p;
        return;
    }

    // Destination died — drop.
    if (nodeAlive[p.dstNode] == 0u) {
        p.flags = 4u;
        packets[gid] = p;
        return;
    }

    if (u.maxHops > 0u && p.hops >= u.maxHops) {
        p.flags = 4u;
        packets[gid] = p;
        return;
    }

    float2 here = positions[p.atNode];
    float2 dest = positions[p.dstNode];
    float bestDist = route_distance(here, dest, u.metric);
    int best = -1;
    uint base = p.atNode * u.neighborsPerNode;
    for (uint t = 0; t < u.neighborsPerNode; ++t) {
        int nb = neighbors[base + t];
        if (nb < 0 || uint(nb) >= u.nodeCount) continue;
        if (nodeAlive[uint(nb)] == 0u) continue;
        float d = route_distance(positions[uint(nb)], dest, u.metric);
        if (d + 1e-7f < bestDist) {
            bestDist = d;
            best = nb;
        }
    }

    if (best >= 0) {
        p.atNode = uint(best);
        p.hops += 1u;
        if (p.atNode == p.dstNode) {
            p.flags = 2u;
        } else if (u.maxHops > 0u && p.hops >= u.maxHops) {
            p.flags = 4u;
        }
    } else {
        // Local minimum: dwell, then optional silent retry (pre-crash recycling).
        p.hops += 1u;
        if (u.stuckRetryHops > 0u && p.hops >= u.stuckRetryHops) {
            p.flags = 8u; // needsRetry — not a physical drop
        }
    }
    packets[gid] = p;
}
