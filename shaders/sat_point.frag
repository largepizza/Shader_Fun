#version 450

layout(location = 0) in vec3  fragColor;
layout(location = 1) in float fragIntensity;

layout(location = 0) out vec4 outColor;

void main() {
    // gl_PointCoord is [0,1] across the point sprite quad.
    vec2  c = gl_PointCoord - 0.5;
    float d = length(c);

    if (d > 0.5) discard;

    // ── Inner glow: tight pinpoint core, fixed size in sprite-space ───────────────
    // sigmaInner is intentionally small so the core stays compact even at large
    // sprite sizes.  At sigma=0.045, the half-power radius is ~7% of the sprite.
    // Brightness scales with intensity so it is always the brightest feature.
    const float sigmaInner = 0.045;
    float inner = exp(-d * d / (2.0 * sigmaInner * sigmaInner))
                  * (0.7 + fragIntensity * 0.8);

    // ── Outer halo: soft bloom, grows with intensity, dimmer than core ────────────
    // sigma grows from 0.10 (dim) to 0.35 (very bright flare).
    // The feathering zone spans d=[0.10, 0.50] — 80% of the disc radius —
    // so the circular sprite boundary is never visible regardless of glow size.
    float sigmaOuter = 0.10 + clamp(fragIntensity * 0.10, 0.0, 0.25);
    float outer = exp(-d * d / (2.0 * sigmaOuter * sigmaOuter))
                  * clamp(fragIntensity * 0.5, 0.0, 0.7);
    outer *= smoothstep(0.50, 0.10, d);

    // ── Diffraction spikes: thin cross with long smooth taper ─────────────────────
    // spikeK=10000 → thinner needles (sub-pixel perpendicular width at typical sizes).
    // spikeRK=3    → slow radial decay; arms remain at ≥68% peak through d=0.35.
    // spikeFade    → soft clip over the outer 30% of disc radius so tips dissolve
    //                into the halo rather than terminating at the circular boundary.
    float spikeAmt  = clamp(fragIntensity - 0.30, 0.0, 1.5);
    const float spikeK  = 10000.0;
    const float spikeRK = 3.0;
    float spikeRad  = exp(-d * d * spikeRK);
    float spikeFade = smoothstep(0.50, 0.35, d);
    float spike = (exp(-c.y * c.y * spikeK) + exp(-c.x * c.x * spikeK))
                  * spikeRad * spikeFade * spikeAmt * 1.8;

    float brightness = inner + outer + spike;
    outColor = vec4(fragColor * brightness, brightness);
}
