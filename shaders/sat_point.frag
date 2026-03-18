// ── sat_point.frag ────────────────────────────────────────────────────────────
// Point-sprite fragment shader for satellite and star rendering.
//
// Each visible satellite is drawn as one point primitive whose size was set by
// the vertex shader from GpuSatVisible.angularSize (computed in sat_flare.comp).
// gl_PointCoord covers [0,1]×[0,1] across that sprite quad.
//
// Rendering model (three additive layers):
//
//   inner — tight Gaussian core (sigmaInner ≈ 4.5% of sprite radius).
//           Always the brightest feature; represents the unresolved point source.
//
//   outer — soft bloom halo.  sigma grows from 0.10 (dim) to 0.35 (bright flare),
//           feathered to zero at the sprite boundary so the circular clip edge is
//           never visible regardless of intensity.
//
//   spike — diffraction spike cross (+).  Only appears when effectFlare > 0.30.
//           spikeK=10000 → sub-pixel perpendicular width (needle-thin arms).
//           spikeRK=3    → slow radial falloff; arms stay bright to d≈0.35.
//           spikeFade    → dissolves arms into the halo before the disc edge.
//
// Performance note:
//   Each fragment evaluates 2–3 exp() calls.  With N visible sats each sporting
//   a large sprite (angSize up to ~57 px radius), fragment work scales as
//   O(N × sprite_area).  Above ~10k simultaneously visible bright satellites the
//   total invocation count can saturate the GPU and trigger a Windows TDR crash.
//   Reduce the `effectFlare * 54.0` sprite-growth coefficient in sat_flare.comp
//   or add a LOD branch (skip spikes when fragIntensity < 0.5) to mitigate this.
//
// All layers are combined additively and written as (rgb*brightness, brightness)
// so the pipeline's additive blend mode accumulates them correctly.
// ─────────────────────────────────────────────────────────────────────────────

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
