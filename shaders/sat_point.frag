// ── sat_point.frag ────────────────────────────────────────────────────────────
// Point-sprite fragment shader for satellite rendering.
//
// This shader deliberately handles ONLY the tight unresolved core of each
// satellite point source.  Camera-style lens flare effects for bright satellites
// are rendered separately in sat_sky.frag using the glowBuf SSBO — that pass
// runs first and draws into the sky/background layer.  This shader draws on top
// of that (additive blend), anchoring the precise satellite position.
//
// fragIntensity range: ~0 (invisible) to ~10+ (ISS-class event).
//
// Two layers:
//
//   inner — tight Gaussian pinpoint (sigmaInner = 4.5% of sprite radius).
//           Log2 brightness compression prevents ISS-class objects from
//           producing an oversized white blob; they still appear brighter than
//           dim satellites but don't blow out.  The lens flare in sat_sky.frag
//           is the primary "this is a very bright object" signal.
//
//   halo  — very soft, dim Gaussian envelope (sigma = 13% of sprite radius).
//           Provides anti-aliased feathering so dim dots don't look like hard
//           pixel squares.  Brightness is tightly capped so it stays invisible
//           on bright objects (lens flare handles bloom there).
//
// Output is (rgb*brightness, brightness) for additive blend accumulation.
// ─────────────────────────────────────────────────────────────────────────────

#version 450

layout(location = 0) in vec3  fragColor;
layout(location = 1) in float fragIntensity;

layout(location = 0) out vec4 outColor;

void main() {
    // gl_PointCoord is [0,1] across the point sprite quad.
    // c is centered at (0,0); d is distance from centre.
    vec2  c = gl_PointCoord - 0.5;
    float d = length(c);

    if (d > 0.5) discard;

    // ── Inner core: tight pinpoint, log-compressed brightness ─────────────────
    //
    // sigmaInner = 0.045:
    //   Core radius is 4.5% of sprite half-width.  At a typical 30 px sprite
    //   this is only a 1-2 px radius -- indistinguishable from a true point
    //   source.  Fixed size so the core doesn't grow into a blob on bright sats.
    //
    // coreScale = 0.9 + log2(1 + intensity) * 0.45:
    //   Calibrated so dim satellites (intensity~0) give coreScale~0.90, and
    //   intensity=1 gives ~1.35.  Above that the log2 curve compresses:
    //     intensity =  1 -> 1.35
    //     intensity =  5 -> 1.94
    //     intensity = 10 -> 2.32  (would be 5.2 with a linear multiplier)
    //   The sat_sky.frag lens flare handles the "this is very bright" signal.
    const float sigmaInner = 0.045;
    float coreScale = 0.9 + log2(1.0 + fragIntensity) * 0.45;
    float inner = exp(-d * d / (2.0 * sigmaInner * sigmaInner)) * coreScale;

    // ── Soft anti-aliasing halo: wide but very dim ─────────────────────────────
    //
    // sigmaHalo = 0.13:
    //   Blurs the hard circular sprite boundary into a smooth taper.
    //   Wide enough to feather cleanly; narrow enough to not look like bloom.
    //
    // haloScale cap at 0.20:
    //   A dim satellite (intensity~0.3) gets just enough halo to look like a
    //   round disc rather than a hard pixel square.  Bright satellites' halos
    //   stay tiny -- the lens flare halos in sat_sky.frag dominate instead.
    //
    // smoothstep(0.50, 0.15, d):
    //   Fade to zero in the outer 35% of the disc so the sprite boundary is
    //   never visible even at large sprite sizes.
    const float sigmaHalo = 0.13;
    float haloScale = clamp(fragIntensity * 0.18, 0.0, 0.20);
    float halo = exp(-d * d / (2.0 * sigmaHalo * sigmaHalo)) * haloScale;
    halo *= smoothstep(0.50, 0.15, d);

    float brightness = inner + halo;
    outColor = vec4(fragColor * brightness, brightness);
}
