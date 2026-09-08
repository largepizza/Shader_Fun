// ── sat_point.frag ────────────────────────────────────────────────────────────
// Point-sprite fragment shader for satellite rendering.
//
// Two layers per sprite, both Gaussians that naturally reach ~0 before the edge:
//
//   inner — tight relative-sigma Gaussian (4.5% of sprite radius).
//           Stays tiny regardless of sprite size — looks like a true point source.
//           Log2-compressed brightness so ISS-class sats don't blow out.
//
//   glow  — outer soft Gaussian in absolute pixel units (fragAngSize converts).
//           Sigma grows with log2(intensity); sprite is sized to 6σ in
//           sat_flare.comp so the Gaussian is ~1% at the sprite boundary —
//           no smoothstep needed, no hard circular edge visible.
//           Zero for intensity ≤ 1; fades in log-scale above that.
//
// The soft glow + godray corona (flare architecture overhaul) is rendered by a separate
// render-to-texture + blur/streak pipeline (flare_source.vert/.frag, composited in
// flare_composite.frag) that draws from the same GpuSatVisible data as this dot sprite. This
// shader's own two-Gaussian falloff is unrelated to that pipeline.
//
// Output is (rgb*brightness, brightness) for additive blend accumulation.
// ─────────────────────────────────────────────────────────────────────────────

#version 450

layout(location = 0) in vec3  fragColor;
layout(location = 1) in float fragIntensity;
layout(location = 2) in float fragAngSize;

layout(location = 0) out vec4 outColor;

// ── Cloud occlusion (C12 follow-up #33) ───────────────────────────────────────
// Same half-res cloud composite targets sat_sky.frag reads (bindings 10/11 there; 5/6 here —
// this pipeline's own descriptor set, see createDescriptors()/createDrawPipeline() in
// SatelliteSim.cpp). Previously satellites (including mirror lens flares) had NO cloud
// awareness at all — this is the first time this pipeline reads cloud data.
layout(set = 0, binding = 5) uniform sampler2D cloudTargetA; // a = tCloudOcclude (>=0 => >=90% opaque)
layout(set = 0, binding = 6) uniform sampler2D cloudTargetB; // rgb = A_total (cloud transmittance)
// Shared terrain/ocean depth (flare architecture overhaul added this binding for
// flare_source.frag's own terrain test — already valid here, no descriptor changes needed).
// Long-exposure trail follow-up: this pipeline's LIVE draw gets terrain occlusion for free from
// the main render pass's hardware depth test, so this binding is normally unused here — only the
// trail draw (no depth attachment of its own) sets pc.manualTerrainTest and pays for the fetch.
layout(set = 0, binding = 7) uniform sampler2D sceneDepthTex;

// Declares a prefix of PointDrawPC (128 bytes — see SatelliteSim.h), through the last field this
// shader reads. debugDisableMask, screenSizePx and manualTerrainTest were relocated here from the
// old shared SatDrawPC tail so both point pipeline layouts fit the 128-byte maxPushConstantsSize
// floor; the sky-only scalars that used to sit between them went to the CloudParams UBO instead.
layout(push_constant) uniform PC {
    mat4  skyView;            // offset 0   — unused here, declared for layout consistency
    float fovYRad;            // offset 64  — unused here
    float aspect;             // offset 68  — unused here
    float waveTime;           // offset 72  — unused here
    float noTwinkle;          // offset 76  — unused here (star_point.vert only)
    vec4  moonDirENU;         // offset 80  — unused here (star_point.vert only)
    vec4  obsECEFDir;         // offset 96  — unused here (star_point.vert only)
    vec2  screenSizePx;       // offset 112 — full-res target size for the cloud/depth UVs below
    uint  debugDisableMask;   // offset 120 — knockout bit 4096 (satellite point cloud occlusion)
    float manualTerrainTest;  // offset 124 — 1 = do the manual sceneDepthTex hit-test below
} pc;

const float kNoSurfaceT = 1e30; // mirrors common.glsl's constant — this shader skips the #include
                                 // machinery for a single value, same idiom flare_source.frag uses

void main() {
    // gl_PointCoord is [0,1] across the point sprite quad; c is centred at (0,0), d is the
    // distance from centre. Done FIRST so fragments in the quad's corners (~21% of them, outside
    // the inscribed circle) exit before paying for the cloud texture fetches below.
    vec2  c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;

    // Knockout bit 4096 (profiling-only, added 2026-08-09): this pipeline had no knockout bit at
    // all before now, unlike every other perf-sensitive block in the codebase (see CLAUDE.md's
    // "GPU Performance Profiling" subsystem) — added specifically to let a per-fragment cost
    // hypothesis (2 cloud texture fetches x every covered pixel of every satellite sprite, up to
    // 120px each, potentially many simultaneously near that cap for Reflect Orbital's near-perfect
    // mirror alignment) be confirmed or ruled out with real GPU FRAME BREAKDOWN data instead of
    // guessed at — see BEAM_CLOUD_PLAN.md's session 2026-08-09 log for the reported satellite/star
    // draw cost jump this is meant to isolate.
    float cloudVis = 1.0;
    if ((pc.debugDisableMask & 4096u) == 0u) {
        // gl_FragCoord.xy must divide by THIS draw's own target size, never an assumed constant —
        // see sat_sky.frag's documented render-scale gotcha (CLAUDE.md "Subsystem: Resolution
        // Scaling"). Satellites always draw at native resolution regardless of renderScale, but the
        // cloud targets are always sized off the true swap extent, so this is the correct divisor
        // in both cases (matches sat_sky.frag's own cloudUV formula exactly).
        vec2 cloudUV = gl_FragCoord.xy / pc.screenSizePx;
        vec4 cloudA  = texture(cloudTargetA, cloudUV);
        vec4 cloudB  = texture(cloudTargetB, cloudUV);
        float tCloudOcclude = cloudA.a;
        float cloudBlock    = dot(cloudB.rgb, vec3(1.0 / 3.0));
        // Two-tier response, mirroring the two ways sat_sky.frag already treats cloud opacity:
        // a hard gate for genuinely opaque cloud (same tCloudOcclude convention that hides the moon
        // disc), and a smooth power-curve dim otherwise (same shape the Milky Way/sun disc use), so
        // satellites join the same existing visual language instead of a new one.
        float cloudHardOcclude = (tCloudOcclude >= 0.0) ? 0.0 : 1.0;
        const float kSatCloudSuppressPower = 2.0;
        cloudVis = cloudHardOcclude * pow(clamp(cloudBlock, 0.0, 1.0), kSatCloudSuppressPower);
    }

    // ── Terrain occlusion (long-exposure trail follow-up) ──────────────────────
    // Only the trail draw sets this — the live draw already gets terrain occlusion for free from
    // the main render pass's hardware depth test (this pipeline is drawn at a fixed depth of 0.5,
    // which loses against any real terrain hit and wins against pure sky — see SatDrawPC's own
    // comment). The trail's offscreen render pass has no depth attachment at all, so it needs this
    // explicit test instead — same technique flare_source.frag already uses for the same reason:
    // any real terrain/ocean hit along this screen ray occludes a satellite, regardless of how far
    // away that hit actually is (satellites/stars have no real depth of their own to compare against).
    float terrainVis = 1.0;
    if (pc.manualTerrainTest >= 0.5) {
        vec2 depthUV = gl_FragCoord.xy / pc.screenSizePx;
        terrainVis = (texture(sceneDepthTex, depthUV).r >= kNoSurfaceT * 0.5) ? 1.0 : 0.0;
    }

    // ── Inner core: tight pinpoint, log-compressed brightness ─────────────────
    // sigmaInner = 0.045 (relative): 4.5% of sprite half-width.
    // exp(-(0.5)²/(2×0.045²)) ≈ 0 — naturally reaches zero at sprite edge, no cutoff.
    // Stays visually small regardless of sprite size; lensFlare() adds the corona.
    const float sigmaInner = 0.045;
    // Sigmoid ramp: x²/(1+x²) where x = intensity × scale.
    // Near-zero at visThresh → smooth fade in/out.
    // Half-max at intensity ≈ 1/scale = 0.05 → ramps quickly above that.
    // Full brightness (3.0) by intensity ≈ 0.2; capped there for bright sats.
    //   intensity=0.008 (visThresh) → coreScale ≈ 0.008  (nearly invisible)
    //   intensity=0.05              → coreScale ≈ 1.75   (comfortably bright)
    //   intensity=0.15              → coreScale ≈ 2.93   (near-full)
    //   intensity=0.5+              → coreScale = 3.0    (capped)
    // Tune scale: higher = ramp starts earlier (brighter at threshold).
    float x = fragIntensity * 30.0;
    float coreScale = min((x * x / (1.0 + x * x)) * 3.5, 3.0);
    float sigmaAbsPx = max(sigmaInner * fragAngSize, 0.4);  // floor at 1px absolute so adjacent pixels always get contribution
    float pixD       = d * fragAngSize;
    float inner      = exp(-pixD * pixD / (2.0 * sigmaAbsPx * sigmaAbsPx)) * coreScale;

    // ── Outer glow: absolute-pixel Gaussian, independent of sprite size ────────
    // pixelDist converts gl_PointCoord distance to real pixels via the sprite size.
    // glowSigmaPx MUST match sat_flare.comp (log2(intensity)*7) so the sprite is
    // sized to exactly 6σ — the Gaussian hits ~1% at the boundary, no edge visible.
    // glowBrightness = 0 for intensity ≤ 1, so dim sats have no soft halo.
    // glowSigmaPx MUST match sat_flare.comp (log2(effectFlare)*10) so the sprite
    // is correctly sized relative to the glow.
    // Lorentzian 1/(1+r²) has a heavier tail than Gaussian — glow spreads more
    // gradually with no abrupt dropoff that could read as a hard circular edge.
    // Full-sprite smoothstep drives glow to exactly 0.0 at the sprite boundary,
    // eliminating any residual circle artifact at any intensity.
    // kGlowMax caps peak brightness so the halo is always dimmer than the inner
    // core — the bright point stays distinguishable inside the diffuse glow.
    // Keep low: many overlapping sprites stack additively; aggregate cluster glow
    // comes from the sky glow bins pass, not from individual sprites.
    // Tune: 0.05 = very subtle, 0.08 = moderate, 0.15 = dramatic single-sat.
    const float kGlowMax = 0.08;
    float pixelDist      = d * fragAngSize;
    float glowSigmaPx    = max(0.5, log2(max(fragIntensity, 1e-4)) * 10.0);
    float glowBrightness = clamp(log2(max(fragIntensity, 1.0)) * 0.5, 0.0, kGlowMax);
    float r              = pixelDist / glowSigmaPx;
    float glow           = (1.0 / (1.0 + r * r)) * glowBrightness;
    glow                *= 1.0 - smoothstep(0.0, 0.5, d);

    float brightness = (inner + glow) * cloudVis * terrainVis;
    outColor = vec4(fragColor * brightness, brightness);
}
