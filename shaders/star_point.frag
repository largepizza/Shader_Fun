// ── star_point.frag ───────────────────────────────────────────────────────────
// Fragment shader for background stars (and, since session 30, planets — same pipeline).
//
// Minimum-size Gaussian (same technique as sat_point.frag):
//   sigmaAbsPx has a 0.7 px floor so even the faintest, smallest sprite
//   spreads across ~2-3 pixels instead of concentrating all flux in one
//   aliased pixel.  For dim stars, coreScale is small → soft dim blob.
//   For bright stars, coreScale is large → bright, still soft core.
//
// Brightness follows sqrt(intensity): correct 0→0 for invisible stars, and
// gives perceptually even compression across the Sirius-to-naked-eye range.
//
// Desaturation: faint stars mix toward white — their B-V colour is
// imperceptible at low flux, and the white tint reduces "coloured pixel" artefacts.
//
// fragTwinkle (from star_point.vert): per-star atmospheric scintillation
// modulator.  Computed in the vertex stage so it is constant across the sprite.
//
// Output: (rgb*brightness, brightness) for additive blend accumulation.
// ─────────────────────────────────────────────────────────────────────────────

#version 450

layout(location = 0) in vec3  fragColor;
layout(location = 1) in float fragIntensity;
layout(location = 2) in float fragAngSize;
layout(location = 3) in float fragTwinkle;  // scintillation modulator [0,~2]; 1 = no twinkle

layout(location = 0) out vec4 outColor;

// ── Cloud occlusion (session 30 bug fix) ──────────────────────────────────────
// Stars/planets previously had NO cloud awareness at all — satellites gained this in C12
// follow-up #33 (sat_point.frag) but it was never mirrored onto this pipeline, so stars always
// rendered through clouds regardless of opacity. Most visible at Medium-and-below presets, where
// clouds are thinner/shorter-range, but the gap was architectural at every preset. Same textures,
// same formula, same "own descriptor set's own binding numbers" convention sat_point.frag already
// uses (its bindings 5/6 there; this pipeline's own set only had binding 1 before this, so 2/3
// here — see starDescLayout in SatelliteSim.cpp).
layout(set = 0, binding = 2) uniform sampler2D cloudTargetA; // a = tCloudOcclude (>=0 => >=90% opaque)
layout(set = 0, binding = 3) uniform sampler2D cloudTargetB; // rgb = A_total (cloud transmittance)
// Shared terrain/ocean depth — new binding (long-exposure trail follow-up, see SatelliteSim.cpp's
// initStars()). This pipeline's LIVE draw gets terrain occlusion for free from the main render
// pass's hardware depth test, so this binding is normally unused here — only the trail draw (no
// depth attachment of its own) sets pc.manualTerrainTest and pays for the fetch.
layout(set = 0, binding = 4) uniform sampler2D sceneDepthTex;

// Declares a prefix of PointDrawPC (128 bytes — see SatelliteSim.h), through manualTerrainTest,
// the last field this shader reads. screenSizePx/manualTerrainTest moved here from the old shared
// SatDrawPC tail so both point pipeline layouts fit the 128-byte maxPushConstantsSize floor.
layout(push_constant) uniform PC {
    mat4  skyView;            // offset 0   — unused here
    float fovYRad;            // offset 64  — unused here
    float aspect;             // offset 68  — unused here
    float waveTime;           // offset 72  — unused here
    float noTwinkle;          // offset 76  — unused here (star_point.vert reads its own copy)
    vec4  moonDirENU;         // offset 80  — unused here
    vec4  obsECEFDir;         // offset 96  — unused here
    vec2  screenSizePx;       // offset 112 — full-res target size for the cloud/depth UVs below
    uint  debugDisableMask;   // offset 120 — unused here
    float manualTerrainTest;  // offset 124 — 1 = do the manual sceneDepthTex hit-test below
} pc;

void main() {

    vec2  c = gl_PointCoord - 0.5;
    float d = length(c);

    if (d > 0.5) discard;

    // ── Minimum-size Gaussian in absolute pixel units ──────────────────────────
    // 0.7 px sigma floor → ~1.65 px FWHM.  Even a 6 px faint-star sprite
    // spreads its flux across 2-3 pixels rather than one harsh spike.
    const float sigmaInner = 0.045;
    float sigmaAbsPx = max(sigmaInner * fragAngSize, 0.2);
    float pixD       = d * fragAngSize;
    float gaussian   = exp(-pixD * pixD / (2.0 * sigmaAbsPx * sigmaAbsPx));

    // ── Brightness: sqrt compression, correct zero for invisible stars ─────────
    // log2(2 + x) * 1.5 never reaches 0 and makes all stars equally bright;
    // sqrt(x) * 2.8 properly attenuates faint stars so they appear as dim blobs
    // rather than bright pinpoints.
    float coreScale = clamp(sqrt(fragIntensity) * 2.8, 0.0, 3.0);

    // ── Colour desaturation for dim stars ─────────────────────────────────────
    // Bright stars (Betelgeuse, Sirius) keep their B-V tint; faint stars
    // fade toward white since their colour is below perceptual threshold anyway.
    float saturation = clamp(sqrt(fragIntensity) * 1.5, 0.0, 1.0);
    vec3  starColor  = mix(vec3(1.0), fragColor, saturation);

    // ── Cloud occlusion ─────────────────────────────────────────────────────
    // gl_FragCoord.xy must divide by THIS draw's own target size, never an assumed constant —
    // stars/planets always draw at native resolution regardless of renderScale, but the cloud
    // targets are always sized off the true swap extent, matching sat_point.frag's identical
    // formula.
    vec2 cloudUV = gl_FragCoord.xy / pc.screenSizePx;
    vec4 cloudA  = texture(cloudTargetA, cloudUV);
    vec4 cloudB  = texture(cloudTargetB, cloudUV);
    float tCloudOcclude = cloudA.a;
    float cloudBlock    = dot(cloudB.rgb, vec3(1.0 / 3.0));
    // Same two-tier response as sat_point.frag: hard gate for genuinely opaque cloud, smooth
    // power-curve dim otherwise.
    float cloudHardOcclude = (tCloudOcclude >= 0.0) ? 0.0 : 1.0;
    const float kStarCloudSuppressPower = 2.0;
    float cloudVis = cloudHardOcclude * pow(clamp(cloudBlock, 0.0, 1.0), kStarCloudSuppressPower);

    // ── Terrain occlusion ─────────────────────────────────────────────────────
    // Set by (a) the long-exposure trail draw, whose offscreen render pass has no depth attachment,
    // and (b) the live draw at renderScale < 1.0, where the sky pass is a low-res prepass that never
    // writes the frame's depth buffer (see buildPointDrawPC). At renderScale 1.0 the live draw
    // instead gets this for free from the hardware depth test against sat_sky.frag's gl_FragDepth.
    //
    // kOcclusionCap MUST match sat_sky.frag's constant of the same name — that shader only writes a
    // point-occluding depth for terrain/ocean within this distance, so this test reproduces its
    // exact behaviour (including the same limitation: a surface farther than the cap does not
    // occlude, which for a point source at infinity is visible only in orbital views).
    float terrainVis = 1.0;
    if (pc.manualTerrainTest >= 0.5) {
        vec2 depthUV = gl_FragCoord.xy / pc.screenSizePx;
        const float kOcclusionCap = 150000.0;
        terrainVis = (texture(sceneDepthTex, depthUV).r < kOcclusionCap) ? 0.0 : 1.0;
    }

    float brightness = gaussian * coreScale * fragTwinkle * cloudVis * terrainVis;
    outColor = vec4(starColor * brightness, brightness);
}
