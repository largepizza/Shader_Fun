#version 450

// ── SSBO: star records written by CPU (updateStars) each frame ────────────────
struct SatVisible {
    vec3  skyDir;         // ENU unit vector
    float flareIntensity; // raw magnitude-based intensity × night factor
    vec3  baseColor;      // B-V derived RGB
    float angularSize;    // point sprite size (pixels)
};
layout(set = 0, binding = 1) readonly buffer SatVisibleBuf {
    SatVisible satellites[];
};

// ── Push constants (prefix of PointDrawPC, 128 bytes — see SatelliteSim.h) ─────────────────────
// Declares through obsECEFDir, the last field this shader reads. noTwinkle moved up to offset 76
// (from 164 in the old shared SatDrawPC) and sunDirENU is gone (it was never read here) — the
// point pipelines carry the smaller PointDrawPC now so their push-constant range fits the 128-byte
// maxPushConstantsSize floor.
layout(push_constant) uniform PC {
    mat4  skyView;
    float fovYRad;
    float aspect;
    float waveTime;   // offset 72 — simSecInDay, used for twinkling
    float noTwinkle;  // offset 76 — 1 = skip scintillation (planet draw only)
    vec4  moonDirENU; // offset 80 — xyz=moon dir ENU, w=illuminated fraction (unused here)
    vec4  obsECEFDir; // offset 96 — xyz=obs ECEF, w=obsHeightOffset (m)
} pc;

// Matches kMoonAngR in sat_sky.frag's moon disc (0.004578 * 3.0) — the Moon is a real opaque
// body much closer than any star, so stars angularly behind its disc must not draw over it.
// The sky pass itself can't express this in the depth buffer: satellites and stars share a
// single fixed clip depth (0.5) with no relative ordering, and giving the moon its own nearer
// depth would incorrectly occlude satellites too (which really are nearer than the Moon and
// should keep drawing over it). Culling here, per-star, sidesteps the shared-depth limitation
// without touching that broader scheme.
const float kMoonAngR = 0.004578 * 3.0;

layout(location = 0) out vec3  fragColor;
layout(location = 1) out float fragIntensity;
layout(location = 2) out float fragAngSize;
layout(location = 3) out float fragTwinkle;  // scintillation modulator [0,~2]; 1 = no twinkle

void main() {
    SatVisible sat = satellites[gl_VertexIndex];

    vec3 cam = (pc.skyView * vec4(sat.skyDir, 0.0)).xyz;

    // Cull stars that fall within the Moon's angular disc — see kMoonAngR comment above.
    bool behindMoon = dot(sat.skyDir, normalize(pc.moonDirENU.xyz)) > cos(kMoonAngR);

    if (sat.flareIntensity <= 0.0 || cam.z >= -0.001 || behindMoon) {
        gl_Position  = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.001;
        fragColor     = vec3(0.0);
        fragIntensity = 0.0;
        fragAngSize   = 0.001;
        fragTwinkle   = 1.0;
        return;
    }

    float tanHalfFov = tan(pc.fovYRad * 0.5);
    gl_Position  = vec4( cam.x / (-cam.z) / (tanHalfFov * pc.aspect),
                        -cam.y / (-cam.z) /  tanHalfFov,
                         0.5, 1.0);
    gl_PointSize = sat.angularSize;

    fragColor     = sat.baseColor;
    fragIntensity = sat.flareIntensity;
    fragAngSize   = sat.angularSize;

    // ── Atmospheric scintillation ─────────────────────────────────────────────
    // Physics: Kolmogorov turbulence → amplitude scales with air mass (1/sin(el)).
    // Space: atmFrac → 0 so twinkling vanishes above the atmosphere.
    //
    // Skipped for planets (pc.noTwinkle=1, see SatDrawPC's comment): real planets are small
    // resolved discs, not point sources, and don't scintillate the way stars do — reusing this
    // code unmodified for the planet draw call would make them flicker like stars, which reads as
    // wrong given the whole point of this feature is real orbital/photometric accuracy.
    if (pc.noTwinkle >= 0.5) {
        fragTwinkle = 1.0;
        return;
    }

    // Precision note: waveTime = simSecInDay (0–86400).  Calling sin() directly
    // with a large argument (43200 × freq × 2π ≈ 1e6) loses all precision in
    // float32 GPU trig.  Fix: reduce to fractional cycle [0,1] with fract()
    // before multiplying by 2π, so the sin argument is always in [0, 2π].

    // Per-star unique phase seeds (hash of ECI direction).
    float h1 = fract(sin(dot(sat.skyDir, vec3(127.1, 311.7,  74.7))) * 43758.5453);
    float h2 = fract(sin(dot(sat.skyDir, vec3(269.5, 183.3, 246.1))) * 65734.7831);

    float t = pc.waveTime;
    const float PI2 = 6.28318530718;

    // Fractional-cycle phase: fract(t*freq) in [0,1], then add per-star hash offset.
    float p1 = fract(fract(t *  3.7) + h1);
    float p2 = fract(fract(t *  7.1) + h2);
    float p3 = fract(fract(t * 11.3) + fract(h1 * 0.3 + h2 * 0.7));

    float tw = sin(p1 * PI2)
             + sin(p2 * PI2) * 0.60
             + sin(p3 * PI2) * 0.30;
    tw /= 1.90;  // normalise to [-1, 1]

    // Atmospheric fraction: 1 at sea level, decays above ~80 km.
    // obsECEFDir.w = obsHeightOffset (user altitude above terrain, metres).
    float obsHeight = max(0.0, pc.obsECEFDir.w);
    float atmFrac   = clamp(exp(-obsHeight / 80000.0), 0.0, 1.0);

    // Air mass proxy: 1/sin(elevation), clamped 1–12.
    // sin(elevation) = sat.skyDir.z (ENU z = Up).
    float sinEl   = max(sat.skyDir.z, 0.05);  // floor at sin(~3°)
    float airMass = clamp(1.0 / sinEl, 1.0, 12.0);

    // Amplitude: 25 % base at zenith, up to 90 % near the horizon, scaled by atmFrac.
    float twAmp = (0.25 + 0.75 * ((airMass - 1.0) / 11.0)) * atmFrac;
    twAmp = min(twAmp, 0.90);

    fragTwinkle = max(0.0, 1.0 + tw * twAmp);
}
