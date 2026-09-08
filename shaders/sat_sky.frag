#version 450

// ── Camera + sun push constants (same layout as C++ SatDrawPC, 172 bytes) ─────
// The pipeline layout declares VK_SHADER_STAGE_VERTEX_BIT|FRAGMENT_BIT so both
// stages share one push constant range.  The fragment uses skyView/fovYRad to
// project glowBuf ENU directions into screen UV for the lens flare pass.
layout(push_constant) uniform PC {
    mat4  skyView;     // ENU -> camera space (rotation, no translation)
    float fovYRad;     // vertical field of view in radians
    float aspect;      // viewport width / height
    float gmst;        // Greenwich Mean Sidereal Time (radians)
    float waveTime;    // wall-clock seconds for wave animation
    vec4  sunDirENU;   // xyz = sun dir in ENU, w = sin(sun elevation)
    vec4  moonDirENU;  // xyz = moon dir in ENU, w = illuminated fraction
    vec4  obsECEFDir;  // xyz = observer ECEF unit vector; w = obsHeightOffset (m)
    uint  debugDisableMask; // profiling-only knockout toggles — see dbgSkip* helpers below
    float pad0;         // explicit — matches C++ SatDrawPC's alignment padding before the vec2 below
    vec2  screenSizePx; // CURRENT render target's pixel size (session 29, resolution scaling) —
                        // gl_FragCoord.xy is relative to THIS draw's own framebuffer, not always
                        // the full swapchain; any [0,1] UV derived from gl_FragCoord must divide
                        // by this, not an assumed full-res constant. See cloud composite sample
                        // below for why this matters.
    float skyGlareVisibility; // eased sun-glare gate for stars/Milky Way, computed CPU-side each
                        // frame in recordCompute() (skyGlareEased) — 0 when the sun is on screen,
                        // sunlitBgVisibility (settings-tunable) when off-screen but the observer
                        // is still in direct sunlight, 1.0 at true night. See Milky Way section.
    // (cloudShadowRangeM and cloudShadowResidualM were here — both only existed to address
    //  cloud_shadow.comp's observer-centred grid and undo its texel snapping. That pass is gone;
    //  the shadow arrives in cloudTargetB.a already evaluated at this pixel's terrain hit point.)
    float beamMaxRangeM; // C12 follow-up #6 — settings-tunable Reflect-Orbital beam render range
    float beamSkyGlowGain; // C12 follow-up #18 — mirrors cloud_march.comp's own copy so the
                        // ground-spot term below and the sky glow march share one brightness
                        // control and read as one continuous effect.
    float beamGlowBleedGain; // C12 follow-up #39 — moved here from cloud_march.comp's CloudMarchPC
                        // (that file's near-field bleed/march was removed entirely); now drives
                        // this shader's own beam sky-illumination wash instead.
    float beamProximityGlow; // C12 follow-up #41 — CPU-computed [0,1] "how close is the observer
                        // to any active beam's actual line" value (SatelliteSim::beamProximityGlow).
                        // Replaces the directional azimuth-sector dome lookup the wash used in
                        // #39/#40 — applied uniformly regardless of view direction.
    float noTwinkle; // offset 164 — unused here (only star_point.vert reads it); MUST still be
                     // declared — a push_constant block must be a byte-exact CONTIGUOUS prefix of
                     // the pushed struct, not a sparse one. Omitting this field entirely (as an
                     // earlier version of this file did) silently shifted mwSuppressEased below to
                     // offset 164 in THIS shader's own layout while the CPU still wrote it at 168,
                     // so the value actually read here was noTwinkle's (always 0.0 for this draw)
                     // instead — the Milky Way's pollution suppression was permanently a no-op.
    float mwSuppressEased; // offset 168 — Milky Way's OWN light-pollution suppression — deliberately NOT the
                        // shared domeVal/kMWPollutionMaxDim shape stars/satellites use below, so
                        // tuning lightPollutionGain for those never requires retuning this. Its
                        // threshold curve and asymmetric fade-in/fade-out hysteresis (moving from
                        // a bright area into a dark one, or into space, fades the Milky Way back
                        // in gradually rather than popping instantly) are entirely CPU-side — see
                        // SatelliteSim.h's mwSuppressEased member and updateLightPollutionDome().
                        // [0,1], 0 = fully visible, 1 = fully suppressed.
} pc;

// ── Perf knockout toggles (profiling-only) ──────────────────────────────────────
// Lets the Display settings tab measure the isolated GPU cost of individual blocks of
// this shader via gpuMsSmoothed deltas (App::drawFrame's timestamp query), without a GPU
// capture tool. Default (mask 0) is bit-identical to normal rendering — every dbgSkip*()
// call compiles to a single AND+compare against a value that's 0 unless a checkbox is on.
bool dbgSkipTerrain()    { return (pc.debugDisableMask & 1u) != 0u; }
bool dbgSkipAtmosphere() { return (pc.debugDisableMask & 2u) != 0u; }
bool dbgSkipSunOD()      { return (pc.debugDisableMask & 4u) != 0u; }
bool dbgSkipOceanRefl()  { return (pc.debugDisableMask & 8u) != 0u; }

layout(location = 0) in  vec3 enuDir;           // interpolated ENU view ray (not normalised)
layout(location = 1) in flat vec4 sunDirENU;    // passed through from vertex (same as pc.sunDirENU)
layout(location = 2) in flat vec4 moonDirENU;   // moon dir + phase pass-through

// Sky glow histogram, written by sat_flare.comp each frame. Must match GpuGlowBuf exactly.
// (flareCount/flareEntries[kFlareMax] lived here — the per-pixel corona loop that read them was
// deleted in the flare architecture overhaul; see FlareSourcePC's comment in SatelliteSim.h. The
// satellite corona/godray effect now comes from a render-to-texture + blur/streak pipeline
// composited once per frame in flare_composite.frag instead.)
layout(std430, set = 0, binding = 0) readonly buffer GlowBuf {
    uint  bins[64]; // sky glow: floatBitsToUint(max effectFlare) per bin
} glowBuf;

// Ocean-glint list (flare architecture overhaul) — matches GpuOceanGlintBuf exactly. Same buffer
// sat_flare.comp's binding 8 writes; read here via skyDescSet's own binding 20.
const uint kOceanGlintMax = 512; // must match kMaxOceanGlints in SatelliteSim.h and
                                  // OCEAN_GLINT_MAX in sat_flare.comp exactly.
layout(std430, set = 0, binding = 20) readonly buffer OceanGlintBuf {
    uint oceanGlintCount;
    uint oceanGlintPad[3];
    vec4 oceanGlintEntries[kOceanGlintMax]; // xyz=ENU dir, w=effectFlare
} oceanGlintBuf;

// RGBA noise texture (binding 1): tiled REPEAT sampler, used for angular corona
// variation in lensFlare().  Replaces the original ShaderToy's iChannel0 lookup.
layout(set = 0, binding = 1) uniform sampler2D noiseTex;

// Moon surface texture (binding 2): near-side face disc image.
// Sampled with an orthographic projection of the surface normal onto the moon's
// local face frame — maps the near hemisphere to the full [0,1] UV range.
layout(set = 0, binding = 2) uniform sampler2D moonTex;

// Earth textures (bindings 3-6): 8K equirectangular maps.
// UV derived from ENU hit point → ECEF → geographic lat/lon.
// earthDayTex:   SRGB colour map (auto-linearised on read).
// earthNightTex: SRGB city-light map.
// earthElevTex:  R8_UNORM land elevation. Ocean baseline = 15/255; land = (p - 15/255) * 8848 m.
// earthSpecTex:  R8_UNORM ocean mask (white=ocean, black=land). Used for wave material.
layout(set = 0, binding = 3) uniform sampler2D earthDayTex;
layout(set = 0, binding = 4) uniform sampler2D earthNightTex;
layout(set = 0, binding = 5) uniform sampler2D earthElevTex;
layout(set = 0, binding = 6) uniform sampler2D earthSpecTex;
layout(set = 0, binding = 7) uniform sampler2D earthCloudsTex;

// City day/night detail textures (bindings 14/15): small tileable maps, REPEAT in both U and V.
// Blended onto dayColor/nightColor near cities (bright earthNightTex pixels) within a fixed
// distance of the observer — see the terrain block in main() below.
layout(set = 0, binding = 14) uniform sampler2D cityDayDetailTex;
layout(set = 0, binding = 15) uniform sampler2D cityNightDetailTex;

// Aurora 3D noise volume (binding 16): 1024x16x256 RGBA8, baked by aurora_noise.comp at init.
// R = curtain fold base, G/B = column-window colA/colB. See that file's header comment for the
// exact UVW layout/frequencies — the sampling code near auroraCurtainNoise/auroraSampleAt below
// must stay in sync with it.
layout(set = 0, binding = 16) uniform sampler3D auroraNoiseTex;

// ── Reflect-Orbital ground beams (C12) — written by sat_orbit.comp, read here for the
// ground-spot direct-lighting term. Capped atomic-append, no site keying/arbitration — see the
// ReflectBeamsBuf comment in sat_orbit.comp for the full history/rationale.
// Struct must match GpuReflectBeam/GpuReflectBeams in SatelliteSim.h and sat_orbit.comp exactly.
#include "reflect_beam.glsl"   // ReflectBeam + BEAM_MAX_ACTIVE
layout(std430, set = 0, binding = 17) readonly buffer ReflectBeamsBuf {
    uint         beamCount;
    uint         beamPad0, beamPad1, beamPad2;
    ReflectBeam  beams[BEAM_MAX_ACTIVE];
};

// Ground-beam compaction (perf follow-up): CPU-built every frame from a fresh readback of
// ReflectBeamsBuf above, filtered to just the entries within pc.beamMaxRangeM of the observer —
// the exact test the ground-spot loop below used to redo, unconditionally, against the FULL raw
// list (up to BEAM_MAX_ACTIVE=2048 entries) for every ground-hit pixel. Consuming this instead
// bounds that loop's trip count to however many beams are actually close enough to matter, capped
// at GROUND_BEAM_MAX. See the CPU aggregation next to lastActiveBeamCount/beamProximityGlow in
// SatelliteSim.cpp, and GpuGroundBeams in SatelliteSim.h.
//
// 2026-08-10: these are no longer raw ReflectBeam records. Everything in the old loop body that did
// not vary per pixel — the range fade, the elevation fade, the shadow attenuation, and the
// obsPos+satENU / raySphere solve for the beam's real ray/ground intersection — is now folded on
// the CPU into `weight` plus a few reciprocals, once per beam instead of once per ground-hit pixel
// at full resolution. Must match GpuGroundBeam in SatelliteSim.h exactly (hand-mirrored).
struct GroundBeam {
    vec2  groundHitXY;    // observer-relative ENU horizontal position of the REAL landing spot
    float invFootprintSq; // 1 / footprintR^2
    float invCoreSq;      // 1 / coreR^2
    float cutoffSq;       // (footprintR * 4)^2 — the loop's first and cheapest reject
    float weight;         // intensity * rangeFade * elevFade * shadowAtten
    float intensity;      // CPU-side top-K ranking only; deliberately unread here
    float pad0;
};
layout(std430, set = 0, binding = 21) readonly buffer GroundBeamsBuf {
    uint        groundBeamCount;
    uint        groundBeamPad0, groundBeamPad1, groundBeamPad2;
    GroundBeam  groundBeams[GROUND_BEAM_MAX];
};

// (binding 18 was cloudShadowTex, cloud_shadow.comp's 128x128 grid. That whole pass is gone —
//  cloud_march.comp now writes a per-pixel shadow into cloudB.a. Bindings 19/20 were compacted
//  down into 18/19 rather than leaving a hole, since the C++ side fills its binding array
//  contiguously.)

// Cloud 3D noise volume (binding 8): 128³ RGBA, baked by cloud_noise.comp at init.
// R = presence (Perlin-Worley), G = pre-summed erosion FBM, B/A = presence at +54/+108 texels
// in Z. See common.glsl's channel-layout block — this packing changed in T2.1, and the only
// reader left in THIS file is the CLOUD_DEBUG==6 visualization below.
layout(set = 0, binding = 8) uniform sampler3D cloudNoiseTex;

// CloudParams UBO + CloudLayer come from the shared header. This block used to be
// hand-copied here; see cloud_params.glsl for why that was a standing hazard.
#define CLOUD_PARAMS_BINDING 9
#include "cloud_params.glsl"

// Half-resolution cloud march output (written by cloud_march.comp, see the "velvet-rolling-
// squirrel" plan / TERRAIN_PLAN.md session 23 log). Replaces the old inline cirrusMarch()/
// cloudMarch() calls in main() below — those functions moved to that compute shader.
// Target A: rgb = combined additive radiance (B_total), a = tCloudOcclude (m, -1 = none).
// Target B: rgb = combined multiplicative attenuation (A_total), a = cloudBlock (sun-dim scalar).
layout(set = 0, binding = 10) uniform sampler2D cloudTargetA;
layout(set = 0, binding = 11) uniform sampler2D cloudTargetB;

// Light pollution dome (binding 12): same buffer as sat_flare.comp's binding 3 (16 azimuth
// sectors, CPU-written each frame). A second, independent read here so the Milky Way skybox can
// be dimmed directionally the same way satellites/stars already are.
layout(std430, set = 0, binding = 12) readonly buffer LightDomeBuf {
    float lightDome[16];
};

// Milky Way skybox (binding 13): 8K equirectangular galactic panorama. Sampled against the
// CPU-computed ENU->galactic basis in cloud.mwBasisRow0/1/2 (see updatePositions() in
// SatelliteSim.cpp for how the fixed orientation, including the longitude mirror, is built).
layout(set = 0, binding = 13) uniform sampler2D milkyWayTex;

// Beam-driven sky-glow suppression dome (binding 19, C12 follow-up #31): same buffer as
// sat_orbit.comp's binding 5 / sat_flare.comp's binding 4 — a SECOND, independent 16-sector dome,
// populated by active Reflect-Orbital beams instead of a static night-lights texture. Stored as
// raw atomicMax'd uint bit-patterns (floatBitsToUint on the write side) — reinterpret via
// uintBitsToFloat, NOT a direct float read like LightDomeBuf above.
layout(std430, set = 0, binding = 18) readonly buffer BeamGlowDomeBuf {
    uint beamGlowDome[16];
};

// Shared scene depth (binding 19) — linear metres to the first terrain/ocean surface along each
// view ray, or kNoSurfaceT for rays that reach space. Written by scene_depth.comp earlier in the
// same recordCompute; this binding was already wired on the C++ side (layout/pool/onResize) but
// never actually declared/sampled here until now (2026-07-29) — used below to test whether
// terrain blocks a lens-flare source's own direction, not this fragment's view ray.
layout(set = 0, binding = 19) uniform sampler2D sceneDepthTex;

layout(location = 0) out vec4 outColor;

// PI, R_EARTH, R_ATMOS, BETA_R/H_R, BETA_M/H_M/G_MIE, SUN_INTENSITY, kCloudHorizFreq/kCloudColFreq,
// raySphere, rotateZ, remap, phaseR/phaseM/phaseCloud and the scene-depth sentinels all live in
// the shared header now. terrain.glsl brings the DEM decode + observer-frame helpers.
#include "common.glsl"
#include "terrain.glsl"

// ── Cloud noise domain frequencies ─────────────────────────────────────────────
// Cloud procedural noise (cloudNoiseTex) is sampled by TRUE 3D unit-sphere position
// (dirECEF = normalize(pECEF)), not by lat/lon UV. A sphere embedded in R^3 carries its
// natural induced metric, so this has no pole singularity and no latitude-dependent scale
// distortion — unlike an equirectangular UV, whose atan2/asin derivatives blow up at the
// poles (causing both the visible polar noise compression and a real perf hit, since the
// raymarch's empty-air skip gets defeated by aliased density near the poles).
// dirECEF is also altitude-invariant by construction (same value straight up/down at a
// given lat/lon), so cloud "presence" shape naturally has no unwanted Z-sweep with no hack
// needed. Frequencies are ~(old UV-space tile count)/(2*PI), since dirECEF isn't normalized
// to a 0-1 globe fraction the way pUV was — retune visually, these are starting points.
// (kCloudHorizFreq / kCloudColFreq are defined in common.glsl — same values, same rationale.)

// ── Domain warp / city upwelling constants: MOVED, not deleted ────────────────────────────
// kWarpFreq/kWarpStrength/kWarpDriftRate/kWarpEvolveRate and kCityUpwellStrength used to live
// here alongside this file's own cloudWarpOffset()/cloudDensity()/cloudMarch(). Those marches
// moved to cloud_march.comp (session 23); the constants and functions were left behind and had
// zero call sites in this file ever since. Removed in the pipeline-unification pass. The live
// definitions — including the full rationale comments this block used to carry — are in
// cloud_march.comp; anything shared migrates to shaders/include/ from there, not from here.

// City-brightness response curve, shared by the cloud upwelling (below) and the atmospheric
// city-glow term (see kNightGlowScale) so both read the same "how bright is this city" signal
// consistently. earthNightTex luminance varies enormously between a small town and a major
// metro core — a LINEAR response (the old cityMask = max(0,cityLum-kNightFloor)) means bright
// cities dominate completely while small towns barely clear the floor and contribute nothing.
// Reinhard-style compression (raw/(raw+k)) has a steep slope near 0 (small towns get a real,
// visible response) and naturally saturates toward 1.0 for large raw values (major metros can't
// run away and blow out) — compresses the huge input dynamic range into a much narrower, more
// even output range. Smaller k = more aggressive compression (steeper low-end boost, earlier
// high-end saturation).
const float kNightFloor    = 0.002;
const float kCityCompressK = 0.08;
float cityBrightness(float lum) {
    float raw = max(0.0, lum - kNightFloor);
    return raw / (raw + kCityCompressK);
}
// Atmospheric city-glow strength (Step 7 / C10 in TERRAIN_PLAN.md — previously deferred,
// implemented here alongside the cloud upwelling fix so both read the same brightness curve
// and sell as one consistent light source instead of bright clouds over a flat-black sky).
// First-pass value, deliberately conservative — kCityUpwellStrength's first guess (50) blew
// out badly, so start low here and raise if the glow reads as too subtle.
const float kNightGlowScale = 0.0000002;

// ── Airglow (C15, TERRAIN_PLAN.md Phase E) ──────────────────────────────────────
// Three altitude-banded emissive nightglow layers, riding the N_VIEW atmosphere loop
// where their peak altitude falls inside it (green/sodium), or a small supplemental
// march where it doesn't (red — peaks at 275km, well past N_VIEW's ~100km ceiling;
// see the airglowRed march after the N_VIEW loop in main()). Density per layer is a
// Gaussian in altitude: exp(-((h-peakAltM)/halfWidthM)^2). Real airglow altitudes are
// near-constant physical constants (not scene-dependent), so they're hardcoded here
// rather than exposed as CloudParams sliders — only per-band brightness (which is a
// legitimate first-pass visual guess, unlike the altitudes) is user-tunable.
const float kAirglowGreenPeakM      = 96000.0;   // O I 557.7nm — dominant visible band
const float kAirglowGreenHalfWidthM = 9000.0;
const vec3  kAirglowGreenColor      = vec3(0.35, 1.0, 0.25);
const float kAirglowSodiumPeakM      = 90000.0;  // Na D 589.3nm — sharp/thin
const float kAirglowSodiumHalfWidthM = 6500.0;
const vec3  kAirglowSodiumColor      = vec3(1.0, 0.65, 0.15);
// kAirglowRedPeakM/HalfWidthM moved to cloud_march.comp with the red band's march itself —
// kAirglowRedColor stays here too, still used by auroraSampleAt's color blend below.
const vec3  kAirglowRedColor      = vec3(1.0, 0.12, 0.05);
// Horizontal patchiness so the bands don't read as a perfectly flat, featureless ring
// around the sky (a pure function of altitude alone has zero horizontal variation).
// Reuses the analytic warpPerlin3 noise already used for cloud domain warp — no new
// texture/binding, matches the "reuse existing noise infra" C15 design directive
// (which pre-dates the cloud warp's migration from a noiseTex lookup to this analytic
// evaluator — see cloudWarpOffset's comment; follow the current code, not the stale plan).
const float kAirglowNoiseFreq = 4.0;
const float kAirglowDriftRate = 0.015; // wall-clock rad/s (pc.waveTime), slow independent drift
// First-pass brightness scale, same convention as kNightGlowScale/kCityUpwellStrength
// above: raw accumulation (density × segLen, summed over qualifying march samples) is
// a large unnormalized number, this brings it into visible range. Deliberately
// conservative — real airglow is famously faint. Tune via cloud.airglowGain (settings
// slider) rather than editing this constant.
const float kAirglowScale = 0.0000005;

// ── 3D volumetric ↔ flat 2D crossfade band ─────────────────────────────────────
// cloudMarch (expensive per-sample 3D shell march) and evalCloudLayer (cheap flat-texture
// paste at layers[0]/[1]'s shellAltM, i.e. the same physical cloud base/top) render the SAME
// shell at two different fidelities. Below kCloud3DFadeStart: pure 3D. Above kCloud3DFadeEnd:
// pure flat 2D (cheap enough for orbit). Both sides read this same pair of constants so the
// crossfade is symmetric — previously the flat paste used a hard `obsEffH < 8000` boolean while
// the volumetric fade didn't reach zero until 180 km, so 8-180 km altitude showed both the flat
// shell AND the still-near-full-strength 3D volume composited at once (visible as the flat
// texture "shell" intersecting the volumetric clouds).
const float kCloud3DFadeStart = 800000.0;
const float kCloud3DFadeEnd   = 3000000.0;

// ── View-march step count vs. altitude ──────────────────────────────────────────
// Step count needed for a glitchless march scales with the shell's ANGULAR size on screen,
// which shrinks with observer altitude — LEO (400-600 km) can look correct with far fewer
// steps than ground level needs. This band is intentionally separate from kCloud3DFadeStart
// (800 km): that constant now sits above typical LEO, so opacity fading alone doesn't reduce
// cost anywhere satellites actually orbit — the reported "LEO cloud perf is awful" case sits
// entirely inside the always-full-3D zone below kCloud3DFadeStart. This band supplies the
// actual LEO-perf lever: steps ramp from full ground quality down to a floor well before 800 km.
const float kMarchStepsAltStart = 2000.0;    // below this: full cloud.marchSteps (ground/aircraft)
const float kMarchStepsAltEnd   = 200000.0;   // at/above this: kMarchStepsFloor (LEO and up)
const float kMarchStepsFloor    = 12.0;       // minimum steps once the shell is angularly tiny

// ── Rayleigh scattering (wavelength-dependent: R=650nm, G=510nm, B=440nm) ─────
// (BETA_R / H_R are defined in common.glsl.)

// ── Mie scattering (aerosols, wavelength-independent) ─────────────────────────
// (BETA_M / H_M / G_MIE are defined in common.glsl.)

// ── Lighting / tone mapping ────────────────────────────────────────────────────
// (SUN_INTENSITY is defined in common.glsl.)
const float EXPOSURE_DAY   =  1.8;   // sun at zenith -- prevents white washout
const float EXPOSURE_NIGHT = 10.0;   // below horizon -- amplifies dim twilight glow

// ── Ray march quality ──────────────────────────────────────────────────────────
// Was fixed const (124/12) — perf follow-up (session 24): the main atmosphere loop runs
// unconditionally on every pixel (terrain, ocean, cloud, satellite, or empty space) before any
// surface-specific work, so this is the single most-paid-for cost in the whole shader. Now
// UBO-tunable ("View samples"/"Light samples" sliders) so the user can empirically test how much
// of the ground-level frame budget this actually costs before investing in a transmittance LUT.
// Defaults (124/12) preserve prior behavior exactly.

// (phaseR / phaseM are defined in common.glsl.)
// (phaseCloud lived here — dead since the cloud march moved to cloud_march.comp, removed in the
// pipeline-unification pass. The live copy is cloud_march.comp's.)
// (raySphere, with its full planetary-scale precision rationale, is in common.glsl.)
// Marches N_LIGHT steps from point p toward direction d over distance segTotal
// and returns (Rayleigh optical depth, Mie optical depth) — i.e. ∫ρ(h) ds for each species.
// Multiply by BETA_R / BETA_M in the caller to convert to actual extinction coefficients.
// Called once per view sample to accumulate the sun-side transmittance at that altitude.
// Kept local, NOT shared with cloud_march.comp's copy of the same integral. See that file's
// optDepth comment for why: sharing forces a runtime trip count, and cloud_march's is a
// compile-time constant it needs to keep. This copy's count is settings-tunable, so it has a
// runtime bound either way and loses nothing by staying here.
vec2 optDepth(vec3 p, vec3 d, float segTotal) {
    // Perf knockout: zero optical depth = "sun ray unattenuated", the same fallback the
    // callers already use when tSun.y <= 0 (no atmosphere intersection) — a safe, already-
    // exercised code path, not a new one. Isolates every optDepth() call site (main N_VIEW
    // loop, ocean sky-reflection loop, and both fixed-count supplemental marches) at once.
    if (dbgSkipSunOD())
        return vec2(0.0);
    int   N_LIGHT = int(max(2.0, cloud.lightSamples));
    float sLen = segTotal / float(N_LIGHT);  // length of each sun-ray sub-step
    float odR = 0.0, odM = 0.0;
    for (int i = 0; i < N_LIGHT; ++i) {
        float h = max(0.0, length(p + d * (float(i) + 0.5) * sLen) - R_EARTH);  // altitude at sub-step midpoint
        odR += exp(-h / H_R);  // Rayleigh density (exponential profile, scale height H_R)
        odM += exp(-h / H_M);  // Mie density (exponential profile, scale height H_M)
    }
    return vec2(odR, odM) * sLen;  // summed densities x step length -> optical depth units
}

// (rotateZ lived here — dead since the cloud march moved to cloud_march.comp, removed in the
// pipeline-unification pass. The live copy is cloud_march.comp's.)

// (remap is defined in common.glsl — still used here by the aurora shell/fold code.)

// ── Analytic 3D gradient noise for the cloud domain warp ───────────────────────
// The warp used to read cloudNoiseTex (a 192³ DISCRETELY STORED texture) at kWarpFreq=0.1,
// which spans only ~38 texels across the whole visible range. Trilinear filtering between
// stored texel values is piecewise-multilinear, not truly smooth — each grid cell interpolates
// as a flat-ish shard, not a curved surface. That's invisible at the texture's intended dense
// sampling rate (kCloudHorizFreq=480+), but reading it this sparsely exposed the underlying
// voxel grid directly as faceted, straight-edged geometry — the reported "tessellating"
// artifacts, baked into the cloud edge wherever the warp perturbed the presence threshold.
// Fix: evaluate gradient noise ANALYTICALLY at the exact continuous query point instead of
// interpolating a coarse discrete grid — same hash/gradient technique cloud_noise.comp uses to
// bake the volume, just run live here instead of pre-baked to a fixed low resolution. No
// texture, no discretization, no grid to facet against, and (bonus) no REPEAT-wrap seam class
// of bug possible at all, since there's no stored tile to wrap.
uvec3 warpHashU(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
vec3 warpGradHash(ivec3 c) {
    return normalize(-1.0 + 2.0 * (vec3(warpHashU(uvec3(c))) * (1.0 / 4294967296.0)));
}
float warpPerlin3(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3  f = fract(p);
    vec3  u = f * f * (3.0 - 2.0 * f);   // smoothstep — gives C1-continuous interpolation
    float v000 = dot(warpGradHash(i),                   f              );
    float v100 = dot(warpGradHash(i + ivec3(1,0,0)), f - vec3(1,0,0));
    float v010 = dot(warpGradHash(i + ivec3(0,1,0)), f - vec3(0,1,0));
    float v110 = dot(warpGradHash(i + ivec3(1,1,0)), f - vec3(1,1,0));
    float v001 = dot(warpGradHash(i + ivec3(0,0,1)), f - vec3(0,0,1));
    float v101 = dot(warpGradHash(i + ivec3(1,0,1)), f - vec3(1,0,1));
    float v011 = dot(warpGradHash(i + ivec3(0,1,1)), f - vec3(0,1,1));
    float v111 = dot(warpGradHash(i + ivec3(1,1,1)), f - vec3(1,1,1));
    return mix(mix(mix(v000, v100, u.x), mix(v010, v110, u.x), u.y),
               mix(mix(v001, v101, u.x), mix(v011, v111, u.x), u.y), u.z);
}

// Airglow coverage patchiness (cloud.airglowCoverageGain) for the green/sodium bands — a
// lower-frequency companion to kAirglowNoiseFreq's mild +-40% shimmer above, remapped through a
// threshold (same idiom as cloud_march.comp's own copy / auroraCoverage) so there are real dim
// gaps between brighter patches instead of a gentle wobble. gain=0 reproduces the old uniform-
// minus-shimmer look exactly; gain=1 is full patchiness. Duplicated in cloud_march.comp (which
// needs its own copy for the red band's supplemental march) rather than shared — matches this
// file's standing convention for small per-shader noise helpers, see the aurora functions above.
float airglowCoverageMask(vec3 dirECEF, float t, vec3 seedOffset) {
    float n = warpPerlin3(dirECEF * (kAirglowNoiseFreq * 0.5)
                          + vec3(t * kAirglowDriftRate * 0.6, 0.0, 0.0) + seedOffset);
    return smoothstep(-0.2, 0.5, n);
}

// (cloudWarpOffset lived here — dead since the cloud march moved to cloud_march.comp, removed in
// the pipeline-unification pass. Worth knowing if you go looking for it: this copy still used
// THREE LIVE warpPerlin3 evaluations, while cloud_march.comp's live version reads the baked
// cloudWarpNoiseTex instead (session 31). They were genuinely different algorithms producing
// different values — the drift was invisible only because this copy was already unreachable.)

// cirrusWindAngleAt/cirrusDomainWarp moved to shaders/cloud_march.comp (C15-perf, half-res cloud
// pass) — they were exclusive to cirrusMarch, which moved there too.

// ── Aurora (C16, TERRAIN_PLAN.md Phase E) ──────────────────────────────────────
// Emissive-only "curtain primitive" centered on the geomagnetic pole (NOT the geographic pole —
// see TERRAIN_PLAN.md C16 for why this matters). Geomagnetic poles are antipodal under a dipole
// model, so one ECEF constant covers both hemispheres (negate for south). Derived with the same
// cos(lat)cos(lon)/cos(lat)sin(lon)/sin(lat) formula the fixed observer ECEF constant in
// CLAUDE.md uses. North geomagnetic pole ≈ 80.7°N, 72.7°W (current epoch; drift ~0.05-0.1°/yr is
// negligible at sim epoch 2036).
const vec3  kGeomagPoleECEF      = vec3(0.0481, -0.1543, 0.9868);
const float kAuroraOvalColatDeg  = 20.0;     // oval centerline, degrees from the geomagnetic pole
const float kAuroraOvalWidthDeg  = 6.0;      // base half-width before storm expansion
const float kAuroraShellInnerM   = 95000.0;  // curtain base altitude (m)
const float kAuroraShellOuterM   = 300000.0; // red-fringe top altitude (m)
const vec3  kAuroraBaseColor     = vec3(0.15, 1.0, 0.35);  // green O I 557.7nm — matches airglow green family
const vec3  kAuroraTopColor      = vec3(0.65, 0.15, 0.45); // red/magenta upper fringe
const float kAuroraRingWarpFreq  = 3.0;   // oval-edge ripple spatial frequency (around the ring)
const float kAuroraOvalWarpDeg   = 4.0;   // oval-edge ripple amplitude, degrees of colatitude
const float kAuroraOvalDriftRate = 0.003; // wall-clock rad/s ripple drift (pc.waveTime) — 10x slower
                                           // than first pass per user feedback (evolution read as
                                           // too fast/frantic for something the size of a continent)
// These three frequencies are NOT comparable as raw numbers — colat/az are RADIANS multiplied by
// Earth's radius (~6.57e6 m) to get physical arc length, while altitude is already METERS. A colat
// frequency of 6 looks "low" next to altitude's 0.00001, but 1/6 radian × 6.57e6 m ≈ 1100 km — a
// physically enormous cell, ~10x longer than altitude's own ~100km cell at the time this was first
// (wrongly) tuned. That's why swapping which noise-space SLOT held colat vs altitude didn't fix
// the "streaks point at the pole" bug: colat was still the physically longest axis by a wide
// margin. Values below are chosen so all three land in the same physical ballpark (~50-170km per
// noise cell) with altitude deliberately the largest (long vertical streaks), tangent/radial both
// short (many separate folds, no unwanted radial elongation).
const float kAuroraTangentFreq   = 40.0;     // ~2π/40 rad cell × R·sin(colat)(~2.2e6 m) ≈ 55 km
const float kAuroraRadialFreq    = 70.0;     // ~1/70 rad cell × R (~6.57e6 m) ≈ 94 km
const float kAuroraAltFreq       = 0.000006; // ~1/0.000006 m cell ≈ 167 km — the long/coherent axis
// Evolution speed is user-tunable — see cloud.auroraShimmerRate (settings window "Fold shimmer
// rate"), mixed into the tangent/azimuthal axis in auroraCurtainNoise below, not here.
// Raw accumulation → visible range, same convention as kAirglowScale. Same order of magnitude as
// kAirglowScale (5e-7) despite aurora being a much brighter phenomenon than nightglow — the segLen
// (meters) × sample-count accumulation this multiplies is the same order for both, so the
// brightness difference belongs on cloud.auroraGain (a much higher default than airglowGain), not
// here. First pass used 0.02 (1e4× too large) and blew out to solid white even at auroraGain=0.01 —
// EXPOSURE_NIGHT (10x) compounds the error further downstream, and once any channel's post-exposure
// value is large the Reinhard-style tonemap saturates every channel to 1.0 together, which reads as
// white instead of an overbright green/red.
const float kAuroraScale         = 0.000001;

// Colatitude + azimuth of a direction relative to a geomagnetic pole, plus the local
// tangent (azimuthal, "around the ring") and radial (colatitude, "toward/away from pole")
// basis vectors so the curtain noise can be stretched anisotropically along each axis.
void auroraFrame(vec3 dirECEF, vec3 poleDir, out float colat, out float az,
                  out vec3 radialT, out vec3 tangentT) {
    colat = acos(clamp(dot(dirECEF, poleDir), -1.0, 1.0));
    vec3 ref = abs(poleDir.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    tangentT = normalize(cross(poleDir, ref));
    radialT  = normalize(cross(tangentT, poleDir));
    vec3 localDir = dirECEF - poleDir * dot(dirECEF, poleDir);
    az = atan(dot(localDir, tangentT), dot(localDir, radialT));
}

// Large-scale "coverage" gate — breaks the oval into discrete arcs/patches instead of a solid,
// uniformly lit ring. MUCH lower frequency than auroraCurtainNoise's fold texture (which only
// varies the brightness WITHIN an already-lit patch); this decides whether a whole multi-degree
// stretch of the ring has any aurora at all, which is what actually reads as "twisty curves that
// come and go" rather than fine internal ray structure. storm strength lowers the threshold (fills
// in gaps) — a strong substorm brightens/fills the whole oval, a quiet aurora is patchy — matching
// real auroral behavior (calm-period aurora genuinely does look like broken arcs, not a full ring).
//
// Varies PRIMARILY with COLATITUDE, only mildly with azimuth — a first version did the opposite
// (noise sampled purely as a function of az, constant across colat) and that was backwards: a
// field that's constant along colat and varies with az has its threshold CROSSINGS at fixed-az
// contours, and constant-azimuth lines are meridians — they point straight at the geomagnetic
// pole by definition. At any real frequency that reads as "cranking up frequency, lines tracing up
// to the pole" — exactly the reported bug. Swapping which coordinate dominates makes the threshold
// crossings fall on near-constant-colatitude contours instead, which run parallel to latitude
// circles — "large tracks... parallel with latitude lines", the explicit ask. The azimuthal term
// is kept small purely to keep the boundary from being a perfect circle (a gentle wave instead).
// kAuroraCoverageSoftness stays a fixed constant (edge softness isn't worth its own slider);
// frequency/az-frequency/drift-rate are user-tunable — see cloud.auroraCoverageFreq/AzFreq/
// DriftRate below (settings window "Coverage freq"/"Coverage az freq"/"Coverage drift").
const float kAuroraCoverageSoftness  = 0.45;  // smoothstep width of each patch's edge
float auroraCoverage(float colat, float az, float t, float storm) {
    // Time is mixed into the AZIMUTHAL embedding (x,y), NOT the colatitude axis (z) — an earlier
    // version added t directly to the colat coordinate, which made the whole pattern visibly
    // TRANSLATE toward/away from the pole over time (reported as "waves marching to the poles").
    // colat here is a pure, time-independent spatial axis. (cos(az)+t·rate, sin(az)+t·rate) is no
    // longer on the unit circle once translated, but that's fine — x,y are just an embedding of az
    // chosen to avoid a seam at az=±π, not a meaningful physical direction, so drifting them
    // doesn't create a directionally-biased slide the way drifting colat did; the noise value at
    // any fixed (az,colat) point instead evolves/shimmers over time.
    vec2  azWarp = vec2(cos(az), sin(az)) * cloud.auroraCoverageAzFreq + t * cloud.auroraCoverageDriftRate;
    float n = warpPerlin3(vec3(azWarp, degrees(colat) * cloud.auroraCoverageFreq));
    float threshold = mix(0.2, -0.6, clamp(storm, 0.0, 1.0));
    return smoothstep(threshold, threshold + kAuroraCoverageSoftness, n);
}

// Ripple-displaces the oval's centerline colatitude as a function of azimuth + time, so the
// band isn't a perfect circle. Sampled on (cos az, sin az) rather than az directly — avoids a
// seam at az=±π, same reason cloudWarpOffset feeds a 3D point into warpPerlin3 instead of a
// raw angle. stormStrength widens the band and pushes it equatorward (larger colatitude),
// matching real substorm behavior. Multiplied by auroraCoverage so the band itself is patchy,
// not just internally textured — this is the "erosion" that turns a solid ring into broken arcs.
float auroraOvalMask(float colat, float az, float t, float storm) {
    // Perf: cheap conservative pre-filter before the two warpPerlin3 calls below (ripple +
    // auroraCoverage's own internal one) — auroraSampleAt calls this once per march step, so
    // these 2 evaluations are paid by every sample whose colatitude is even plausibly near the
    // oval, which is the majority of samples along a long oblique ray (the reason the aurora
    // march dominated frame cost — see this session's profiling). centerDeg's worst-case range
    // is [20-6, 20+6+8] = [14,34] (ripple bounded generously at ±1.5*kAuroraOvalWarpDeg=±6°,
    // storm*8.0 up to +8° at storm's slider-enforced max of 1.0); the fade zone reaches
    // widthDeg*2, worst case 2*(6*(1+1.5))=30° at storm=1. So no parameter combination can light
    // any colatitude beyond 34+30=64°, worst case — 70° below is that bound plus margin, kept as
    // a plain arithmetic comparison (no noise, no branches inside a loop already paid for) so it
    // costs nothing on the samples that DO need the real calculation.
    if (degrees(colat) > 70.0)
        return 0.0;
    vec2  ringP  = vec2(cos(az), sin(az)) * kAuroraRingWarpFreq;
    float ripple = warpPerlin3(vec3(ringP, t * kAuroraOvalDriftRate)) * kAuroraOvalWarpDeg;
    float centerDeg = kAuroraOvalColatDeg + ripple + storm * 8.0;
    float widthDeg  = kAuroraOvalWidthDeg * (1.0 + storm * 1.5);
    // Full brightness out to half of widthDeg, then a WIDE gradual fade out to 2x widthDeg —
    // previously the whole 0..widthDeg span was the falloff, which hit exactly zero right at the
    // oval's nominal edge. Airglow (green/sodium, similar altitude) has no such cutoff at all, so
    // that hard zero read as a visible seam where the aurora clipped against it. Widening the fade
    // zone (without changing the bright "core" size) blends the two smoothly instead.
    float distDeg = abs(degrees(colat) - centerDeg);
    float band = smoothstep(widthDeg * 2.0, widthDeg * 0.5, distDeg);
    return band * auroraCoverage(colat, az, t, storm);
}

// Curtain fold structure: many thin vertical sheets standing up off the surface, distributed
// around the ring — NOT rays radiating toward/away from the pole (two earlier attempts produced
// exactly that "spokes pointing at the pole" look: first by anisotropically stretching the wrong
// pair of axes, then by picking a colat frequency that LOOKED small as a raw number but was still
// physically enormous once multiplied by Earth's radius — see the frequency constants' comment
// above for the physical-cell-size reasoning that fixed it for real). Built from two anisotropic
// warpPerlin3 samples: HIGH frequency along the tangent (azimuthal) axis gives many separate folds
// distributed around the ring; comparably HIGH frequency along colatitude keeps the band's
// cross-section from adding unwanted radial coherence; LOW frequency along ALTITUDE is the one
// genuinely long axis, keeping each fold an unbroken streak running vertically. storm increases
// fold frequency (more chaotic structure).
//
// Time (cloud.auroraShimmerRate) drives the TANGENT/azimuthal axis, NOT altitude — a first version
// added it directly to the altitude coordinate, which made every fold visibly scroll monotonically
// top-to-bottom (reported as "columns flicker from top to bottom" — the same axis-conflation
// mistake auroraCoverage's colat/time bug was, one layer down). Altitude is a real physical
// direction (up), so translating it with time reads as a directional slide.
//
// It's mixed in via a WARP PHASE (a separate warpPerlin3 evaluation with t as one of its inputs),
// NOT added directly as `+ t*rate` — a second version did the direct-add version, which fixed the
// top-to-bottom slide but was reported as looking like "a spinning texture moving east to west":
// still a rigid, linear translation, just along a different (harmless-direction) axis instead of a
// wrong one. A pure additive shift is mechanical regardless of which axis it's on — real curtains
// don't slide sideways at constant velocity, they morph. Routing time through its own noise
// evaluation first (same technique cloudWarpOffset already uses for cloud shape) means the phase
// itself changes non-monotonically over time, and — since colat/altM feed the SAME warp evaluation
// — varies smoothly across the curtain instead of shifting every fold by an identical amount in
// lockstep, so it reads as evolving structure rather than the whole pattern sliding as one rigid
// sheet.
float auroraCurtainNoise(float colat, float az, float altM, float t, float storm) {
    float shimmerPhase = warpPerlin3(vec3(colat * kAuroraRadialFreq * 0.3,
                                           altM * kAuroraAltFreq * 0.3,
                                           t * cloud.auroraShimmerRate)) * 2.5;
    vec3 p = vec3(az * kAuroraTangentFreq * (1.0 + storm * 0.8) + shimmerPhase,
                  altM * kAuroraAltFreq,
                  colat * kAuroraRadialFreq);
    // Perf (this session): "base" is now a texture read against auroraNoiseTex's R channel
    // (baked once by aurora_noise.comp) instead of a live warpPerlin3 call — the biggest single
    // piece of "why is aurora so much more expensive than clouds, which look more complex"
    // (clouds' noise was already baked in a prior session; aurora's never was). shimmerPhase above
    // stays LIVE and cheap (1 call) — it's the animation driver, offsetting the sample coordinate
    // into the static baked texture, same "warp the lookup, bake the detail" split cloudWarpOffset
    // already uses for clouds. (Also dropped the "detail" octave earlier this session — one fewer
    // call — before this replaced the remaining "base" call entirely.)
    //
    // U wraps at the TRUE physical period (kAuroraTangentFreq*2*PI≈251.3), independent of the
    // bake's own internal resolution (256 cells/loop — a power-of-2 approximation of that period,
    // required for correct tiling, see aurora_noise.comp). fract() here handles both the wrap AND
    // storm's frequency-scaling of p.x correctly: a p.x that completes more physical cycles per
    // loop (storm scales the az term) just wraps U more times, which reads as "more folds" — the
    // same visual effect storm produced before, now via repetition of the baked pattern rather
    // than genuinely new higher-frequency noise (an accepted approximation).
    // V/W are clamped to the bake's fixed ranges (see aurora_noise.comp): p.y in [0.24, 2.46]
    // (= [kAuroraMarchInnerM, kAuroraMarchOuterM] * kAuroraAltFreq), p.z in [0, 91.6]
    // (= [0, 75deg] * kAuroraRadialFreq).
    const float kAuroraCurtainPeriod = kAuroraTangentFreq * 2.0 * PI;
    float texU = fract(p.x / kAuroraCurtainPeriod);
    float texV = clamp((p.y - 0.24) / (2.46 - 0.24), 0.0, 1.0);
    float texW = clamp(p.z / 91.6, 0.0, 1.0);
    float base = texture(auroraNoiseTex, vec3(texU, texV, texW)).r * 2.0 - 1.0;
    return remap(base, -0.3, 0.9, 0.0, 1.0); // bias toward bright folds over dark gaps
}

// Combined density + color sample at a point given in the observer's local ENU-scaled frame
// (rp; same frame obsPos/dir/main()'s march points use), converted to a true ECEF direction via
// the enuX/enuY/enuZ basis — mirrors the rDirECEF idiom the airglowRed march already uses.
// Picks whichever geomagnetic pole (north or south) the point is nearer to, so one code path
// covers both hemispheres.
//
// Day-gated per-SAMPLE on that point's own geographic day/night state (same twilight window
// airglowRed's rDayness/rNight uses), NOT on the observer's local sky brightness as originally
// planned. The observer-based gate (a single smoothstep on pc.sunDirENU.w, applied once outside
// the march) was wrong for an orbital view near the terminator: an observer whose own local sun
// angle reads "daylight" can still be looking at a geographically dark limb with a large visible
// night-side portion, and the old gate blacked out the aurora there entirely instead of letting
// it fade in over the genuinely dark samples along that same ray.
vec3 auroraSampleAt(vec3 rp, vec3 enuX, vec3 enuY, vec3 enuZ, vec3 sunDirECEF, float t, float storm) {
    vec3 pDirECEF = normalize(rp.x * enuX + rp.y * enuY + rp.z * enuZ);
    float dayness = clamp((dot(pDirECEF, sunDirECEF) + 0.15) / 0.3, 0.0, 1.0);
    float night   = 1.0 - dayness;
    if (night <= 0.001) return vec3(0.0); // cheapest test first: this patch of sky is in daylight
    vec3 poleDir  = (dot(pDirECEF, kGeomagPoleECEF) > 0.0) ? kGeomagPoleECEF : -kGeomagPoleECEF;
    float colat, az; vec3 radialT, tangentT;
    auroraFrame(pDirECEF, poleDir, colat, az, radialT, tangentT);
    float oval = auroraOvalMask(colat, az, t, storm);
    if (oval <= 0.001) return vec3(0.0); // cheap early-out before the pricier fold noise below
    float altM = length(rp) - R_EARTH;
    // Inner/outer edges kept as SEPARATE terms (not pre-multiplied into one `vert`) so both the
    // fold-contrast fade and the color blend below can use each edge's own weight independently —
    // innerVert also does double duty as the early-out gate via the combined `vert` product.
    // SIGMOID falloff, not smoothstep — smoothstep(edge0,edge1,x) is EXACTLY zero at and below
    // edge0 no matter how far apart edge0/edge1 are; widening the transition just moves where that
    // hard floor sits; it can never remove it. That's why the previous fix (widening to 80-110km)
    // just relocated the visible cut from 95km to exactly 80km instead of eliminating it. A sigmoid
    // asymptotically approaches 0/1 without ever exactly reaching either — no floor to hit at any
    // altitude, so there's nothing left to read as a hard edge. `kAuroraInnerFalloffM`/
    // `kAuroraOuterFalloffM` set the transition's rough WIDTH (effective ~4x this value from ~12%
    // to ~88%), analogous to smoothstep's old span but without the hard endpoint. The march's own
    // bounds (main()) were extended further to match — a sigmoid's tail is still finite in practice
    // (the vert<=0.001 early-out below still culls it eventually), but that cull point needs to
    // actually be reachable by the march, not clipped off before the tail gets sampled at all.
    const float kAuroraInnerFalloffM = 7500.0;
    const float kAuroraOuterFalloffM = 15000.0;
    float innerVert = 1.0 / (1.0 + exp(-(altM - kAuroraShellInnerM) / kAuroraInnerFalloffM));
    float outerVert = 1.0 / (1.0 + exp((altM - kAuroraShellOuterM) / kAuroraOuterFalloffM));
    float vert = innerVert * outerVert;
    if (vert <= 0.001) return vec3(0.0);
    // Fold contrast is blended toward a flat 1.0 as `vert` approaches its edges (mix(1.0,fold,vert),
    // NOT raw fold) — softening `vert` alone (the earlier fix) made the DENSITY/ALPHA transition
    // gradual, but the fold NOISE's own structure stayed at full contrast right up until vert hit
    // zero, so the curtain's sharply-textured folds still snapped straight to airglow's smooth,
    // uniform glow at the boundary — a texture/character discontinuity, not a brightness one, and
    // exactly why softening the density alone didn't read as a real blend. Fading the structure
    // itself alongside the density means the curtain smooths out into a uniform glow BEFORE it
    // fades away, so it hands off to airglow's own uniform character instead of cutting to it.
    // Per-column elevation window: without this, every column spans the FULL inner-to-outer shell
    // (vert/innerVert/outerVert above are the same at every colat/az), so every fold shows the same
    // complete base->top gradient — reads as suspiciously uniform. Real curtains vary in height: some
    // barely lift off the ~95km base, others tower to the full ~300km extent. Sampled as a function of
    // (colat, az) ONLY — no altitude, no time — so it's constant all the way up a given column (that's
    // the definition of "this column's own height range") and stable frame to frame rather than
    // flickering. Low frequency (kAuroraColumnFreq, well below the fold texture's own tangent/radial
    // frequencies) so one "column" here bundles many individual fold-noise folds together, matching
    // the real scale where dozens of thin folds share one taller or shorter structure.
    //
    // Deliberately kept SEPARATE from vert/innerVert/outerVert rather than replacing them — those two
    // still drive the color blend and the airglow hand-off at the TRUE shell bounds (kAuroraShellInnerM/
    // OuterM), which must stay physically anchored there regardless of any one column's random window.
    // This only gates final visibility/opacity on top.
    const float kAuroraColumnFreq = 9.0;
    // Perf (this session): colA/colB are now a single texture read against auroraNoiseTex's G/B
    // channels (baked by aurora_noise.comp) instead of 2 separate live warpPerlin3 calls — no
    // per-frame animation here to preserve (the live version had none either: no altitude, no
    // time — constant all the way up a given column, see the comment above), so this is a
    // straightforward bake with no runtime coordinate warp needed. az's column-specific frequency
    // (kAuroraColumnFreq) cancels out of the wrap-period division below (az*freq / (freq*2*PI) =
    // az/(2*PI)), so the U mapping is independent of it; W reuses the exact same colatitude-
    // fraction formula as the curtain sample above (colat / radians(75)) — both channels were
    // baked over the identical [0,75deg] colatitude range, just at different internal frequencies.
    float colTexU = fract(az / (2.0 * PI));
    float colTexW = clamp(colat / radians(75.0), 0.0, 1.0);
    vec2  colSample = texture(auroraNoiseTex, vec3(colTexU, 0.5, colTexW)).gb;
    float colA = colSample.x;
    float colB = colSample.y;
    // Sorting two decorrelated samples into lo/hi (instead of deriving lo/hi from one center+halfwidth)
    // naturally produces the full requested spread: when colA/colB land close together the window is
    // narrow (low OR high depending on where), when they land far apart (near 0 and near 1) the window
    // covers nearly the whole shell — "some just low, some just high, others span the full distance"
    // falls out of this without needing separate special cases.
    float colLoFrac = clamp(min(colA, colB) - 0.08, 0.0, 1.0);
    float colHiFrac = clamp(max(colA, colB) + 0.08, 0.0, 1.0);
    float colLoM = mix(kAuroraShellInnerM, kAuroraShellOuterM, colLoFrac);
    float colHiM = mix(kAuroraShellInnerM, kAuroraShellOuterM, colHiFrac);
    const float kAuroraColumnFalloffM = 12000.0;
    float columnWindow = (1.0 / (1.0 + exp(-(altM - colLoM) / kAuroraColumnFalloffM)))
                        * (1.0 / (1.0 + exp((altM - colHiM) / kAuroraColumnFalloffM)));
    if (columnWindow <= 0.001) return vec3(0.0);
    float fold = mix(1.0, auroraCurtainNoise(colat, az, altM, t, storm), vert);
    vec3  col  = mix(kAuroraBaseColor, kAuroraTopColor,
                      clamp(remap(altM, kAuroraShellInnerM, kAuroraShellOuterM, 0.0, 1.0), 0.0, 1.0));
    // Hue also blends toward the nearby airglow band's own color in each edge zone — green/sodium
    // near the inner edge (matching their real 83-105km presence), red near the outer edge (matching
    // its real 200-350km presence) — instead of aurora's own base/top gradient just stopping short
    // and handing off to a completely differently-colored airglow with no shared transition at all.
    vec3 innerAirglowCol = mix(kAirglowSodiumColor, kAirglowGreenColor, 0.5);
    col = mix(innerAirglowCol, col, innerVert);
    col = mix(kAirglowRedColor, col, outerVert);
    return col * (oval * fold * vert * columnWindow * night);
}

// Representative altitude for auroraGlowAt's fold-noise texture — the ground-glow term doesn't
// march a real altitude, it just needs *a* fixed altitude to sample the curtain's horizontal
// structure at (mid-shell, roughly where the green base is brightest).
const float kAuroraGroundGlowAltM = 150000.0;

// Local aurora ambient light AT A GIVEN GEOGRAPHIC POINT (terrain/ocean hit point, ocean-reflection
// sample, etc.) — evaluates the SAME oval mask + curtain-fold noise the sky curtain itself uses,
// but keyed on that point's own geographic location instead of the observer's. This is what makes
// aurora ground-lighting properly local, the same way moonlight is: a patch of ground directly
// under an active curtain lights up regardless of where the observer is standing, and moving the
// observer somewhere else doesn't turn it off. (The first implementation computed a single CPU-side
// value from the OBSERVER's position and applied it to everything in view — from LEO that lit the
// entire visible Earth uniformly green whenever the observer's orbit passed over the oval, and shut
// off instantly the moment it didn't, regardless of what was actually under the curtain. See
// TERRAIN_PLAN.md session 28 follow-up #5.)
vec3 auroraGlowAt(vec3 posDirECEF, vec3 sunDirECEF, float t, float storm) {
    float dayness = clamp((dot(posDirECEF, sunDirECEF) + 0.15) / 0.3, 0.0, 1.0);
    float night   = 1.0 - dayness;
    if (night <= 0.001) return vec3(0.0);
    vec3 poleDir  = (dot(posDirECEF, kGeomagPoleECEF) > 0.0) ? kGeomagPoleECEF : -kGeomagPoleECEF;
    float colat, az; vec3 radialT, tangentT;
    auroraFrame(posDirECEF, poleDir, colat, az, radialT, tangentT);
    float oval = auroraOvalMask(colat, az, t, storm);
    if (oval <= 0.001) return vec3(0.0);
    float fold = auroraCurtainNoise(colat, az, kAuroraGroundGlowAltM, t, storm);
    return kAuroraBaseColor * oval * fold * night;
}

// ── Lens flare (adapted from "Lens Flare Example" by peterekepeter, public domain)
// ─────────────────────────────────────────────────────────────────────────────
// Produces the visible corona/bloom around the source AND the reflected ghost
// artifacts that appear along the flare axis (source -> screen centre -> beyond).
// Diffraction spikes are intentionally omitted; instead the irregular corona
// shape (human-eye / dirty-lens airy-disk pattern) is produced entirely by the
// noise texture lookup on f0.
//
// Coordinate space: ShaderToy-style UV.
//   x in [-0.5*aspect, +0.5*aspect],  y in [-0.5, +0.5].
//
// Parameters:
//   uv     -- current fragment position in flare UV space
//   pos    -- source (satellite / sun) position in flare UV space
//   intens -- normalised brightness [0,1]; controls f0 scale and ghost strength
//
// Returns an HDR additive RGB contribution.
// The call site multiplies by a tint and an overall scale factor.
// ─────────────────────────────────────────────────────────────────────────────
// bokehMult: independent brightness scalar for the ghost/bokeh elements (f2–f6).
// Use a small value (e.g. 0.3) for satellites, larger (e.g. 2.0) for the sun.
// Separates corona brightness (intens) from artifact brightness (bokehMult).
vec3 lensFlare(vec2 uv, vec2 pos, float intens, float bokehMult) {

    // uvd: radially distorted UV -- uv * |uv|.
    // Near screen centre uvd ~= 0; toward edges it bends outward.
    // Ghost artifacts use uvd so their positions follow the curved optical path
    // of real multi-element lens reflections.
    vec2 uvd = uv * length(uv);

    // d: displacement from current fragment to source.
    vec2 d = uv - pos;

    // dist: radius^0.1 -- nearly 1.0 everywhere, dips to 0 right at the source.
    // Used as a small radial term in f0's shimmer modulation.
    float dist = pow(length(d), 0.1);

    // ang: polar angle [-pi, +pi] around the source.
    // Used to sample the noise texture angularly so the corona has irregular lobes.
    float ang = atan(d.y, d.x);

    // ── Angular corona noise via texture lookup ────────────────────────────────
    // Replicates the original ShaderToy formula:
    //   noise(sin(ang*4 + pos.x)*4 - cos(ang*3 + pos.y))
    //
    // The argument is a smoothly-varying scalar that changes both with the angle
    // around the source (ang) and with the source's screen position (pos.x, pos.y).
    // This means each satellite at a different screen position has a unique corona
    // shape -- the lobes don't align between adjacent satellites.
    //
    // Mapping the scalar to a UV coordinate for noiseTex:
    //   We use a 1D slice along the texture's x-axis (v = 0.5, middle row).
    //   The u coordinate wraps via the REPEAT sampler so any float value is valid.
    //   The noise value (red channel) is then passed into sin(...*16)*0.1 which
    //   creates fine angular variation (+/-10%) around the corona rim.
    // noiseSeed is a smoothly-varying float that changes with angle and source position.
    // We map it into [0,1] UV space by dividing by the expected range (~8) and adding
    // 0.5 to centre it, then rely on REPEAT wrapping for values outside [0,1].
    // Using fract() explicitly makes the wrapping behaviour unambiguous.
    // The v coordinate is fixed at 0.25 (upper quarter of texture, away from the
    // edge to avoid any border artifacts on some hardware).
    float noiseSeed = sin(ang * 4.0 + pos.x) * 4.0 - cos(ang * 3.0 + pos.y);
    float noiseU    = fract(noiseSeed * 0.125 + 0.5); // map [-8,+8] -> [0,1], wrapping
    float angNoise  = texture(noiseTex, vec2(noiseU, 0.25)).r;

    // ── Source glow: Lorentzian corona centered on the source ─────────────────
    // The Lorentzian  1/(r * scale + 1)  is wider and softer than a Gaussian,
    // matching real lens-coating scatter on a bright point source.
    //
    float scale = 1200.0; // corona radius: higher = tighter. 60 = wide (visible at 200px), 1200 = tight (visible at ~15px)
    //   r = 0.005 (~5px at 1080p):  f0 = 1/(0.005*60+1) = 0.77
    //   r = 0.02  (~22px):          f0 = 1/(0.02 *60+1) = 0.45
    //   r = 0.05  (~54px):          f0 = 1/(0.05 *60+1) = 0.25
    //   r = 0.10  (~108px):         f0 = 1/(0.10 *60+1) = 0.14
    //   r = 0.20  (~216px):         f0 = 1/(0.20 *60+1) = 0.077
    // This gives a wide, visible corona that extends well past the satellite dot
    // and fades naturally without a hard edge.  The old scale of 200 fell to
    // <0.05 at only 50px, making the corona invisible at our additive blend scale.
    //
    // The modulation line applies the noise-driven angular shimmer:
    //   sin(angNoise * 16) * 0.1  -- fine ripple from texture (+/- 10% per lobe)
    //   dist * 0.1                -- barely-there radial taper (~constant ~1)
    //   + 0.8                     -- base boost so the corona is always bright
    // sin(noise*16) oscillates rapidly around the corona, creating 8-16 irregular
    // bright lobes -- the airy-disk / human-eye diffraction pattern.
    float f0 = 1.0 / (length(d) * scale + 1.0);
    f0 = f0 + f0 * (sin(angNoise * 16.0) * 20.8 + dist);
    // Scale by intensity so dimmer satellites have a proportionally smaller corona.
    f0 *= 0.1;// + intens * 0.5);

    // ── Large near-source bloom: soft blob mirrored through screen centre ──────
    // Placed at -1.2*pos (reflected slightly beyond centre).
    // Represents light that bounced backward through the lens and re-emerged near
    // the entrance pupil.  Multiplier 4.0 (reduced from original 7.0) and
    // contribution capped below to prevent peripheral over-saturation.
    float f1 = max(0.01 - pow(length(uv + 1.2 * pos), 1.9), 0.0) * 4.0;
    f1 *= 0.6;

    // ── Ghost artifacts: fade when source is near screen centre ───────────────
    // When pos ~= (0,0) (looking directly at the source), uvd + k*pos ~= uvd,
    // which is nearly zero everywhere near centre.  The Lorentzian denominator
    // (1 + 32*r^2) then approaches 1 everywhere, lighting up the entire screen.
    //
    // ghostFade = smoothstep(0.03, 0.12, |pos|):
    //   source within ~3% screen height of centre: ghosts = 0
    //   source more than 12% screen height off-centre: ghosts full
    // This also makes physical sense: looking directly at the source means ghost
    // reflection paths don't form visible off-axis elements.
    float ghostFade = smoothstep(0.03, 0.12, length(pos));

    // ── Bokeh halos: large circular rings reflected through screen centre ──────
    // Classic rainbow-ringed bokeh circles opposite the source.
    // Lorentzian  1/(1 + 32*r^2)  matches wide, soft real ghost disc profiles.
    // Three slightly offset RGB positions produce chromatic aberration fringing.

    float f2  = max(1.0/(1.0 + 32.0*pow(length(uvd + 0.80*pos), 2.0)), 0.0) * 0.25 * bokehMult;
    float f22 = max(1.0/(1.0 + 32.0*pow(length(uvd + 0.85*pos), 2.0)), 0.0) * 0.23 * bokehMult;
    float f23 = max(1.0/(1.0 + 32.0*pow(length(uvd + 0.90*pos), 2.0)), 0.0) * 0.21 * bokehMult;

    // ── Star-shaped secondary bokeh (between source and centre) ───────────────
    // uvx = 1.5*uv - 0.5*uvd.  The 2.4 exponent gives a slightly star-shaped
    // profile (intermediate between circle and square).
    // RGB variants at 0.40/0.45/0.50*pos create a second tier of chromatic split.
    vec2 uvx = mix(uv, uvd, -0.5);
    float f4  = max(0.01 - pow(length(uvx + 0.40*pos), 2.4), 0.0) * 6.0;
    float f42 = max(0.01 - pow(length(uvx + 0.45*pos), 2.4), 0.0) * 5.0;
    float f43 = max(0.01 - pow(length(uvx + 0.50*pos), 2.4), 0.0) * 3.0;

    // ── Compact sparkle dots along the flare axis ─────────────────────────────
    // High exponent (5.5) = sharp dropoff = tight bright pinpoints at 0.2/0.4/0.6*pos.
    uvx = mix(uv, uvd, -0.4);
    float f5  = max(0.01 - pow(length(uvx + 0.20*pos), 5.5), 0.0) * 2.0;
    float f52 = max(0.01 - pow(length(uvx + 0.40*pos), 5.5), 0.0) * 2.0;
    float f53 = max(0.01 - pow(length(uvx + 0.60*pos), 5.5), 0.0) * 2.0;

    // ── Broad streaks on the camera-side of centre ────────────────────────────
    // Negative multiplier places these between centre and the source.
    // Low exponent (1.6) = broad, diffuse -- reads as a smear on the front element.
    uvx = mix(uv, uvd, -0.5);
    float f6  = max(0.01 - pow(length(uvx - 0.300*pos), 1.6), 0.0) * 6.0;
    float f62 = max(0.01 - pow(length(uvx - 0.325*pos), 1.6), 0.0) * 3.0;
    float f63 = max(0.01 - pow(length(uvx - 0.350*pos), 1.6), 0.0) * 5.0;

    // (A "radial ray fan" sunburst stand-in for screen-space godrays lived here briefly,
    // 2026-07-29 — reverted same day per user feedback: too sharply-defined/star-like for the
    // soft, astigmatism-like human-eye look this flare is going for, and made the many-satellite
    // Reflect-Orbital case look worse, not better. The prior streak/bokeh terms above (f4-f6) are
    // the preferred "spike" look. Screen-space godrays remain a real, unimplemented want — see
    // TERRAIN_PLAN.md's follow-up log for the "threshold + depth-subtract + radial blur" direction
    // proposed instead.)

    // ── Assemble ──────────────────────────────────────────────────────────────
    vec3 c = vec3(0.0);

    // Source corona -- achromatic (warm white set by call-site tint).
    c += vec3(f0);
    c += vec3(f1 * 0.5);  // bloom at -1.2*pos

    // Ghost terms: chromatic, gated by ghostFade to prevent centre blowout.
    // bokehMult independently scales all ghost/reflection artifacts from the corona (f0).
    c.r += (f2  + f4  + f5  + f6)  * 0.4 * ghostFade * bokehMult;
    c.g += (f22 + f42 + f52 + f62) * 0.4 * ghostFade * bokehMult;
    c.b += (f23 + f43 + f53 + f63) * 0.4 * ghostFade * bokehMult;

    // Slight vignette: outer screen positions have more lens distortion.
    c = c * 1.3 - vec3(length(uvd) * 0.05);

    return max(c, vec3(0.0));
}

// ── Ocean wave functions (adapted from "Seascape" by Alexander Alekseev aka TDM, 2014)
// License: CC-BY-NC-SA 3.0 — tdmaav@gmail.com
// posM = ENU East/North metres + geographic phase offset (observer-relative, ~Earth-fixed);
// pHeight = metres above R_EARTH; seaTime = 1.0 + pc.waveTime * kSeaSpeed.

const mat2  kOctaveM       = mat2(1.6, 1.2, -1.2, 1.6);
const float kSeaFreq       = 0.056;
const float kSeaHeight     = 2;
const float kSeaChoppy     = 3.0;   // 4.0 → 2.0: rounder crests, less plateau cliffs
const float kSeaSpeed      = 1.5;
const vec3  kSeaBase = vec3(0.01, 0.04, 0.08);   // dark, desaturated blue
const vec3  kSeaWaterColor = vec3(0.2, 0.50, 0.85) * 0.1;

// Hash without Sine (Dave Hoskins, MIT): stable for all float input magnitudes.
// The original fract(sin(dot(p, large_vec))*large_num) loses GPU sin() precision
// once the dot product exceeds ~10^4 (happens at 4th-5th octave where kOctaveM
// doubles UV scale each iteration), producing the angular banding artifact.
float seaHash(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973) * 0.1); //vec3(0.1031, 0.1030, 0.0973)
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
float seaNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return -1.0 + 2.0 * mix(
        mix(seaHash(i + vec2(0.0, 0.0)), seaHash(i + vec2(1.0, 0.0)), u.x),
        mix(seaHash(i + vec2(0.0, 1.0)), seaHash(i + vec2(1.0, 1.0)), u.x),
        u.y);
}

float seaOctave(vec2 uv, float choppy) {
    uv += seaNoise(uv);
    vec2 wv  = 1.0 - abs(sin(mod(uv, vec2(2.0 * PI, 2.0 * PI))));
    vec2 swv = abs(cos(mod(uv, vec2(2.0 * PI, 2.0 * PI))));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

// Geometry pass (3 octaves default): used in height-map trace. Octave count is UBO-tunable
// (cloud.oceanSeaOctaves, perf session 24) — this is called up to 10x per ocean pixel by
// heightMapTracing's secant refinement, so it's a direct multiplicative cost lever.
float seaMap(vec2 posM, float pHeight, float seaTime) {
    float freq = kSeaFreq, amp = kSeaHeight, choppy = kSeaChoppy;
    vec2  uv   = posM; uv.x *= 0.75;
    float h    = 0.0;
    int   nOct = int(max(1.0, cloud.oceanSeaOctaves));
    for (int i = 0; i < nOct; i++) {
        float d  = seaOctave((uv + seaTime) * freq, choppy);
              d += seaOctave((uv - seaTime) * freq, choppy);
        h  += d * amp;
        uv *= kOctaveM; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return pHeight - h;
}

// Fragment pass (5 octaves default): used for high-quality normal computation. Octave count is
// UBO-tunable (cloud.oceanDetailOctaves, perf session 24).
float seaMapDetail(vec2 posM, float pHeight, float seaTime) {
    float freq = kSeaFreq, amp = kSeaHeight, choppy = kSeaChoppy;
    vec2  uv   = posM; uv.x *= 0.75;
    float h    = 0.0;
    int   nOct = int(max(1.0, cloud.oceanDetailOctaves));
    for (int i = 0; i < nOct; i++) {
        float d  = seaOctave((uv + seaTime) * freq, choppy);
              d += seaOctave((uv - seaTime) * freq, choppy);
        h  += d * amp;
        uv *= kOctaveM; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return pHeight - h;
}

// ── Thin-shell cloud layer evaluator ─────────────────────────────────────────
// Intersects a sphere shell at R_EARTH + shellAltM, samples earthCloudsTex at the
// hit point's geographic lat/lon (Earth-fixed UV + per-layer longitude drift), and
// blends the result into `color`.
//
// Lighting uses dot(normalize(cloudPointECEF), sunDirECEF) — the sun angle at the
// cloud's own geographic location, NOT the observer's sun elevation.  This ensures
// clouds on the dark side of Earth are dark regardless of where the observer is.
void evalCloudLayer(
    vec3  obsPos,  vec3 dir,  float tSurface,
    vec3  enuX,    vec3 enuY, vec3  enuZ,
    vec3  sunDirECEF,
    float odRcam,  float odMcam,
    float coverage, float density, float sunGain, float sunGainZenith,
    float shellAltM, float driftMult, float alphaMax, float mipLod,
    float cloudPhase,
    float obsEffH, float volumetricPair,
    vec3  skyOnlyColor,
    inout vec3 color)
{
    // Runtime-tunable scattering strength — shadows the physical base constants (common.glsl)
    // with the user-facing "Rayleigh gain"/"Mie/haze gain" sliders. See cloud_params.glsl's
    // atmosRayleighGain/atmosMieGain comment for what each one does perceptually.
    //
    // flatRayleighGain ("2D Rayleigh gain") stacks on top of the global Rayleigh gain and applies
    // to THIS function only — the flat 2D paste — so the 3D->2D crossfade can be matched without
    // moving the sky/volumetric look. It lands on both uses of BETA_R below, which is the point:
    // sunColorFlat (the sun's own colour arriving AT the cloud) and attn/airlight (the
    // camera->cloud path) together are what set how red and how washed-out the flat layer reads
    // relative to the volumetric shell it fades into. 1.0 = the previous coupled behaviour
    // exactly. See cloud_params.glsl's flatRayleighGain comment.
    vec3  BETA_R = BETA_R_BASE * cloud.atmosRayleighGain * cloud.flatRayleighGain;
    float BETA_M = BETA_M_BASE * cloud.atmosMieGain;

    vec2  tc = raySphere(obsPos, dir, R_EARTH + shellAltM);
    float t  = (tc.x > 0.001) ? tc.x : tc.y;
    if (t <= 0.001) return;
    if (tSurface > 0.0 && t >= tSurface) return;

    // Crossfade against the volumetric pass, for the two layers it also renders (0 and 1). This
    // has to be computed HERE rather than at the call site, because it depends on t — the
    // distance to the shell along THIS ray — and the call site does not have it. That is the whole
    // point of moving to a distance-keyed fade: it varies across the screen, so near clouds can be
    // volumetric while horizon clouds on the same frame are flat. The weight is the exact
    // complement of cloudMarchCS's, so the two always sum to one.
    if (volumetricPair > 0.5) {
        float altFade  = 1.0 - smoothstep(kCloud3DFadeStart, kCloud3DFadeEnd, obsEffH);
        float distFade = 1.0 - smoothstep(cloud.cloudDistFadeStartM, cloud.cloudDistFadeEndM, t);
        alphaMax *= 1.0 - min(altFade, distFade);
        if (alphaMax < 0.001) return;
    }

    // Hit point in ENU → convert to ECEF for geographic UV and sun-dot
    vec3  hitENU = obsPos + t * dir;
    vec3  cECEF  = hitENU.x * enuX + hitENU.y * enuY + hitENU.z * enuZ;
    float cL     = length(cECEF);
    float cLon   = atan(cECEF.y, cECEF.x);
    float cLat   = asin(clamp(cECEF.z / cL, -1.0, 1.0));

    // Earth-fixed UV with per-layer longitude drift
    vec2  uv    = vec2(fract((cLon + PI) / (2.0*PI) + cloudPhase * driftMult / (2.0*PI)),
                       (0.5*PI - cLat) / PI);
    float raw   = textureLod(earthCloudsTex, uv, mipLod).r;
    float alpha = clamp((raw - (1.0 - coverage)) * density, 0.0, alphaMax);
    if (alpha <= 0.0) return;

    // Sun angle at the cloud's geographic position — independent of observer location
    float cloudSunDot  = dot(normalize(cECEF), sunDirECEF);
    float cloudDayFrac = smoothstep(-0.1, 0.15, cloudSunDot);
    // See cloud_march.comp's cloudMarchCS sunGainCurve comment — same horizon/zenith blend.
    float sunGainCurve = mix(sunGain, sunGainZenith,
                             smoothstep(0.0, max(cloud.sunGainElevBand, 0.02),
                                        clamp(cloudSunDot, 0.0, 1.0)));
    // Soft-compressed rather than the old raw product. This term fed straight into the composite
    // unbounded, so a sunGain tuned to give the VOLUMETRIC path good sunsets drove the flat layer
    // hard into pure white — the two paths respond to the same slider completely differently,
    // because the volumetric accumulates through transmittance while this is a single multiply.
    // 1-exp(-x) is ~x for small x (dim clouds unchanged) and asymptotes to 1 instead of clipping,
    // which is also about right physically: a fully lit cloud's albedo is ~0.7-0.9, not unbounded.
    // Spectral sun colour at the CLOUD's own geographic position. This term used to be a pure
    // white vec3, which is precisely why raising sun gain only ever made 2D clouds BRIGHTER and
    // never orange: the flat path had no wavelength-dependent factor anywhere in it, while the
    // volumetric path multiplies by sunColorCloud — its own optDepth-derived sun transmittance at
    // shell entry. This is the same integral, evaluated in ECEF from the cloud point toward the
    // sun, so at a grazing sun the long air path extinguishes blue and green and what actually
    // reaches the cloud is genuinely orange/red rather than white scaled up.
    //
    // The soft 1-exp(-x) compression below then works per channel, so at high gain red saturates
    // first and the colour survives instead of washing to white the way a scaled grey did.
    //
    // dbgSkipSunOD() makes optDepth return 0, hence vec3(1.0) — exactly the old white behaviour —
    // so the existing sun-OD knockout bit isolates this cleanly at zero extra plumbing.
    // Two details here are copied deliberately from cloudMarchCS's sunColorCloud block rather
    // than reinvented, because getting either one wrong shifts the HUE relative to the volumetric
    // path — which is exactly what the first version of this did (2D read distinctly redder):
    //
    //   * BETA_M * 1.1, not BETA_M. Mie extinction is wavelength-neutral, so it is the term that
    //     dilutes Rayleigh's red with grey. Using a 10% weaker Mie coefficient than the volumetric
    //     leaves proportionally less grey in the mix, i.e. a MORE saturated red, for the same
    //     geometry.
    //   * The Earth-shadow gate. raySphere against R_EARTH first: if the sun ray from this cloud
    //     point enters the planet, the cloud is in Earth's shadow and gets no sunlight at all.
    //     Without it, optDepth happily integrates a chord that passes underground — where its
    //     max(0, length-R_EARTH) altitude clamp reads maximum air density the whole way — yielding
    //     an enormous, extremely red-shifted colour just past the terminator. The volumetric has
    //     already gone to zero there, so that band was the "dark red" tail with no 3D counterpart.
    //     cloudDayFrac's smoothstep only fades this out by cloudSunDot = -0.1, leaving the whole
    //     sunset band running on the ungated value.
    //
    // SUN_INTENSITY is deliberately NOT copied — the flat path's magnitude calibration lives in
    // sunGain * flatSunGainScale instead, and folding in a second large constant would just move
    // where those sit. Only the wavelength RATIO has to match, and it now does.
    vec3  sunColorFlat = vec3(0.0);
    vec2  tSunEarth    = raySphere(cECEF, sunDirECEF, R_EARTH);
    if (!(tSunEarth.x > 0.0 && tSunEarth.y > 0.0)) {
        vec2 tSunAtm = raySphere(cECEF, sunDirECEF, R_ATMOS);
        if (tSunAtm.y > 0.0) {
            vec2 odSun = optDepth(cECEF, sunDirECEF, tSunAtm.y);
            sunColorFlat = exp(-(BETA_R * odSun.x + BETA_M * 1.1 * odSun.y));
        }
    }
    vec3  cloudLit     = sunColorFlat * (max(0.0, cloudSunDot + 0.1) * sunGainCurve) * cloudDayFrac;

    // Moonlight — this flat path had no night-side light source at all (cloudDayFrac zeroes
    // cloudLit once the sun sets), unlike the volumetric shell's moonContrib (cloud_march.comp)
    // or terrain's own moonContribTerrain. Same geographic-dot gate as those, and reuses
    // cloud.moonGain so all three stay calibrated to the same brightness.
    vec3  moonDir3     = normalize(moonDirENU.xyz);
    vec3  moonDirECEF  = moonDir3.x * enuX + moonDir3.y * enuY + moonDir3.z * enuZ;
    float cloudMoonDot = dot(normalize(cECEF), moonDirECEF);
    float moonLit      = max(0.0, cloudMoonDot) * moonDirENU.w;
    vec3  moonContrib  = vec3(0.92, 0.95, 1.0) * moonLit * cloud.moonGain;

    // ── Twilight sky ambient (flat 2D path) ──────────────────────────────────────────────────
    // Until this was added the flat layer had NO ambient term of any kind — its only light
    // sources were direct sun (cloudLit, which cloudDayFrac drives to zero at the terminator) and
    // moonlight. The volumetric shell has had a twilight ambient since session 28: sky-lit cloud
    // at dusk/dawn, so clouds read consistently against the sky-lit TERRAIN beside them instead
    // of dropping dark while the ground is still visibly blue. Across the 3D->2D crossfade that
    // asymmetry read as flat clouds going black through twilight while the volumetric ones next
    // to them stayed blue — the airlight term below is NOT a substitute, since that is light
    // scattered in FRONT of the cloud (so it grows with distance and vanishes on near clouds),
    // not downwelling sky light falling ON it.
    //
    // Deliberately a near-verbatim copy of cloudMarchCS's twilightAmbient + skyAmbientBase blocks
    // (same bell, same fixed widths, same UBO edges, same 6-step zenith integral, same
    // cloud-anchored p0) — matching the crossfade is the entire purpose of this term, so anything
    // reinvented here would just have to be re-matched by hand afterwards. Keep the two in sync;
    // same standing rule as the rest of the duplicated cloud code in this file.
    //
    // Three deliberate differences from the volumetric copy:
    //   * The volumetric's `mix(0.3, 0.9, hNorm)` becomes the constant kFlatAmbientHeightMix.
    //     That factor is its vertical shading ramp across a cloud COLUMN, and a flat shell has no
    //     column — 0.6 is that ramp's own midpoint, i.e. what a full column averages to.
    //   * The anchor is this layer's own shell hit point rather than a march entry point, and
    //     needs no altitude clamp: a cloud shell sits at 2-11 km by construction, decades inside
    //     R_ATMOS, so the guard the volumetric keeps has nothing to guard against here.
    //   * The whole block is gated on twilightWeight > 0. That is EXACT, not an approximation —
    //     the weight is a bell, zero in full daylight and zero deep into night — and it keeps six
    //     optDepth calls off every daylit cloud pixel, which at full resolution matters.
    //
    // BETA_R here is the flat-scaled one (flatRayleighGain), so this term's colour tracks the 2D
    // Rayleigh slider the same way sunColorFlat and attn do. That is intended: all three are "how
    // this path sees the atmosphere", and splitting them would make the 2D slider shift hue.
    vec3 twilightAmbient = vec3(0.0);
    {
        // Widths are fixed; only the edges move. Both constants match cloudMarchCS exactly.
        const float kTwilightRiseWidth = 0.2;
        const float kTwilightFallWidth = 0.3;
        const float kFlatAmbientHeightMix = 0.6;
        float twiHi = cloud.twilightBandHi;
        float twiLo = cloud.twilightBandLo;
        float twilightRise = 1.0 - smoothstep(twiHi - kTwilightRiseWidth, twiHi, cloudSunDot);
        float twilightFall = smoothstep(twiLo, twiLo + kTwilightFallWidth, cloudSunDot);
        float twilightWeight = twilightRise * twilightFall;
        float twiGain = cloud.cloudTwilightAmbientGain * cloud.flatTwilightAmbientGain;

        if (twilightWeight > 0.0 && twiGain > 0.0) {
            // Downwelling sky light AT the cloud: a short zenith-ray single-scattering integral
            // anchored on the cloud point itself, not on the observer. Anchoring at the observer
            // is only equivalent when the two are co-located (true from the ground, badly false
            // from orbit, where you can be over the night side looking at a cloud that is still
            // in full twilight) — see the long note on this in cloud_march.comp.
            vec3 skyAmbientBase = vec3(0.0);
            vec3 p0     = cECEF;
            vec3 zenith = normalize(p0);
            vec2 tSA    = raySphere(p0, zenith, R_ATMOS);
            if (tSA.y > 0.0) {
                const int N_Z = 6;
                float zSeg   = tSA.y / float(N_Z);
                float cosAUp = dot(zenith, sunDirECEF);
                float pR_up  = 0.75 * (1.0 + cosAUp * cosAUp);
                float odR_z = 0.0, odM_z = 0.0;
                float skyAmbientBaseM = 0.0;
                for (int zi = 0; zi < N_Z; ++zi) {
                    vec3  sp = p0 + zenith * ((float(zi) + 0.5) * zSeg);
                    float h  = max(0.0, length(sp) - R_EARTH);
                    float dR = exp(-h / H_R) * zSeg;
                    float dM = exp(-h / H_M) * zSeg;
                    odR_z += dR;  odM_z += dM;
                    vec2 tSE = raySphere(sp, sunDirECEF, R_EARTH);
                    if (tSE.x > 0.0 && tSE.y > 0.0) continue;   // this step is in Earth's shadow
                    vec2 tSun  = raySphere(sp, sunDirECEF, R_ATMOS);
                    vec2 sunOD = (tSun.y > 0.0) ? optDepth(sp, sunDirECEF, tSun.y) : vec2(0.0);
                    vec3 tau      = BETA_R * (odR_z + sunOD.x) + BETA_M * 1.1 * (odM_z + sunOD.y);
                    vec3 attnStep = exp(-tau);
                    skyAmbientBase  += attnStep * dR;
                    skyAmbientBaseM += dot(attnStep, vec3(1.0 / 3.0)) * dM;
                }
                float pM_up = phaseM(cosAUp);
                skyAmbientBase = SUN_INTENSITY * (pR_up * BETA_R * skyAmbientBase
                                                + vec3(pM_up * BETA_M * skyAmbientBaseM));
            }
            twilightAmbient = skyAmbientBase * kFlatAmbientHeightMix * twilightWeight * twiGain;
        }
    }

    // Compression applied to LUMINANCE, then scaled back onto the original colour, rather than
    // per channel. This is what caused the reported yellow -> orange -> red -> dark red march
    // across a single sunset while the volumetric only went orange -> dark orange.
    //
    // Per-channel 1-exp(-x) is a saturating curve, so it compresses BRIGHT channels harder than
    // dim ones. On a sunset colour like (1.0, 0.5, 0.2) it lifts green and blue relative to red —
    // it DESATURATES. How much depends on absolute magnitude: near x=0 the curve is the identity
    // (no desaturation at all, full sunset red), and at large x every channel pins near 1 (heavy
    // desaturation, toward yellow/white). So as the sun sets and cloudLit's magnitude falls, the
    // flat layer slides continuously from the compressed end of that curve to the linear end,
    // traversing the whole hue range on the way down. The volumetric never does this because it
    // has no per-channel compressor at all — it accumulates linearly through transmittance.
    // flatSunGainScale (~4x) put this path even deeper into the compressed regime to begin with,
    // which is why its bright end read as yellow rather than orange.
    //
    // Compressing the luminance and rescaling preserves chromaticity exactly, so the hue is now
    // whatever sunColorFlat says it is at every brightness — the same thing the volumetric does.
    // The magnitude roll-off (the reason this compressor exists: an unbounded product drove the
    // flat layer to pure white at volumetric-tuned sun gains) is unchanged.
    //
    // Note this can leave an individual channel above 1.0 for a strongly tinted colour, where the
    // per-channel form could not. That is correct for an HDR value feeding the tone map below, but
    // it is a real change in what this function can return.
    vec3  litSum       = cloudLit + moonContrib + twilightAmbient;
    float litLum       = dot(litSum, vec3(0.2126, 0.7152, 0.0722));
    vec3  cloudColor   = litSum * ((1.0 - exp(-litLum)) / max(litLum, 1e-4));

    // Aerial perspective. `color` at this point already holds the atmosphere's own inscattered
    // light along this ray (the N_VIEW loop that ran before this function), the same value the
    // terrain/ocean surfAttn path adds its own lit color on top of. Without this, a straight
    // mix() to cloudColor*attn at high alpha throws that inscattered light away and replaces it
    // with a plain Beer-Lambert-dimmed cloud color, which trends to black/dark-red at grazing
    // angles (attn shrinks, and BETA_R/BETA_M dim blue harder than red) instead of fading into
    // the horizon haze the way the volumetric clouds do via their additive cloudA/cloudB
    // composite. Blending toward `color` by (1-attn) — the fraction of light scattered INTO the
    // path between the cloud and the camera — fixes that.
    //
    // BUT `color` here is the FULL accumulated ray color, not just sky inscatter — over a terrain
    // hit it already has the ground's own lit surface (including night-lights, `surfColor *
    // surfAttn`, added earlier in main()) folded in. Blending toward it by a flat `(1-attn)`
    // therefore leaked ground light through this layer via a path that has nothing to do with
    // `alpha`/`alphaMax` at all — reported as "opacity scale has zero effect once alpha already
    // saturates," and worst from orbital altitude specifically because `attn` (camera-to-cloud
    // atmosphere) is naturally lower over that much longer path, so the leaked fraction is larger
    // exactly where this layer is doing the most work (the 3D->2D crossfade's flat regime).
    // Gating the leak by `(1-alpha)` fixed THAT leak, but it over-corrected: `(1-alpha)` also
    // zeroes out the airlight (atmosphere-scattered sunlight added INTO the camera-to-cloud
    // segment) for exactly the fully-opaque case, which is the one case that most needs it — a
    // solid horizon-hugging deck at real distance (this is the common case at the true horizon:
    // the shell's own curvature-limited entry point is already 100+ km out at ground level) has
    // `attn` crushed toward zero by BETA_R/BETA_M over that path, so with no airlight term
    // `cloudColor * attn` alone collapses toward black and, because BETA_R/BETA_M extinguish
    // blue/green harder than red, does so through an increasingly saturated yellow/red before
    // going dark — reported as horizon clouds rendering "too yellow, red, and dark" instead of
    // fading into the same orange sunset glow the sky around them already correctly shows.
    //
    // Fix: split `color` into its sky-only part (`skyOnlyColor`, captured in main() right after
    // the N_VIEW atmosphere loop and before any ground/surface light is added) and whatever
    // ground contributed on top (`groundLeak = color - skyOnlyColor`). Airlight drawn from the
    // sky-only part is legitimate regardless of alpha — it represents light scattered in FRONT of
    // the cloud, not whatever is behind it — so it is weighted only by `(1-attn)` and, to match
    // the volumetric shell's own airlight (cloud_march.comp's `airlight`, weighted by how opaque
    // the cloud itself is), by `alpha`: a fully opaque cloud gets the full sky-glow substitute for
    // its own extinguished light, a clear pixel gets none. `groundLeak` keeps exactly the previous
    // `(1-alpha)`-gated treatment, so the original leak fix is untouched.
    vec3 attn       = exp(-(BETA_R * odRcam + BETA_M * 1.1 * odMcam));
    vec3 groundLeak = color - skyOnlyColor;
    vec3 airlight   = skyOnlyColor * (1.0 - attn) * alpha;
    vec3 cloudSeen  = cloudColor * attn + airlight + groundLeak * (1.0 - attn) * (1.0 - alpha);
    color = mix(color, cloudSeen, alpha);
}

// (cloudDensity lived here — dead since the cloud march moved to cloud_march.comp, removed in
//  the pipeline-unification pass. Worth knowing if you go looking for it: this copy blended only
//  TWO fixed Z-slices (0.0 / 0.5) and used a fixed 0.2*erosion floor, while cloud_march.comp's
//  live version blends THREE (0.0 / 0.28 / 0.56) piecewise and takes its erosion floor from the
//  UBO. Another genuine drift that stayed invisible because this copy was already unreachable.)

// ── Cloud raymarch diagnostics ────────────────────────────────────────────────
// Set CLOUD_DEBUG to 1-5 to replace cloud output with a diagnostic overlay.
// Set to 0 for normal rendering.
//
//  1 = 2D coverage at column entry — white=overcast, black=clear.
//      Question: Is the coverage map causing a solid overcast everywhere?
//
//  2 = hNorm of FIRST cloud hit — red=hit near base, green=hit near top, dark-blue=no hit.
//      Question: Are all clouds at the same altitude (should show varying colour if 3D)?
//
//  3 = Fraction of march steps where d>0 — white=solid cloud in this column, black=clear.
//      Question: Is the march mostly in cloud (overcast) or mostly clear (scattered)?
//
//  4 = noiseUVW at march midpoint — R=X, G=Y, B=Z noise coords.
//      Question: Is the noise actually varying in 3D, or is one axis stuck constant?
//
//  5 = posZ value at first cloud hit — greyscale [0,1].
//      Question: Is posZ (the Z anti-banding offset) actually varying across the image?
//      UNIFORM GREY = posZ is constant for all visible pixels = geographic UV barely changes
//      within the visible cloud footprint from ground level. This confirms the Z-layer
//      banding is caused by posZ not spanning enough geographic range from the surface.
#define CLOUD_DEBUG 0

// NOTE: the description below is HISTORY. cloud_shadow.comp no longer exists — the shadow is now
// marched per pixel inside cloud_march.comp from this pixel's own terrain hit point (see
// cloudGroundShadow there) and arrives in cloudTargetB.a. Kept because the reasoning about why
// full-res per-pixel shadowing was too expensive in session 23 still explains why the current
// version runs at half resolution rather than here.
//
// cloudShadowFactor() removed (C15-perf follow-up, session 23) — it was the dominant
// remaining surface-level cloud cost (full-res, ~every terrain/ocean pixel, up to 64
// steps each). Its CloudParams UBO slot reverted to `pad0`, and the CLOUD_ISOLATE_COLH/SHADOW
// debug switches that existed only to isolate this function's seam bugs went with it.
// Cloud shadowing on terrain/ocean is BACK as of C12 (session 32+) via a different mechanism:
// cloud_shadow.comp precomputes the same shadow transmittance once per low-res grid texel
// (128x128) instead of once per screen pixel, sampled here as an O(1) texture read — see
// directSun below.

// cloudMarch()/cirrusMarch() moved to shaders/cloud_march.comp (half-res compute pass —
// C15-perf). main() below now samples cloudTargetA/cloudTargetB (bindings 10/11) instead
// of calling these directly. See TERRAIN_PLAN.md session 23 log for the design.

void main() {
    // Runtime-tunable scattering strength — shadows the physical base constants (common.glsl)
    // with the user-facing "Rayleigh gain"/"Mie/haze gain" sliders, visible to every use of
    // BETA_R/BETA_M below (atmosphere loop, terrain ambient, ocean reflection, moon/sun
    // attenuation). See cloud_params.glsl's atmosRayleighGain/atmosMieGain comment.
    vec3  BETA_R = BETA_R_BASE * cloud.atmosRayleighGain;
    float BETA_M = BETA_M_BASE * cloud.atmosMieGain;

    vec3 dir    = normalize(enuDir);
    vec3 sunDir = normalize(sunDirENU.xyz);

    // ENU→ECEF rotation built from observer ECEF direction (needed early for terrain UV).
    vec3 enuZ = normalize(pc.obsECEFDir.xyz); // observer Up in ECEF
    vec3 enuX = normalize(cross(vec3(0.0, 0.0, 1.0), enuZ)); // East
    vec3 enuY = cross(enuZ, enuX);            // North

#if CLOUD_DEBUG == 6
    // Minimal, direct test: sample cloudNoiseTex straight from the view direction, bypassing
    // the entire raymarch/threshold/lighting pipeline entirely. If the seam still appears here,
    // it's unambiguously baked into the volume itself (or in this one-line coordinate build);
    // if it's clean, something between here and the raymarch is still the culprit.
    {
        vec3 dirECEFView = normalize(dir.x * enuX + dir.y * enuY + dir.z * enuZ);
        outColor = vec4(texture(cloudNoiseTex, dirECEFView * kCloudHorizFreq).rgb, 1.0);
        return;
    }
#endif

    // Sun direction in ECEF — used by evalCloudLayer for per-cloud-point illumination.
    // Transforms ENU sunDir into ECEF so cloud day/night is geographically correct,
    // not relative to the observer's view of the sun.
    vec3 sunDirECEF = sunDir.x * enuX + sunDir.y * enuY + sunDir.z * enuZ;

    // Elevation encoding constants (kElevRange / kMaxTerrain / kElevOffset) and the GPU-side
    // observer ground-height lookup both live in terrain.glsl now — see that file's header for
    // the DEM encoding, which is the single most re-broken piece of knowledge in this project.
    float obsEffH = observerEffHeight(earthElevTex, earthSpecTex, pc.obsECEFDir);

    // Observer position: +2 m eye height above ground.
    vec3 obsPos = observerPos(obsEffH);

    // For elevated observers the visible region extends below the geometric horizon.
    // limbZ = sin(Earth-limb depression angle) — negative, approaches 0 at sea level.
    float obsR  = length(obsPos);
    float limbZ = (obsR > R_EARTH) ? -sqrt(max(0.0, 1.0 - (R_EARTH / obsR) * (R_EARTH / obsR))) : 0.0;
    float hClip = smoothstep(limbZ - 0.02, limbZ + 0.03, dir.z);

    // ── Phase 1: terrain march (runs before atmosphere so we can truncate tEnd) ─

    vec2 tBase  = raySphere(obsPos, dir, R_EARTH);
    vec2 tShell = raySphere(obsPos, dir, R_EARTH + kMaxTerrain);

    // March for rays that could plausibly intersect terrain (up to ~44° above horizon).
    // Beyond that angle no terrain on Earth is geometrically reachable from any altitude.
    float tHit      = -1.0;
    float tSeaLvl   = (tBase.x > 0.0) ? tBase.x : -1.0;
    vec2  hitUV     = vec2(0.0);
    vec3  terrainNorm = vec3(0.0, 0.0, 1.0); // overwritten on terrain hit

    if (!dbgSkipTerrain() && dir.z < 0.7 && tShell.y > 0.0) {
        float tExit = (tBase.x > 0.0) ? tBase.x
                    : (tShell.y > 0.0  ? tShell.y : 0.0);
        // Cap scales with observer altitude so terrain is visible from LEO.
        // At ground the old 250 km limit is preserved (horizon is close anyway).
        // At LEO (400 km) it extends to 900 km, covering ~65° off-nadir views.
        // Limb-grazing rays that would otherwise generate very long tShell paths
        // are still clamped so the march stays bounded.
        float tCap = mix(250000.0, 3600000.0, clamp(obsEffH / 400000.0, 0.0, 1.0));
        tExit = min(tExit, tCap);

        // Quadratic step distribution: steps grow proportionally to their index so
        // near terrain gets fine resolution while far terrain gets coarser steps.
        // Perf (this session): step count now scales with this RAY's actual march distance
        // (tExit), not observer altitude — same principle as the aurora march's path-length
        // scaling (N_AURORA/kAuroraMaxStepM). The old altitude-only mix() gave a steep near-
        // vertical ray and a long grazing-horizon ray from the SAME observer altitude identical
        // step budgets, even though the grazing ray covers far more ground and needs more steps
        // to avoid undersampling, while the steep ray was overpaying. With quadratic spacing the
        // COARSEST step lands at the far end (frac≈1): dt ≈ 2*(tExit-2)/kN, so solving for kN
        // at a target coarsest-step size reproduces the same ~2.8km-at-far-end calibration the
        // original altitude-based tuning aimed for (see the historical "up to 320 at LEO...
        // ~2.8km" comment this replaced). Min/max (64/164) are the user-validated range from the
        // preliminary altitude-only test — jittery-but-passable at 64, fine at 164 — kept as-is,
        // only the scaling variable changed from obsEffH to tExit.
        const float kTerrainStepTargetM = 2800.0;
        const int   kTerrainStepsMin = 64;
        const int   kTerrainStepsMax = 164;
        int kNFull = clamp(int(2.0 * (tExit - 2.0) / kTerrainStepTargetM), kTerrainStepsMin, kTerrainStepsMax);

        // S4 (RELEASE_v1_1_PLAN.md, session 31): tCap above grows the march REACH with altitude
        // faster than kNFull's budget grows, so kNFull pins at its 164 ceiling on essentially
        // every screen pixel from LEO (measured via debugDisableMask bit 1 recovering a decent
        // frame rate, more so from altitude) — because terrain relief is genuinely sub-pixel at
        // the reach this ray is capable of. Fade the step budget down as tExit (this ray's own
        // reach, not just observer altitude — a grazing ray pays more than a steep one at the
        // SAME altitude) grows past terrainDistFadeStartM, and skip the march outright past
        // terrainDistFadeEndM: tHit stays -1 and the code below falls back to tSeaLvl, already
        // computed above at zero extra cost — the same "smooth textured Earth" result the
        // terrain debug knockout already produces, not a pop to nothing.
        float terrainReachFade = 1.0 - smoothstep(cloud.terrainDistFadeStartM, cloud.terrainDistFadeEndM, tExit);
        int kN = int(float(kNFull) * terrainReachFade);

        float jitter  = textureLod(noiseTex, gl_FragCoord.xy * (1.0/128.0), 0.0).r;
        float tPrev   = 2.0;

        for (int i = 0; i < kN; ++i) {
            if (tHit >= 0.0) break;
            float frac = (float(i) + jitter) / float(kN);
            float t    = 2.0 + (tExit - 2.0) * frac * frac;
            if (t > tExit) break;
            vec3  p = obsPos + t * dir;
            float rayH = length(p) - R_EARTH;
            if (rayH <= 0.0) break;

            vec3  pE  = p.x * enuX + p.y * enuY + p.z * enuZ;
            float terrainH = terrainHeightAtUV(earthElevTex, earthSpecTex, posToUV(pE), 0.0);

            if (rayH < terrainH) {
                float tLo = tPrev, tHi = t;
                for (int j = 0; j < 12; ++j) {
                    float tM  = (tLo + tHi) * 0.5;
                    vec3  pm  = obsPos + tM * dir;
                    float mH  = length(pm) - R_EARTH;
                    vec3  pmE = pm.x * enuX + pm.y * enuY + pm.z * enuZ;
                    float mT  = terrainHeightAtUV(earthElevTex, earthSpecTex, posToUV(pmE), 0.0);
                    if (mH < mT) tHi = tM; else tLo = tM;
                }
                tHit = (tLo + tHi) * 0.5;
                vec3  ph  = obsPos + tHit * dir;
                vec3  phE = ph.x * enuX + ph.y * enuY + ph.z * enuZ;
                float phL = length(phE);
                hitUV = vec2((atan(phE.y, phE.x) + PI) / (2.0*PI),
                             (0.5*PI - asin(clamp(phE.z / phL, -1.0, 1.0))) / PI);

                // Terrain normal from elevation gradient (central differences).
                // Builds hit-point local East/North/Up in ECEF, then maps to observer ENU.
                {
                    const float kTexU = 1.0 / 21600.0;
                    const float kTexV = 1.0 / 10800.0;
                    float hE2 = max(0.0, textureLod(earthElevTex, hitUV + vec2(kTexU, 0.0), 0.0).r * kElevRange - kElevOffset);
                    float hW2 = max(0.0, textureLod(earthElevTex, hitUV - vec2(kTexU, 0.0), 0.0).r * kElevRange - kElevOffset);
                    float hN2 = max(0.0, textureLod(earthElevTex, hitUV - vec2(0.0, kTexV), 0.0).r * kElevRange - kElevOffset);
                    float hS2 = max(0.0, textureLod(earthElevTex, hitUV + vec2(0.0, kTexV), 0.0).r * kElevRange - kElevOffset);
                    float hitLat2 = PI * 0.5 - hitUV.y * PI;
                    float texLon  = max(100.0, 2.0 * PI * R_EARTH * abs(cos(hitLat2)) / 21600.0);
                    float texLat  = PI * R_EARTH / 10800.0; // ~1853 m/texel
                    float dE2     = (hE2 - hW2) / (2.0 * texLon);
                    float dN2     = (hN2 - hS2) / (2.0 * texLat);
                    vec3 hUpE     = phE / phL;
                    vec3 hEsE     = normalize(vec3(-hUpE.y, hUpE.x, 0.0)); // East in ECEF
                    vec3 hNrE     = cross(hUpE, hEsE);                      // North in ECEF
                    vec3 nECEF    = normalize(-dE2 * hEsE + -dN2 * hNrE + hUpE);
                    terrainNorm   = normalize(vec3(dot(nECEF, enuX), dot(nECEF, enuY), dot(nECEF, enuZ)));
                }
            }
            tPrev = t;
        }
    }

    // Effective surface distance: terrain if found, else sea level
    float tSurface = (tHit > 0.0) ? tHit : tSeaLvl;

    // ── Half-resolution cloud composite sample (hoisted early) ─────────────────
    // Sampled here — ahead of the moon disc below — so the moon can be occluded by opaque
    // cloud the same way it's occluded by terrain. The actual multiplicative/additive
    // composite (`color = color * cloudB.rgb + cloudA.rgb`) still applies later, after the
    // 2D flat cloud-layer overlay and satellite glow so those get attenuated too; this early
    // sample only reads the alpha channels needed for occlusion tests.
    vec2  cloudUV        = gl_FragCoord.xy / pc.screenSizePx;
    vec4  cloudACenter   = texture(cloudTargetA, cloudUV);
    vec4  cloudBCenter   = texture(cloudTargetB, cloudUV);
    // 3x3 box-blurred over cloudTargetA/B's own half-res texels — same idiom as the
    // cloudGroundShadow 5x5 blur further down. A cloud edge is a single evaluation per half-res
    // texel with no spatial supersampling, so its silhouette is genuinely stair-stepped at that
    // resolution; ordinarily hidden by the natural softness of a sky-color-to-cloud-color
    // transition, but a beam's glow riding inside cloudA.rgb (B_total, see cloud_march.comp's
    // beam pointing-ray loop) turns that same stair-step into a hard, jagged edge wherever a
    // cloud silhouette passes in front of a beam (reported in-app — the edge tracks the cloud's
    // real shape and is fixed in world space, i.e. genuine spatial aliasing of the half-res
    // alpha field, not a screen-space or depth-gate artifact). Only .rgb is blurred — .a
    // (tCloudOcclude / the shadow channel, which gets its own separate 5x5 blur below) keeps its
    // single-tap value, since occlusion tests want the exact per-pixel distance, not a blend of
    // neighbors.
    vec3  cloudARgb = vec3(0.0);
    vec3  cloudBRgb = vec3(0.0);
    {
        vec2 cloudTexel = 1.0 / vec2(textureSize(cloudTargetA, 0));
        for (int sy = -1; sy <= 1; ++sy)
            for (int sx = -1; sx <= 1; ++sx) {
                vec2 uv = cloudUV + vec2(sx, sy) * cloudTexel;
                cloudARgb += texture(cloudTargetA, uv).rgb;
                cloudBRgb += texture(cloudTargetB, uv).rgb;
            }
        cloudARgb *= (1.0 / 9.0);
        cloudBRgb *= (1.0 / 9.0);
    }
    vec4  cloudA         = vec4(cloudARgb, cloudACenter.a);
    vec4  cloudB         = vec4(cloudBRgb, cloudBCenter.a);
    float tCloudOcclude  = cloudA.a;
    // cloudB.a used to carry tEnterCombined, the fused entry distance this shader compared
    // against tSurface to suppress the whole composite. Every volumetric layer is now clamped to
    // the shared scene depth inside cloud_march.comp instead, so there is nothing to test here.
    // The channel is free; the next step gives it the per-pixel cloud shadow.
    //
    // Local cloud opacity at THIS pixel's view ray (0 = clear sky/no cloud, 1 = fully opaque),
    // same formula `cloudBlock` derives from cloudB.rgb further below — computed here too so the
    // night-lights blur-through-cloud blend (city detail block below) can use it before that
    // later point. A soft edge, not a hard cutoff on purpose: even a thin/wispy cloud should
    // diffuse city light a little, not switch abruptly from sharp to blurred.
    float localCloudOpacity = 1.0 - clamp(dot(cloudB.rgb, vec3(1.0 / 3.0)), 0.0, 1.0);

    // ── Phase 2: atmosphere integration, truncated at the surface ─────────────
    vec2  tAtmos = raySphere(obsPos, dir, R_ATMOS);
    // Clamped to 0: when the observer is above R_ATMOS (reachable via the uncapped "Raise
    // Elevation" control) and looking outward/away from Earth, raySphere's forward root
    // (tAtmos.y) goes negative — the 100km shell is now entirely behind the camera. Without
    // this clamp, segLen would go negative and the whole loop below would march backward from
    // the observer instead of contributing nothing, corrupting the sky colour along that ray.
    float tEnd   = max(0.0, (tSurface > 0.0) ? min(tAtmos.y, tSurface) : tAtmos.y);

    // Adaptive N_VIEW (perf follow-up, session 24 round 2): a FIXED sample count over segLen =
    // tEnd/N_VIEW badly serves this loop, because tEnd itself varies enormously with viewing
    // geometry, not just altitude — a straight-up ray from the ground has tEnd~100km, but a
    // near-horizon ray (the same geometry as looking toward a low sun near the terminator) can
    // graze a chord of 2000+ km through the thin atmosphere shell, and most rays from orbit hit
    // the shell at a similarly grazing angle. A fixed low N_VIEW gives a razor-thin, accurate
    // step for the short case but a step many times the ~8km Rayleigh scale height (H_R) for the
    // long case — under-resolving the exponential falloff exactly where it's most sensitive,
    // which is what read as rainbow banding near the terminator and a vanishing atmosphere past
    // MEO. This is undersampling, not a floating-point precision problem (precision loss
    // wouldn't recover just by raising the sample count the way this does).
    //
    // Fix: derive a target step length from the user's validated "looks convincing" ground-level
    // case (cloud.viewSamplesMin steps over the reference ~100km straight-up path — kAtmosRefTEnd
    // below is exactly R_ATMOS-R_EARTH), then scale N_VIEW to hold that SAME step length for
    // whatever tEnd this particular ray actually has, clamped to cloud.viewSamplesMax so a
    // pathologically long grazing ray can't balloon the cost unboundedly.
    const float kAtmosRefTEnd = 100000.0; // R_ATMOS - R_EARTH: ground-level straight-up reference path
    float viewSamplesMin = max(2.0, cloud.viewSamplesMin);
    float viewSamplesMax = max(viewSamplesMin, cloud.viewSamplesMax);
    float targetStepLen  = kAtmosRefTEnd / viewSamplesMin;
    int   N_VIEW = dbgSkipAtmosphere() ? 0
                 : int(clamp(ceil(tEnd / targetStepLen), viewSamplesMin, viewSamplesMax));
    float segLen = tEnd / float(N_VIEW);
    float cosA   = dot(dir, sunDir);
    float pR     = phaseR(cosA);
    float pM     = phaseM(cosA);

    vec3  accumR  = vec3(0.0);
    float accumM  = 0.0;
    float accumCity = 0.0;
    vec3  accumAirglow = vec3(0.0); // green + sodium bands (C15) — ride these same samples
    float odR_cam = 0.0;
    float odM_cam = 0.0;

    // ── Orbital terminator gate (artistic) ───────────────────────────────────
    // Strength of the per-sample terminator suppression applied to accumR/accumM below, faded in
    // by observer altitude so ground level is bit-identical to having no gate at all. Reuses the
    // same 40-100 km fade constants as the Milky Way block further down; below 40 km this is 0 and
    // the whole feature compiles out to a multiply by 1.0.
    //
    // Why altitude-gated rather than global: the gate suppresses samples sitting over night-side
    // ground, which from orbit is exactly the unwanted twilight wash, but from the GROUND is the
    // post-sunset sky itself. Measured at sun 2 degrees below the horizon, applying it globally
    // took the western afterglow down 4-8x and removed the Belt of Venus entirely. It is an
    // orbital art knob, not a scattering correction, and it is scoped to say so.
    float atmTermSpace = clamp((obsEffH - 40000.0) / 60000.0, 0.0, 1.0)
                       * clamp(cloud.atmosTermStrength, 0.0, 1.0);
    float atmTermW     = max(1e-4, cloud.atmosTermWidth);

    // ── Single-scattering atmosphere integration (Rayleigh + Mie) ────────────
    // N_VIEW uniform steps from the observer toward the atmosphere exit (or truncated at
    // the surface).  At each step the scattered sunlight is accumulated using:
    //   - Two running totals (odR_cam, odM_cam): optical depth from the CAMERA to this step.
    //     These are also read back after the loop to attenuate the surface/moon/cloud colours.
    //   - Per-step sun optical depth (sunOD): optical depth from THIS STEP to the SUN,
    //     computed by calling optDepth along the sun direction.
    //   - Phase functions pR/pM: angular weighting of how much scatter points toward the camera.
    for (int i = 0; i < N_VIEW; ++i) {
        vec3  sp  = obsPos + dir * ((float(i) + 0.5) * segLen);  // midpoint of this atmosphere step
        float len = length(sp);
        if (len < R_EARTH) sp *= R_EARTH / len;  // clamp underground samples to Earth surface
        float h = max(0.0, length(sp) - R_EARTH);  // altitude above sea level (metres)

        // Running camera-side optical depth: accumulated from step 0 to this step.
        // densR/densM = density × step length = optical depth contribution of this step alone.
        float densR = exp(-h / H_R) * segLen;   // Rayleigh: peaks at sea level, scale height H_R
        float densM = exp(-h / H_M) * segLen;   // Mie: concentrated near surface, scale height H_M
        odR_cam += densR;
        odM_cam += densM;

        // sin(sun elevation) at THIS SAMPLE's own geographic point — the same quantity the cloud
        // march gates on (cloudSunDotRaw), which is the whole point: it is what lets the
        // atmosphere cut off on the same variable the clouds already do. Set inside the block
        // below (which computes spDirECEF for city glow / airglow anyway) and consumed at the
        // accumulation, so it costs one dot product that was already being taken.
        float sampleSunDotGeo = 1.0;

        // City light-pollution upwelling (Step 7 / C10, TERRAIN_PLAN.md). An INDEPENDENT light
        // source, not derived from sunlight, so it's computed here — BEFORE the sun-shadow test
        // below, which specifically triggers when this sample is in Earth's shadow (i.e. at
        // night, exactly when city glow matters). Uses camera-side attenuation only (no
        // sun-side optical depth term — irrelevant for a non-solar light source). densR weights
        // near-surface atmosphere heavily, so an observer directly over a city gets strong
        // zenith glow while a distant observer only picks up dim glow from low horizon samples
        // — both fall out naturally from the same accumulation used for Rayleigh/Mie above.
        {
            vec3  spECEF    = sp.x * enuX + sp.y * enuY + sp.z * enuZ;
            float spLen     = length(spECEF);
            vec3  spDirECEF = spECEF / spLen;
            float spLat     = asin(clamp(spDirECEF.z, -1.0, 1.0));
            float spLon     = atan(spDirECEF.y, spDirECEF.x);
            vec2  spUV      = vec2((spLon + PI) / (2.0*PI), (0.5*PI - spLat) / PI);
            float spLum     = dot(textureLod(earthNightTex, spUV, 4.0).rgb, vec3(0.2126, 0.7152, 0.0722));
            vec3  attnCam   = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));
            accumCity += cityBrightness(spLum) * densR * dot(attnCam, vec3(1.0 / 3.0));

            // Airglow (C15): green (96km) + sodium (90km) bands both fall inside this loop's
            // own altitude range (h spans 0..~100km along an open-sky ray), so they ride these
            // existing samples for free — no dedicated march needed (unlike red, see below the
            // loop). Gated by the SAMPLE's own geographic day/night (not the observer's), same
            // dot-product test cloud lighting uses (evalCloudLayer/cloudMarch sampleDayness) —
            // physically correct since the glow originates at that geographic point, not at the
            // observer. Horizontal patchiness from a slow analytic domain warp avoids a flat,
            // featureless ring (a pure function of altitude alone has none).
            sampleSunDotGeo  = dot(spDirECEF, sunDirECEF);
            float airDayness = clamp((sampleSunDotGeo + 0.15) / 0.3, 0.0, 1.0);
            float airNight   = 1.0 - airDayness;
            if (airNight > 0.001) {
                float airPatch = 0.6 + 0.4 * warpPerlin3(spDirECEF * kAirglowNoiseFreq
                                    + vec3(pc.waveTime * kAirglowDriftRate, 17.0, -5.0));
                // Coverage patchiness — independent noise samples per band so green and sodium
                // don't brighten/dim in perfect lockstep (real thermospheric density variation
                // doesn't affect both emission layers identically).
                float coverageG = airglowCoverageMask(spDirECEF, pc.waveTime, vec3(3.0, 29.0, -11.0));
                float coverageS = airglowCoverageMask(spDirECEF, pc.waveTime, vec3(-37.0, 6.0, 44.0));
                float covGainC  = clamp(cloud.airglowCoverageGain, 0.0, 1.0);
                float dzG = (h - kAirglowGreenPeakM) / kAirglowGreenHalfWidthM;
                float dzS = (h - kAirglowSodiumPeakM) / kAirglowSodiumHalfWidthM;
                float densAirG = exp(-dzG * dzG) * segLen;
                float densAirS = exp(-dzS * dzS) * segLen;
                accumAirglow += (kAirglowGreenColor  * cloud.airglowGreenGain  * densAirG * mix(1.0, coverageG, covGainC)
                                + kAirglowSodiumColor * cloud.airglowSodiumGain * densAirS * mix(1.0, coverageS, covGainC))
                                * airNight * airPatch;
            }
        }

        // Shadow test: skip samples in Earth's shadow.
        // If the sun-ray from this point has TWO positive intersections with R_EARTH, the sun
        // is behind Earth from here → no direct sunlight → no in-scatter contribution.
        vec2 tSunEarth = raySphere(sp, sunDir, R_EARTH);
        if (tSunEarth.x > 0.0 && tSunEarth.y > 0.0) continue;

        // Compute sun-side optical depth from this sample to the atmosphere boundary.
        vec2 tSun  = raySphere(sp, sunDir, R_ATMOS);
        vec2 sunOD = (tSun.y > 0.0) ? optDepth(sp, sunDir, tSun.y) : vec2(0.0);

        // Combined transmittance τ = BETA × (cam_depth + sun_depth):
        //   cam_depth: how much atmosphere light must traverse from here to the camera.
        //   sun_depth: how much atmosphere sunlight must traverse from the sun to here.
        // Mie multiplied by 1.1 to account for aerosol absorption (σ_ext > σ_scat).
        vec3 tau  = BETA_R       * (odR_cam + sunOD.x)
                  + BETA_M * 1.1 * (odM_cam + sunOD.y);
        vec3 attn = exp(-tau);  // total transmittance: sun → this sample → camera

        // Orbital terminator gate. Deliberately NOT physical: the scattering integral itself was
        // measured (against a clean-room reimplementation of this exact loop) to fall about one
        // decade per 6 degrees across the terminator, which is real twilight's rate. The problem
        // it solves is a tone-mapping mismatch, not a scattering error — this renderer composites
        // an eyeballed HDR scene rather than shooting at a fixed exposure the way the orbital
        // photography it is being compared against does, so the physically-correct twilight tail
        // survives tone mapping far more visibly than a camera would record it. That leaves the
        // clouds (which cut off hard on their own geographic sun angle) reading as oddly dark
        // patches against a still-bright atmosphere. Suppressing the atmosphere is the correct
        // direction to close that gap; raising the clouds' twilight ambient to meet the
        // atmosphere, which was tried first, drives them to full-bright neon seen from the ground.
        //
        // Weighting by the SAMPLE's own geographic sun elevation is what makes this free of
        // daylight cost: samples over daylit ground never enter the rolloff at all, so the day
        // side is untouched to six decimal places while SZA 92 drops ~23x at width 0.08.
        float atmTermW8 = mix(1.0, smoothstep(-atmTermW, atmTermW, sampleSunDotGeo), atmTermSpace);

        // Accumulate in-scattered radiance for each particle type.
        // Multiplying by density (densR/densM) weights by how many particles are at this altitude.
        // accumCity and accumAirglow above are deliberately NOT gated — they are independent
        // emissive sources, not scattered sunlight, and suppressing them past the terminator is
        // the exact opposite of what they exist to do.
        accumR += attn * densR * atmTermW8;                  // Rayleigh: wavelength-dependent (blue sky)
        accumM += dot(attn, vec3(1.0 / 3.0)) * densM * atmTermW8; // Mie: wavelength-neutral (white haze/corona)
    }

    vec3 color = SUN_INTENSITY * (pR * BETA_R * accumR + vec3(pM * BETA_M * accumM));

    // City light-pollution glow dome, composited once here (see accumCity comment in the loop
    // above). nightFactor fades it out through the day — cheap local gate rather than reusing
    // any later-computed day/night variable, since none exists yet at this point in main().
    float nightFactor = 1.0 - smoothstep(-0.05, 0.1, sunDirENU.w);
    color += accumCity * vec3(1.0, 0.72, 0.42) * nightFactor * kNightGlowScale;

    // C12 follow-up #41: replaced the directional (azimuth-sector-dome-based) wash from #39/#40
    // with a simple non-directional "sky is brighter near an active beam" term. The directional
    // version read as a narrow rising pillar from one bearing (right for a city's broad glow dome,
    // wrong for one concentrated, often low-angle beam) and measured distance to the beam's GROUND
    // TARGET only, incorrectly fading out as the observer climbed up alongside a beam away from the
    // ground while staying right next to its actual line. pc.beamProximityGlow (CPU-computed in
    // SatelliteSim.cpp from true point-to-segment distance to the nearest active beam's line,
    // one-frame-stale like every other reflectBeamsBuf readback) already carries the complete
    // [0,1] falloff — applied equally regardless of view direction, so standing near a beam
    // brightens the WHOLE visible sky, not one patch of it. beamGlowDomeBuf itself is untouched —
    // still used below (Milky Way section) and in sat_flare.comp/updateStars() for its original
    // purpose, suppressing other sky objects near an active beam.
    const float kBeamSkyGlowScale = 1.0; // C12 follow-up #40 — see that follow-up's note on why
                                          // this needed to be O(1), not O(1e-6): it multiplies an
                                          // already-[0,1]-normalized value, unlike kNightGlowScale.
    // Reuses the SAME nightFactor just computed for city glow above — an accepted first-pass
    // simplification (an active beam's target is always night-side by construction, but this
    // doesn't separately check the OBSERVER's own day/night).
    color += vec3(1.0, 0.95, 0.9) * pc.beamProximityGlow * pc.beamGlowBleedGain * kBeamSkyGlowScale
           * nightFactor;

    // ── Airglow (C15) ─────────────────────────────────────────────────────────
    // Green + sodium bands (accumAirglow) rode the N_VIEW loop above for free.
    color += accumAirglow * kAirglowScale * cloud.airglowGain;

    // Red band (630nm) supplemental march: peaks at 275km, well past N_VIEW's ~100km ceiling
    // (R_ATMOS), so it never rode those samples the way green/sodium do above. Perf (this
    // session): the dedicated march itself moved to cloud_march.comp (airglowRedMarchCS),
    // alongside aurora — half resolution instead of full-res. Rides along inside cloudA.rgb
    // (B_total) the same way aurora does now; no separate handling needed here. See that
    // function for the full march logic and the entry/exit classification history.

    // ── Aurora (C16, TERRAIN_PLAN.md Phase E) ──────────────────────────────────
    // Perf (this session): the sky curtain march itself moved to cloud_march.comp, alongside
    // clouds/cirrus, so it runs at half resolution instead of full-res (one of the two big
    // remaining levers from "why is aurora so much more expensive than clouds" — the other,
    // baking its noise, is done too — see aurora_noise.comp). Its result now arrives already
    // folded into cloudTargetA's B_total (additive radiance), composited below alongside
    // cirrus/cloud with no separate handling needed here. auroraFrame/auroraCoverage/
    // auroraOvalMask/auroraCurtainNoise/auroraSampleAt (above) and auroraGlowAt (below) STAY here
    // — still used by terrain/ocean ambient lighting and the ocean sky-reflection's aurora sample,
    // both full-resolution. dbgSkipAurora() now lives in CloudMarchPC (mirrors debugDisableMask)
    // since that's where the actual march runs; see cloud_march.comp's auroraMarchCS.

    // ── Moon disc ─────────────────────────────────────────────────────────────
    // kMoonTexRotDeg: rotates the texture CW in the UV plane to align the image's
    // north pole with the physical lunar north pole as seen from the observer.
    // Tune this until the terminator's shadow boundary matches the image poles.
    const float kMoonTexRotDeg = 180.0;
    const float kMoonAngR      = 0.004578 * 3.0;
    const float kMoonBright    = 0.54;
    // Set below on an actual ray-disc hit; used later to block the Milky Way skybox (and
    // nothing else — stars are culled per-vertex in star_point.vert) from showing through the
    // Moon's opaque disc on a clear-sky ray. Terrain/cloud occluding the Moon itself already
    // separately block the Milky Way via their own existing visibility terms, so this only
    // needs to be the pure geometric hit, not discFade.
    bool moonDiscHit = false;
    if (moonDirENU.z > limbZ - kMoonAngR * 2.0) {
        vec3  moonDir3 = normalize(moonDirENU.xyz);

        // ── Atmospheric refraction squish ─────────────────────────────────────
        // Near the horizon, differential refraction lifts the bottom limb more
        // than the top, compressing the apparent disc height.  The Bennett formula
        // gives refraction R(el) in arcminutes; the squish fraction is the
        // difference in R across the disc diameter, divided by the disc diameter.
        float squish = 0.0;
        float elDeg  = degrees(asin(clamp(moonDirENU.z, -1.0, 1.0)));
        if (elDeg < 15.0) {
            float r   = degrees(kMoonAngR);             // disc angular radius, degrees
            float elo = max(elDeg - r, 0.2);            // lower limb elevation (clamped off ground)
            float ehi = elDeg + r;                      // upper limb elevation
            float Rlo = 1.02 / tan(radians(elo + 10.3 / (elo + 5.11))); // arcmin
            float Rhi = 1.02 / tan(radians(ehi + 10.3 / (ehi + 5.11)));
            squish = 0; //clamp((Rlo - Rhi) / (2.0 * r * 60.0), 0.0, 0.5);
        }
        // Stretching dir.z before intersection maps screen pixels into a
        // vertically compressed disc-space — the silhouette becomes a physical
        // ellipse (shorter in elevation) matching the naked-eye refraction effect.
        vec3  dirR  = normalize(vec3(dir.xy, dir.z * (1.0 + squish)));

        vec3  oc    = -moonDir3;
        float bm    = dot(oc, dirR);
        float cm    = 1.0 - kMoonAngR * kMoonAngR;
        float discm = bm * bm - cm;
        float tm    = -bm - sqrt(max(discm, 0.0));
        if (discm >= 0.0 && tm > 0.0) {
            moonDiscHit = true;
            vec3  hp = tm * dirR;
            vec3  n  = normalize(hp - moonDir3);
            float diffuse  = max(0.0, dot(n, sunDir)) * moonDirENU.w;
            float mu       = max(0.0, dot(n, -moonDir3));
            float limbDark = 0.35 + 0.65 * sqrt(mu);
            // Earthshine inversely follows moon phase: new moon (full Earth) = maximum.
            float earthshine = 0.0008 * mu * (1.0 - moonDirENU.w);

            // Build the moon's local face frame: moonZ points toward the observer
            // (tidally locked near side), moonX/moonY span the visible face plane.
            // refUp = celestial north pole in ENU: converts ECEF (0,0,1) to observer ENU
            // by dotting with the ENU basis vectors (enuX/Y/Z are in ECEF-space).
            // This correctly rotates the texture with parallactic angle as the observer
            // moves across Earth, instead of always aligning north with local zenith.
            vec3 moonZ = -moonDir3;
            vec3 northCelENU = vec3(enuX.z, enuY.z, enuZ.z);
            vec3 refUp = (abs(dot(northCelENU, moonZ)) < 0.99) ? northCelENU : vec3(1.0, 0.0, 0.0);
            vec3 moonX = normalize(cross(refUp, moonZ));
            vec3 moonY = cross(moonZ, moonX);

            // Orthographic projection of the surface normal onto the face plane.
            // At the disc centre n == moonZ → UV (0.5, 0.5); at the limb UV spans [0,1].
            vec2 moonUV = vec2(dot(n, moonX), dot(n, moonY)) * 0.5 + 0.5;

            // Rotate UV around disc centre by kMoonTexRotDeg to align image north pole
            // with the physical lunar north pole. Positive = CCW rotation of the texture.
            float rotRad = radians(kMoonTexRotDeg);
            float cosR = cos(rotRad), sinR = sin(rotRad);
            vec2  uvc  = moonUV - 0.5;
            moonUV = vec2(cosR * uvc.x - sinR * uvc.y,
                          sinR * uvc.x + cosR * uvc.y) + 0.5;

            vec3 texColor = texture(moonTex, moonUV).rgb;

            // Occluded by terrain OR by opaque cloud (tCloudOcclude, ≥90% opaque along this ray —
            // same threshold satellite/star depth occlusion uses below). Without the cloud term
            // the moon's raw disc brightness survived the later multiplicative cloud attenuation
            // visibly intact even under a thick deck — the composite dims it but doesn't blank
            // the fine albedo detail the way a genuinely opaque cloud should.
            float discFade = (tSurface > 0.0 || tCloudOcclude >= 0.0) ? 0.0 : 1.0;
            vec3 moonColor = texColor * (diffuse + earthshine) * limbDark * kMoonBright;
            vec3 moonAttn  = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));
            color += discFade * moonColor * moonAttn;
        }
    }

    // ── Satellite constellation sky glow (pre-tonemap) ────────────────────────
    // Wide Gaussian (kSig = 0.90 rad ~= 51 deg) over 64 sky bins.
    // Each occupied bin represents the brightest satellite in that 45°×11.25° cell.
    // Runs pre-tonemap so the exposure system scales it: invisible at noon,
    // visible at dusk, prominent at night.
    // Knockout bit 65536 (2026-08-10): 64 iterations with an acos() each, on EVERY full-res pixel,
    // unconditionally — the only fixed-cost loop in this shader with no knockout and no quality
    // slider behind it, so its share of the "sky background draw" bucket was previously
    // unmeasurable. Skipping just leaves `color` without the glow term, which is exactly what an
    // all-empty glowBuf already produces.
    if ((pc.debugDisableMask & 65536u) == 0u) {
        const float TWO_PI = 6.28318530718;
        vec3  flareAttn = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));
        const float kSig = 0.90;
        for (int gi = 0; gi < 64; ++gi) {
            uint fluxBits = glowBuf.bins[gi];
            if (fluxBits == 0u) continue;
            float flux   = uintBitsToFloat(fluxBits);
            // Derive bin-centre ENU direction from bin index.
            // azBin=0 is North, increasing toward East (matches atan(x,y) convention).
            float az     = (float(gi / 8) + 0.5) * (TWO_PI / 8.0);
            float elSin  = (float(gi % 8) + 0.5) / 8.0; // z = sin(elevation)
            float elCos  = sqrt(max(0.0, 1.0 - elSin * elSin));
            vec3  fd     = vec3(sin(az) * elCos, cos(az) * elCos, elSin);
            if (fd.z < limbZ - 0.05) continue;
            float angle  = acos(clamp(dot(dir, fd), -1.0, 1.0));
            float glow   = exp(-angle * angle / (2.0 * kSig * kSig)) * 0.01;
            float gElev  = smoothstep(-0.08, 0.02, fd.z);
            float intens = clamp(log2(max(flux, 1.0)) / 4.0, 0.0, 1.5);
            float atmosW = 1.0 - exp(-odR_cam / 5000.0);
            color += hClip * gElev * glow * intens * 0.06 * vec3(1.0, 0.96, 0.88) * flareAttn * atmosW;
        }
    }

    // ── Phase 3: ground / terrain composite ──────────────────────────────────
    // The atmosphere was truncated at tSurface, so odR_cam/odM_cam represent
    // optical depth from the observer to the surface. Transmittance = e^(-tau).
    // We ADD attenuated surface colour to the atmosphere scatter already in `color`.
    if (tSurface > 0.0) {
        vec3 surfAttn = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));

        vec2 uvSurf;
        vec3 hitPt;
        if (tHit > 0.0) {
            uvSurf = hitUV;
            hitPt  = obsPos + tHit * dir;
        } else {
            hitPt        = obsPos + tSeaLvl * dir;
            vec3  hE     = hitPt.x * enuX + hitPt.y * enuY + hitPt.z * enuZ;
            float geoLat = asin(clamp(hE.z / R_EARTH, -1.0, 1.0));
            float geoLon = atan(hE.y, hE.x);
            uvSurf = vec2((geoLon + PI) / (2.0*PI), (0.5*PI - geoLat) / PI);
        }

        vec3  shadingN   = (tHit > 0.0) ? terrainNorm : normalize(hitPt);
        float sunDot     = dot(shadingN, sunDir);
        // Geographic horizon gate: dot(normalize(hitPt), sunDir) is the sun's elevation above
        // the local horizon at the TERRAIN POINT's geographic location.  The slope normal
        // (shadingN) cannot be used for this — a steep slope can face toward the sun even
        // when the terrain point is on the night side of Earth.  Gate dayFrac by the radial
        // direction so no illumination leaks past the terminator regardless of slope angle.
        // Margin [-0.03, 0.02]: thin alpenglow zone for mountain peaks that see the sun just
        // past the flat horizon; anything below -1.7° geographic is forced to zero.
        float geoSunDot   = dot(normalize(hitPt), sunDir);
        float horizonGate = smoothstep(-0.03, 0.02, geoSunDot);
        float dayFrac     = smoothstep(-0.15, -0.12, sunDot) * horizonGate;
        // directSun combines day/night blend for all sun-driven contributions.
        float directSun   = dayFrac;
        // Cloud shadow: cloud_march.comp already marched sunward from this pixel's own terrain
        // hit point and stored the transmittance in cloudB.a (sampled earlier, alongside the rest
        // of the composite).
        //
        // This replaced a 128x128 observer-centred tangent-plane grid, which needed all three of
        // those things plus a texel-snapping residual to stop shadows swimming as the observer
        // moved, and which silently stopped shadowing anything past cloudShadowRangeM. The
        // replacement has no range limit and nothing to snap, because the value is a function of
        // the world point being shaded rather than of where the camera happens to be.
        //
        // 5x5 box-blurred over cloudTargetB's own half-res texels rather than a single tap.
        // cloudGroundShadow (cloud_march.comp) is a 12-step raymarch dithered by one noise-texture
        // lookup per half-res pixel, with no temporal accumulation to average it away (single
        // frame in flight) — so the raw value reads as the noise texture itself stamped onto the
        // ground, worst on the ocean where there's no other high-frequency detail to hide it in.
        // A small spatial blur here is the same fix the light-pollution dome already uses (its own
        // 5-tap blur) for the same kind of per-sector/per-texel sampling noise. Started at 3x3;
        // still visibly grainy up close on the ocean, widened to 5x5 (radius 2 half-res texels,
        // ~4 screen pixels at renderScale=1) — only ground-hit pixels pay for this, and 25 taps
        // against an already-small half-res target is cheap next to the sky-ambient zenith
        // integration and city-detail sampling this same branch already does.
        float cloudShadowT = 0.0;
        {
            vec2 shadowTexel = 1.0 / vec2(textureSize(cloudTargetB, 0));
            for (int sy = -2; sy <= 2; ++sy)
                for (int sx = -2; sx <= 2; ++sx)
                    cloudShadowT += texture(cloudTargetB, cloudUV + vec2(sx, sy) * shadowTexel).a;
            cloudShadowT *= (1.0 / 25.0);
        }
        directSun *= cloudShadowT;
        // Antimeridian seam fix: longitude wraps at ±PI so dFdx(uvSurf.x) jumps by ~1.0
        // across that boundary. The GPU would pick the highest mip level, blurring a
        // vertical strip. Clamp the derivative to the small expected value instead.
        vec2 uvd_dx = dFdx(uvSurf);
        vec2 uvd_dy = dFdy(uvSurf);
        if (uvd_dx.x >  0.5) uvd_dx.x -= 1.0;
        if (uvd_dx.x < -0.5) uvd_dx.x += 1.0;
        if (uvd_dy.x >  0.5) uvd_dy.x -= 1.0;
        if (uvd_dy.x < -0.5) uvd_dy.x += 1.0;
        vec3 dayColor   = textureGrad(earthDayTex,   uvSurf, uvd_dx, uvd_dy).rgb;
        vec3 nightColor = textureGrad(earthNightTex, uvSurf, uvd_dx, uvd_dy).rgb;
        // Blur city lights toward a coarser mip under cloud (see localCloudOpacity above and
        // cloud.cityLightBlurLod) — real light passing through cloud droplets is diffused, not a
        // clean pass-through of whatever's behind it, so a sharp copy of earthNightTex's city
        // silhouette bleeding through a hazy/thin cloud reads as an artifact even when the
        // OPACITY itself is physically reasonable. Skipped below the horizon (dayFrac path has no
        // night lights anyway) is unnecessary — nightColor is cheap and already computed for the
        // dayFrac mix regardless of whether it's actually night at this point.
        if (cloud.cityLightBlurLod > 0.01 && localCloudOpacity > 0.001) {
            vec3 nightColorBlur = textureLod(earthNightTex, uvSurf, cloud.cityLightBlurLod).rgb;
            nightColor = mix(nightColor, nightColorBlur, localCloudOpacity);
        }

        // ── City detail texture blend ───────────────────────────────────────────
        // Fades in a tileable high-frequency detail texture over bright earthNightTex pixels
        // (cities) within a fixed distance of the observer: dayDetail replaces dayColor,
        // nightDetail replaces the night emissive term. Beyond kCityFadeFarM, or over
        // non-city terrain, this is a no-op and dayColor/nightColor pass through unchanged.
        {
            const float kCityDetailTileM = 20000.0;  // metres per texture tile repeat
            const float kCityMaskLo      = 0.01;   // nightColor luminance where detail starts
            const float kCityMaskHi      = 0.3;   // luminance where detail is fully blended in
            const float kCityFadeNearM   = 30000.0; // full detail strength inside this distance
            const float kCityFadeFarM    = 300000.0; // detail fully faded out beyond this distance
            float cityDistFade = 1.0 - smoothstep(kCityFadeNearM, kCityFadeFarM, tSurface);
            if (cityDistFade > 0.001)
            {
                // Cheap reject using the already-fetched, full-detail nightColor (no extra texture
                // fetch) before paying for the blurred sample + noise below — most terrain within
                // kCityFadeFarM isn't near a city at all.
                float cityLumFast = dot(nightColor, vec3(0.2126, 0.7152, 0.0722));
                float cityMask = 0.0;
                vec2  worldXY  = vec2(0.0);
                if (cityLumFast > kCityMaskLo - 0.05)
                {
                    // World-fixed ground coordinate — see the detailUV comment below for why this
                    // (not hitPt.xy directly) is what stays glued to the terrain as the observer
                    // moves. Computed here too so the edge-jitter noise below can share it.
                    worldXY = hitPt.xy + vec2(cloud.pad1, cloud.pad2);

                    // earthNightTex is only ~8K across the whole globe (~5 km/texel) — city
                    // silhouettes are inherently blocky at that resolution, and no amount of
                    // filtering recovers detail that was never captured. Two cheap tricks disguise
                    // it instead of trying to resolve it:
                    //   1. Sample luminance at a deliberately coarser LOD than nightColor's own
                    //      (already-mip-selected) sample, so the mask transition isn't chasing the
                    //      raw texel grid — a wider, softer step instead of a hard mip-block edge.
                    //   2. Jitter the smoothstep threshold with the same analytic 3D Perlin noise
                    //      the clouds use (warpPerlin3 — pure ALU, no texture, no tiling seam at any
                    //      zoom), sampled in the world-fixed ground plane at a frequency well above
                    //      the source texture's resolution. This breaks the mip-grid-aligned
                    //      blockiness into an organic, irregular wobble. It can't recover the TRUE
                    //      city boundary (there's no more data than the low-res mask has) — it's
                    //      purely a masking/edge-shape disguise, independent of the (already
                    //      world-fixed) detail texture UV below.
                    const float kCityMaskLod          = 1.5;
                    const float kCityEdgeNoiseFreqInv = 1.0 / 8000.0; // ~800 m noise period
                    const float kCityEdgeNoiseAmt     = 0.05;        // threshold jitter (luminance units)
                    float cityLumBlur = dot(textureLod(earthNightTex, uvSurf, kCityMaskLod).rgb,
                                             vec3(0.2126, 0.7152, 0.0722));
                    float edgeNoise = warpPerlin3(vec3(worldXY * kCityEdgeNoiseFreqInv, 0.0));
                    float jitter    = edgeNoise * kCityEdgeNoiseAmt;
                    // max(sharp, blurred), not blurred alone: blurring dilutes peak brightness (a
                    // city core averaged with its darker surroundings at LOD 1.5 may never cross
                    // kCityMaskHi on its own), which was capping cityMask well below 1.0 even deep
                    // in bright cores — nightColor could never fully hand off to nightDetail, so the
                    // detail pattern was always fighting a still-visible base underneath. The sharp
                    // sample lets genuinely bright pixels reach full mask strength; the blurred+
                    // jittered sample still governs the soft, noisy edge in between.
                    float cityLumForMask = max(cityLumFast, cityLumBlur);
                    // Once kCityEdgeNoiseAmt exceeds kCityMaskLo, jitter can push the lower
                    // threshold below zero — and a negative lower threshold means truly-dark
                    // (cityLumForMask≈0) pixels read as "above threshold", producing spurious light
                    // patches in genuinely unpopulated areas. Shift the whole [lo,hi] band up
                    // together when that would happen (rather than clamping loT alone, which would
                    // narrow or invert the transition width) — preserves the organic per-edge jitter
                    // everywhere it's safe, without ever letting true darkness read as lit.
                    float loT = kCityMaskLo + jitter;
                    float hiT = kCityMaskHi + jitter;
                    const float kCityMaskLoFloor = kCityMaskLo * 0.5;
                    if (loT < kCityMaskLoFloor)
                    {
                        float shift = kCityMaskLoFloor - loT;
                        loT += shift;
                        hiT += shift;
                    }
                    cityMask = smoothstep(loT, hiT, cityLumForMask) * cityDistFade;
                }
                if (cityMask > 0.001)
                {
                    // hitPt.xy (observer-local ENU tangent plane) is a real orthogonal projection —
                    // a physical square patch of ground always looks square in it, at any latitude,
                    // any distance — unlike a lon/lat-derived (plate-carrée-style) UV, which is only
                    // exactly square-scale at the one latitude its metric was evaluated at and
                    // visibly skews elsewhere (tried; no derivative fix rescues it, it's the wrong
                    // coordinate system). So a local ENU tangent plane is the right shape. hitPt.xy's
                    // only flaw is being tied to the observer's live position (drifts as the observer
                    // moves). A fixed-basis anchor was tried to cancel that exactly, but re-deriving
                    // the basis at each grid-snap silently rotated the axes a little, not just
                    // translated them — a visible pop at every snap instead of a seamless jump.
                    //
                    // The actual fix is simpler: hitPt.xy's drift, for any point near the observer,
                    // is to leading order just a uniform shift equal to the observer's OWN north/east
                    // motion (shifting the reference frame doesn't rotate nearby points relative to
                    // each other, it moves them all together). So track the observer's cumulative
                    // north/east displacement on the CPU (cityOffsetEastM/NorthM in
                    // SatelliteSim.cpp, packed into cloud.pad1/pad2) and add it straight back —
                    // a plain per-frame-constant translation, no basis, no trig, no snap events.
                    // worldXY (computed above) is this same hitPt.xy + cloud.pad1/pad2 offset —
                    // reused here rather than recomputed.
                    vec2 detailUV = worldXY / kCityDetailTileM;
                    vec2 duv_dx = dFdx(detailUV);
                    vec2 duv_dy = dFdy(detailUV);
                    vec3 dayDetail   = textureGrad(cityDayDetailTex,   detailUV, duv_dx, duv_dy).rgb;
                    vec3 nightDetail = textureGrad(cityNightDetailTex, detailUV, duv_dx, duv_dy).rgb;
                    // Same blur-through-cloud treatment as the base nightColor sample above —
                    // this is the HIGHER-frequency of the two textures, so left sharp it would be
                    // the more visible half of the "sharp texture cutting through cloud" artifact.
                    if (cloud.cityLightBlurLod > 0.01 && localCloudOpacity > 0.001) {
                        vec3 nightDetailBlur = textureLod(cityNightDetailTex, detailUV, cloud.cityLightBlurLod).rgb;
                        nightDetail = mix(nightDetail, nightDetailBlur, localCloudOpacity);
                    }
                    dayColor   = mix(dayColor,   dayDetail,   cityMask);
                    nightColor = mix(nightColor, nightDetail, cityMask);
                }
            }
        }

        // ── Spectral sun color at terrain hit ─────────────────────────────────
        // Sun light arriving at the terrain is orange at low angles (long atmospheric path).
        // Normalized so the brightest channel = 1.0 (preserves hue; noon ≈ warm white).
        vec3 sunSpecTint = vec3(1.0);
        {
            vec2 tSET = raySphere(hitPt, sunDir, R_EARTH);
            if (!(tSET.x > 0.0 && tSET.y > 0.0)) {  // terrain not in Earth's shadow
                vec2 tSAT = raySphere(hitPt, sunDir, R_ATMOS);
                if (tSAT.y > 0.0) {
                    vec2 sODT    = optDepth(hitPt, sunDir, tSAT.y);
                    vec3 attnT   = exp(-(BETA_R * sODT.x + BETA_M * 1.1 * sODT.y));
                    float maxAttn = max(max(attnT.r, attnT.g), max(attnT.b, 0.001));
                    sunSpecTint  = attnT / maxAttn;  // hue-normalized: max channel → 1.0
                }
            }
        }

        // ── Sky ambient at terrain hit ─────────────────────────────────────────
        // 4-step zenith integration gives the scattered sky color illuminating terrain
        // faces: blue during day, warm-orange during twilight. Mirrors the cloud skyAmbientBase.
        vec3 skyAmbientTerrain = vec3(0.0);
        {
            vec3  zenT = normalize(hitPt);  // terrain zenith (radially outward)
            vec2  tSAT = raySphere(hitPt, zenT, R_ATMOS);
            if (tSAT.y > 0.0) {
                const int N_ZT = 4;
                float zSegT    = tSAT.y / float(N_ZT);
                float cosAT    = dot(zenT, sunDir);
                float pR_upT   = 0.75 * (1.0 + cosAT * cosAT);
                float pM_upT   = phaseM(cosAT);
                float odR_zt = 0.0, odM_zt = 0.0;
                float skyAmbTM = 0.0;
                for (int zi = 0; zi < N_ZT; ++zi) {
                    vec3  sp = hitPt + zenT * ((float(zi) + 0.5) * zSegT);
                    float h  = max(0.0, length(sp) - R_EARTH);
                    float dR = exp(-h / H_R) * zSegT;
                    float dM = exp(-h / H_M) * zSegT;
                    odR_zt += dR;  odM_zt += dM;
                    vec2 tSE  = raySphere(sp, sunDir, R_EARTH);
                    if (tSE.x > 0.0 && tSE.y > 0.0) continue;
                    vec2 tSun = raySphere(sp, sunDir, R_ATMOS);
                    vec2 sunOD = (tSun.y > 0.0) ? optDepth(sp, sunDir, tSun.y) : vec2(0.0);
                    vec3 tau      = BETA_R * (odR_zt + sunOD.x) + BETA_M * 1.1 * (odM_zt + sunOD.y);
                    vec3 attnStep = exp(-tau);
                    skyAmbientTerrain += attnStep * dR;
                    skyAmbTM         += dot(attnStep, vec3(1.0 / 3.0)) * dM;
                }
                skyAmbientTerrain = SUN_INTENSITY * (pR_upT * BETA_R * skyAmbientTerrain
                                                   + vec3(pM_upT * BETA_M * skyAmbTM));
            }
        }

        // ── Moonlight at terrain hit ────────────────────────────────────────────
        // Mirrors the sun's own direct-light pattern above (shadingN·dir Lambertian +
        // geographic horizon gate) rather than cloud_march.comp's self-shadow/phase model —
        // terrain has no volumetric self-occlusion to model, so the sun scaffolding already
        // in this block is the closer fit. cloud.moonGain is shared with cloud_march.comp's
        // moonContrib so terrain and moonlit clouds stay calibrated to the same brightness.
        vec3  moonDir3t        = normalize(moonDirENU.xyz);
        float moonDot          = dot(shadingN, moonDir3t);
        float geoMoonDot       = dot(normalize(hitPt), moonDir3t);
        float moonHorizonGate  = smoothstep(-0.03, 0.02, geoMoonDot);
        float moonLitTerrain   = max(0.0, moonDot) * moonHorizonGate * moonDirENU.w;
        vec3  moonContribTerrain = dayColor * vec3(0.92, 0.95, 1.0) * moonLitTerrain * cloud.moonGain;

        // Aurora ground-glow: soft ambient wash from the curtain overhead, evaluated LOCALLY at
        // this hit point (auroraGlowAt — same oval mask + fold noise the sky curtain itself uses)
        // rather than a single observer-position proxy, so lighting is properly local like
        // moonlight: only ground actually under an active curtain lights up. Modulated by how much
        // the surface faces "up" (toward the glow), same spirit as skyAmbientTerrain's fill above.
        //
        // auroraGlowAt needs a TRUE ECEF direction (it compares against the fixed geomagnetic-pole
        // ECEF constant) — hitPt itself is in the observer-local ENU-ish frame (same convention
        // rp/obsPos/dir all use), so it must go through the enuX/enuY/enuZ basis first, same as
        // every other geographic lookup in this file (e.g. the terrain-hit lat/lon UV above). A
        // first version passed normalize(hitPt) directly, which is a fine "local up" vector for the
        // Lambertian dot product just below (shadingN is in the SAME local frame, so that part was
        // already correct) but wrong for auroraGlowAt specifically — it made the computed
        // "geographic" position track the OBSERVER's own frame instead of the terrain point's true
        // location, so the noise pattern appeared to follow the observer instead of the ground.
        vec3  hitDirECEF           = normalize(hitPt.x * enuX + hitPt.y * enuY + hitPt.z * enuZ);
        vec3  auroraGlowTerrain    = auroraGlowAt(hitDirECEF, sunDirECEF, pc.waveTime, cloud.stormStrength);
        // Same cloud-awareness gap as the ocean reflection fix (see that block's comment) — this
        // is a plain ambient wash, so the simplest gate is reusing cloudB, already sampled for
        // THIS pixel's own camera ray up top: an overcast view of this ground point dims its
        // aurora glow along with everything else, no new march or texture read needed.
        float auroraGroundCloudOccl = dot(cloudB.rgb, vec3(1.0 / 3.0));
        vec3  auroraContribTerrain = dayColor * auroraGlowTerrain
                                    * max(dot(shadingN, normalize(hitPt)), 0.0)
                                    * cloud.auroraGroundGain * auroraGroundCloudOccl;

        // cloudShadowT gates only the direct-sun term (real cloud shadows block the sun, not the
        // diffuse skylight) — skyAmbientTerrain stays outside it, same split the ocean branch
        // below already uses (directSun on the sun specular/diffuse terms, plain dayFrac on the
        // sky reflection). Previously this term used dayFrac alone with no cloud-shadow factor at
        // all, so cloud shadows never appeared on land — only on the ocean/sea-level branch, which
        // is the only place `directSun` (dayFrac * cloudShadowT) was actually consumed.
        vec3 surfColor  = mix(nightColor * 0.12,
                              dayColor * sunSpecTint * clamp(sunDot * 1.5, 0.05, 1.0) * cloudShadowT
                            + dayColor * skyAmbientTerrain * 0.4,  // sky ambient fill (blue day, orange dusk)
                              dayFrac)
                        + moonContribTerrain
                        + auroraContribTerrain;

        // ── Ocean wave material (sea-level hits only, not terrain) ─────────────
        // ShaderToy "Seascape" by TDM adapted to Earth ENU/ECEF space.
        // heightMapTracing: 8-step secant refinement around the sea-sphere hit.
        // getNormal: central differences on seaMapDetail (5 octaves).
        // getSeaColor: kSeaBase refraction + atmosphere reflection + specular.
        float oceanMask = textureGrad(earthSpecTex, uvSurf, uvd_dx, uvd_dy).r;
        if (oceanMask > 0.5 && tHit < 0.0) {
            vec3  surfUp  = normalize(hitPt);
            float dist    = tSeaLvl;
            float seaTime = 1.0 + pc.waveTime * kSeaSpeed;

            

            // Altitude fade: full 3D waves at low altitude, smooth specular from orbit.
            float altFade = 1.0 - smoothstep(3000.0, 8000.0, obsEffH);

            // Wave UV strategy:
            //   posM = hitPt.xy (ENU East/North metres from observer nadir) — always small,
            //   so the 0.5 m normal epsilon is hundreds of float steps above ULP.
            //   An observer-geographic phase offset (modulo first-octave wave period ≈ 39.3 m)
            //   is added so the pattern is approximately Earth-fixed without accumulating
            //   large absolute coordinates. Derived entirely from enuZ (observer ECEF unit vec).
            vec2 obsPhase = vec2(0.0);
            if (altFade > 0.01) {
                const float wvScale = 2.0 * PI / kSeaFreq;
                float oLat  = asin(clamp(enuZ.z, -1.0, 1.0));
                float oLon  = atan(enuZ.y, enuZ.x);
                obsPhase.x  = fract(oLon * R_EARTH * cos(oLat) / wvScale) * wvScale;
                obsPhase.y  = fract(oLat * R_EARTH           / wvScale) * wvScale;
            }
            vec2 posM = hitPt.xy + obsPhase;

            // ── heightMapTracing (low altitude only) ──────────────────────────
            // Bracket: ±2.5 m vertical around the sea-sphere intersection.
            // hm > 0 at the near end (above waves), hx < 0 at the far end (inside).
            if (altFade > 0.01 && dist < 5000.0) {
                float cosEl  = max(0.05, abs(dot(dir, surfUp)));
                float traceR = min(60.0, 2.5 / cosEl);
                float tm     = tSeaLvl - traceR;
                float tx     = tSeaLvl + traceR;

                // Height above sea level computed as obsEffH + 2 + t*dir.z — avoids
                // catastrophic cancellation in length(p)-R_EARTH at sea level (float
                // ULP at 6.37 M m is 0.76 m, which quantises 1.5 m waves into ~2 steps).
                vec3  plo = obsPos + tm * dir;
                float hm  = seaMap(plo.xy + obsPhase, obsEffH + 2.0 + tm * dir.z, seaTime);
                vec3  phi = obsPos + tx * dir;
                float hx  = seaMap(phi.xy + obsPhase, obsEffH + 2.0 + tx * dir.z, seaTime);

                if (hx < 0.0) {
                    for (int i = 0; i < 8; i++) {
                        float tmid = mix(tm, tx, hm / (hm - hx));
                        vec3  pm   = obsPos + tmid * dir;
                        float hmid = seaMap(pm.xy + obsPhase, obsEffH + 2.0 + tmid * dir.z, seaTime);
                        if (hmid < 0.0) { tx = tmid; hx = hmid; }
                        else             { tm = tmid; hm = hmid; }
                        if (abs(hmid) < 0.001) break;
                    }
                    float tWave = mix(tm, tx, hm / (hm - hx));
                    hitPt  = obsPos + tWave * dir;
                    surfUp = normalize(hitPt);
                    posM   = hitPt.xy + obsPhase;
                    dist   = tWave;
                }
            }

            // Same precision fix: obsEffH + 2 + dist*dir.z instead of length(hitPt)-R_EARTH.
            float pHeight = obsEffH + 2.0 + dist * dir.z;
            vec3  viewDir = normalize(-dir);

            // ── getNormal (central differences on seaMapDetail) ───────────────
            // posM is in ENU East/North metres, so +eps in x = East, +eps in y = North.
            // Normal in ENU = normalize(East_slope, North_slope, Up_component).
            //
            // Perf (session 24 round 3, low-angle/horizon views): the blend factor below only
            // needs `dist`/`altFade`, both already known here — compute it FIRST and skip the
            // three seaMapDetail calls (15 octave evaluations total) entirely when it's already
            // ~1 (result blends back to flat `surfUp` regardless). The original version always
            // paid full detail cost then discarded most of it for distant ocean — exactly the
            // case that dominates horizon views, where most visible ocean is far past the 8km
            // distance fade. Bitwise-identical result for blend<0.99; below that threshold the
            // discarded detail was already imperceptible (>99% blended to flat).
            vec3 waveN = surfUp;
            if (altFade > 0.01) {
                float distFade = smoothstep(3000.0, 8000.0, dist);
                float blend    = max(distFade, 1.0 - altFade);  // 0 = full detail, 1 = flat
                if (blend < 0.99) {
                    float eps = max(0.5, dist * 0.0008);
                    float n0  = seaMapDetail(posM,                    pHeight, seaTime);
                    float nX  = seaMapDetail(posM + vec2(eps, 0.0),  pHeight, seaTime) - n0;
                    float nY  = seaMapDetail(posM + vec2(0.0,  eps), pHeight, seaTime) - n0;
                    waveN = normalize(vec3(nX, nY, 0.0) + eps * surfUp);
                    waveN = normalize(mix(waveN, surfUp, blend));
                }
            }

            // ── getSeaColor ────────────────────────────────────────────────────
            // Fresnel: cubic ramp, capped at 0.5 (ShaderToy formula)
            float fresnel = min(pow(clamp(1.0 - dot(waveN, viewDir), 0.0, 1.0), 3.0), 0.5);

            // Sky reflection — 6-sample atmosphere, distance-gated
            vec3 reflDir   = reflect(dir, waveN);
            vec3 reflColor = vec3(0.12, 0.28, 0.50) * dayFrac;
            float reflStr  = fresnel * exp(-dist / 40000.0);
            if (!dbgSkipOceanRefl() && dot(reflDir, surfUp) > 0.0 && reflStr > 0.005) {
                vec2 tAR = raySphere(hitPt, reflDir, R_ATMOS);
                if (tAR.y > 0.0) {
                    int   N_REFL = int(max(1.0, cloud.oceanReflSamples)); // perf session 24, was const 6
                    float rStart = max(0.0, tAR.x);
                    float rSeg   = (tAR.y - rStart) / float(N_REFL);
                    float rcosA  = dot(reflDir, sunDir);
                    float rpR    = phaseR(rcosA);
                    float rpM    = phaseM(rcosA);
                    vec3  rAccR  = vec3(0.0);
                    float rAccM  = 0.0;
                    float rodR   = 0.0, rodM = 0.0;
                    for (int ri = 0; ri < N_REFL; ++ri) {
                        vec3  rp   = hitPt + reflDir * (rStart + (float(ri) + 0.5) * rSeg);
                        float rh   = max(0.0, length(rp) - R_EARTH);
                        float rdR  = exp(-rh / H_R) * rSeg;
                        float rdM  = exp(-rh / H_M) * rSeg;
                        rodR += rdR; rodM += rdM;
                        vec2 tSE  = raySphere(rp, sunDir, R_EARTH);
                        if (tSE.x > 0.0 && tSE.y > 0.0) continue;
                        vec2 tSun = raySphere(rp, sunDir, R_ATMOS);
                        vec2 sOD  = (tSun.y > 0.0) ? optDepth(rp, sunDir, tSun.y) : vec2(0.0);
                        vec3 rtau  = BETA_R * (rodR + sOD.x) + BETA_M * 1.1 * (rodM + sOD.y);
                        vec3 rattn = exp(-rtau);
                        rAccR += rattn * rdR;
                        rAccM += dot(rattn, vec3(1.0 / 3.0)) * rdM;
                    }
                    reflColor = SUN_INTENSITY * (rpR * BETA_R * rAccR + vec3(rpM * BETA_M * rAccM));
                }

                // Aurora reflection: literally march the curtain shell along the REFLECTED ray
                // instead of the camera ray — reuses the exact same auroraSampleAt() the primary
                // sky view uses, so the aurora shows up as a genuine mirror-like glint on the
                // water (visible in the reflection itself, not just a flat ambient wash) whenever
                // a wave happens to reflect toward it. hitPt is always below the shell here (ocean
                // surface), so only the "observer below shell" entry/exit case applies — no need
                // for the full obsEffH-keyed classification the primary march uses.
                vec2 rAuroraFar = raySphere(hitPt, reflDir, R_EARTH + kAuroraShellOuterM);
                vec2 rAuroraIn  = raySphere(hitPt, reflDir, R_EARTH + kAuroraShellInnerM);
                if (rAuroraIn.y > 0.0 && rAuroraIn.y < rAuroraFar.y) {
                    const int N_AURORA_REFL = 6;
                    float raSeg = (rAuroraFar.y - rAuroraIn.y) / float(N_AURORA_REFL);
                    vec3  accumAuroraRefl = vec3(0.0);
                    for (int ai = 0; ai < N_AURORA_REFL; ++ai) {
                        vec3 rap = hitPt + reflDir * (rAuroraIn.y + (float(ai) + 0.5) * raSeg);
                        accumAuroraRefl += auroraSampleAt(rap, enuX, enuY, enuZ, sunDirECEF,
                                                           pc.waveTime, cloud.stormStrength);
                    }
                    // Same atmospheric extinction as the primary sky march (see that block's
                    // comment) — using the REFLECTED ray's own elevation, since that's the
                    // direction the aurora's light actually traveled through the atmosphere before
                    // bouncing off the water toward the camera. Ocean views skew toward low-angle
                    // reflections by construction (Fresnel favors grazing angles), so this matters
                    // here at least as much as it does for the direct view.
                    float sinElAuroraRefl   = clamp(reflDir.z, 0.0, 1.0);
                    float elDegAuroraRefl   = degrees(asin(sinElAuroraRefl));
                    float airmassAuroraRefl = 1.0 / (sinElAuroraRefl + 0.50572 * pow(elDegAuroraRefl + 6.07995, -1.6364));
                    float extinctMagAuroraRefl = cloud.extinctionCoeff * (airmassAuroraRefl - 1.0);
                    float extinctionAuroraRefl = pow(10.0, -0.4 * extinctMagAuroraRefl);

                    // Cloud occlusion: this march reused auroraSampleAt() (the raw curtain
                    // function) directly rather than going through cloud_march.comp's
                    // auroraMarchCS, so it had none of that pass's cloud-suppression — the water
                    // mirrored the aurora right through an overcast sky. Same fix as the
                    // satellite/sun lens-flare occlusion above: project reflDir into screen space
                    // with the same skyView transform and sample the transmittance
                    // cloud_march.comp already wrote for that direction into cloudTargetB, rather
                    // than re-marching cloud density along reflDir here.
                    float auroraReflCloudOccl = 1.0;
                    vec3  reflCam = mat3(pc.skyView) * reflDir;
                    if (reflCam.z < -0.01) {
                        float tanHFRefl = tan(pc.fovYRad * 0.5);
                        vec2  reflUV       = vec2(reflCam.x, -reflCam.y) / (-reflCam.z * tanHFRefl * 2.0);
                        vec2  reflScreenUV = vec2(reflUV.x / pc.aspect + 0.5, reflUV.y + 0.5);
                        auroraReflCloudOccl = dot(texture(cloudTargetB, reflScreenUV).rgb, vec3(1.0 / 3.0));
                    }

                    reflColor += accumAuroraRefl * raSeg * kAuroraScale * cloud.auroraGain
                               * extinctionAuroraRefl * auroraReflCloudOccl;
                }
            }

            // Refracted subsurface color (SEA_BASE + diffuse * SEA_WATER_COLOR)
            // directSun replaces dayFrac for all sun-driven contributions so clouds shadow the ocean.
            float diff    = pow(max(0.0, dot(waveN, sunDir)) * 0.4 + 0.6, 80.0) * directSun;
            vec3 refracted = kSeaBase * directSun + diff * kSeaWaterColor * 0.12;

            // Fresnel blend (distance-attenuated to prevent orbit-scale glowing ring)
            surfColor = mix(refracted, reflColor, reflStr);

            // Wave-height crest shading: raised crests catch more water-color light
            float atten = max(1.0 - dist * dist * 1e-5, 0.0);
            surfColor += kSeaWaterColor * max(pHeight - kSeaHeight, 0.0) * 0.18 * atten * directSun;

            // Specular: shininess narrows close-up, broadens with distance
            float specPow = clamp(600.0 / max(1.0, sqrt(dist)), 8.0, 600.0);
            float nrm     = (specPow + 8.0) / (PI * 8.0);
            surfColor    += pow(max(0.0, dot(reflect(dir, waveN), sunDir)), specPow) * nrm * directSun;

            // Moon glint on ocean — nighttime only, dims with phase (new moon = brightest Earth).
            if (moonDirENU.z > limbZ && moonDirENU.w > 0.01) {
                vec3  moonDir3o = normalize(moonDirENU.xyz);
                float mSpecPow  = 120.0;
                float mNrm      = (mSpecPow + 8.0) / (PI * 8.0);
                surfColor += pow(max(0.0, dot(reflect(dir, waveN), moonDir3o)), mSpecPow)
                           * mNrm * moonDirENU.w * clamp(moonDirENU.z, 0.0, 1.0)
                           * 0.006 * (1.0 - dayFrac);
            }

            // Aurora ground-glow: soft ambient tint from the curtain overhead, evaluated LOCALLY at
            // this ocean point (auroraGlowAt — same function terrain uses) rather than a single
            // observer-position proxy, distance-attenuated by the same `atten` the wave-crest
            // shading above uses so it doesn't glow uniformly out to the horizon.
            //
            // surfUp is the observer-local "up" (same frame as hitPt/obsPos/dir) — auroraGlowAt
            // needs a TRUE ECEF direction instead (see the terrain block's own comment on this same
            // bug), so it goes through enuX/enuY/enuZ first rather than being passed straight in.
            vec3 surfUpECEF = normalize(surfUp.x * enuX + surfUp.y * enuY + surfUp.z * enuZ);
            // Same cloud gate as the terrain ground-glow and the aurora reflection above — this
            // is why the water kept turning aurora-green straight through an overcast sky.
            float auroraGroundCloudOcclOcean = dot(cloudB.rgb, vec3(1.0 / 3.0));
            surfColor += auroraGlowAt(surfUpECEF, sunDirECEF, pc.waveTime, cloud.stormStrength)
                       * cloud.auroraGroundGain * 0.5 * atten * auroraGroundCloudOcclOcean;
            // Mirror satellite flare glints — own small independent capped atomic-append list
            // (flare architecture overhaul), decoupled from the deleted per-pixel corona system.
            // Now also occlusion-aware (previously had NONE at all): sampled at each entry's own
            // screen position, the same technique already proven this session for the corona loop.
            {
                uint fCount = min(oceanGlintBuf.oceanGlintCount, kOceanGlintMax);
                float tanHFg = tan(pc.fovYRad * 0.5);
                for (uint fi = 0u; fi < fCount; ++fi) {
                    float flux = oceanGlintBuf.oceanGlintEntries[fi].w;
                    if (flux < 2.0) continue;
                    vec3 fe = normalize(oceanGlintBuf.oceanGlintEntries[fi].xyz);
                    if (fe.z < limbZ - 0.02) continue;
                    vec3 feCam = mat3(pc.skyView) * fe;
                    if (feCam.z >= -0.01) continue;
                    vec2 feUV = vec2(feCam.x, -feCam.y) / (-feCam.z * tanHFg * 2.0);
                    vec2 feScreenUV = vec2(feUV.x / pc.aspect + 0.5, feUV.y + 0.5);
                    float feCloudOccl = dot(texture(cloudTargetB, feScreenUV).rgb, vec3(1.0 / 3.0));
                    float feTerrainOccl = (texture(sceneDepthTex, feScreenUV).r >= kNoSurfaceT * 0.5) ? 1.0 : 0.0;
                    float fSpecPow = 80.0;
                    float fNrm     = (fSpecPow + 8.0) / (PI * 8.0);
                    float fIntens  = clamp(log2(max(flux, 1.0)) / 10.0, 0.0, 1.0);
                    surfColor += pow(max(0.0, dot(reflect(dir, waveN), fe)), fSpecPow)
                               * fNrm * fIntens * 0.008 * vec3(1.2, 1.1, 1.0) * (1.0 - dayFrac) * altFade
                               * feCloudOccl * feTerrainOccl;
                }
            }
        }

        // ── Reflect-Orbital beam ground-spot (C12) ──────────────────────────────────────────
        // Applies uniformly to whichever branch above produced surfColor (terrain or ocean) —
        // deliberately placed after both, not inside either. Physically a different quantity
        // from cloud_march.comp's volumetric in-scatter term above the surface: this is direct
        // irradiance landing ON the ground, so unlike the volumetric term it does NOT get
        // dimmer in fully clear air — the shadow lookup below only accounts for intervening
        // cloud, not "is there scattering medium to see the beam in" (there's no beam here to
        // see, just a lit patch of ground).
        if ((pc.debugDisableMask & 128u) == 0u) {
            const float kBeamGroundScale = 4e-8;
            // Normalized against the slider's default (0.05, see SatelliteSim.h) so existing
            // footprint brightness is unchanged at default gain, while still scaling together
            // with cloud_march.comp's sky glow (C12 follow-up #18) — one shared control instead
            // of two independently-tuned pieces, so raising/lowering it visually reads as one
            // continuous beam rather than a mismatched ground patch under an unrelated sky ray.
            float skyGlowNorm = pc.beamSkyGlowGain / 0.05;
            // Site-referenced (C12 follow-up #5): beams are now written unconditionally by any
            // satellite above the OBSERVER's own orbital horizon, not gated by the ground
            // target's local horizon — so pc.beamMaxRangeM (settings-tunable, follow-up #6) is
            // the render-time "is the observer close enough to this site" cutoff. Perf follow-up:
            // that cutoff is now applied ONCE, CPU-side, when GroundBeamsBuf is built each frame
            // (see its declaration above) rather than redone here per ground-hit pixel against
            // the full raw list — this loop's trip count is the real cost, not the comparison.
            // 2026-08-10: the loop body is now almost entirely per-pixel work. Everything that
            // did not vary across the screen — the range fade keyed to the chosen target's site,
            // the 5-degree elevation fade, the per-beam cloud shadow attenuation, and the
            // obsPos+satENU / raySphere solve for the beam's REAL ray/ground intersection (two
            // sqrts) — was hoisted to the CPU, which already visits these entries once per frame
            // when it builds GroundBeamsBuf. That loop measured 1.59 ms of this shader at Medium in
            // the Anchorage worst-case sweep, on a list sitting at its full GROUND_BEAM_MAX cap, so
            // every ground-hit pixel paid all of it 256 times.
            //
            // The squared-distance reject is now FIRST rather than last: a pixel nowhere near a
            // landing spot costs one 2D subtract, one dot and one compare per beam, and the two
            // Gaussians (the only genuinely per-pixel maths left) are reached only by pixels that
            // are actually inside a footprint. Working in squared distance also drops the
            // length() sqrt the old reject needed.
            int activeBeamCount = int(min(groundBeamCount, GROUND_BEAM_MAX));
            for (int bi = 0; bi < activeBeamCount; ++bi) {
                vec2  d  = hitPt.xy - groundBeams[bi].groundHitXY;
                float d2 = dot(d, d);
                if (d2 > groundBeams[bi].cutoffSq) continue;
                float w = groundBeams[bi].weight;
                if (w <= 0.0) continue;

                // Tight bright "hotspot" core (the mirror's own true physical size) on top of the
                // soft halo (the full sun-disk-broadened extent around it) — C12 follow-up #18/#34,
                // and an isotropic circle rather than #35's reverted ellipse (follow-up #43).
                // Reciprocals come precomputed so this is two multiplies and two exps.
                float footprint = exp(-0.5 * d2 * groundBeams[bi].invFootprintSq);
                float core      = exp(-0.5 * d2 * groundBeams[bi].invCoreSq);

                surfColor += vec3(kBeamGroundScale * w * (footprint + core * 2.0) * skyGlowNorm);
            }
        }

        color += surfColor * surfAttn;
    }

    // Sky-only snapshot for evalCloudLayer's aerial-perspective term below — everything folded
    // into `color` up to this point is atmosphere/sky (Rayleigh/Mie inscatter, city/beam glow,
    // airglow, moon disc/corona) with no ground/terrain/ocean surface light yet, since that was
    // just added immediately above. See evalCloudLayer's own comment for why this needs to be
    // kept separate from the ground-inclusive `color`.
    vec3 skyOnlyColor = color;

    // ── Cloud layers (C3/C4 unified: thin-shell 2D overlays) ─────────────────
    // Layers 0/1 double as the volumetric shell's base/top (same shellAltM values cloudMarch
    // reads below), so their flat paste here must crossfade against cloudMarch's own fade using
    // the SAME kCloud3DFadeStart/End band — not an independent threshold — or the two renders
    // overlap. Layers 2/3 (e.g. a standalone high cirrus deck) are always flat, at full weight.
    //
    // Iterate HIGH INDEX -> LOW INDEX: layers are conventionally ordered by increasing altitude
    // (layer0 = low/near, layer1 = cirrus/far, and any future layer2/3 should follow the same
    // convention). evalCloudLayer composites each call ON TOP of whatever `color` already holds,
    // so the farthest-from-a-ground-observer shell must be drawn FIRST (as background) and the
    // nearest drawn LAST (on top) for correct back-to-front compositing — ascending-index order
    // had this backwards (cirrus drew over the low deck regardless of which was actually nearer).
    for (int li = 3; li >= 0; --li) {
        if (cloud.layers[li].enabled < 0.5) continue;
        // The 3D->2D weight used to be computed here from observer altitude alone — one value for
        // the entire screen. It now lives inside evalCloudLayer, which knows this ray's own
        // distance to the shell; see the note there.
        float volumetricPair = (li < 2) ? 1.0 : 0.0;
        evalCloudLayer(
            obsPos, dir, tSurface, enuX, enuY, enuZ, sunDirECEF,
            odR_cam, odM_cam,
            // flatCoverageScale / flatSunGainScale calibrate the shared sliders onto this path.
            // Without them one set of values could only ever suit the volumetric OR the flat
            // layer, never both — which is what made the 3D->2D crossfade untunable and forced
            // kCloud3DFadeStart out to 800 km to hide the mismatch.
            cloud.coverage * cloud.layers[li].coverageMult * cloud.flatCoverageScale,
            // flatDensityScale decouples the flat layer's opacity from the volumetric one. They
            // reach opacity by completely different routes — the volumetric accumulates
            // transmittance over many samples, so lowering `density` to soften its shading
            // necessarily thins it; the flat layer multiplies once, so the same value drops
            // straight out as translucency. One shared slider could only ever satisfy one of them.
            cloud.density  * cloud.layers[li].densityMult * cloud.flatDensityScale,
            cloud.sunGain      * cloud.flatSunGainScale,
            cloud.sunGainZenith * cloud.flatSunGainScale,
            cloud.layers[li].shellAltM,
            cloud.layers[li].driftMult,
            cloud.layers[li].alphaMax,
            cloud.layers[li].mipLod,
            cloud.cloudPhase,
            obsEffH, volumetricPair,
            skyOnlyColor,
            color);
    }

    // ── Half-resolution cloud composite (C15-perf) ───────────────────────────────
    // cirrusMarch/cloudMarch ran in shaders/cloud_march.comp at half resolution; sample the
    // precomputed result here instead of marching per full-res pixel. Target A: rgb=B_total
    // (combined additive radiance), a=tCloudOcclude (m, -1=none, only set when the cloud is
    // ≥90% opaque — used below for satellite/star depth occlusion, NOT for terrain suppression).
    // Target B: rgb=A_total (combined multiplicative attenuation), a=per-pixel cloud shadow
    // (currently 1.0 — the channel was freed by deleting tEnterCombined and is claimed in the
    // next step).
    // Perf (session 29, resolution-scaling fix): was `gl_FragCoord.xy / (textureSize(
    // cloudTargetA,0)*2.0)`, silently assuming gl_FragCoord always spans the full swap extent —
    // true before renderScale existed, false once sat_sky.frag can render into a SMALLER low-res
    // framebuffer (recordPrePass) while cloudTargetA/B stay sized off the TRUE swap extent
    // (cloud_march.comp's own dispatch is unaffected by renderScale). Dividing by the wrong,
    // larger denominator compressed cloudUV into a shrinking corner of [0,1] as renderScale
    // dropped — reported as clouds drifting off-center and distorting. pc.screenSizePx is always
    // this draw's OWN actual target size, so this now maps to [0,1] correctly regardless of scale.
    // cloudUV/cloudA/cloudB/tCloudOcclude were sampled earlier (right after tSurface), so the
    // moon disc above can be occluded by opaque cloud too — not resampled here.
    // cloudBlock (post-tonemap sun-disc dimming, used below) derived from A_total's luminance
    // rather than a separate stored scalar — A_total already tracks combined opacity closely
    // (→0 when opaque, →1 when clear).
    float cloudBlock    = dot(cloudB.rgb, vec3(1.0/3.0));
    // No terrain-suppression test any more. cloud_march.comp now clamps every layer (cloud,
    // cirrus, aurora, airglow-red) and every beam to the shared scene depth at march time, so
    // whatever reached this composite is already correctly occluded — per layer, and with real
    // partial truncation where a ridge pokes into a shell, which the old single-scalar gate
    // could not express at all.
    // Aurora rides along inside cloudA.rgb (B_total) now — folded in by cloud_march.comp's
    // auroraMarchCS, with its own cloud-suppression already applied there using the local cloud
    // opacity. No separate aurora term needed here; it is terrain-occluded at march time along
    // with everything else in the composite.
    color = color * cloudB.rgb + cloudA.rgb;

    // ── Auto-exposure tone mapping ─────────────────────────────────────────────
    float dayness  = clamp((sunDirENU.w + 0.2) / 1.2, 0.0, 1.0);
    float exposure = mix(EXPOSURE_NIGHT, EXPOSURE_DAY, pow(dayness, 0.4));
    color = vec3(1.0) - exp(-exposure * color);

    // ── Night ambient floor ────────────────────────────────────────────────────
    float nightAmt = 1.0 - clamp(dayness * 5.0, 0.0, 1.0);
    color += vec3(0.0008, 0.001, 0.002) * nightAmt;

    // ── Milky Way skybox ───────────────────────────────────────────────────────
    // Diffuse galactic-plane glow behind the discrete star catalog (star_point.vert/frag).
    // Visible only from truly dark sites or space, using the same gating shape as CPU's
    // updateStars(): sun-elevation/space detection, moonlight, directional light-pollution dome,
    // and atmospheric extinction. Added post-tonemap like the ambient terms around it (comparably
    // faint) rather than folded into the HDR atmosphere accumulation above.
    {
        // Space detection: mirrors CPU's updateStars()/atmFrac — linear fade over the last
        // stretch of the simulated atmosphere shell (40-100km, R_ATMOS-R_EARTH=100km) rather
        // than an 80km-scale-height exponential, which decayed too fast and leaked Milky Way
        // brightness into a clear daytime sky by cloud-deck altitude. Keep in sync with that copy.
        const float kMWSpaceFadeStartM = 40000.0;
        const float kMWSpaceFadeEndM   = 100000.0;
        float atmFracSky = 1.0 - clamp((obsEffH - kMWSpaceFadeStartM)
                                        / (kMWSpaceFadeEndM - kMWSpaceFadeStartM), 0.0, 1.0);
        float nightFactorSky = clamp(-sunDirENU.w * 5.0, 0.0, 1.0);
        // pc.skyGlareVisibility (CPU-eased sun-glare gate, see its push-constant comment) replaces
        // the old flat 1.0 space target — matches the same replacement in CPU's updateStars().
        float nightFactorEffSky = mix(pc.skyGlareVisibility, nightFactorSky, atmFracSky);

        // Moonlight suppression — same shape as CPU's moonBrightStar.
        float tm = clamp(moonDirENU.z / 0.5, 0.0, 1.0);
        float moonBrightSky = tm * tm * moonDirENU.w;
        const float kMWMoonMaxDim = 0.95;

        // Directional dome geometry — copied from sat_flare.comp's domeVal computation (session
        // 26), using this fragment's own view direction (dir) in place of a satellite's skyDir.
        // Only sec0w/sec1w/secFrac/elevFalloffMW are still needed here (for beamDomeVal below) —
        // the shared domeAz/kIsotropicFrac/domeVal curve itself is NOT used for the Milky Way's
        // own pollution response any more (see mwSuppressEased below for why).
        float azLP    = mod(atan(dir.x, dir.y) + 6.283185307, 6.283185307);
        float secF    = azLP * (16.0 / 6.283185307) - 0.5;
        int   sec0    = int(floor(secF));
        float secFrac = secF - float(sec0);
        int   sec0w   = ((sec0 % 16) + 16) % 16;
        int   sec1w   = (sec0w + 1) % 16;
        float elevFalloffMW = 0.35 / (max(dir.z, 0.0) + 0.35);

        // Milky Way-specific pollution response (replaces the old shared domeVal/kMWPollutionMaxDim
        // linear dim, which started dimming the Milky Way the instant ANY light pollution was
        // present and never fully hid it — 0.99 max dim still lets 1% through). The Milky Way
        // should read as invisible until skies are genuinely dark, and pc.mwSuppressEased is a
        // CPU-side value already hysteresis-eased over mwPollutionThresholdLo/Hi (SatelliteSim.h,
        // see updateLightPollutionDome()) — a single non-directional "is it dark enough here" gate,
        // deliberately not per-pixel like domeVal/beamDomeVal: the Milky Way is a large diffuse
        // feature, not something whose suppression should pop differently by look direction the
        // way a single city's horizon glow does.
        float mwPollutionSuppress = pc.mwSuppressEased;

        // C12 follow-up #31: same suppression shape, second independent source — a nearby
        // Reflect-Orbital beam should wash out the Milky Way the same way real light pollution
        // does. beamGlowDome[] holds raw atomicMax'd uint bit-patterns (floatBitsToUint on the
        // write side in sat_orbit.comp) — reinterpret via uintBitsToFloat, unlike lightDome[].
        float beamDomeAz = mix(uintBitsToFloat(beamGlowDome[sec0w]), uintBitsToFloat(beamGlowDome[sec1w]), secFrac);
        float beamDomeVal = clamp(beamDomeAz * elevFalloffMW, 0.0, 1.0);
        const float kMWBeamPollutionMaxDim = 0.99;

        // Atmospheric extinction — same Kasten & Young 1989 airmass approximation used by
        // sat_flare.comp/updateStars(), reusing cloud.extinctionCoeff (was pad0 — see CloudParams).
        // Deliberately NOT atmFracSky (that one is keyed on the OBSERVER's own altitude, correct
        // for "is there scattering atmosphere near me to explain a bright sky" but wrong for "how
        // much atmosphere does THIS RAY pass through" — a ray aimed near the horizon from orbit can
        // still graze deep into the atmosphere even though the observer itself is far above it, see
        // rayTangentAltM in common.glsl). Uses this ray's own tangent altitude instead, same shape
        // (linear fade over kMWSpaceFadeStartM/EndM) as atmFracSky so near-zenith views (where the
        // two coincide) are unchanged.
        float atmFracExtinctMW = 1.0 - clamp((rayTangentAltM(obsPos, dir) - kMWSpaceFadeStartM)
                                              / (kMWSpaceFadeEndM - kMWSpaceFadeStartM), 0.0, 1.0);
        float sinElMW  = clamp(dir.z, 0.0, 1.0);
        float elDegMW  = degrees(asin(sinElMW));
        float airmassMW = 1.0 / (sinElMW + 0.50572 * pow(elDegMW + 6.07995, -1.6364));
        float extinctMagMW = cloud.extinctionCoeff * (airmassMW - 1.0) * atmFracExtinctMW;
        float extinctionMW = pow(10.0, -0.4 * extinctMagMW);

        // Sun glare: even in space (where nightFactorEffSky doesn't suppress anything — there's
        // no atmosphere to scatter sunlight into a uniform "day" sky), staring straight at the
        // sun should still wash out the Milky Way — real eye/camera dazzle, not atmospheric
        // scattering, so this is unconditional rather than atmFracSky-gated. On the ground it's
        // mostly redundant with nightFactorEffSky already zeroing everything once the sun is up,
        // but it also correctly dims the Milky Way near the sun during twilight, when the rest of
        // the sky is still dark enough to show it.
        // Gated on the sun actually being above the spherical horizon (same limbZ test the sun
        // disc's own visibility uses below) — a pure angle-to-sunDir test has no notion of what's
        // along that line of sight, so without this it kept dimming the Milky Way toward the
        // sunset point (or, from orbit, toward wherever the Earth hides the sun) long after the
        // sun itself was fully Earth-occluded and no real glare could exist.
        float sunAngleMW = acos(clamp(dot(dir, sunDir), -1.0, 1.0));
        float sunGlareSuppress = (sunDirENU.w > limbZ) ? smoothstep(0.12, 0.5, sunAngleMW) : 1.0; // 0 within ~7deg, 1 beyond ~29deg or sun occluded

        // Project the view ray into the galactic frame and sample the panorama.
        vec3 dirGal = vec3(dot(dir, cloud.mwBasisRow0.xyz),
                            dot(dir, cloud.mwBasisRow1.xyz),
                            dot(dir, cloud.mwBasisRow2.xyz));
        float lonGal = atan(dirGal.y, dirGal.x);
        float latGal = asin(clamp(dirGal.z, -1.0, 1.0));
        vec2  mwUV   = vec2(0.5 + lonGal / (2.0 * PI), 0.5 + latGal / PI);
        vec3  mwColor = texture(milkyWayTex, mwUV).rgb * cloud.mwBasisRow0.w;

        // Cloud suppression: CUBED, not linear — same reasoning as the aurora's auroraCloudSuppress
        // above (session 28 follow-up #11). A plain `* cloudBlock` still let the Milky Way show
        // clearly through cloud that reads as visually solid, since a "mostly opaque" transmittance
        // of e.g. 0.25 only cuts brightness to a quarter. Reuses the same opacity scalar the sun
        // disc is already dimmed by (see "Sun/moon disc" above); this term is added post-tonemap
        // (deliberately, see comment above) so it can't be folded into the HDR cloud composite the
        // same way aurora/atmosphere are, but a steeper power curve on the same continuous value
        // works without needing that.
        const float kMWCloudSuppressPower = 3.0;
        float visibility = nightFactorEffSky
                          * (1.0 - mwPollutionSuppress)
                          * (1.0 - beamDomeVal * kMWBeamPollutionMaxDim)
                          * (1.0 - moonBrightSky * kMWMoonMaxDim)
                          * extinctionMW
                          * sunGlareSuppress
                          * (tSurface > 0.0 ? 0.0 : 1.0) // blocked by terrain/ocean
                          * (moonDiscHit ? 0.0 : 1.0)    // blocked by the Moon's own opaque disc
                          * pow(clamp(cloudBlock, 0.0, 1.0), kMWCloudSuppressPower);
        color += mwColor * visibility;
    }

    // ── Zodiacal light ─────────────────────────────────────────────────────────
    // Sunlight scattered by interplanetary dust in the ecliptic plane — a faint, warm-white
    // diffuse cone brightest a few degrees beyond the sun's own corona, fading with elongation,
    // plus a much fainter "gegenschein" patch directly opposite the sun. Pure analytic falloff,
    // no texture: elongation from the sun needs only sunDir (already in scope); ecliptic latitude
    // needs the one new CPU-computed basis vector, cloud.eclipticPoleENU (see updatePositions()
    // and cloud_params.glsl). Added post-tonemap for the same reason the Milky Way/sun disc/moon
    // corona are — a faint additive term should read at a consistent brightness regardless of
    // whatever exposure the raw HDR atmosphere integral needed that frame, not get compressed or
    // blown out differently by EXPOSURE_DAY/EXPOSURE_NIGHT depending on sky brightness that frame.
    // Deliberately does NOT reuse the Milky Way's sunGlareSuppress above — that exists specifically
    // to dim things NEAR the sun, which is exactly where zodiacal light is brightest; innerFade
    // below handles the seam against the corona for an unrelated reason (avoiding a double-bright
    // ring, not glare). Locals below duplicate the Milky Way block's day/night/moon/extinction/
    // dome shapes rather than reaching across the closed `{}` above — same per-block-local
    // convention this file already uses everywhere else.
    {
        float theta = acos(clamp(dot(dir, sunDir), -1.0, 1.0));
        float beta  = asin(clamp(dot(dir, cloud.eclipticPoleENU.xyz), -1.0, 1.0));

        // Main cone: innerFade clears the sun corona's own falloff (coronaSig maxes at ~0.08 rad
        // above), outerFade closes it out by cloud.zodiacalOuterFadeDeg. The ecliptic-latitude
        // sigma narrows with elongation — wide and low near the sun/horizon, narrowing further
        // out — the real cone shape.
        float innerFade = smoothstep(0.09, 0.17, theta);
        float outerR1   = radians(cloud.zodiacalOuterFadeDeg);
        float outerR0   = outerR1 * 0.75;
        float outerFade = 1.0 - smoothstep(outerR0, outerR1, theta);
        float sigmaNear = radians(cloud.zodiacalWidthDeg);
        float sigmaFar  = sigmaNear * 0.45;
        float sigmaLat  = mix(sigmaNear, sigmaFar, smoothstep(0.0, 1.2, theta));
        float latFalloff = exp(-(beta * beta) / (2.0 * sigmaLat * sigmaLat));
        float zodMain = innerFade * outerFade * latFalloff;

        // Gegenschein: same shape mirrored around the antisolar direction, tight and dim, no
        // separate slider — real zodiacal light's opposition brightening is subtle.
        float thetaAnti = acos(clamp(dot(dir, -sunDir), -1.0, 1.0));
        float outerFadeAnti  = 1.0 - smoothstep(radians(15.0), radians(25.0), thetaAnti);
        float sigmaAnti       = sigmaNear * 0.6;
        float latFalloffAnti  = exp(-(beta * beta) / (2.0 * sigmaAnti * sigmaAnti));
        const float kZodGegenscheinRatio = 0.07;
        float gegenschein = outerFadeAnti * latFalloffAnti * kZodGegenscheinRatio;
        float zodShape = zodMain + gegenschein;

        // Color: pale warm-white, far less saturated than the sun disc/corona (vec3(1.8,0.7,0.2)
        // above) — a faint wash, not a second sun. Mirrors the sun disc's own sunsetT shape for a
        // consistent warm shift near the horizon; the gegenschein (night-side) stays unwarmed.
        float sunsetTZod = clamp(1.0 - (sunDirENU.w - limbZ) / 0.15, 0.0, 1.0);
        vec3  zodColMain = mix(vec3(1.00, 0.98, 0.92), vec3(1.05, 0.88, 0.68), sunsetTZod * 0.5);
        vec3  zodColAnti = vec3(1.0, 0.98, 0.95);
        vec3  zodCol = (zodMain * zodColMain + gegenschein * zodColAnti) / max(zodShape, 1e-4);

        // Visibility chain — same day/night, moonlight, terrain/cloud/moon-disc occlusion shape
        // the Milky Way uses above, minus sunGlareSuppress (see header comment). Pollution reuses
        // the shared per-pixel dome (S2c isotropic-floor shape from sat_flare.comp/updateStars),
        // not the Milky Way's own decoupled mwSuppressEased hysteresis state — real zodiacal light
        // tolerates more light pollution than the Milky Way, so it gets a gentler max-dim ceiling
        // instead of its own eased CPU state.
        float atmFracSkyZ    = 1.0 - clamp((obsEffH - 40000.0) / (100000.0 - 40000.0), 0.0, 1.0);
        float nightFactorSkyZ = clamp(-sunDirENU.w * 5.0, 0.0, 1.0);
        float nightFactorEffZ = mix(pc.skyGlareVisibility, nightFactorSkyZ, atmFracSkyZ);

        float tmZ = clamp(moonDirENU.z / 0.5, 0.0, 1.0);
        float moonBrightZ = tmZ * tmZ * moonDirENU.w;
        const float kZodMoonMaxDim = 0.9;

        // Ray's own tangent-altitude extinction (not the observer-altitude atmFracSkyZ above) —
        // same rayTangentAltM correction documented for aurora/satellites/stars/planets/Milky Way.
        float atmFracExtinctZ = 1.0 - clamp((rayTangentAltM(obsPos, dir) - 40000.0) / (100000.0 - 40000.0), 0.0, 1.0);
        float sinElZ    = clamp(dir.z, 0.0, 1.0);
        float elDegZ    = degrees(asin(sinElZ));
        float airmassZ  = 1.0 / (sinElZ + 0.50572 * pow(elDegZ + 6.07995, -1.6364));
        float extinctMagZ = cloud.extinctionCoeff * (airmassZ - 1.0) * atmFracExtinctZ;
        float extinctionZ = pow(10.0, -0.4 * extinctMagZ);

        // Directional light-pollution dome — same azimuth-sector lookup + isotropic-floor blend
        // as sat_flare.comp/updateStars(), duplicated locally (own view direction, own constant).
        float azZ      = mod(atan(dir.x, dir.y) + 6.283185307, 6.283185307);
        float secFZ    = azZ * (16.0 / 6.283185307) - 0.5;
        int   sec0Z    = int(floor(secFZ));
        float secFracZ = secFZ - float(sec0Z);
        int   sec0wZ   = ((sec0Z % 16) + 16) % 16;
        int   sec1wZ   = (sec0wZ + 1) % 16;
        float elevFalloffZ = 0.35 / (max(dir.z, 0.0) + 0.35);
        float domeAzZ  = mix(lightDome[sec0wZ], lightDome[sec1wZ], secFracZ);
        const float kZodIsotropicFrac = 0.4;
        float domeValZ = clamp(domeAzZ * (kZodIsotropicFrac + (1.0 - kZodIsotropicFrac) * elevFalloffZ), 0.0, 1.0);
        const float kZodPollutionMaxDim = 0.85;

        const float kZodCloudSuppressPower = 2.0;
        float visibilityZod = nightFactorEffZ
                             * (1.0 - moonBrightZ * kZodMoonMaxDim)
                             * extinctionZ
                             * (tSurface > 0.0 ? 0.0 : 1.0) // blocked by terrain/ocean
                             * (moonDiscHit ? 0.0 : 1.0)    // blocked by the Moon's own opaque disc
                             * pow(clamp(cloudBlock, 0.0, 1.0), kZodCloudSuppressPower)
                             * (1.0 - domeValZ * kZodPollutionMaxDim);

        color += zodCol * zodShape * cloud.eclipticPoleENU.w * visibilityZod;
    }

    // ── Moonlight ambient ──────────────────────────────────────────────────────
    float moonEl    = clamp(moonDirENU.z, 0.0, 1.0);
    float moonIllum = moonDirENU.w;
    // Atmosphere weight: glow and ambient fade to zero above the atmosphere.
    float atmosWeight = 1.0 - exp(-odR_cam / 5000.0);
    color += vec3(0.0025, 0.003, 0.004) * moonIllum * moonEl * nightAmt * atmosWeight;

    // ── Moon glow: tight corona + wide diffuse halo (atmosphere-only) ─────────
    if (moonDirENU.z > limbZ - 0.05) {
        vec3  moonDir3  = normalize(moonDirENU.xyz);
        float moonAngle = acos(clamp(dot(dir, moonDir3), -1.0, 1.0));
        float moonFade  = smoothstep(limbZ - 0.006, limbZ + 0.002, moonDirENU.z);

        // Tight inner corona — peaks at disc edge, falls off quickly.
        float corona = exp(-moonAngle * moonAngle / (2.0 * 0.012 * 0.012)) * nightAmt;
        color += hClip * moonFade * corona * vec3(0.92, 0.94, 1.00) * moonIllum * 0.04 * atmosWeight;

        // Wide diffuse halo — scattered moonlight glow, atmosphere-only.
        float scale = 100.0;
        float halo  = exp(-moonAngle * moonAngle / (2.0 * 0.018 * 0.018 * scale * scale));
        color += hClip * moonFade * halo * vec3(0.88, 0.90, 1.00) * moonIllum * 0.012 * atmosWeight;
    }

    // ── Sun disc + atmospheric corona ─────────────────────────────────────────
    if (sunDirENU.w > limbZ - 0.1) {
        float angle      = acos(clamp(cosA, -1.0, 1.0));
        const float kSunAngR = 0.00466; // solar angular radius (~0.267°)
        // Geometric fade: smooth transition as sun centre crosses the geometric limb.
        float geomFade   = smoothstep(limbZ - kSunAngR, limbZ + kSunAngR, sunDirENU.w);
        // Hard gate: terrain/ocean OR genuinely-opaque (>=90%) cloud on THIS fragment's own view
        // ray — same ray the sun disc/corona are drawn along, and the same tSurface/tCloudOcclude
        // pair the moon disc gates on (discFade above). Fixed 2026-07-29: previously `corona` (the
        // wide atmospheric halo, ~10-20x the disc's radius) had NO gate at all — only `discVis`
        // checked tSurface, and neither checked tCloudOcclude — so a mountain or a genuinely opaque
        // cloud deck correctly hid the disc but left its halo glowing right through. The only
        // attenuation either term got was `cloudBlock` (A_total's soft luminance dimming) below,
        // which fades but never reaches zero for real cloud, so it read as "doesn't occlude."
        float sunGate    = (tSurface > 0.0 || tCloudOcclude >= 0.0) ? 0.0 : 1.0;
        // Disc pixel: hard-clipped by terrain/ocean/opaque-cloud hit for this fragment direction.
        float discVis    = (1.0 - smoothstep(0.007, 0.010, angle)) * sunGate;
        // Sunset shift: redden and widen corona as sun approaches the limb.
        float sunsetT    = clamp(1.0 - (sunDirENU.w - limbZ) / 0.15, 0.0, 1.0);
        vec3  sunCol     = mix(vec3(1.5, 1.3, 1.0), vec3(1.8, 0.7, 0.2), sunsetT * 0.7);
        float coronaSig  = mix(0.035, 0.08, sunsetT * sunsetT);
        float corona     = exp(-angle * angle / (2.0 * coronaSig * coronaSig)) * sunGate;
        // Remaining soft dimming (cloudBlock, A_total's luminance) still applies on top for thin/
        // translucent cloud the hard gate above doesn't trip on — a hazy, dimmed disc through mist
        // is correct; sunGate only handles the "actually opaque" case that dimming alone can't.
        color += (discVis * geomFade * sunCol + corona * geomFade * sunCol * 0.12) * cloudBlock;
    }

    // ── Camera lens flares (post-tonemap) ─────────────────────────────────────
    // Applied after all physics-based rendering so they read as pure camera
    // optical artifacts on top of the scene.
    //
    // UV space: x in [-0.5*aspect, +0.5*aspect], y in [-0.5, +0.5].
    //
    // Fragment projection:
    //   fragCamDir = mat3(skyView) * enuDir  (camera-space ray, z ~= -1)
    //   fragUV = vec2(camDir.x, -camDir.y) * invTanHF2
    //   No perspective divide since z ~= -1 throughout the fullscreen tri.
    //
    // Source projection (satellite or sun):
    //   satCam = mat3(skyView) * normalize(enu)
    //   satUV  = vec2(satCam.x, -satCam.y) / (-satCam.z * tanHF * 2)
    //   Perspective divide by -satCam.z is required here.
    {
        float tanHF     = tan(pc.fovYRad * 0.5);
        float invTanHF2 = 1.0 / (tanHF * 2.0);

        vec3 fragCamDir = mat3(pc.skyView) * enuDir;
        vec2 fragUV     = vec2(fragCamDir.x, -fragCamDir.y) * invTanHF2;

        vec3 flareAccum = vec3(0.0);

        // (Satellite lens flares — the per-pixel loop over glowBuf.flareEntries — lived here.
        // Deleted in the flare architecture overhaul: satellites now get their soft glow+godray
        // treatment from a render-to-texture + blur/streak pipeline, composited separately in
        // flare_composite.frag. Only the sun keeps a hand-authored lensFlare() ghost/corona call,
        // per explicit user decision — see FlareSourcePC's comment in SatelliteSim.h.)

        // ── Sun lens flare ──────────────────────────────────────────────────────
        // Gate on limbZ (sin of geometric limb depression, already accounts for observer
        // altitude) so the flare correctly persists when the sun is visible past the
        // curved Earth from orbit — not just when sunDirENU.w > 0.
        if (pc.sunDirENU.w > limbZ - 0.05) {
            float above        = pc.sunDirENU.w - limbZ;
            float sunIntensity = 10.0 * clamp(above / 0.5, 0.0, 1.0);
            vec3 sunCam = mat3(pc.skyView) * normalize(pc.sunDirENU.xyz);
            if (sunCam.z < -0.01) {
                vec2 sunUV    = vec2(sunCam.x, -sunCam.y) / (-sunCam.z * tanHF * 2.0);
                float sunFade = clamp(above * 8.0, 0.0, 1.0);
                vec3  sunTint = vec3(1.4, 1.2, 0.9);
                // Same cloud-occlusion fix as the satellite flares above — sampled at the SUN's
                // own screen position, not this fragment's (`cloudBlock`, used for the sun disc
                // itself right above, is deliberately not reused here for the same reason).
                vec2  sunScreenUV  = vec2(sunUV.x / pc.aspect + 0.5, sunUV.y + 0.5);
                float sunCloudOccl = dot(texture(cloudTargetB, sunScreenUV).rgb, vec3(1.0/3.0));
                // Terrain occlusion, same technique/reasoning as the satellite loop above — a
                // local ridge or mountain blocking the sun's own direction (distinct from limbZ's
                // Earth-curvature horizon test above) now correctly hides its flare too.
                float sunTerrainOccl = (texture(sceneDepthTex, sunScreenUV).r >= kNoSurfaceT * 0.5) ? 1.0 : 0.0;
                flareAccum += lensFlare(fragUV, sunUV, sunIntensity, 2.0) * sunTint * sunFade * 0.45
                            * sunCloudOccl * sunTerrainOccl;
            }
        }

        // Lens flares are a screen-space camera-optics artifact, not light literally travelling
        // to each pixel — source visibility is already handled per-source above (sun: limbZ
        // gate at its `if`; satellites: satDir.z horizon cull + camera-facing check). Do NOT
        // gate flareAccum by this fragment's OWN terrain hit (tHit): that tests whether THIS
        // pixel's unrelated view ray hit land, not whether the source is occluded. The old
        // `tHit > 0.0 ? 0.0 : 1.0` mask zeroed the flare's additive glow on every terrain pixel
        // anywhere on screen — invisible at ground level (terrain only fills the lower frame),
        // but at LEO twilight, where terrain fills most of the screen under a large sun flare,
        // it hard-clipped the raymarched terrain silhouette out of the middle of the glow.
        color += flareAccum;
    }

    outColor = vec4(color, 1.0);

    // Terrain/ocean occlusion depth for subsequent satellite/star passes.
    // Satellites and stars are drawn with gl_Position.z = 0.5 (fixed) and tested with LESS.
    // Close surface hits write [0, 0.5) so they block those overlays; sky writes 1.0 so they pass.
    // The 150 km cap prevents space-view terrain from incorrectly culling near satellites.
    // tSeaLvl covers ocean pixels that have no terrain hit but still block satellites.
    const float kOcclusionCap = 150000.0;
    float tOcclude = (tHit >= 0.0) ? tHit : tSeaLvl;
    // Opaque cloud also occludes satellites/stars behind it (cloud is above terrain so
    // tCloudOcclude is only used when no terrain/ocean is closer).
    if (tOcclude < 0.0 && tCloudOcclude >= 0.0) tOcclude = tCloudOcclude;
    gl_FragDepth = (tOcclude >= 0.0 && tOcclude < kOcclusionCap)
                   ? tOcclude / (kOcclusionCap * 2.0)
                   : 1.0;
}
