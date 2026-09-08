#pragma once
#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE // see VulkanContext.h — must be repeated at every raw glfw3.h include site
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../Simulation.h"
#include "../UIRenderer.h" // WindowChrome — used by member state below (needs complete type)

// Forward declaration only — savePerfSnapshot/buildPerfSnapshotJson are the sole users and both
// live in SatelliteSimUI.cpp, which includes the real header. Pulling all of nlohmann/json.hpp in
// here would put it in every TU that touches this simulation for two method signatures.
#include <nlohmann/json_fwd.hpp>

#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

// ── Maximum satellites per frame ──────────────────────────────────────────────
static constexpr uint32_t MAX_SATELLITES = 10'000'000;

// ── Satellite attitude model ───────────────────────────────────────────────────
enum class AttitudeMode
{
    NadirPointing,     // flat face toward Earth (Starlink bus/antenna) — brief intense flares
    SunTracking,       // panel normal tracks sun for power — opposition flares
    Tumbling,          // uncontrolled random tumble — chaotic flashes (debris)
    Perpendicular,     // 90° to primary surface in the nadir plane — along orbital track
                       // (secondary only: normal = cross(surfN0, satNadir))
    AntiNadir,         // facing away from Earth center — deep-space-pointing radiator panels
                       // (secondary only: normal = -satNadir)
                       // Brightest to observers at the satellite's horizon; nearly invisible
                       // to observers directly beneath (at satellite's zenith), because the
                       // radiator face points away from them toward cold space.
    FlatMirror45,      // flat mirror oriented to reflect sunlight straight toward Earth center.
                       // Normal = normalize(sunDir + satNadir) — bisects incoming sun and
                       // outgoing nadir directions.  The reflected beam hits the ground below
                       // the satellite; angle between mirror and nadir ≈ 45° when sun is
                       // on the orbital horizon.  Models space-mirror illumination proposals
                       // (e.g. Reflect Orbital 55m mirrors).
    TargetedReflector, // as FlatMirror45 but aimed at a specific ground point on the terminator
                       // rather than straight down.  Target parametrized by SatOrbit::targetTerminatorAngle
                       // (radians along the terminator great-circle).  Each satellite's mirror
                       // normal is: normalize(sunDir + toTarget), so that reflected sunlight
                       // hits the chosen surface spot.  Multiple satellites sharing the same
                       // angle converge their beams onto one ground location — focused illumination.
    KnifeEdge,         // roll around the along-track (velocity) axis to put the sun edge-on
                       // to the flat panel, minimising its reflective cross-section.
                       // Picks whichever of the two edge-on orientations requires less roll
                       // from nadir, then clamps to ±kKnifeMaxRollDeg (solar panel gimbal limit).
                       // Models the SpaceX roll-angle adjustment adopted in 2020 (Mallama 2023):
                       // ~90% brightness reduction at standard distance when unclamped.
    SunPerp,           // normal = normalize(cross(sunDirECI, satNadir))
                       // Panel is edge-on to the sun AND edge-on to nadir; normal points
                       // perpendicular to the sun-nadir plane.  Used for thermal radiator panels
                       // on nadir-locked buses (e.g. AI1 datacenter): the bus yaws so solar
                       // panels face the sun, which constrains the hard-mounted radiators to
                       // this orientation.  irr = |dot(sun, normal)| = 0 always — the radiator
                       // intentionally never receives direct sunlight (correct thermal design).
                       // Visual contribution is through the overall diffuse scatter parameter.
    SunTrackingTilted, // flare mitigation: as SunTracking, but the sun-facing normal is pitched
                       // away from nadir toward zenith by the global flareMitigationTiltDeg
                       // setting, so the specular lobe (and the Earthshine reflect term in
                       // sat_flare.comp) point up and away from the ground instead of straight
                       // down at twilight. Pure rotation of sunDirECI within the plane it spans
                       // with zenith (-satNadir), so dot(result, sunDirECI) == cos(tiltDeg)
                       // exactly — that cosine IS the operator's real power loss vs. optimal
                       // sun-facing (Lambert's cosine law), used to compute the "power output"
                       // readout in the satellite selection panel. tiltDeg=0 is bit-identical to
                       // plain SunTracking. Used by the SpaceX AI Sats (datacenter) primary
                       // surface — see loadHardcoded()/constellations.json.
};

// ── Display unit system (right HUD panel altitude readout, settings Display tab) ──
enum class UnitSystem
{
    Metric,   // km
    Imperial, // mi
};

// NEW-7 (RELEASE_v1_1_PLAN.md) — Settings > Display "Frame limiter". Off/Cap30/Cap60/Cap120 all
// use VkPresentModeKHR MAILBOX/IMMEDIATE (uncapped submission) and rely on App::mainLoop's
// sleep-based pacing (see Simulation::targetFpsCap) for the numeric caps; VSync uses FIFO and
// needs no manual pacing at all. See VulkanContext::presentModePreference for the present-mode
// side of this.
enum class FpsCapMode
{
    Off, // uncapped, present mode IMMEDIATE (tearing allowed) — max perf regardless of comfort
    Cap30,
    Cap60,
    Cap120,
    VSync, // default — FIFO, paced by the display's own refresh rate
};

// UC1 (RELEASE_v1_1_PLAN.md) — Settings > Display graphics preset. Applying a named preset
// (anything but Custom) overwrites debugDisableMask, renderScale, and the "advanced" Clouds/
// Ocean/Terrain/Aurora sliders wholesale — see SatelliteSim::applyGraphicsPreset for the table.
// Custom is not user-selectable directly; it is set automatically the instant any of those
// advanced sliders is edited by hand (see buildCloudSliderRows), and simply means "trust
// whatever is currently loaded/set — don't overwrite it with a preset table."
enum class GraphicsPreset
{
    Planetarium, // v1.0 experience: flat textured Earth, stars, satellites, atmosphere. No clouds.
    Low,         // integrated graphics / old laptops — flat 2D cloud paste, tight terrain reach
    Medium,      // mainstream discrete GPU — volumetric clouds/terrain at reduced budgets
    High,        // today's tuned defaults
    Ultra,       // uncapped for showcase/screenshots
    Custom,      // user has hand-edited an advanced slider since the last named preset was applied
    Potato,      // below Planetarium — very weak GPUs / MoltenVK translation. Atmosphere scattering,
                 // the scene-depth pass, and the beam/sky-glow loops are ALL knocked out; 0.5 render
                 // scale. Appended after Custom on purpose: the persisted int index (saveSettings
                 // writes (int)graphicsPreset) of every pre-existing preset stays unchanged.
};

// ── Orbit distribution type ────────────────────────────────────────────────────
enum class OrbitDistribution
{
    Walker,      // regular Walker constellation: numPlanes planes × perPlane satellites
    RandomShell, // randomly distributed: random RAAN, random incl in [0, incl], jittered alt
    Disk,        // ring or concentric disk in a fixed orbital plane (incl + raan)
                 // Set alignTerminator=true to auto-derive the plane from sunDirECI
};

// ── One reflective surface of a satellite ─────────────────────────────────────
// A SatelliteType is composed of a primary surface plus an optional secondary
// surface (e.g. radiator panels perpendicular to solar panels) and an optional
// isotropic diffuse floor (structural body scatter).
struct SurfaceSpec
{
    AttitudeMode attitude; // how the surface normal is oriented each frame
    float specExp;         // specular exponent (0 = Lambertian diffuse)
    float weight;          // contribution weight relative to primary (0 = disabled)
};

// ── Per-type satellite parameters (CPU-side, drives GpuSatInput fields) ───────
struct SatelliteType
{
    std::string name;
    glm::vec3 baseColor;   // visual tint
    float crossSectionM2;  // total reflective area (m²); brightness ∝ sqrt(area/10)
    SurfaceSpec primary;   // always active (solar panels, antenna face, etc.)
    SurfaceSpec secondary; // optional second surface — set weight=0 to disable
    float diffuse;         // isotropic Lambertian floor: always visible fraction [0,1]
                           // models structural body scatter; applied after litFactor
    float mirrorFrac;      // fraction of primary surface that is near-perfect mirror [0,1]
                           // adds ultra-narrow specular spike (MIRROR_BOOST×) on top of Phong lobe
                           // 0.0 = no mirror peak; 0.05 = Starlink; 0.15 = ISS solar panels
};

// ── Constellation descriptor ───────────────────────────────────────────────────
// One entry per shell. Multiple shells can share the same SatelliteType.
// orbitStart/Count are filled by initConstellation() — do not set manually.
struct ConstellationConfig
{
    std::string name;
    float altM;       // orbital altitude above surface (meters)
    float incl;       // Walker: fixed inclination; RandomShell: max inclination (radians)
    int numPlanes;    // Walker: number of planes; other: total satellite count
    int perPlane;     // Walker: satellites per plane; other: ignored (use numPlanes as total)
    uint32_t typeIdx; // index into satTypes[]
    bool enabled;     // visibility toggle (hot-swappable)
    OrbitDistribution distribution = OrbitDistribution::Walker;
    float altJitterM = 0.0f;      // RandomShell: ±altitude jitter; Disk: ±per-satellite alt scatter
    float raan = 0.0f;            // Disk: orbital plane RAAN (radians); ignored if alignTerminator
    bool alignTerminator = false; // Disk: derive incl+raan from sunDirECI at init time
    int numRings = 1;             // Disk: number of concentric rings (1 = single ring)
    float ringSpacingM = 0.0f;    // Disk: altitude step between consecutive rings (meters)
    bool highlight = false;       // highlight mode: show all sats at fixed brightness, ignoring lighting
    // Populated by initConstellation():
    uint32_t orbitStart = 0; // first index into satOrbits[]
    uint32_t orbitCount = 0; // number of orbits belonging to this constellation
};

// ── GPU data structures ───────────────────────────────────────────────────────
// std430 packing: vec3 alignment=16 size=12, so vec3+float fills one 16-byte block.
// Five vec3+float blocks (80 bytes) + one float4 tail = 80 bytes total.
//
// Byte map:
//   [  0] eciRelPos (vec3) + range (float)         — position data
//   [ 16] surfN0    (vec3) + elevation (float)      — primary surface normal
//   [ 32] surfN1    (vec3) + specExp0 (float)       — secondary surface normal
//   [ 48] baseColor (vec3) + specExp1 (float)       — colour + secondary specular
//   [ 64] crossSection + w1 + diffuse + _pad (float4) — photometric scalars
//   Total: 80 bytes

struct GpuSatInput
{
    glm::vec3 eciRelPos; // observer-relative ECI position (meters)
    float range;         // distance (meters)
    glm::vec3 surfN0;    // primary surface normal in ECI (attitude-dependent unit vector)
    float elevation;     // elevation above local horizon (radians), pre-computed on CPU
    glm::vec3 surfN1;    // secondary surface normal in ECI (radiators, body, etc.)
    float specExp0;      // primary surface specular exponent (0 = Lambertian)
    glm::vec3 baseColor; // satellite tint from SatelliteType
    float specExp1;      // secondary surface specular exponent (0 = Lambertian)
    float crossSection;  // sqrt(crossSectionM2 / 10.0): area brightness scale (~1 = 10 m²)
    float w1;            // secondary surface weight relative to primary (0 = disabled)
    float diffuse;       // isotropic Lambertian floor — structural body scatter [0,1]
    float mirrorFrac;    // fraction of primary surface that is near-perfect mirror [0,1]
};
static_assert(sizeof(GpuSatInput) == 80, "GpuSatInput layout mismatch");

struct GpuSatVisible
{
    glm::vec3 skyDir;     // unit vector in ENU (x=East, y=North, z=Up)
    float flareIntensity; // [0, 1+]
    glm::vec3 baseColor;  // satellite tint
    float angularSize;    // point sprite size hint (pixels)
};
static_assert(sizeof(GpuSatVisible) == 32, "GpuSatVisible layout mismatch");

// Mercury..Uranus — the naked-eye-relevant classical planets plus Uranus (mag ~5.7-5.9, right at
// the edge of the star catalog's own mag-6.5 floor). Neptune excluded: never naked-eye (~mag 7.8).
enum PlanetId
{
    kMercury = 0,
    kVenus,
    kMars,
    kJupiter,
    kSaturn,
    kUranus,
    kPlanetCount
};
extern const char *const kPlanetNames[kPlanetCount];

// Per-planet ephemeris state, recomputed every frame in updatePositions() from the Keplerian
// elements in SatelliteSim.cpp (kPlanetElements/keplerEclipticPos) — see "Subsystem: Planets" in
// CLAUDE.md. Distinct from GpuSatVisible: this is the astronomy (direction/distance/phase), not
// the render-ready record (brightness/color/size), which updatePlanets() derives from it each
// frame into planetBuf.
struct PlanetState
{
    glm::vec3 eciDir{0, 1, 0};  // unit vector from Earth toward the planet, ECI
    float distanceAU = 0.0f;    // Earth-planet distance (AU)
    float sunDistAU = 0.0f;     // Sun-planet distance (AU)
    float phaseAngleDeg = 0.0f; // Sun-Planet-Earth angle (illumination phase)
};

// Compute push constants (must match sat_flare.comp push_constant block exactly).
// GLSL std430 layout, vec3 aligned to 16 bytes:
//   enuX/Y/Z (vec4): offsets 0,16,32
//   sunDirECI (vec3): offset 48  (48 is 16-aligned ✓)
//   satCount  (uint): offset 60
//   obsECI    (vec3): offset 64  (64 is 16-aligned ✓)
//   pad       (float): offset 76
//   total: 80 bytes
struct SatFlarePC
{
    glm::vec4 enuX;      // East  basis in ECI (w unused)
    glm::vec4 enuY;      // North basis in ECI (w unused)
    glm::vec4 enuZ;      // Up    basis in ECI (w unused)
    glm::vec3 sunDirECI; // unit vector toward sun in ECI
    uint32_t satCount;
    glm::vec3 obsECI; // observer ECI position (meters) for shadow test
    float elevCutoff; // sin(Earth-limb angle) — horizon cull threshold (≤ -0.01)
    // Photometry tuning — runtime-adjustable via the settings window.
    float brightnessScale; // global flux multiplier (mirrors BRIGHTNESS_SCALE in shader)
    float daySuppression;  // sky background suppression ratio (mirrors DAY_SUPPRESSION)
    float mirrorBoost;     // mirror peak multiplier (mirrors MIRROR_BOOST)
    float visThresh;       // visibility cull threshold (mirrors VIS_THRESH)
    float highlightFlare;  // fixed flare for constellation census (mirrors HIGHLIGHT_FLARE)
    float extinctionCoeff; // atmospheric extinction, magnitudes per airmass (reuses the slot that
                           // was lightPollution — see SatelliteSim::updateLightPollutionDome for
                           // why that moved to the lightDomeBuf SSBO instead of push-constant space)
    float moonSuppression; // sky background suppression ratio from moonlight (mirrors daySuppression's
                           // role, much smaller in practice — moon is ~14 magnitudes dimmer than the sun)
    float pad0;            // reserved — pads moonDirECI to 16-byte (vec3) alignment
    glm::vec3 moonDirECI;  // unit vector from Earth toward Moon in ECI
    float sunRefIntensity; // was pad1 — S3 (RELEASE_v1_1_PLAN.md): soft ceiling reference so no
                           // satellite's effectFlare can render brighter than the sun; mirrors
                           // FlareSourcePC's own sunRefIntensity (SatelliteSim::sunFlareRefIntensity)
}; // total: 128 bytes
static_assert(sizeof(SatFlarePC) == 128, "SatFlarePC layout mismatch");

// Sky-background draw push constants (sat_sky.vert + sat_sky.frag / _lite / _minimal, via
// skyBgPipeLayout). Exactly 128 bytes — the Vulkan-guaranteed maxPushConstantsSize floor, which
// the oldest AMD integrated parts sit right at. Everything that used to trail past offset 128 here
// (debugDisableMask, screenSizePx, skyGlareVisibility, the four beam knobs, mwSuppressEased) was
// per-frame-uniform, so it moved into the CloudParams frame UBO (cloud_params.glsl / GpuCloudParams
// — sat_sky.frag already binds it) rather than a push constant. See the "Push-constant relief"
// block at the end of GpuCloudParams. The two per-draw flags that lived here (noTwinkle,
// manualTerrainTest) were only ever read by the point shaders, which now carry PointDrawPC below.
// GLSL std430 layout:
//   skyView    (mat4):  offset 0
//   fovYRad    (float): offset 64
//   aspect     (float): offset 68
//   gmst       (float): offset 72  — Greenwich Mean Sidereal Time (radians)
//   waveTime   (float): offset 76  — sim seconds; scales with time warp, pauses when paused
//   sunDirENU  (vec4):  offset 80   xyz=direction, w=sin(elevation)
//   moonDirENU (vec4):  offset 96   xyz=moon dir in ENU, w=illuminated fraction
//   obsECEFDir (vec4):  offset 112  xyz=observer ECEF unit vector, w=obsHeightOffset (m)
//   total: 128 bytes
struct SatDrawPC
{
    glm::mat4 skyView;         // ENU → camera space
    float fovYRad;             // vertical field of view (radians)
    float aspect;              // viewport width / height
    float gmst;                // Greenwich Mean Sidereal Time (radians)
    float waveTime;            // sim seconds for wave animation
    glm::vec4 sunDirENU;       // sun direction in ENU (xyz unit vec, w = sin(elevation))
    glm::vec4 moonDirENU;      // moon direction in ENU (xyz unit vec, w = illuminated fraction)
    glm::vec4 obsECEFDir;      // xyz = observer ECEF unit vector (lets sat_sky.frag convert ENU hit →
                               // ECEF → geographic lat/lon for texture UV); w = obsHeightOffset (m,
                               // user altitude offset above terrain — maxed with the GPU's own
                               // ground-height lookup as obsEffH). Despite the field's original "w
                               // unused" comment (stale — corrected here), it IS read.
}; // total: 128 bytes
static_assert(sizeof(SatDrawPC) == 128, "SatDrawPC layout mismatch");

// Point-sprite draw push constants — satellites (sat_point.vert/.frag via drawPipeLayout) AND
// stars/planets (star_point.vert/.frag via starPipeLayout). A separate, smaller struct from
// SatDrawPC so both point pipeline layouts declare a push-constant range that fits the 128-byte
// floor while still carrying the two per-draw flags (noTwinkle, manualTerrainTest) that genuinely
// cannot go in the frame UBO — they differ between the star draw and the planet draw, and between
// the live draw and the long-exposure trail draw, all within one frame. Each point shader declares
// only the prefix it reads (a push_constant block only has to be a contiguous PREFIX of the pushed
// bytes). GLSL std430 layout:
//   skyView    (mat4):  offset 0    — both verts
//   fovYRad    (float): offset 64   — both verts
//   aspect     (float): offset 68   — both verts
//   waveTime   (float): offset 72   — star_point.vert (twinkle phase)
//   noTwinkle  (float): offset 76   — star_point.vert; 1 on the planet draw only
//   moonDirENU (vec4):  offset 80   — star_point.vert (moon-disc star cull)
//   obsECEFDir (vec4):  offset 96   — star_point.vert (w = obsHeightOffset for atmFrac)
//   screenSizePx (vec2): offset 112 — sat_point.frag / star_point.frag (cloud/depth UV; always
//                                     ctx.swapExtent — point draws never render at renderScale<1)
//   debugDisableMask (uint): offset 120 — sat_point.frag (knockout bit 4096 only)
//   manualTerrainTest (float): offset 124 — both frags; 1 on the trail draw only
//   total: 128 bytes
struct PointDrawPC
{
    glm::mat4 skyView;
    float fovYRad;
    float aspect;
    float waveTime;
    float noTwinkle;
    glm::vec4 moonDirENU;
    glm::vec4 obsECEFDir;
    glm::vec2 screenSizePx;
    uint32_t debugDisableMask;
    float manualTerrainTest;
}; // total: 128 bytes
static_assert(sizeof(PointDrawPC) == 128, "PointDrawPC layout mismatch");

// ── Push constants for cloud_march.comp (half-res cloud compute pass, C15-perf) ──────────────
// Matches the layout(push_constant) block in cloud_march.comp exactly. A separate struct from
// SatDrawPC (own pipeline layout, own push-constant range) — carries only the fields the moved
// cloudMarch/cirrusMarch bodies actually use, plus obsEffH (CPU-computed; the compute shader has
// no elevation-texture lookup of its own, see recordCompute()).
//
// Exactly 128 bytes — the 128-byte maxPushConstantsSize floor (see SatDrawPC). The five tail
// fields that used to sit past offset 128 (debugDisableMask, beamMaxRangeM, showBeamDebugRays,
// beamSkyGlowGain, cloudShadowRangeM) are all per-frame-uniform and moved into the CloudParams
// UBO this shader already binds — see the "Push-constant relief" block at the end of
// GpuCloudParams. cloud_march.comp reads them as cloud.dbgDisableMask / cloud.beamMaxRangeM /
// cloud.showBeamDebugRays / cloud.beamSkyGlowGain / cloud.cloudShadowRangeM.
struct CloudMarchPC
{
    glm::mat4 skyView;
    float fovYRad;
    float aspect;
    float waveTime;
    float obsEffH;
    glm::vec4 sunDirENU;
    glm::vec4 moonDirENU;
    glm::vec4 obsECEFDir;
}; // total: 128 bytes.
static_assert(sizeof(CloudMarchPC) == 128, "CloudMarchPC layout mismatch");

// ── Reflect-Orbital beam->cloud light sources (host-visible) ─────────────────────────────────
// 2026-08-09, fourth design for this feature. First was a per-target CPU aggregation anchored at
// the idealized targetENU (retired same day once beam_self_march.comp made per-beam values real).
// Second was a true per-beam screen-space glow folded into cloud_march.comp's per-pixel debug-ray
// loop (distance from the CAMERA's own view ray to the beam's line) — cheap, but WRONG: that
// distance's iso-contours are not soft blobs, they're hyperbola-like curves, which read in-app as
// large white rings intersecting the beams. Third was a true per-beam list fed forward into the
// per-sample volumetric march (the correct architecture, kept since) — but at a small cap (16) the
// cloud effect was barely visible, and getting "appreciable" coverage needed raising the cap to
// 512, which tanked frame rate (the per-sample loop's cost is linear in the cap).
//
// This fourth version keeps the third's per-sample architecture and its two geometric fixes
// (`posENU` is the beam's REAL ground intersection — `satENU + reflectDirENU` traced to `R_EARTH`,
// computed on the CPU below via the same rotation-invariant local-frame `raySphere` trick the GPU
// side uses, no ECEF conversion needed; `dirToSource` is the beam's REAL direction, ground toward
// satellite), but adds clustering: a beam LOCKED onto its target (`aimErrorRad` below a threshold
// — the same convergence signal `cloud_march.comp`'s own `aimFade` already uses) is folded into a
// shared per-target cluster via intensity-weighted running average (same technique already used
// for the blockAltM/blockOpacity flicker fix); a beam still SLEWING keeps its own individual slot,
// since its geometry is genuinely unique right now. Per explicit user direction: converged beams
// at a busy site are redundant with each other and should combine; transiting beams shouldn't.
// This recovers full coverage — the effective distinct-light count is bounded near the active-
// target count (times however many DISTINCT approach directions are simultaneously converged on
// one target — see the direction-similarity gate on the cluster match, added same day: merging by
// ground position alone let satellites arriving from very different parts of the sky get blended
// into one nonsensical averaged direction) — not total active-satellite count. In-app testing
// after the clustering fix showed this cap is no longer the cost driver it was as a flat
// (non-clustered) list, so it was raised back to 512 for headroom.
//
// 2026-08-12 — this struct is UNCHANGED, but it is no longer built from scratch each frame. It is
// now the per-frame EMIT of the persistent TrackedBeamLight pools below: same clustering intent,
// same two reserved budgets, same ordering (clusters then individuals), but each entry has a stable
// cross-frame identity and its values are temporally eased rather than snapped. The parts of the
// history above that describe the intra-frame epsilon match and the running-average direction gate
// are retained as background — that mechanism is gone. See TrackedBeamLight for what replaced it
// and why.
//
// Struct must match BeamCloudLight/BeamCloudLightBuf in cloud_march.comp exactly — including
// kMaxCloudBeamLights itself, hand-duplicated there (not shared via a header); a mismatch
// silently over/under-reads the buffer with no compile or validation error.
static constexpr int kMaxCloudBeamLights = 512;
// 2026-08-09 (in-app finding: "I do not see cloud lighting on singular beams anymore" after
// clustering became stable): a converged cluster's intensity is the SUM of every beam folded into
// it, so a busy target with dozens of locked satellites is legitimately dozens of times brighter
// than any one transiting beam's own single intensity — that part is physically correct, not a
// bug. But the CPU build below used one shared top-K-by-intensity eviction pool for both
// categories: once total distinct entries (clusters + individuals) reached kMaxCloudBeamLights,
// eviction always removes the globally weakest entry, and a lone transiting beam is essentially
// always weaker than an established multi-satellite cluster — so under any capacity pressure,
// individual beams are evicted first, potentially entirely, while clusters are never at risk.
// Split the budget into two reserved, independently-evicted pools so transiting beams keep a
// guaranteed floor of visibility regardless of how many targets are simultaneously busy: clusters
// are bounded near the real target count (kNumReflectorTargets capacity is 201, so 256 leaves
// headroom), individuals get the rest.
static constexpr int kMaxClusterCloudLights = 256;
static constexpr int kMaxIndividualCloudLights = kMaxCloudBeamLights - kMaxClusterCloudLights;
struct GpuBeamCloudLight
{
    glm::vec3 posENU;      // meters, observer-relative — REAL ground intersection of this beam
    float intensity;       // groundIrradiance * beamGain for this ONE satellite
    glm::vec3 dirToSource; // unit direction from posENU toward the satellite (real beam direction)
    float footprintRadM;
    float blockAltM;    // altitude (m) where THIS beam's own path first drops below 50% transmittance
    float blockOpacity; // 0 = clear column, 1 = fully opaque
    float pad0, pad1;   // std430 array-of-vec4-pairs alignment
};
static_assert(sizeof(GpuBeamCloudLight) == 48, "GpuBeamCloudLight layout mismatch");

struct GpuBeamCloudLights
{
    uint32_t count;
    uint32_t pad0, pad1, pad2;
    GpuBeamCloudLight entries[kMaxCloudBeamLights];
};
static_assert(sizeof(GpuBeamCloudLights) == 16 + kMaxCloudBeamLights * 48, "GpuBeamCloudLights layout mismatch");

// ── TrackedBeamLight: persistent, cross-frame cloud-light identity (2026-08-12) ────────────────
// CPU-only — this is the state the GPU-facing GpuBeamCloudLights above is DERIVED from each frame,
// not a mirror of it.
//
// Why this exists. Until now the light list was rebuilt from scratch every frame, and a cluster's
// identity was EMERGENT: it was seeded by whichever beam the scan happened to reach first at a
// target, and admitted members by comparing them against a running partial average that changed as
// members were added. That partition is discontinuous in its own inputs — when the seed satellite
// drops out (a lock-window reassignment, or setting below minBeamElevSin) the survivors repartition
// from scratch, so cluster count, every direction and every summed intensity can all change in one
// frame. Since posENU/dirToSource/blockAltM/blockOpacity are all intensity-weighted means, and the
// cloud phase function is very sharply forward-peaked (hgG~0.99), that reads as lights popping in
// and out and re-aiming instantly. See BEAM_CLOUD_PLAN.md for the four in-app test rounds that
// established this, and for the 2026-08-11 attempt that tried to fix it purely by adding a fade
// on top — reverted, because easing a slot whose MEANING changes underneath it does not help, and
// recovering identity by proximity-searching a pool cost O(rawClusters x 256) per frame (20 FPS).
//
// The fix is to DECLARE identity instead of deriving it, which is possible now that sat_orbit.comp
// carries its `bestIdx` through as GpuReflectBeam::targetIdx:
//   * a CLUSTER is keyed by (targetIdx, direction bucket). The bucket is a FIXED quantization of
//     the beam's approach direction in the TARGET SITE's own local frame (reflectorSiteEnu*[]) —
//     so it depends only on that beam's own geometry, never on scan order, on a seed, or on what
//     else is in the cluster, and never on where the observer is. Nothing repartitions when a
//     member leaves.
//   * an INDIVIDUAL (transiting) beam is keyed by its originating satellite's dispatch index,
//     which sat_orbit.comp already guarantees is stable (GpuReflectBeam::debugPad).
// Both are exact integer keys, so cross-frame matching is an O(1) hash lookup rather than a
// proximity search, and a beam crossing the converged/transiting boundary simply moves between two
// eased entries — a crossfade, not a pop, with no hysteresis needed.
//
// Geometry is stored in EARTH-FIXED ECEF, in double precision, and re-projected into the current
// frame's observer ENU only at emit time. This is the single most important detail: a ground site
// is stationary in ECEF, so an entry that goes unmatched for the whole release window cannot drift,
// regardless of how far or fast the observer moves. The 2026-08-11 attempt burned three rounds
// rediscovering this — it stored observer-relative ENU and tried to correct it incrementally with
// rebase(), which is rotation-only and built for exactly ONE frame of GPU-readback staleness. Do
// not reintroduce observer-relative storage here. (doubles, not floats, because ECEF magnitudes are
// ~6.4e6 m and these values are differenced against the observer's own position at emit time.)
//
// Accepted: the eased list is history-dependent, so it is not bit-reversible under time reversal,
// unlike the orbital pipeline. Same class and precedent as skyGlareEased/mwSuppressEased — a
// visual smoothing of a derived quantity, not simulation state.
struct TrackedBeamLight
{
    // 0 = free slot. Real keys always set a high bit so that (targetIdx=0, bucket=0) and
    // (satIdx=0) can't collide with "free".
    uint32_t key = 0;
    // Eased state — what actually gets emitted.
    glm::dvec3 posECEF{0.0}; // Earth-fixed ground position of this light
    glm::dvec3 dirECEF{0.0}; // Earth-fixed unit direction, ground -> satellite
    float easedIntensity = 0.0f;
    float easedFootprintRadM = 0.0f;
    float easedBlockAltM = 0.0f;
    float easedBlockOpacity = 0.0f;
    // Per-frame accumulators (this frame's target values), zeroed at the top of every build. A slot
    // with tgtIntensity == 0 after the scan had no contributing beam this frame and is fading out
    // while HOLDING its geometry — which is exactly why the geometry has to be drift-free.
    float tgtIntensity = 0.0f;
    float tgtFootprintRadM = 0.0f;
    glm::dvec3 tgtPosSum{0.0}; // intensity-weighted
    glm::dvec3 tgtDirSum{0.0}; // intensity-weighted
    float tgtAltSum = 0.0f;
    float tgtOpacitySum = 0.0f;
};
// Power-of-two open-addressed index (key -> slot), rebuilt from the live slots at the top of every
// build. Rebuilding is O(live) <= 256 and sidesteps tombstones entirely, which is the only fiddly
// part of open addressing with deletion. Sized well above the pool so probe chains stay short.
static constexpr int kTrackedLightHashSize = 1024;
// A slot is retired once its eased intensity falls below this AND nothing is feeding it. Keeps
// fully-faded entries from occupying pool slots (and GPU light-list entries) indefinitely.
static constexpr float kTrackedLightEpsilon = 1e-4f;

// (CloudShadowPC lived here — push constants for cloud_shadow.comp's 128x128 grid, including the
//  shadowResidualM texel-snapping term that stopped shadows swimming as the observer moved. The
//  whole pass is gone; the per-pixel replacement in cloud_march.comp needs no snapping because
//  its value is a function of the world point being shaded, not of the camera's position.)

// ── Push constants for beam_self_march.comp (2026-08-09) ─────────────────────────────────────
// Replaces beam_cloud_block.comp's per-TARGET vertical march (BeamCloudBlockPC, now retired) with
// a per-BEAM slant march — needs the observer frame (obsECEFDir/obsEffH) to reconstruct each
// beam's true ECEF endpoints from its observer-relative ENU offsets, which the per-target version
// never needed (it worked in absolute ECEF target coordinates directly). See that shader's own
// header for the full design and BEAM_CLOUD_PLAN.md for the session history.
struct BeamSelfMarchPC
{
    glm::vec4 obsECEFDir; // xyz = observer ECEF unit direction (w unused)
    float obsEffH;
    float waveTime;
    float cloudPhase;
    float pad0;
}; // total: 32 bytes
static_assert(sizeof(BeamSelfMarchPC) == 32, "BeamSelfMarchPC layout mismatch");

// ── Push constants for scene_depth.comp (pipeline unification) ───────────────────────────────
// Camera-only: this pass marches terrain, so it needs the view ray and the observer, nothing
// about sun/moon/clouds. `aspect` is ALWAYS the true swapchain aspect (never a render-scaled
// one) — that is what makes the resulting depth buffer well-defined for consumers rendering at a
// different resolution than this pass.
struct SceneDepthPC
{
    glm::mat4 skyView;         // offset 0  — ENU → camera space
    float fovYRad;             // offset 64
    float aspect;              // offset 68
    uint32_t debugDisableMask; // offset 72 — bit 1 skips the terrain march; bit 1024 skips the
                               //             whole pass (fills kNoSurfaceT = nothing occludes
                               //             anything, reproducing pre-unification behaviour, so
                               //             the entire architecture A/Bs from one checkbox)
    float pad0;                // offset 76 — explicit, aligns the vec4 below to 16
    glm::vec4 obsECEFDir;      // offset 80 — xyz = observer ECEF unit vector, w = height offset
}; // total: 96 bytes
static_assert(sizeof(SceneDepthPC) == 96, "SceneDepthPC layout mismatch");

// Per-frame sky glow data, written by sat_flare.comp each frame.
//
//   bins[64] — Spatial histogram (45°×11.25° cells, 8 az × 8 el).
//              atomicMax(floatBitsToUint(effectFlare)) per bin.
//              Used for the wide-Gaussian aggregate sky glow pass in sat_sky.frag. Unrelated to,
//              and unaffected by, the flare/corona architecture overhaul below — a separate
//              phenomenon that happens to share this buffer for historical reasons.
//
// std430: bins[64]×uint(256).
//
// (flareCount/flareEntries[kMaxFlares] lived here — a capped atomic-append list of "bright"
// satellites that sat_sky.frag looped over PER SCREEN PIXEL to draw lens-flare coronas. Deleted in
// the flare architecture overhaul (see TERRAIN_PLAN.md): raising the cap enough to stop flicker
// with "100s" of simultaneously-bright Reflect-Orbital satellites made the per-pixel cost scale
// with it — confirmed by measurement to tank framerate to 24fps. Replaced by actually RENDERING
// every visible satellite (plus the sun) as a point into a small offscreen texture
// (flareSourceImg/flareScratchImg, see those members below), which a couple of cheap compute
// blur/streak passes turn into the corona+godray texture composited once per frame instead of once
// per pixel per bright satellite. See GpuOceanGlintBuf below for the one consumer that still needed
// a small, independent capped list of its own.)
static constexpr int kGlowBins = 64;
struct GpuGlowBuf
{
    uint32_t bins[kGlowBins];
};
static_assert(sizeof(GpuGlowBuf) == kGlowBins * 4, "GpuGlowBuf layout mismatch");

// ── Ocean satellite-glint list (flare architecture overhaul) ─────────────────────────────────
// A small, INDEPENDENT capped atomic-append list — same pattern the old flareEntries used, just
// decoupled from corona rendering and much smaller, since ocean specular glint is a minor highlight
// effect, not the primary visibility signal. Written by sat_flare.comp alongside GpuSatVisible;
// read by sat_sky.frag's ocean-glint block, which ALSO gained cloud + terrain occlusion tests at
// each entry's own screen position this round (the previous version had none at all).
static constexpr int kMaxOceanGlints = 512;
struct GpuOceanGlintBuf
{
    uint32_t count;
    uint32_t pad0, pad1, pad2;
    glm::vec4 entries[kMaxOceanGlints]; // xyz=ENU dir, w=effectFlare
};
static_assert(sizeof(GpuOceanGlintBuf) == 16 + kMaxOceanGlints * 16, "GpuOceanGlintBuf layout mismatch");

// ── Flare/corona render-to-texture pipeline (flare architecture overhaul) ────────────────────
// Replaces the deleted per-pixel flareEntries loop. Three stages, all inside recordCompute()/
// recordDraw() (no new Simulation interface hook needed):
//   1. flare_source.vert/.frag: every visible satellite (from GpuSatVisible, exactly like
//      sat_point.vert) plus one virtual point for the sun is drawn into a small offscreen target
//      (flareSourceImg), quarter the swap extent, independent of renderScale (same sizing
//      rationale as scene_depth.comp/cloud_march.comp's half-res targets — this one goes one step
//      smaller since it is deliberately going to be blurred, not sampled 1:1). Cloud + terrain
//      occlusion are tested per-point in the fragment shader (reusing sat_point.frag's existing
//      technique, plus a new terrain test this pass needs since it has no shared depth buffer of
//      its own to rely on).
//   2. flare_blur.comp: one pipeline, three dispatches, ping-ponging between flareSourceImg and
//      flareScratchImg — two separable-Gaussian passes (corona softness) then one multi-directional
//      streak pass (the godray mechanism: operates uniformly on the whole buffer, so it naturally
//      produces shafts from ANY bright, unoccluded region — sun or satellites alike — with no
//      "which light source" tracking needed).
//   3. flare_composite.vert/.frag: one additive fullscreen-triangle draw, appended at the end of
//      recordDraw(), sampling the final blurred/streaked texture into the frame.
// The sun keeps its existing disc/atmospheric-corona block and its lensFlare() ghost/streak call
// in sat_sky.frag UNCHANGED — "lens elements" (ghosts, chromatic streaks) stay sun-only, per
// explicit user decision; it ALSO becomes one bright point in this new pipeline, so it gains real
// godray shafts through cloud/terrain gaps on top of its existing hand-authored treatment.
struct FlareSourcePC
{
    glm::mat4 skyView;        // offset 0
    float fovYRad;            // offset 64
    float aspect;             // offset 68
    uint32_t satCount;        // offset 72 — gl_VertexIndex >= satCount means "this is the sun"
    float sunRefIntensity;    // offset 76 — fixed reference brightness for the sun's virtual point
    glm::vec4 sunDirENU;      // offset 80 — xyz=dir, w=sin(elevation)
    glm::vec2 screenSizePx;   // offset 96 — THIS pass's own (small) target size, for the fragment
                              //             shader's occlusion UV — never assume a constant, see
                              //             the resolution-scaling gotcha this codebase already
                              //             learned once for sat_sky.frag.
    float resScale;           // offset 104 — flareExtent / ctx.swapExtent ratio; scales point size
                              //              down to this smaller target so satellites don't look
                              //              oversized once the composite upsamples back to full res
    float sunDayCompensation; // offset 108 — was pad0. The shared streak/composite gain is scaled
                              // by flareEyeAdaptGain (see recordCompute) to keep satellite glow tame
                              // in daylight — but the sun's OWN godray/corona is meant to read
                              // strongest at low sun angles/daytime, the opposite intent. This is
                              // 1/flareEyeAdaptGain, applied ONLY to the sun's own written brightness
                              // in flare_source.vert, so it nets out to full strength after the later
                              // uniform daytime suppression — satellites are unaffected.
}; // total: 112 bytes
static_assert(sizeof(FlareSourcePC) == 112, "FlareSourcePC layout mismatch");

// Push constants for flare_blur.comp — one pipeline, three dispatches per frame (see the
// architecture comment above): direction picks which image is source vs destination this
// dispatch, mode picks horizontal-gaussian / vertical-gaussian / streak.
struct FlareBlurPC
{
    uint32_t direction; // 0 = read flareSourceImg write flareScratchImg, 1 = reverse
    uint32_t mode;      // 0 = horizontal gaussian, 1 = vertical gaussian, 2 = streak
    float streakGain;   // user-tunable streak/godray strength (Settings > Display)
    float pad0;
}; // total: 16 bytes
static_assert(sizeof(FlareBlurPC) == 16, "FlareBlurPC layout mismatch");

// Push constants for flare_composite.frag — the final additive draw into the main frame.
struct FlareCompositePC
{
    float gain; // user-tunable overall glow gain (Settings > Display)
}; // total: 4 bytes
static_assert(sizeof(FlareCompositePC) == 4, "FlareCompositePC layout mismatch");

// Push constants for trail_fade.comp — see the long-exposure trail member block below.
struct TrailFadePC
{
    float decayFactor; // exp(-dt / trailDecaySeconds); dt = real wall-clock frame delta, never simDt
    float ceiling;     // per-channel soft cap on the decayed value (defense-in-depth; the primary
                       // ceiling is trail_composite.frag's own hard clamp)
    float pad0, pad1;
}; // total: 16 bytes
static_assert(sizeof(TrailFadePC) == 16, "TrailFadePC layout mismatch");

// Push constants for trail_composite.frag — the final additive draw into the main frame.
struct TrailCompositePC
{
    float gain; // user-tunable overall trail gain (Settings > Photometry, "Trail gain")
}; // total: 4 bytes
static_assert(sizeof(TrailCompositePC) == 4, "TrailCompositePC layout mismatch");

// ── GPU orbital parameters (uploaded once per buildOrbits, device-local) ─────
// 28 × 4-byte fields = 112 bytes.  All plain floats/uints — no vec3 — so
// C++ struct packing matches GLSL std430 without any alignment padding.
// Must match the SatOrbit struct in sat_orbit.comp exactly.
struct GpuSatOrbit
{
    float raan;    // right ascension of ascending node at epoch
    float u0;      // epoch-baked initial phase: fmod(orig_u0 + meanMot*epochT0, 2π)
    float R_sat;   // kEarthRadius + altM (meters)
    float meanMot; // sqrt(GM/R³) (rad/s)

    float cosI;    // cos(inclination)
    float sinI;    // sin(inclination)
    float cosRaan; // cos(raan); valid when !alignTerminator
    float sinRaan; // sin(raan); valid when !alignTerminator

    float tumbleRate;      // rotation rate (rad/s); 0 if not tumbling
    float tumblePhase;     // epoch-baked angle: fmod(phase + rate*epochT0, 2π)
    float alignTerminator; // 1.0 = SSO (RAAN precesses); 0.0 = fixed RAAN
    float tumbleAxisX;

    float tumbleAxisY;
    float tumbleAxisZ;
    uint32_t primaryAttitude; // AttitudeMode cast to uint
    uint32_t secondaryAttitude;

    float baseColorR;
    float baseColorG;
    float baseColorB;
    float crossSection; // sqrt(crossSectionM2 / 10)

    float specExp0;
    float specExp1;
    float w1; // secondary surface weight
    float diffuse;

    float mirrorFrac;
    uint32_t constIdx; // constellation index for enabled/highlight masks
    uint32_t pad0;
    uint32_t pad1;
    // Total: 112 bytes
};
static_assert(sizeof(GpuSatOrbit) == 112, "GpuSatOrbit layout mismatch");

// GpuReflectorTarget / the per-frame CPU-compacted night-side buffer it described were removed
// 2026-08-06: sat_orbit.comp now reads the static per-target ECEF buffer directly (the same one
// beam_cloud_block.comp already used) and rotates to live ECI itself via SatOrbitPC::gmstNow — see
// CLAUDE.md's TargetedReflector section for why the CPU-side compaction became dead code.

// ── Reflect-Orbital ground beams (device-local, written by sat_orbit.comp) ───
// History: originally azimuth-sector-keyed (16 sectors, GpuGlowBuf's flareEntries/sectorBright
// idiom) — let unrelated satellites aiming at different real targets collide whenever they
// shared a 22.5° bearing wedge. Re-keyed by TARGET IDENTITY (bestIdx, one slot per site) to fix
// that — but then a well-serviced site (many satellites simultaneously choosing it) could only
// ever show its single brightest satellite via atomicMax, the rest silently dropped. Tried 4
// sub-slots per site next — still arbitrary and still glitchy for a site serviced by "dozens"
// of satellites (user report, 2026-07-21): whichever 4 happened to win kept changing as
// brightness/geometry shifted, so the beam still visibly jumped between represented satellites.
//
// **Current design (2026-07-21): plain capped atomic-append, no site keying and no
// deduplication/arbitration at all.** Every satellite that reaches the TargetedReflector branch
// with a real target (bestIdx >= 0) claims its OWN slot via a single global atomicAdd counter
// and writes its own beam unconditionally — no competition, so nothing to glitch between AS LONG
// AS the cap isn't actually hit.
//
// kMaxActiveBeams (C12 follow-up #11, 2026-07-21: 256->2048): the first cap (256) was too tight
// and reintroduced arbitration by a different door — when genuinely more satellites are eligible
// than the cap, only the first N (by atomicAdd claim order, which correlates with GPU dispatch
// order and therefore with satellite ARRAY INDEX, not anything geometric) get written, and every
// later-indexed satellite is silently dropped EVERY frame. Reflect Orbital is a "Disk" (per
// CLAUDE.md — a single orbital plane, 10 concentric altitude rings, not spread across many
// planes/RAANs like a Walker constellation), so the whole constellation sweeps together along
// essentially one great circle — from a fixed observer, the VISIBLE fraction of that one ring is
// a specific, fairly large arc (not a small scattered sample of the whole sphere), and as
// low-index satellites within that eligible set changed which physical satellites they were
// (the ring sweeping across the sky over time / with observer motion), the rendered subset
// visibly "stacked" toward whichever side of the sky currently held the lower-indexed eligible
// satellites — reported as flares fading in from one side and stacking on the other, reversing
// when time direction reversed, and lagging when moving quickly into a new region. Re-estimated
// the realistic worst case at ~900 simultaneously eligible for a 5,000-satellite version of this
// constellation (a single-plane arc estimate, not the much smaller uniform-sphere estimate
// follow-up #9 used) — kMaxActiveBeams raised to 2048 for comfortable headroom over that, making
// overflow effectively never happen rather than trying to make the overflow's bias smaller.
// GpuReflectBeams is still tiny at this size (~96KB, after the debug reflectDirENU field below
// grew each entry to 48 bytes). A satellite's slot index isn't stable frame
// to frame (a race on the counter) but that no longer matters when nothing is being dropped:
// nothing else depends on WHICH index a given satellite lands in, only that everything currently
// eligible gets rendered, every frame. Zeroed every frame via vkCmdFillBuffer (resets beamCount
// to 0 too), same pattern as glowBuf.
static constexpr int kMaxActiveBeams = 2048;
struct GpuReflectBeam
{
    glm::vec3 satENU;          // meters, observer-relative (East, North, Up)
    float intensity;           // groundIrradiance * beamGain — NOT the view-dependent
                               // mirrorPeak specular term; see sat_orbit.comp writer comment
    glm::vec3 targetENU;       // meters, observer-relative; exact 3D ENU projection of the
                               // chosen ground target — correctly encodes Earth curvature
    float footprintRadM;       // ground footprint radius
    glm::vec3 reflectDirENU;   // unit direction, observer ENU basis — the mirror's ACTUAL current
                               // reflected-sunlight direction (reflect(-sunDirECI, surfN0)), which
                               // may differ from normalize(targetENU-satENU) while the mirror is
                               // still slewing toward bestIdx's target (MIRROR_ROT_RATE-limited).
                               // Debug-only (C12 follow-up #12): drawn as a long "pointing ray" from
                               // the satellite so a busy site's convergence — and any satellites
                               // still mid-slew and not yet converged — can be seen directly.
    float debugPad;            // Repurposed (C12 follow-up #20): carries the originating satellite's
                               // own stable dispatch index (written as float(i) in sat_orbit.comp) —
                               // used by cloud_march.comp's sky glow to downsample by a STABLE subset
                               // of satellites rather than by the atomic-append slot index (which
                               // isn't stable frame-to-frame). Name kept for minimal diff; no longer
                               // debug-only or padding.
    float blockAltM;           // Altitude (m above sea level) at which THIS BEAM's own real 3D path
                               // (ground intersection -> satellite) first drops below 50%
                               // transmittance. Written by beam_self_march.comp (2026-08-09), a
                               // per-beam slant march — was beam_cloud_block.comp's per-TARGET
                               // vertical-column approximation (C12 follow-up #33) before that. See
                               // beam_self_march.comp's own header for the full design and why this
                               // (not the per-target version) is what beam physics actually needs.
                               // Irrelevant when blockOpacity==0.
    float blockOpacity;        // 0 = clear path, 1 = fully opaque — see beam_self_march.comp.
                               // Consumed by cloud_march.comp (fades the volumetric glow and, as of
                               // 2026-08-09, the visible pointing ray below the cloud) and
                               // sat_sky.frag (ground-spot dimming).
    float mirrorRadiusM;       // C12 follow-up #34: repurposed from padding — equivalent-circle radius
                               // of the physical mirror (sqrt(mirrorAreaM2/PI)). Consumed by
                               // cloud_march.comp (sky tube radius) and sat_sky.frag (ground-spot core).
    float aimErrorRad;         // Repurposed from padding (2026-08-06): angle, in radians, between the
                               // mirror's current attitude and the exact ideal half-vector toward
                               // `bestIdx`'s target this frame. 0 outside a window-boundary crossfade
                               // (the mirror aims exactly at its target); non-zero only during the
                               // brief blend between two different targets at a lock-window boundary
                               // (see sat_orbit.comp's TargetedReflector block and CLAUDE.md) — lets a
                               // consumer distinguish "mid-crossfade" from "locked and settled" the
                               // same way it always could, just driven by the new stateless windowed
                               // selection instead of the old rate-limited slew.
    uint32_t targetIdx;        // 2026-08-12: sat_orbit.comp's own `bestIdx` — the resolved ground
                               // target's ORIGINAL index into reflectorTargetsECEF[]/
                               // reflectorTargetsRadiusM[]/reflectorSiteEnu*[]. Always valid
                               // ([0, reflectorTargetCount)) for any entry that exists at all, since
                               // sat_orbit.comp only writes a beam inside `if (bestIdx >= 0)`.
                               // This is a STABLE INTEGER IDENTITY, which is the entire reason it
                               // exists: the cloud-light build below keys clusters on it directly
                               // instead of epsilon-matching targetENU against whichever beam
                               // happened to be scanned first. See TrackedBeamLight's comment.
    uint32_t pad0, pad1, pad2; // MUST be declared explicitly, in both this struct and ReflectBeam
                               // in shaders/include/reflect_beam.glsl. std430 rounds the GLSL struct
                               // up to its 16-byte alignment (vec3 members) whether or not the pads
                               // are written; C++ does NOT, because glm::vec3 is 4-aligned. Before
                               // targetIdx the total happened to be exactly 64 and the two agreed by
                               // luck. Same silent-permutation hazard as GpuCloudParams — no compile
                               // error, every field past the divergence point reads its neighbour.
};
static_assert(sizeof(GpuReflectBeam) == 80, "GpuReflectBeam layout mismatch"); // 64 -> 80, 2026-08-12 (targetIdx + explicit pads)

struct GpuReflectBeams
{
    uint32_t beamCount;                      // atomicAdd counter — total claims this frame, may exceed kMaxActiveBeams
    uint32_t pad0, pad1, pad2;               // std430 array-of-16-byte-aligned-struct alignment padding
    GpuReflectBeam entries[kMaxActiveBeams]; // only entries[0 .. min(beamCount,kMaxActiveBeams)) are valid
};
static_assert(sizeof(GpuReflectBeams) == 16 + kMaxActiveBeams * 80, "GpuReflectBeams layout mismatch");

// Ground-beam compaction (perf follow-up, RELEASE_v1_1_PLAN.md): CPU-built every frame from
// ReflectBeamsBuf's readback (the same loop that already computes lastActiveBeamCount/
// beamProximityGlow), filtered to entries within beamMaxRangeM of the observer — the exact cull
// sat_sky.frag's ground-spot loop used to redo per-pixel, against the FULL raw (up to
// kMaxActiveBeams=2048) buffer, for every ground-hit pixel on screen. Consumed by sat_sky.frag
// instead of ReflectBeamsBuf directly, so that loop's trip count is bounded by how many beams are
// actually within range of the CAMERA, not by how many are active anywhere across the whole
// visible constellation (measured: disabling the "Reflect-Orbital beams" debug knockout bit nearly
// doubled frame rate — this is the dominant cost that knockout was hiding). Entries are raw,
// unaggregated ReflectBeam records (per-satellite, not summed by target) because the ground-spot
// term needs each satellite's own satENU for its elevation fade.
//
// 2026-08-10: entries are no longer raw GpuReflectBeam records — they are a PRE-SOLVED
// GpuGroundBeam, which is what makes the shader loop cheap. The Anchorage sweep measured that loop
// at 1.59 ms of sky_background_draw at Medium, and the reason was that essentially its whole body
// was view-INDEPENDENT and being recomputed for every ground-hit pixel at full resolution: a
// length(targetENU) plus a smoothstep for the range fade, an obsPos+satENU and a raySphere (two
// sqrts) for the real ray/ground intersection, the elevation fade, and the shadow attenuation. Only
// the horizontal distance from the shaded point to the landing spot, and the two Gaussians built
// from it, genuinely vary per pixel. All the rest is now folded on the CPU — once per beam, at most
// kMaxGroundBeams times per frame, in the readback loop that was already visiting these entries —
// into `weight` and a few reciprocals, so the shader's per-beam work drops to a 2D difference, a
// dot, a squared-distance reject and two exps, with the reject FIRST rather than last.
//
// Struct must match GroundBeam/GroundBeamsBuf in sat_sky.frag exactly (hand-mirrored, same
// convention and same hazard as GpuCloudParams — all plain floats so std430 needs no padding
// beyond what is written here).
static constexpr int kMaxGroundBeams = 256;
struct GpuGroundBeam
{
    float groundHitX, groundHitY; // observer-relative ENU horizontal position of the beam's REAL
                                  // ray/ground intersection (not the chosen target's site)
    float invFootprintSq;         // 1 / footprintR^2   — halo Gaussian
    float invCoreSq;              // 1 / coreR^2        — hotspot Gaussian
    float cutoffSq;               // (footprintR * 4)^2 — the shader's first and cheapest reject
    float weight;                 // intensity * rangeFade * elevFade * shadowAtten, i.e. every
                                  // view-independent multiplier the old shader loop applied
    float intensity;              // raw pre-fade intensity — CPU-side top-K ranking ONLY, never read
                                  // by the shader. Kept so eviction ranks on exactly the same
                                  // quantity it did before this rework (see the top-K comment at
                                  // the insertion site for why ranking stability matters).
    float pad0;
};
static_assert(sizeof(GpuGroundBeam) == 32, "GpuGroundBeam layout mismatch");
struct GpuGroundBeams
{
    uint32_t count;
    uint32_t pad0, pad1, pad2;
    GpuGroundBeam entries[kMaxGroundBeams];
};
static_assert(sizeof(GpuGroundBeams) == 16 + kMaxGroundBeams * 32, "GpuGroundBeams layout mismatch");

// ── Per-layer cloud shell descriptor (std140: 32 bytes, 2 × vec4) ─────────────
// Each layer is an infinitely thin sphere-shell sample of earthCloudsTex.
// Layers 0+ are evaluated in order; disabled layers (enabled=0) are skipped.
struct GpuCloudLayerParams
{
    float shellAltM;    // sphere-shell altitude above R_EARTH (m)
    float driftMult;    // cloudPhase longitude multiplier (1.0 = same speed as surface)
    float alphaMax;     // maximum opacity of this layer [0,1]
    float mipLod;       // fixed texture LOD (0=sharp, 2=soft/wispy)
    float coverageMult; // scales global coverage for this layer
    float densityMult;  // scales global density for this layer
    float enabled;      // 1.0 = active, 0.0 = skip
    float pad;
};
static_assert(sizeof(GpuCloudLayerParams) == 32, "GpuCloudLayerParams layout mismatch");

static constexpr int kNumCloudLayers = 4;

// ── Cloud parameters UBO (binding 9 in sky descriptor set) ───────────────────
// Matches the layout(binding=9) uniform CloudParams block in sat_sky.frag.
// Global tunables + per-layer descriptors.  cloudPhase is CPU-computed each frame.
// std140 layout: 96-byte global section (6×vec4) + 4 × 32-byte layer = 224 bytes.
struct GpuCloudParams
{
    // Global controls — shared across all layers
    float coverage;           // global coverage gate [0,1]
    float density;            // global density sharpness scale
    float driftRate;          // base longitude drift rate (rad/s sim-time)
    float sunGain;            // global sun brightness multiplier
    float ambientGain;        // night-side ambient (for future use in volumetrics)
    float hgG;                // Henyey-Greenstein g (C7+ volumetric march)
    float marchSteps;         // volumetric march step count (C7+)
    float lightSteps;         // volumetric light-cone step count (C7+)
    float cloudPhase;         // CPU: fmod(driftRate * simTime, 2π) — uploaded each frame
    float extinctionCoeff;    // was pad0 (freed session 23 when cloudShadowFactor was removed); now
                              // carries the same atmospheric-extinction coefficient sat_flare.comp
                              // gets via push constant, so sat_sky.frag's Milky Way term can apply
                              // identical Kasten & Young dimming without its own push-constant field
    float cirrusWindAngle;    // C13: cirrus streak wind axis, radians (was pad1)
    float cirrusStretch;      // C13: cirrus noise anisotropic elongation factor (was pad2)
    float airglowGain;        // C15: master airglow brightness multiplier
    float airglowGreenGain;   // C15: green (557.7nm) band gain
    float airglowRedGain;     // C15: red (630.0nm) band gain
    float airglowSodiumGain;  // C15: sodium (589.3nm) band gain — keep dim relative to green
    float shadowMaxDistM;     // cloudMarch's sun self-shadow cone fades out beyond this distance (m)
    float maxRenderDistM;     // cloudMarch's tExit distance cap (was a hardcoded 80km)
    float viewSamplesMin;     // perf (session 24 round 2): N_VIEW floor for short rays (was pad2)
    float lightSamples;       // perf (session 24): N_LIGHT optDepth sub-march count (was pad3)
    float oceanSeaOctaves;    // perf (session 24): seaMap() octave count (height-trace geometry)
    float oceanDetailOctaves; // perf (session 24): seaMapDetail() octave count (wave normal)
    float oceanReflSamples;   // perf (session 24): ocean sky-reflection loop sample count (N_REFL)
    float viewSamplesMax;     // perf (session 24 round 2): N_VIEW ceiling for long/grazing rays (was pad4)
    float sunGainZenith;      // was pad3 — sun-gain multiplier near sun zenith, blended against
                              // `sunGain` (effectively the near-horizon/sunset value) by sun
                              // elevation in both cloudMarchCS/cirrusMarchCS and evalCloudLayer.
    float moonGain;           // shared moonlight brightness master — terrain direct term AND
                              // cloud_march.comp's moonContrib (was a hardcoded 0.015 there) both
                              // read this, so one slider keeps moonlit terrain and moonlit clouds
                              // calibrated to the same brightness instead of drifting apart
    float pad1;               // ACTUALLY repurposed (see SatelliteSim.cpp) — city-detail world-fixed
                              // east offset (m); kept named pad1 since sat_sky.frag/cloud_march.comp
                              // don't need to read it (CPU computes the offset, GPU just samples the
                              // resulting texture), name is stale but layout-critical, don't rename
    float pad2;               // ACTUALLY repurposed — city-detail world-fixed north offset (m); see pad1
    // Milky Way skybox (session 27): CPU-computed ENU->galactic basis rows (fixed orientation,
    // confirmed by eye against the real star field), mirroring the eciX/Y/Z basis-vector
    // convention already used for SatOrbitPC/SatFlarePC. dirGal = dot(enuDir, mwBasisRowN.xyz)
    // for N=0,1,2. .w of row0 carries a fixed gain of 1.0 (spare otherwise).
    glm::vec4 mwBasisRow0;
    glm::vec4 mwBasisRow1;
    glm::vec4 mwBasisRow2;
    // Per-layer descriptors
    GpuCloudLayerParams layers[kNumCloudLayers];
    // Aurora (C16, TERRAIN_PLAN.md Phase E): geomagnetic curtain primitive.
    float stormStrength;    // [0,1] drives oval equatorward expansion, brightness, fold chaos
    float auroraGain;       // master aurora brightness multiplier (sky curtain itself)
    float auroraCloudGain;  // master gain for LOCAL aurora ambient upwelling on CLOUDS only —
                            // split from auroraGroundGain (session 28 follow-up #6) because the
                            // two formulas' magnitudes aren't comparable: clouds have no albedo
                            // term at all (roughly full reflectivity assumed) while terrain/ocean
                            // multiply by the surface's own dark albedo, so one shared slider
                            // couldn't hit "plausible" for both at once.
    float auroraGroundGain; // master gain for the LOCAL, per-point aurora ambient/reflection
                            // lighting on TERRAIN/OCEAN (evaluated in-shader per pixel/sample,
                            // mirroring how moonlight is local) — distinct from auroraGain above
                            // (the sky curtain's own brightness) and auroraCloudGain (clouds).
    // Aurora "erosion" coverage gate (session 28 follow-up #8) — breaks the oval into patchy arcs.
    // See auroraCoverage() in sat_sky.frag/cloud_march.comp for the full design.
    float auroraCoverageFreq;      // per-degree colatitude frequency — patch size across the band
    float auroraCoverageAzFreq;    // azimuthal wobble frequency — keeps the boundary non-circular
    float auroraCoverageDriftRate; // wall-clock rad/s evolution speed
    float auroraShimmerRate;       // curtain fold noise evolution speed (wall-clock rad/s) — was a
                                   // fixed kAuroraShimmerRate constant (session 28 follow-up #9)
    // Struct grew 320->336 here (session 30): appended rather than reusing pad1/pad2 above, which
    // turned out to already be repurposed (city-detail world offset, read by name as cloud.pad1/
    // pad2 in sat_sky.frag) despite their stale "reserved" comments — do not repurpose those.
    float cloudTwilightAmbientGain; // gain on cloud_march.comp's twilightAmbient term — sky-lit
                                    // cloud at dusk/dawn only (twilightWeight is a bell, so this
                                    // contributes nothing in daylight or full night). Same UBO slot
                                    // that used to drive a non-decaying night floor; that term was
                                    // the wrong effect and is gone. Deliberately separate from
                                    // ambientGain, which also drives city-light upwelling.
                                    // Unused in sat_sky.frag — kept for layout parity.
    float cloudBaseVariance;        // was pad4 — noise-driven cloud base height undulation (hNorm
                                    // units, 0 = old perfectly flat base). See cloudMarchCS.
    float cloudErosionEdge;         // was pad5 — cloudDensity() erosion strength at the silhouette
                                    // edge (base near 0).
    float cloudErosionCore;         // was pad6 — cloudDensity() erosion strength at the dense core
                                    // (base near 1); kept lower than cloudErosionEdge.
    // Struct grew 336->352. std140 rounds a uniform block up to a multiple of 16, so a single
    // trailing float would have left the GLSL block at 352 while this struct stayed 340 — a
    // silent mismatch. The three pads keep both at 352; take one when the next field is needed.
    float sunGainElevBand; // sin(sun elevation) at which sunGainZenith fully replaces
                           // sunGain. Was a hardcoded smoothstep(0,1,sinElev): only
                           // half-way to the zenith value at 30 degrees up, so a sunset-
                           // tuned gain stayed dominant through most of the morning.
    float twilightBandHi;  // sin(sun elev) above which twilight cloud ambient is zero.
                           // Raise to bring the term FORWARD into sunset — the original
                           // hardcoded 0.15 left a gap where direct sun had faded but the
                           // sky term had not arrived (clouds briefly went black).
    float twilightBandLo;  // sin(sun elev) below which it is zero — how far into night it
                           // carries.
    // ORDER BELOW MUST MATCH shaders/include/cloud_params.glsl EXACTLY, field for field.
    //
    // This is the one pairing the shared header cannot protect — GLSL and C++ can't share a
    // declaration, so this struct is a hand-maintained mirror. It was gotten wrong immediately
    // after that header landed: these four were appended AFTER coverageMipLod in the GLSL but
    // inserted BEFORE it here. Nothing failed to compile and the static_assert still passed,
    // because the total size was right; every field from coverageMipLod onward simply read its
    // neighbour's value. The visible result was flatSunGainScale reading pad10 (0 -> black
    // clouds) and flatCoverageScale reading 4.0 (-> coverage x4, clouds swallowing the Earth).
    //
    // A size check cannot catch a permutation. When adding a field, add it in the same position
    // in both files, and prefer appending at the end of both.
    float coverageMipLod;      // mip the volumetric march samples earthCloudsTex at. Was a
                               // hardcoded 4.0 (~78 km/texel on the 8K source): the volumetric
                               // shape could only ever follow large blobs, while the flat 2D
                               // layer sampled sharply — which is why the two never matched
                               // and the 3D->2D crossfade had to be pushed out to 800 km.
    float flatCoverageScale;   // see cloud_params.glsl — maps the shared Coverage slider onto
                               // the flat 2D layer, which needs a lower value than the
                               // volumetric for the same apparent cloud amount.
    float flatSunGainScale;    // same idea for Sun gain: the flat layer is a single multiply
                               // while the volumetric accumulates through transmittance, so
                               // the same slider lands ~4x dimmer on the flat path.
    float fogTopAltM;          // C11 (repurposed from pad10) — ground fog shell top altitude
                               // (m above sea level); see fogMarchCS in cloud_march.comp.
    float fogDensity;          // C11 (repurposed from pad11) — fog density scale.
    float cloudDistFadeStartM; // distance-based 3D->2D crossfade: fully volumetric nearer than
                               // this, fully flat-2D beyond cloudDistFadeEndM. Keyed on the
                               // per-ray distance to the cloud shell, so it actually bounds the
                               // march — maxRenderDistM caps march LENGTH from the shell entry
                               // and so does nothing from orbit, where that span is just the
                               // ~9 km shell crossing.
    float cloudDistFadeEndM;
    float fogCoverage; // C11 (repurposed from pad12) — ground fog global coverage gate.
    float fogSunGain;  // C11 (repurposed from pad13) — fog sun-lit brightness gain,
                       // separate from cloud.sunGain per [[feedback_shared_gain_sliders]].
    // Terrain march distance fade (S4, RELEASE_v1_1_PLAN.md session 31) — see cloud_params.glsl
    // for the full design rationale. Fades out sat_sky.frag's terrain-relief march step budget
    // as this ray's own march reach (tExit) grows, skipping it outright beyond End and falling
    // back to the sea-level sphere, which already exists as the "no hit" result.
    float terrainDistFadeStartM;
    float terrainDistFadeEndM;
    float cloudOpacityScale; // repurposed from pad14 — see cloud_params.glsl for the full
                             // rationale: scales the volumetric cloud march's extinction
                             // coefficient directly, since per-sample density `d` is clamped to
                             // [0,1] before extinction is derived, so the `density` slider alone
                             // has a hard ceiling on achievable opacity. Default 1.0.
    float cityLightBlurLod;  // repurposed from pad15 — see cloud_params.glsl. Mip LOD blend
                             // target for earthNightTex/cityNightDetailTex under cloud, at full
                             // local cloud opacity. Default 3.0; 0 disables the blur.
    // Atmospheric scattering strength gains — see cloud_params.glsl for the full rationale.
    // Scale BETA_R_BASE/BETA_M_BASE (common.glsl) uniformly across every shader that shadows them.
    // Default 1.0 each reproduces the original hardcoded constants exactly.
    float atmosRayleighGain;
    float atmosMieGain;
    float cloudWarpStrength; // domain-warp amplitude / spatial frequency — their PRODUCT is the
    float cloudWarpFreq;     // shear. See cloud_params.glsl for the folding threshold.
    // Erosion redesign, 416 -> 432. APPENDED rather than reusing pad1/pad2 above, which are
    // already repurposed as the city-detail world offsets despite their stale names.
    float cloudSurfaceCarve;   // 0 = heightProfile multiplies the eroded field (original),
                               // 1 = it is subtracted, displacing the top/bottom SURFACE by the
                               // erosion noise instead of fading the noise out at it
    float cloudErosionBillow;  // 0 = original inverted-Worley polarity (removes round blobs,
                               // leaves a web), 1 = flipped (removes the web, leaves lumps)
    float cloudErosionBillowH; // normalised column height over which polarity ramps wispy->billowy
    float cloudErosionFreq;    // erosion lookup coordinate scale (was a hardcoded 1.5)
    // Directional-shading contrast, 432 -> 448. See cloud_params.glsl for why all four of these
    // were flattening the only term that carries lit-side/dark-side variation.
    float cloudMultiScatter; // 0 = single scatter (hard shadows), 1 = original 3-octave sum
    float cloudShadowFloorT; // hard floor on sun transmittance (was a fixed 0.05)
    float cloudGrazeShadow;  // sunOptDepth multiplier at zero sun elevation (was a fixed 0.35)
    float cloudConeLenScale; // multiplier on the self-shadow cone's length cap
    // Shape-aware shading + flat-layer decoupling, 448 -> 464. See cloud_params.glsl.
    float cloudVertShadeGain; // strength of the height-only shading ramp ("lasagna" banding)
    float cloudDensityAO;     // occlusion driven by local density — carries real cloud shape
    float cloudAOPower;       // where 1-exp(-d*power) bites; tune against `density`
    float flatDensityScale;   // flat 2D layer opacity, decoupled from volumetric `density`
    // Flat-2D Rayleigh decoupling, 464 -> 480. See cloud_params.glsl for why the flat layer needs
    // its own Rayleigh multiplier to make the 3D->2D crossfade match.
    float flatRayleighGain;        // extra BETA_R multiplier for evalCloudLayer only; 1.0 = no-op
    float flatTwilightAmbientGain; // flat layer's share of the twilight sky ambient, stacked on
                                   // cloudTwilightAmbientGain; 1.0 = volumetric-equal strength
    // Orbital terminator gate — claimed the last two pads, so the block is still 480 but there is
    // no slack left. The NEXT field added here has to grow this struct AND cloud_params.glsl by a
    // full 16 bytes together. See cloud_params.glsl for what these do and why they are altitude-
    // gated.
    float atmosTermStrength; // 0 = previous behaviour exactly; 1 = full suppression past the cut
    float atmosTermWidth;    // rolloff half-width in sin(sun elev); smaller = harder cliff
    // Airglow coverage + polar boost, 480 -> 496. See cloud_params.glsl for the full design.
    float airglowCoverageGain; // 0 = uniform shimmer (old behaviour), 1 = full patchy coverage
    float airglowPolarGain;    // RED band only: extra boost ramping toward the geomagnetic pole
    // ── Push-constant relief (496 -> 544) ────────────────────────────────────────────────────
    // These were push-constant fields on SatDrawPC / CloudMarchPC until both structs had to fit
    // the 128-byte maxPushConstantsSize floor (oldest AMD integrated GPUs). All are per-frame-
    // uniform (one value for the whole frame, not per-draw), which is exactly what a frame UBO is
    // for. Consumers: sat_sky.frag (all of them) and cloud_march.comp (dbgDisableMask,
    // showBeamDebugRays, beamMaxRangeM, beamSkyGlowGain, cloudShadowRangeM). Filled in
    // recordCompute()'s CloudParams block; MUST match cloud_params.glsl field-for-field (run
    // tools/check_cloud_params.py). Appended, not folded into pad21/pad22 — same rule as every
    // prior growth here.
    uint32_t dbgDisableMask;    // = debugDisableMask; profiling knockout bitmask
    uint32_t showBeamDebugRays; // 0/1 — draw each mirror's live pointing ray (cloud_march.comp)
    float skyGlareVisibility;   // eased sun-glare gate for sat_sky.frag's Milky Way
    float beamMaxRangeM;        // Reflect-Orbital beam render-range cutoff (m)
    float beamSkyGlowGain;      // beam->cloud illumination brightness (shared: sky frag + march)
    float beamGlowBleedGain;    // beam-driven sky-illumination wash gain (sat_sky.frag)
    float beamProximityGlow;    // CPU [0,1] observer-near-a-beam-line wash (sat_sky.frag)
    float mwSuppressEased;      // Milky Way's own light-pollution suppression [0,1] (sat_sky.frag)
    float cloudShadowRangeM;    // per-pixel terrain cloud-shadow fade distance (cloud_march.comp)
    float skyScreenW;           // sat_sky.frag render-target width  (skyLowResExtent at renderScale<1,
    float skyScreenH;           // else ctx.swapExtent) — for gl_FragCoord->UV; the point shaders
                                // carry their own screenSizePx (always full-res) in PointDrawPC
    float pad21;
    float pad22;
    float pad23;
};
static_assert(sizeof(GpuCloudParams) == 544, "GpuCloudParams layout mismatch");

// ── Push constants for sat_orbit.comp ────────────────────────────────────────
// Offsets verified against the push_constant block in sat_orbit.comp.
// Total: 96 bytes.
struct SatOrbitPC
{
    glm::vec4 enuX;               // East  basis in ECI (w unused) — offset 0
    glm::vec4 enuY;               // North basis in ECI (w unused) — offset 16
    glm::vec4 enuZ;               // Up    basis in ECI (w unused) — offset 32
    glm::vec3 sunDirECI;          // unit vector toward sun — offset 48
    float deltaT;                 // simTime - epochT0 (float precision) — offset 60
    glm::vec3 obsECI;             // observer ECI position (meters) — offset 64
    uint32_t satCount;            // total satellite count — offset 76
    uint32_t highlightMask;       // bit i = constellation i in highlight mode — offset 80
    uint32_t enabledMask;         // bit i = constellation i is enabled — offset 84
    float simDt;                  // simulated seconds this frame — offset 88
    float elevCutoff;             // sin(Earth-limb angle) — horizon cull threshold (≤ -0.01) — offset 92
    float beamGain;               // Reflect-Orbital ground-beam intensity multiplier — offset 96
    float reflectorLockWindowS;   // offset 100 — 2026-08-06 reversibility rework: fixed-width sim-time
                                  // window (seconds) a TargetedReflector satellite commits to one target
                                  // for. Target IDENTITY is chosen per-window (see sat_orbit.comp), not
                                  // by a persisted per-satellite lock — this is also the extrapolation
                                  // step the shader uses to reach a window's midpoint from `deltaT`/
                                  // `gmstNow` (see those fields below), so it doubles as "windowS".
                                  // Replaces the old rate-limited-slew design's mirrorSlewDegPerSec
                                  // (same push-constant slot, same settings-window slider).
    uint32_t targetCount;         // offset 104 — total loaded reflector targets (reflectorTargetCount).
                                  // sat_orbit.comp scans the full static ECEF set itself now (rotating
                                  // by gmstNow/derived gmstEval on the fly) instead of reading a
                                  // per-frame CPU-compacted night-side buffer — see CLAUDE.md's
                                  // TargetedReflector section. Replaces the old activeTargetCount.
    float minBeamElevSin;         // offset 108 — sin(reflectorMinElevDeg), precomputed on CPU. Candidate
                                  // targets below this local-elevation-at-target angle are rejected
                                  // outright (grazing, not just deprioritized).
    float gmstNow;                // offset 112 — current-frame GMST (rad), for rotating a target's
                                  // static ECEF entry to its LIVE ECI position (the actual aim/beam
                                  // position — see CLAUDE.md). Replaces the old mirrorSnap flag; nothing
                                  // persists across frames anymore, so there is no snap state to force.
    float windowFrac;             // offset 116 — fract(simTimeAbs / reflectorLockWindowS): this
                                  // frame's fractional position within the current lock window, in
                                  // [0,1). sat_orbit.comp uses -windowFrac*reflectorLockWindowS to
                                  // extrapolate deltaT/gmstNow back to the current window's START
                                  // instant (and one window further back for the previous window's),
                                  // exact rather than approximate for the same reason deltaT/gmstNow
                                  // themselves are (see that shader). Replaces the old
                                  // minBeamElevSinRelease — hysteresis is gone; a rate-limited ease
                                  // sized off windowFrac is what smooths a target change instead.
    float mirrorMaxRateDegPerSec; // offset 120 — real angular-rate cap for the closed-form ease
                                  // above: sat_orbit.comp derives an ease duration from the actual
                                  // angle between the previous and current window's targets (at the
                                  // current window's start instant) and this rate, then eases the
                                  // live orientation over that duration — a genuine max attitude
                                  // rate, unlike the old design's fixed-fraction-of-window crossfade
                                  // (2026-08-06 same-day follow-up: satellites were visibly snapping
                                  // to target because that crossfade only covered ONE of several
                                  // transition cases and wasn't derived from real angular distance).
    float flareMitigationTiltRad; // offset 124 — was pad2. Global flare-mitigation tilt angle
                                  // (radians) for AttitudeMode::SunTrackingTilted satellites — see
                                  // that enum value's comment and computeNormal() in sat_orbit.comp.
                                  // Deliberately global rather than per-satellite-type: it's an
                                  // operator policy knob, not orbital geometry, so it belongs beside
                                  // the other live photometry sliders (flareMitigationTiltDeg,
                                  // SatelliteSim.h) rather than growing GpuSatOrbit.
}; // 128 bytes
static_assert(sizeof(SatOrbitPC) == 128, "SatOrbitPC layout mismatch");

// ── Sky camera ────────────────────────────────────────────────────────────────
// Azimuth/elevation look direction in the local ENU frame.
// Right-click to capture mouse; WASD-style look via mouse deltas.
struct SkyCamera
{
    float azDeg = 0.0f;    // azimuth of look direction (0=North, 90=East), degrees
    float elDeg = 30.0f;   // elevation of look direction, degrees
    float fovYDeg = 70.0f; // vertical field of view, degrees
    float sens = 0.12f;    // mouse sensitivity (degrees per pixel)
    bool captured = false;

    // Returns a view matrix that transforms ENU directions into camera space.
    // Camera convention: +X=right, +Y=up, -Z=forward (standard OpenGL).
    glm::mat4 viewMatrix() const
    {
        float az = glm::radians(azDeg);
        float el = glm::radians(elDeg);
        // Forward vector in ENU (x=East, y=North, z=Up)
        glm::vec3 fwd{sinf(az) * cosf(el), cosf(az) * cosf(el), sinf(el)};
        // World up = ENU Up. Fall back to North when near zenith/nadir.
        glm::vec3 worldUp = (fabsf(elDeg) > 88.0f)
                                ? glm::vec3{0.0f, 1.0f, 0.0f}  // North when near zenith
                                : glm::vec3{0.0f, 0.0f, 1.0f}; // ENU Up otherwise
        return glm::lookAt(glm::vec3(0.0f), fwd, worldUp);
    }

    void update(GLFWwindow *win, float dmx, float dmy)
    {
        if (!captured)
        {
            if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            {
                captured = true;
                glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            return;
        }
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
        {
            captured = false;
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        azDeg += dmx * sens;
        elDeg -= dmy * sens; // screen Y down → mouse up = negative dmy = increase el
        elDeg = glm::clamp(elDeg, -89.0f, 89.0f);
    }
};

// ── Fixed orbital parameters (one per satellite, computed once at init) ───────
struct SatOrbit
{
    float raan;           // right ascension of ascending node (radians)
    float incl;           // inclination (radians) — per-satellite so RandomShell can vary
    float u0;             // initial mean argument of latitude (radians)
    uint32_t typeIdx;     // index into satTypes[]
    float altM;           // orbital altitude above surface (meters) — per-satellite
    float tumbleRate;     // rotation rate (rad/s); 0 = not tumbling
    float tumblePhase;    // initial rotation angle (radians)
    glm::vec3 tumbleAxis; // fixed body tumble axis (unit vector in ECI)
    bool alignTerminator; // if true, incl/raan are recomputed from sunDirECI each frame
    float targetTerminatorAngle = 0.0f;
    // ── Precomputed frame-invariant constants (set once in buildOrbits) ────────
    float R_sat = 0.0f;    // kEarthRadius + altM
    float meanMot = 0.0f;  // sqrt(kGM / R_sat^3)
    float cosI = 0.0f;     // cos(incl)
    float sinI = 0.0f;     // sin(incl)
    float cosRaan = 0.0f;  // cos(raan) — valid when !alignTerminator
    float sinRaan = 0.0f;  // sin(raan) — valid when !alignTerminator // TargetedReflector: angle (rad) along the terminator great-circle
    uint32_t constIdx = 0; // index into constellations[] — set by buildOrbits()
                           // that selects the ground target this mirror aims at.
                           // Terminator basis: t1=cross(sunDir,ref), t2=cross(sunDir,t1).
                           // Target = kEarthRadius × (cos(angle)×t1 + sin(angle)×t2).
};

// ── SatelliteSim ──────────────────────────────────────────────────────────────
class SatelliteSim : public Simulation
{
public:
    const char *name() const override { return "SAT LIGHT SIM"; }

    void init(VulkanContext &ctx) override;
    void onResize(VulkanContext &ctx) override;
    void recordCompute(VkCommandBuffer cmd, VulkanContext &ctx, float dt) override;
    SatDrawPC buildSkyDrawPC(VulkanContext &ctx);     // sky background (recordPrePass + recordDraw Pass 1)
    PointDrawPC buildPointDrawPC(VulkanContext &ctx); // satellite / star / planet / trail point draws
    void recordPrePass(VkCommandBuffer cmd, VulkanContext &ctx, float dt, uint32_t imgIdx) override;
    VkRenderPass activeRenderPass(VulkanContext &ctx) override { return renderScale < 0.999f ? ctx.renderPassLoad : ctx.renderPass; }
    void recordDraw(VkCommandBuffer cmd, VulkanContext &ctx, float dt) override;
    void buildUI(float dt, UIRenderer &ui) override;
    void setAudio(AudioSystem *audio) override;
    void setWindow(GLFWwindow *w) override { win = w; }
    VkClearValue clearColor() const override { return {{{0.0f, 0.0f, 0.015f, 1.0f}}}; }
    // NEW-7: numeric caps (Off/Cap30/Cap60/Cap120 all run MAILBOX/IMMEDIATE present, uncapped
    // submission) need App::mainLoop to pace them manually; VSync (FIFO) paces itself.
    float targetFpsCap() const override
    {
        switch (fpsCapMode)
        {
        case FpsCapMode::Cap30:
            return 30.0f;
        case FpsCapMode::Cap60:
            return 60.0f;
        case FpsCapMode::Cap120:
            return 120.0f;
        default:
            return 0.0f; // Off, VSync
        }
    }
    bool consumeSwapchainRebuildRequest() override
    {
        if (!fpsCapSwapchainRebuildPending)
            return false;
        fpsCapSwapchainRebuildPending = false;
        return true;
    }
    // UC6: see Simulation.h for the calling convention (peek before ui.record(), record the copy
    // after the render pass ends, finalize at the top of the next frame).
    bool wantsCleanScreenshot() const override { return screenshotRequested; }
    void recordScreenshotCopy(VkCommandBuffer cmd, VulkanContext &ctx, VkImage image) override;
    void finalizeScreenshot() override;
    // UC6: shared by KB_SCREENSHOT (dispatchKeyAction) and the left HUD panel's camera button —
    // builds screenshotPath and sets screenshotRequested, or no-ops if a capture is already in
    // flight (copy pending or still encoding).
    void requestScreenshot();
    void cleanup(VkDevice device) override;
    void onKey(GLFWwindow *w, int key, int action) override;
    void onCursorPos(GLFWwindow *w, double x, double y) override;

private:
    // ── SSBOs ─────────────────────────────────────────────────────────────────
    VkBuffer satInputBuf = VK_NULL_HANDLE; // device-local; sat_orbit.comp writes, sat_flare.comp reads
    VkDeviceMemory satInputMem = VK_NULL_HANDLE;
    VkBuffer satVisibleBuf = VK_NULL_HANDLE; // device-local, sat_flare.comp→vertex
    VkDeviceMemory satVisibleMem = VK_NULL_HANDLE;

    // ── Satellite picking / selection tracking ────────────────────────────────
    // pickedVisibleBuf mirrors just the selected satellite's 32-byte GpuSatVisible entry
    // each frame (host-visible, mapped once like glowBuf) so buildUI can reproject its
    // screen position without ever reading back the full (device-local) satVisibleBuf
    // except at the moment of an initial click. See pickSatelliteAt/projectSkyDirToScreen.
    VkBuffer pickedVisibleBuf = VK_NULL_HANDLE;
    VkDeviceMemory pickedVisibleMem = VK_NULL_HANDLE;
    void *pickedVisibleMapped = nullptr;
    int selectedSatIndex = -1;        // index into satOrbits[]/satVisibleBuf; -1 = no selection
    glm::vec3 lastPickedSkyDir{0.0f}; // previous frame's ENU sky direction for the selection
    float lastPickedFlare = 0.0f;     // previous frame's flareIntensity for the selection (>0 = on screen)
    // Cached info text, reformatted only when selectedSatIndex changes (see formatSelectedSatInfo).
    // Separate per-line buffers, not one multi-line string — Clay/UIRenderer text draws a single
    // line per CLAY_TEXT call with no embedded-newline support.
    // 7th slot (session follow-up): optional "Power output" line, filled only for satellites whose
    // primary surface uses AttitudeMode::SunTrackingTilted (see formatSelectedSatInfo); left empty
    // (and skipped by buildSelectedSatPanel's render loop) for everything else, including planets —
    // formatSelectedPlanetInfo never touches planetInfoLine[6].
    static constexpr int kSelInfoLines = 7;
    char selInfoLine[kSelInfoLines][40] = {};
    char planetInfoLine[kSelInfoLines][40] = {}; // same shape, filled by formatSelectedPlanetInfo

    // ── Orbit pipeline buffers ────────────────────────────────────────────────
    VkBuffer satOrbitBuf = VK_NULL_HANDLE; // device-local, uploaded once at init
    VkDeviceMemory satOrbitMem = VK_NULL_HANDLE;
    // mirrorNormalsBuf (persistent mirror lock/slew state) and reflectorTargetsBuf (the per-frame
    // CPU-compacted night-side ECI buffer) were removed 2026-08-06 — see CLAUDE.md's
    // TargetedReflector section. sat_orbit.comp now derives everything it needs (live AND
    // window-eval target positions) directly from reflectorTargetsECEFBuf below plus
    // SatOrbitPC::gmstNow, with no per-frame CPU rotation/compaction step and no persisted GPU
    // state — target selection and mirror orientation are both pure functions of sim time.
    //
    // C12 follow-up #33: reflectorTargetsECEF[]/reflectorTargetsRadiusM[] never change after
    // target generation, so this is uploaded ONCE (right after initConstellation() in init(), see
    // that call site) rather than refreshed per frame. xyz = unit ECEF direction, w = ground
    // radius incl. terrain elevation. Read by sat_orbit.comp (TargetedReflector target search).
    // beam_cloud_block.comp, the buffer's other former reader, was retired 2026-08-09 —
    // beam_self_march.comp doesn't need it (reconstructs ECEF from ReflectBeamsBuf's own ENU
    // offsets instead — see that shader's header).
    VkBuffer reflectorTargetsECEFBuf = VK_NULL_HANDLE; // host-visible+coherent, written once
    VkDeviceMemory reflectorTargetsECEFMem = VK_NULL_HANDLE;
    void *reflectorTargetsECEFMapped = nullptr;
    // beamCloudBlockBuf (per-TARGET cloud occlusion result, beam_cloud_block.comp's own output)
    // retired 2026-08-09 — beam_self_march.comp now writes blockAltM/blockOpacity directly into
    // each beam's own ReflectBeamsBuf entry (per BEAM, not per target), no intermediate buffer.
    // Reflect-Orbital ground beams (host-visible+coherent; written by sat_orbit.comp, indexed
    // by target identity + atomicMax, zeroed every frame via vkCmdFillBuffer — same pattern as
    // glowBuf, including CPU readback of the previous frame's contents for a diagnostic). Read by
    // cloud_march.comp (debug pointing rays only, as of C12 follow-up #44) and sat_sky.frag
    // (ground-spot direct lighting). See GpuReflectBeams.
    VkBuffer reflectBeamsBuf = VK_NULL_HANDLE;
    VkDeviceMemory reflectBeamsMem = VK_NULL_HANDLE;
    void *reflectBeamsMapped = nullptr;
    // beamCloudLightBuf — host-visible+coherent, written by the CPU each frame in recordCompute()
    // (a small capped list of individual real-beam light sources, no per-target aggregation — see
    // GpuBeamCloudLights' own comment for the 2026-08-09 design history), read by
    // cloud_march.comp's cloudMarchCS as a real per-sample directional light source.
    VkBuffer beamCloudLightBuf = VK_NULL_HANDLE;
    VkDeviceMemory beamCloudLightMem = VK_NULL_HANDLE;
    void *beamCloudLightMapped = nullptr;
    // Ground-beam compaction (perf follow-up) — host-visible+coherent, written by the CPU each
    // frame from reflectBeamsBuf's readback (same loop lastActiveBeamCount/beamProximityGlow use).
    VkBuffer groundBeamsBuf = VK_NULL_HANDLE;
    VkDeviceMemory groundBeamsMem = VK_NULL_HANDLE;
    void *groundBeamsMapped = nullptr;
    // Diagnostic readback (C12): how many of the 16 sectors currently hold an active beam, and
    // the straight-line distance (meters) from the observer to the nearest one's ground target —
    // one-frame-stale, same idiom as peakMagnitude. -1 = no active beams this/last frame.
    int lastActiveBeamCount = 0;
    // How many of those survived the observer-range cull into the compacted ground-spot list
    // (kMaxGroundBeams cap). Logged in perf snapshots alongside lastActiveBeamCount because the two
    // drive different shaders: lastActiveBeamCount bounds cloud_march.comp's per-pixel pointing-ray
    // loop, this one bounds sat_sky.frag's per-ground-pixel spot loop, and at a site like Anchorage
    // (itself a reflector target, few neighbouring targets) they diverge sharply.
    int lastGroundBeamCount = 0;
    float lastNearestBeamDistM = -1.0f;
    // 2026-08-09 debug instrumentation (BEAM_CLOUD_PLAN.md): raw min/max/avg blockOpacity across
    // every active beam this frame, computed straight from reflectBeamsBuf's readback with no
    // per-target aggregation/argmax in the way — see the Beams settings tab. If Max stays ~0.00
    // regardless of visible cloud cover, beam_self_march.comp's own march isn't finding cloud
    // (a shell/geometry/binding bug upstream of every consumer). If Max is meaningfully >0 but
    // rendering still looks unaffected, the bug is downstream in a consumer instead.
    float dbgBeamOpacityMin = 0.0f;
    float dbgBeamOpacityMax = 0.0f;
    float dbgBeamOpacityAvg = 0.0f;
    int dbgBeamOccludedCount = 0; // count of beams this frame with blockOpacity > 0.1
    int dbgBeamSampleCount = 0;   // same as lastActiveBeamCount, kept alongside for clarity
    // Beam-driven sky-glow "pollution dome" (C12 follow-up #31) — parallel to lightDomeBuf's
    // 16-sector scheme, but populated by ACTIVE Reflect-Orbital beams instead of a static
    // night-lights texture, so satellites/stars/Milky Way dim near a bright beam the same way
    // they already dim near real light pollution. Written by sat_orbit.comp (atomicMax per
    // sector, host-visible+coherent so the CPU can also read back the previous frame's contents —
    // same one-frame-stale idiom as reflectBeamsBuf/glowBuf). Read directly on the GPU by
    // sat_flare.comp and sat_sky.frag's Milky Way section; updateStars() reads the CPU-side
    // beamGlowDomeAz[] copy below. Deliberately a SEPARATE buffer from lightDomeBuf, not merged —
    // two independent phenomena that happen to share a consumption pattern.
    static constexpr int kNumBeamGlowSectors = 16;
    VkBuffer beamGlowDomeBuf = VK_NULL_HANDLE;
    VkDeviceMemory beamGlowDomeMem = VK_NULL_HANDLE;
    void *beamGlowDomeMapped = nullptr;
    float beamGlowDomeAz[kNumBeamGlowSectors]{}; // CPU-side copy of the previous frame's contents

    // ── sat_flare.comp descriptors / pipeline ─────────────────────────────────
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    // ── sat_orbit.comp descriptors / pipeline ─────────────────────────────────
    VkDescriptorSetLayout orbitDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool orbitDescPool = VK_NULL_HANDLE;
    VkDescriptorSet orbitDescSet = VK_NULL_HANDLE;
    VkPipelineLayout orbitPipeLayout = VK_NULL_HANDLE;
    VkPipeline orbitPipeline = VK_NULL_HANDLE;

    // ── Pipelines ─────────────────────────────────────────────────────────────
    VkPipelineLayout compPipeLayout = VK_NULL_HANDLE;
    VkPipeline compPipeline = VK_NULL_HANDLE;
    VkPipelineLayout skyBgPipeLayout = VK_NULL_HANDLE; // sky/ground background
    VkPipeline skyBgPipeline = VK_NULL_HANDLE;
    // Minimal stand-in used when debugDisableMask bit 262144 is set (Potato preset): sat_sky.frag
    // is ~490 ms/frame on a 2015 AMD GPU via MoltenVK and no quality knob touches that. Same
    // pipeline layout / descriptor set / render pass as skyBgPipeline — only the fragment module
    // differs (shaders/sat_sky_minimal.frag). Created and resized alongside skyBgPipeline.
    VkPipeline skyBgMinimalPipeline = VK_NULL_HANDLE;
    // Planetarium-tier stand-in used when debugDisableMask bit 524288 is set: sat_sky.frag compiled
    // a second time with -DSKY_LITE (shaders/sat_sky_lite.frag.spv) — heaviest subsystems #ifdef'd
    // out for GPUs where the full shader collapses occupancy but the minimal shader is too austere.
    // Same layout / descriptor set / render pass; created and resized alongside skyBgPipeline.
    VkPipeline skyBgLitePipeline = VK_NULL_HANDLE;
    VkPipelineLayout drawPipeLayout = VK_NULL_HANDLE;
    VkPipeline drawPipeline = VK_NULL_HANDLE;

    // ── Resolution scaling (session 29) ─────────────────────────────────────────
    // Below 100%, sky_bg renders to a low-res offscreen target then gets blitted (linear-
    // filtered upscale) into the swapchain image before the main render pass opens — see
    // recordPrePass/activeRenderPass. At the default 1.0 this whole path is skipped and behavior
    // is byte-identical to before this feature existed. Reuses skyBgPipeLayout (same push
    // constants/descriptor set) — only the pipeline's viewport/render-pass/depth-state differ.
    // Depth is deliberately NOT blitted (depth-format blit support isn't spec-guaranteed, a real
    // portability concern specifically on the lower-end hardware this feature targets) — a known,
    // accepted tradeoff: satellites/stars are not occluded by terrain while scaled below 100%.
    float renderScale = 1.0f;                          // [0.5, 1.0], Settings > Display "Render scale"
    VkRenderPass skyLowResRenderPass = VK_NULL_HANDLE; // color-only, CLEAR, finalLayout=TRANSFER_SRC_OPTIMAL
    VkImage skyLowResColorImg = VK_NULL_HANDLE;
    VkDeviceMemory skyLowResColorMem = VK_NULL_HANDLE;
    VkImageView skyLowResColorView = VK_NULL_HANDLE;
    VkFramebuffer skyLowResFramebuffer = VK_NULL_HANDLE;
    VkPipeline skyLowResPipeline = VK_NULL_HANDLE; // low-res viewport variant of skyBgPipeline
    VkExtent2D skyLowResExtent{};

    // Moon state (updated each frame in updatePositions)
    glm::vec3 moonDirECI{1, 0, 0};    // unit vector toward moon in ECI (equatorial orbit)
    glm::vec4 moonDirENU{0, 1, 0, 0}; // xyz = moon dir in ENU, w = illuminated fraction

    // ── Stars ─────────────────────────────────────────────────────────────────
    struct StarRecord
    {
        glm::vec3 eciDir;   // unit vector toward star in ECI (J2000)
        float rawIntensity; // magnitude-derived brightness (no night factor)
        glm::vec3 color;    // spectral color from B-V index
        float angSize;      // point sprite size in pixels
    };
    std::vector<StarRecord> starRecords;
    VkBuffer starBuf = VK_NULL_HANDLE;
    VkDeviceMemory starMem = VK_NULL_HANDLE;
    void *starMapped = nullptr;
    VkDescriptorSetLayout starDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool starDescPool = VK_NULL_HANDLE;
    VkDescriptorSet starDescSet = VK_NULL_HANDLE;
    VkPipelineLayout starPipeLayout = VK_NULL_HANDLE;
    VkPipeline starPipeline = VK_NULL_HANDLE;
    uint32_t starCount = 0;

    // ── Planets ──────────────────────────────────────────────────────────────
    // Reuses starPipeline/starPipeLayout/starDescLayout unchanged (same GpuSatVisible-shaped
    // point-sprite pipeline) — only a second small host-mapped buffer + descriptor set, drawn with
    // a second vkCmdDraw call. See "Subsystem: Planets" in CLAUDE.md.
    VkBuffer planetBuf = VK_NULL_HANDLE;
    VkDeviceMemory planetMem = VK_NULL_HANDLE;
    void *planetMapped = nullptr;
    VkDescriptorPool planetDescPool = VK_NULL_HANDLE; // starDescPool is sized maxSets=1, so this
                                                      // gets its own tiny pool for one more set
    VkDescriptorSet planetDescSet = VK_NULL_HANDLE;   // allocated with starDescLayout (same shape)

    // ── Simulation state ──────────────────────────────────────────────────────
    SkyCamera camera;
    // simTime is split into an integer day counter and a double-precision
    // seconds-within-day value.  This avoids accumulated float precision loss
    // when a large J2000 epoch base is added to a small per-frame delta.
    // simSecInDay is re-based to [0, 86400) each frame so it stays small.
    // Use simTimeDouble() wherever a full-precision double is needed.
    int64_t simDayJ2000 = 0;     // integer days since J2000 (2000-01-01 12:00 TT)
    double simSecInDay = 0.0;    // seconds within current day [0, 86400)
    int64_t simInitDayJ2000 = 0; // values at construction — used for display
    double simInitSecInDay = 0.0;
    // Bug fix: this was 1 ("10x"), so a launch that never touched settings.json's "time" key
    // (or replayed the intro, which doesn't re-run loadSettings) came up at 10x instead of the
    // intended 1x default — the inconsistency the user reported ("time base doesn't seem to be
    // consistently set on bootup"). loadSettings() still overrides this with whatever was
    // persisted; this is only the compiled-in fallback.
    int timeScaleIdx = 0;
    bool timePaused = false;
    float timeDir = 1.0f; // +1 = forward, -1 = reverse
    // Observer position/facing in Earth-fixed ECEF — canonical movement state.
    // obsLatDeg / obsLonDeg are display caches derived each frame; camera.azDeg is also derived.
    // Initial: lat=67°S lon=67°W, facing north.
    //   obsDir    = (cos(-67°)cos(-67°), cos(-67°)sin(-67°), sin(-67°))
    //             ≈ (0.1527, -0.3596, -0.9205)
    //   obsFacing = (-sin(-67°)cos(-67°), -sin(-67°)sin(-67°), cos(-67°))
    //             ≈ (0.3596, -0.8473, 0.3907)
    glm::vec3 obsDir = {0.1527f, -0.3596f, -0.9205f}; // unit position vector
    glm::vec3 obsFacing = {1, 0, 0};                  // unit tangent (forward direction, north)
    float obsLatDeg = -67.0f;                         // display cache — derived from obsDir
    float obsLonDeg = -67.0f;                         // display cache — derived from obsDir
    float obsTerrainH = 0.0f;                         // terrain elevation at observer lat/lon (m)
    float obsHeightOffset = 0.0f;                     // user-controlled height above terrain (m, Q/E/Z)
    // 2026-08-09 (in-app finding: beam/cloud-glow ground spots visibly "drag" behind the observer
    // while moving): satENU/reflectDirENU/targetENU read back from reflectBeamsBuf each frame are
    // expressed in the East/North/Up basis at whatever obsDir PRODUCED them (beam_self_march.comp's
    // own push-constant fill, one frame ago) — that basis ROTATES as obsLatDeg/obsLonDeg change via
    // WASD movement, so reinterpreting last frame's ENU numbers against this frame's obsPos without
    // re-projecting them is a real rotation error, not just ordinary 1-frame staleness. These cache
    // exactly the obsDir/obsEffH that produced the data currently sitting in reflectBeamsMapped, so
    // recordCompute's readback loop can un-rotate it into the current frame's basis before use.
    // Defaulted to match obsDir/0 so the very first frame's correction is a no-op.
    glm::vec3 lastBeamObsDir = {0.1527f, -0.3596f, -0.9205f};
    float lastBeamObsEffH = 0.0f;
    // Persistent cloud-light pools (2026-08-12) — see TrackedBeamLight's own comment for the whole
    // design. Two pools with independently reserved eviction budgets, exactly as the previous
    // per-frame build had, so a busy target's summed intensity can never starve lone transiting
    // beams out of the list. NOTE these are the one piece of cross-frame state in this subsystem;
    // everything else in the readback loop is rebuilt from scratch each frame. rebase() must NOT be
    // applied to them — they are stored in Earth-fixed ECEF precisely so that they need no such
    // correction (the 2026-08-11 revert is the cautionary tale).
    TrackedBeamLight trackedClusters[kMaxClusterCloudLights]{};
    TrackedBeamLight trackedIndividuals[kMaxIndividualCloudLights]{};
    // key -> slot+1 (0 = empty), rebuilt from live slots at the top of every build.
    uint32_t trackedClusterHash[kTrackedLightHashSize]{};
    uint32_t trackedIndividualHash[kTrackedLightHashSize]{};
    // Live-slot counts, for the Beams settings tab's diagnostics readout.
    int lastClusterLightCount = 0;
    int lastIndividualLightCount = 0;
    uint32_t activeSatCount = 0;
    uint32_t visibleCount = 0;   // above-horizon sats this frame (UI display)
    uint32_t gpuSatCount = 0;    // in-frustum sats written to GPU buffer
    float loopMs = 0.0f;         // satellite loop time last frame (milliseconds)
    float peakMagnitude = 99.0f; // brightest steady-state sat magnitude this frame

    // ── GPU frame timing (perf HUD, Display settings tab) ──────────────────────
    // EMA-smoothed breakdown of ctx.timestampMs, updated once per frame in
    // updateGpuTimingStats(). One-frame-stale by construction (same pattern as
    // peakMagnitude above): App resolves the query pool right after the fence wait,
    // before buildUI/recordCompute run, so these hold the previous completed frame's
    // GPU time when buildUI reads them, and get refreshed for the following frame's
    // display at the top of recordCompute().
    float gpuMsSmoothed[8] = {};     // scene depth, beam cloud block, orbit compute, cloud march,
                                     // flare compute, sky background draw, satellite+star draw,
                                     // UI overlay — order fixed by VulkanContext's slot table;
                                     // kPerfLabels[] and savePerfSnapshot() mirror it
    float gpuMsTotalSmoothed = 0.0f; // whole-frame GPU time
    // UNSMOOTHED copies of the same eight deltas, written by the same updateGpuTimingStats() call.
    // The EMA above exists so the HUD numbers don't flicker; the automated knockout sweep wants
    // the opposite (a clean per-frame sample it can average over its OWN fixed window), and reading
    // an EMA there would have meant either waiting out its ~40-frame settle at every one of the
    // sweep's kDebugToggleCount+1 steps or silently letting the previous step bleed into the next.
    float gpuMsRaw[8] = {};
    float gpuMsRawTotal = 0.0f;

    // ── CPU frame timing (2026-08-10) ─────────────────────────────────────────
    // The GPU side of the Anchorage worst-case work brought Medium to 15.5 ms GPU, at which point
    // the Release wall-clock frame was 18.3 ms — a ~2.8 ms non-GPU remainder that was NEARLY
    // IDENTICAL at Planetarium (2.73 ms against a 6.0 ms GPU frame, i.e. 31% of that frame). Fixed
    // per-frame cost that doesn't scale with rendering load is exactly the shape the GPU timestamp
    // buckets were built to expose, and nothing equivalent existed on the CPU side — so the
    // remainder was a single opaque number and any attempt to shrink it would have been guesswork.
    // This is the same instrument, same conventions: raw per-frame values, an EMA for the HUD, and
    // the sweep/snapshot logging the raw ones.
    //
    // ONE-FRAME-STALE, and deliberately so: timers accumulate into cpuAccumMs[] as the frame runs,
    // and beginCpuFrameTiming() (first thing in buildUI, the first sim call of a frame) publishes
    // the completed previous frame into cpuMsRaw[]/cpuMsSmoothed[] and clears the accumulator. That
    // matches gpuMsRaw[]'s own staleness, so a sweep step samples a CPU frame and a GPU frame from
    // the same moment rather than one lagging the other.
    enum CpuBucket
    {
        CPU_BUILD_UI = 0,     // Clay layout for the whole HUD/settings window
        CPU_UPDATE_POSITIONS, // sun/moon/planets/observer basis (O(1), not per-satellite)
        CPU_BEAM_READBACK,    // reflectBeamsBuf readback + sort + clustering + ground-beam top-K
        CPU_UPDATE_STARS,     // per-star ENU + suppression chain over the catalogue
        CPU_LIGHT_DOME,       // updateLightPollutionDome's 16 sectors x 4 radii
        CPU_UPDATE_PLANETS,   // 6 planets, render-ready entries
        CPU_COUNT,
    };
    float cpuAccumMs[CPU_COUNT] = {};    // accumulating, current (incomplete) frame
    float cpuMsRaw[CPU_COUNT] = {};      // last COMPLETE frame, unsmoothed — sweep reads this
    float cpuMsSmoothed[CPU_COUNT] = {}; // EMA, for the HUD
    void beginCpuFrameTiming();

    // Scoped accumulator. Deliberately adds rather than assigns, so a bucket whose work happens in
    // several places (or inside a loop) can be timed with several instances in the same frame.
    struct CpuTimer
    {
        float *acc;
        std::chrono::steady_clock::time_point t0;
        explicit CpuTimer(float &a) : acc(&a), t0(std::chrono::steady_clock::now()) {}
        ~CpuTimer()
        {
            *acc += std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        }
        CpuTimer(const CpuTimer &) = delete;
        CpuTimer &operator=(const CpuTimer &) = delete;
    };

    // ── Perf knockout toggles (profiling-only; not persisted) ──────────────────
    // Bitmask sent to sat_sky.frag as SatDrawPC::debugDisableMask, so the individual cost of
    // the terrain march / atmosphere loop / sun optical-depth sub-march / ocean sky-reflection
    // loop / airglow-red supplemental march / aurora curtain march / cloud self-shadow light cone
    // / Reflect-Orbital beams / cloud shadow map can be measured in isolation via gpuMsSmoothed
    // deltas, without needing a GPU capture tool. See the dbgSkip* helpers near the top of
    // sat_sky.frag for the bit assignments (1,2,4,8,16,32); bit 64 (cloud self-shadow cone) and
    // bit 128 (Reflect-Orbital beam volumetric term) are checked directly in cloud_march.comp;
    // bit 128 (beam ground-spot term) and bit 256 (cloud shadow map) are also checked in
    // sat_sky.frag (128 gates both consumers of the same feature, checked in both shaders).
    // Bit 512 gates the beam_self_march.comp DISPATCH itself (2026-08-09 — repurposed from
    // beam_cloud_block.comp's own identical producer-side skip bit, now retired along with that
    // pass), in recordCompute() — no shader reads it. Bits 256 and 512 are the two producer-side
    // knockouts; every other bit disables a consumer block inside a shader.
    //
    // 2026-08-10 (Anchorage worst-case profiling session) added four bits for blocks that had no
    // knockout at all and were therefore permanently invisible inside a lumped timestamp bucket:
    // 8192 = the per-pixel Reflect-Orbital beam POINTING-RAY loop in cloud_march.comp (up to 2048
    // iterations per half-res pixel; bit 128 never reached it — CLAUDE.md flagged that gap and this
    // closes it), 16384 = cirrusMarchCS, 32768 = cloudMarchCS (the volumetric low/mid march itself),
    // 65536 = sat_sky.frag's unconditional 64-bin satellite sky-glow loop. 8192 and 65536 are also
    // the two blocks no graphics preset could reach, so they were paid in full even at Planetarium.
    //
    // The authoritative bit/label/json-key table is kDebugToggles at the top of SatelliteSimUI.cpp.
    uint32_t debugDisableMask = 0;

    // ── Automated knockout sweep (profiling-only) ──────────────────────────────
    // "Run knockout sweep" (Display tab) walks the kDebugToggles table on its own — baseline first,
    // then one step per bit — holding each mask for a fixed frame window and averaging gpuMsRaw[]
    // over it, then appends ONE profile_log.jsonl record carrying every step's bucket breakdown
    // plus its delta against the baseline.
    //
    // Why this exists: doing it by hand is ~18 checkbox toggles x 8 numbers read off a HUD, which is
    // both slow and unreliable — the scene has to hold still for the whole session or the steps
    // aren't comparable, and the HUD values are EMA-smoothed, so a hand capture taken too soon
    // after a toggle silently reads a blend of two configurations. The sweep pauses sim time for
    // its duration (restored afterwards) so every step measures the same frame.
    //
    // THE BASELINE IS THE CURRENT MASK, NOT ZERO — this was a real defect in the first version
    // (2026-08-10) and is the single thing most likely to be "fixed" back into a bug. Zeroing the
    // mask for the baseline means that on any preset which knocks effects out (Planetarium disables
    // eight of them; Low three), the sweep silently RE-ENABLES all of them and then measures the
    // cost of removing each one again. That is a valid cost table for "this preset's slider values
    // with every effect on", but it is NOT a profile of the preset as shipped, and it inflates the
    // baseline — the Planetarium sweep read 18.73 ms against a real 10.4 ms. Anchoring to the live
    // mask instead makes every step answer the question actually being asked: "of what this preset
    // still renders, what does each piece cost me?" Bits already set in the baseline have nothing
    // left to remove, so they are skipped outright and reported as already_disabled rather than
    // logged as a meaningless ~0 ms row.
    //
    // kDebugToggleSlots sizes hovDebugToggle[] and the accumulators below; the static_assert in
    // startKnockoutSweep() keeps it honest.
    static constexpr int kDebugToggleSlots = 18;
    static constexpr int kSweepSettleFrames = 6;  // discard after a mask change — covers the
                                                  // one-frame-stale timestamp readback plus a
                                                  // little driver/clock hysteresis
    static constexpr int kSweepSampleFrames = 24; // then average this many consecutive frames
    // A sweep now measures the baseline TWICE — step 0 at the start and one extra step at the end,
    // both with the baseline mask — and logs the pair plus their drift. Two reasons, both seen in
    // real captures: (1) the 2026-08-10 Release Medium sweep showed a systematic ~+0.32 ms
    // orbit_compute delta on ten unrelated knockout rows, i.e. the FIRST window alone was elevated,
    // which silently inflates every cost_ms by the same amount; (2) "the scene drifted between
    // captures" has been the standing caveat on every cross-sweep comparison this session, and a
    // start/end pair measures that drift instead of leaving it to be argued about. Large drift
    // means the whole sweep is suspect and should be retaken.
    bool sweepActive = false;
    int sweepStep = 0;           // 0 = baseline, 1..sweepBitCount = sweepBits[step-1],
                                 // sweepBitCount+1 = baseline re-measure
    int sweepFrame = 0;          // frames elapsed within the current step
    uint32_t sweepSavedMask = 0; // the baseline mask — restored when the sweep finishes
    // Indices into kDebugToggles for the rows this sweep will actually measure: everything NOT
    // already disabled by sweepSavedMask. Sized for the whole table (the mask==0 case).
    int sweepBits[kDebugToggleSlots] = {};
    int sweepBitCount = 0;
    bool sweepSavedPaused = false;
    // Sized +2, not +1: baseline, every measured bit, and the trailing baseline re-measure.
    float sweepAccum[kDebugToggleSlots + 2][8] = {}; // [step][bucket] running sum, ms
    float sweepAccumTotal[kDebugToggleSlots + 2] = {};
    // CPU wall-clock frame time, averaged over the SAME windows as the GPU buckets. Originally the
    // record carried only `cpuDt` — a single instantaneous frame sampled on whichever frame the
    // sweep happened to finish, right after a mask change, while every GPU number beside it was a
    // 24-frame average. Comparing the two produced a GPU-vs-CPU gap that jumped around by several
    // ms between otherwise-identical captures. Now that the GPU side of the Anchorage work is done
    // and CPU frame time is the binding constraint, that scalar has to be measured as carefully as
    // the GPU ones are.
    float sweepAccumCpu[kDebugToggleSlots + 2] = {};
    // Per-bucket CPU breakdown, same windows again. A knockout bit should barely move these (they
    // are almost all GPU-independent work), so a row that DOES move one is telling you the bit has
    // a CPU-side consumer you didn't know about — which is the other half of why this is logged.
    float sweepAccumCpuBucket[kDebugToggleSlots + 2][CPU_COUNT] = {};
    bool hovRunSweep = false;
    float sweepDoneMsgTimer = 0.0f; // seconds remaining to show the "Sweep saved" confirmation
    // NAME IS STALE, BEHAVIOR IS NOT A BUG — read this before "fixing" the default again. This
    // started as a literal debug-only visualization (a green line) back when the volumetric tube
    // it now replaces was live. When that tube was thrown out for graphics/perf reasons, this ray
    // was reworked in a later session (realistic color, altitude attenuation) into THE actual beam
    // visual — it is the genuine, production Reflect-Orbital beam rendering now, not a diagnostic
    // overlay, and correctly defaults to true. (A same-session pass mistakenly "fixed" this to
    // false, reasoning from the still-stale shader-side header comment in cloud_march.comp that
    // calls it "Opt-in and off by default" — reverted; see BEAM_CLOUD_PLAN.md 2026-08-09.) It
    // draws each active beam's ACTUAL current mirror-pointing direction as a long ray, occluded by
    // terrain ONLY — genuinely no cloud awareness at all, which is the real, still-open bug (this
    // ray is what a player actually sees, so its lack of cloud occlusion is why beams read as
    // passing straight through cloud regardless of the ground-spot/glow fixes elsewhere). Being
    // replaced by the per-beam march project — see BEAM_CLOUD_PLAN.md. Not persisted to
    // settings.json (unlike debugDisableMask, which is).
    // See GpuReflectBeam::reflectDirENU and cloud_march.comp (C12 follow-up #12).
    bool showBeamDebugRays = true;

    // ── Sky glow SSBO ─────────────────────────────────────────────────────────
    // Written by sat_flare.comp each frame via binned atomicMax; read by sat_sky.frag.
    VkBuffer glowBuf = VK_NULL_HANDLE;
    VkDeviceMemory glowMem = VK_NULL_HANDLE;
    void *glowMapped = nullptr;

    // ── Light pollution dome SSBO ─────────────────────────────────────────────
    // Host-visible, updated each frame by updateLightPollutionDome() (CPU); read by
    // sat_flare.comp (satellites) and directly by updateStars() (stars, no upload needed —
    // same array, CPU already has it). 16 azimuth sectors (bumped from 8 — session 26 follow-up:
    // 8 hard-edged 45° wedges showed visible blocky transitions over wide, fairly uniform bright
    // regions like Europe); both consumers additionally interpolate between sector centers rather
    // than hard-binning, so this no longer needs to match GlowBuf's own (unrelated) 8-sector
    // azBin scheme — decoupled on purpose.
    static constexpr int kNumLightSectors = 16;
    VkBuffer lightDomeBuf = VK_NULL_HANDLE;
    VkDeviceMemory lightDomeMem = VK_NULL_HANDLE;
    void *lightDomeMapped = nullptr;
    float lightDomeAz[kNumLightSectors]{}; // CPU-side copy, shared with updateStars()
    // Raw (pre-lightPollutionGain) max local city-brightness across all sectors, written each
    // frame by updateLightPollutionDome() and consumed immediately after by recordCompute()'s
    // mwSuppressEased hysteresis step — see that member's comment. Transient, not persisted.
    float mwPollutionRaw = 0.0f;
    VkDescriptorSetLayout skyDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool skyDescPool = VK_NULL_HANDLE;
    VkDescriptorSet skyDescSet = VK_NULL_HANDLE;
    // Noise texture (binding 1): RGBA PNG tiled for lens-flare angular corona variation.
    VkImage noiseTex = VK_NULL_HANDLE;
    VkDeviceMemory noiseTexMem = VK_NULL_HANDLE;
    VkImageView noiseTexView = VK_NULL_HANDLE;
    VkSampler noiseSampler = VK_NULL_HANDLE;
    // Moon texture (binding 2): near-side face disc image for surface detail.
    VkImage moonTex = VK_NULL_HANDLE;
    VkDeviceMemory moonTexMem = VK_NULL_HANDLE;
    VkImageView moonTexView = VK_NULL_HANDLE;
    VkSampler moonSampler = VK_NULL_HANDLE;
    // Earth day texture (binding 3): 8K equirectangular colour map.
    VkImage earthDayImg = VK_NULL_HANDLE;
    VkDeviceMemory earthDayMem = VK_NULL_HANDLE;
    VkImageView earthDayView = VK_NULL_HANDLE;
    VkSampler earthDaySampler = VK_NULL_HANDLE;
    uint32_t earthDayMips = 1;
    // Earth night texture (binding 4): 8K equirectangular night-lights map.
    VkImage earthNightImg = VK_NULL_HANDLE;
    VkDeviceMemory earthNightMem = VK_NULL_HANDLE;
    VkImageView earthNightView = VK_NULL_HANDLE;
    VkSampler earthNightSampler = VK_NULL_HANDLE;
    uint32_t earthNightMips = 1;
    // City-detail world-fixed offset (metres): the observer's own cumulative north/east
    // displacement, accumulated every frame from consecutive obsLatDeg/obsLonDeg deltas. Added
    // to hitPt.xy in sat_sky.frag's "City detail texture blend" to cancel that coordinate's
    // observer-relative drift with a plain translation — see the comment there for why a
    // translation is sufficient (no basis/anchor-snap machinery needed). Packed into CloudParams
    // pad1/pad2. Double precision on CPU is cheap insurance against long play sessions; only
    // cast to float when uploading.
    double cityOffsetEastM = 0.0;
    double cityOffsetNorthM = 0.0;
    bool cityOffsetInit = false;
    double cityPrevObsLatRad = 0.0;
    double cityPrevObsLonRad = 0.0;
    // City day/night detail textures (bindings 14/15): small tileable high-frequency maps,
    // blended onto dayColor/nightColor near cities (see terrain block in sat_sky.frag). Hardcoded
    // tiling scale + distance fade, no CloudParams UBO fields.
    VkImage cityDayDetailImg = VK_NULL_HANDLE;
    VkDeviceMemory cityDayDetailMem = VK_NULL_HANDLE;
    VkImageView cityDayDetailView = VK_NULL_HANDLE;
    VkSampler cityDayDetailSampler = VK_NULL_HANDLE;
    uint32_t cityDayDetailMips = 1;
    VkImage cityNightDetailImg = VK_NULL_HANDLE;
    VkDeviceMemory cityNightDetailMem = VK_NULL_HANDLE;
    VkImageView cityNightDetailView = VK_NULL_HANDLE;
    VkSampler cityNightDetailSampler = VK_NULL_HANDLE;
    uint32_t cityNightDetailMips = 1;
    // Earth specular texture (binding 6): 8K R8_UNORM ocean mask (white=ocean, black=land).
    // Used to gate the wave normal + specular glint material on sea-level sphere hits.
    VkImage earthSpecImg = VK_NULL_HANDLE;
    VkDeviceMemory earthSpecMem = VK_NULL_HANDLE;
    VkImageView earthSpecView = VK_NULL_HANDLE;
    VkSampler earthSpecSampler = VK_NULL_HANDLE;
    uint32_t earthSpecMips = 1;
    // Earth cloud map (binding 7): 8K R8_UNORM grayscale cloud coverage map.
    VkImage earthCloudsImg = VK_NULL_HANDLE;
    VkDeviceMemory earthCloudsMem = VK_NULL_HANDLE;
    VkImageView earthCloudsView = VK_NULL_HANDLE;
    VkSampler earthCloudsSampler = VK_NULL_HANDLE;
    uint32_t earthCloudsMips = 1;
    // Cloud 3D noise volume (binding 8): 128³ RGBA Perlin-Worley/Worley, baked once at init.
    VkImage cloudNoiseImg = VK_NULL_HANDLE;
    VkDeviceMemory cloudNoiseMem = VK_NULL_HANDLE;
    VkImageView cloudNoiseView = VK_NULL_HANDLE;
    VkSampler cloudNoiseSampler = VK_NULL_HANDLE;
    // Cloud/cirrus domain-warp 3D noise volume (cloud_march.comp binding 9): 128³ RGB, tiling
    // period 16 cells, baked once at init. Replaces cloudWarpOffset's old 3 live warpPerlin3
    // calls with a single texture read — see cloud_warp_noise.comp for the tiling/repetition
    // trade-off this bake deliberately accepted.
    VkImage cloudWarpNoiseImg = VK_NULL_HANDLE;
    VkDeviceMemory cloudWarpNoiseMem = VK_NULL_HANDLE;
    VkImageView cloudWarpNoiseView = VK_NULL_HANDLE;
    VkSampler cloudWarpNoiseSampler = VK_NULL_HANDLE;
    // Aurora 3D noise volume (binding 16): 1024x16x256 RGBA8 — R=curtain fold base,
    // G/B=column-window colA/colB, baked once at init. See aurora_noise.comp.
    VkImage auroraNoiseImg = VK_NULL_HANDLE;
    VkDeviceMemory auroraNoiseMem = VK_NULL_HANDLE;
    VkImageView auroraNoiseView = VK_NULL_HANDLE;
    VkSampler auroraNoiseSampler = VK_NULL_HANDLE;
    // Milky Way skybox texture (binding 13): 8K equirectangular galactic panorama.
    VkImage milkyWayImg = VK_NULL_HANDLE;
    VkDeviceMemory milkyWayMem = VK_NULL_HANDLE;
    VkImageView milkyWayView = VK_NULL_HANDLE;
    VkSampler milkyWaySampler = VK_NULL_HANDLE;
    uint32_t milkyWayMips = 1;
    // Cloud params UBO (binding 9): host-visible, persistently mapped, updated each frame.
    VkBuffer cloudParamsBuf = VK_NULL_HANDLE;
    VkDeviceMemory cloudParamsMem = VK_NULL_HANDLE;
    void *cloudParamsMapped = nullptr;
    // ── Half-resolution cloud march output (C15-perf) ─────────────────────────
    // Written by cloud_march.comp each frame at half ctx.swapExtent; sampled by sat_sky.frag as
    // bindings 10/11 (skyDescSet). Recreated in onResize (swapchain-size-dependent, unlike every
    // other image in this class). Target A: rgb=B_total additive radiance, a=tCloudOcclude.
    // Target B: rgb=A_total multiplicative attenuation, a=cloudBlock sun-dimming scalar.
    VkImage cloudMarchTargetAImg = VK_NULL_HANDLE;
    VkDeviceMemory cloudMarchTargetAMem = VK_NULL_HANDLE;
    VkImageView cloudMarchTargetAView = VK_NULL_HANDLE;
    VkImage cloudMarchTargetBImg = VK_NULL_HANDLE;
    VkDeviceMemory cloudMarchTargetBMem = VK_NULL_HANDLE;
    VkImageView cloudMarchTargetBView = VK_NULL_HANDLE;
    VkSampler cloudMarchSampler = VK_NULL_HANDLE; // shared by both targets; resolution-independent
    VkDescriptorSetLayout cloudMarchDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool cloudMarchDescPool = VK_NULL_HANDLE;
    VkDescriptorSet cloudMarchDescSet = VK_NULL_HANDLE;
    VkPipelineLayout cloudMarchPipeLayout = VK_NULL_HANDLE;
    VkPipeline cloudMarchPipeline = VK_NULL_HANDLE;
    // ── Shared scene depth (pipeline unification) ─────────────────────────────
    // Half ctx.swapExtent, R32_SFLOAT, written by scene_depth.comp at the very top of
    // recordCompute. Holds the LINEAR distance in metres along each view ray to the first
    // terrain/ocean surface, or kNoSurfaceT (1e30) for rays that reach space.
    //
    // Same sizing rule as cloudMarchTargetA/B — half of the SWAP extent, deliberately
    // independent of renderScale — so cloud_march.comp reads it 1:1 with texelFetch on its own
    // dispatch grid, while fragment consumers use gl_FragCoord.xy / pc.screenSizePx.
    //
    // R32, not R16: distances reach 3.6e6 m from LEO and half-float saturates at 65504. That
    // overflow is exactly the bug this buffer retires (see tEnterCombined) — do not shrink it.
    // Recreated in onResize alongside the cloud targets; see there for the descriptor patches.
    VkImage sceneDepthImg = VK_NULL_HANDLE;
    VkDeviceMemory sceneDepthMem = VK_NULL_HANDLE;
    VkImageView sceneDepthView = VK_NULL_HANDLE;
    VkSampler sceneDepthSampler = VK_NULL_HANDLE; // resolution-independent; created once
    VkDescriptorSetLayout sceneDepthDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool sceneDepthDescPool = VK_NULL_HANDLE;
    VkDescriptorSet sceneDepthDescSet = VK_NULL_HANDLE;
    VkPipelineLayout sceneDepthPipeLayout = VK_NULL_HANDLE;
    VkPipeline sceneDepthPipeline = VK_NULL_HANDLE;

    // ── Flare/corona render-to-texture pipeline (flare architecture overhaul) ─────────────────
    // See FlareSourcePC's comment above for the three-stage design. flareExtent is a QUARTER of
    // ctx.swapExtent (one step smaller than scene_depth/cloud_march's half-res convention — this
    // buffer is deliberately going to be blurred, not sampled 1:1), independent of renderScale,
    // recreated in onResize alongside those.
    VkExtent2D flareExtent{};
    VkImage flareSourceImg = VK_NULL_HANDLE; // RGBA16F, COLOR_ATTACHMENT|STORAGE|SAMPLED
    VkDeviceMemory flareSourceMem = VK_NULL_HANDLE;
    VkImageView flareSourceView = VK_NULL_HANDLE;
    VkImage flareScratchImg = VK_NULL_HANDLE; // RGBA16F, STORAGE|SAMPLED — compute ping-pong + final composite source
    VkDeviceMemory flareScratchMem = VK_NULL_HANDLE;
    VkImageView flareScratchView = VK_NULL_HANDLE;
    VkSampler flareSampler = VK_NULL_HANDLE;             // shared by both images; resolution-independent, created once
    VkRenderPass flareSourceRenderPass = VK_NULL_HANDLE; // single color attachment, CLEAR, finalLayout=COLOR_ATTACHMENT_OPTIMAL
    VkFramebuffer flareSourceFramebuffer = VK_NULL_HANDLE;
    // Stage 1 (render): reuses the EXISTING descLayout/descSet (satVisibleBuf binding 1,
    // cloudTargetA/B bindings 5/6 are already there for sat_point.frag; binding 7 (sceneDepthTex)
    // is new — see createDescriptors()) — own pipeline layout only because the push constant type
    // (FlareSourcePC) differs from drawPipeLayout's SatDrawPC.
    VkPipelineLayout flareSourcePipeLayout = VK_NULL_HANDLE;
    VkPipeline flareSourcePipeline = VK_NULL_HANDLE;
    // Stage 2 (blur/streak): own tiny descriptor set — two STORAGE_IMAGE bindings, nothing this
    // codebase's existing sets already provide.
    VkDescriptorSetLayout flareBlurDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool flareBlurDescPool = VK_NULL_HANDLE;
    VkDescriptorSet flareBlurDescSet = VK_NULL_HANDLE;
    VkPipelineLayout flareBlurPipeLayout = VK_NULL_HANDLE;
    VkPipeline flareBlurPipeline = VK_NULL_HANDLE;
    // Stage 3 (composite): own tiny descriptor set — one COMBINED_IMAGE_SAMPLER binding
    // (flareScratchImg, sampled directly in VK_IMAGE_LAYOUT_GENERAL — legal, and this image is
    // small enough that skipping a layout-transition barrier costs nothing measurable). Targets
    // the MAIN render pass (ctx.renderPass), same render-pass-compatibility trick drawPipeline
    // already relies on to also work under ctx.renderPassLoad at renderScale<1.
    VkDescriptorSetLayout flareCompositeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool flareCompositeDescPool = VK_NULL_HANDLE;
    VkDescriptorSet flareCompositeDescSet = VK_NULL_HANDLE;
    VkPipelineLayout flareCompositePipeLayout = VK_NULL_HANDLE;
    VkPipeline flareCompositePipeline = VK_NULL_HANDLE;
    // Ocean-glint list (see GpuOceanGlintBuf) — device-local, zeroed every frame like glowBuf.
    VkBuffer oceanGlintBuf = VK_NULL_HANDLE;
    VkDeviceMemory oceanGlintMem = VK_NULL_HANDLE;
    // User tunables (Settings > Display), persisted in settings.json. First-pass defaults —
    // expected to need retuning once seen in-app, same as every other constant this session.
    float flareGlowGain = 0.005f;       // post-composite overall multiplier — was 1.0f, 100x the
                                        // UI slider's actual [0, 0.01] range (SatelliteSimUI.cpp),
                                        // so a fresh settings.json (or any older save predating
                                        // this key) booted the flare glow fully maxed out
    float flareStreakGain = 0.35f;      // per-tap streak/godray strength (flare_blur.comp mode=2)
    float sunFlareRefIntensity = 40.0f; // fixed reference brightness for the sun's virtual point
                                        // in the flare-source buffer — NOT a slider (kept small in
                                        // scope this round); tune in code if the sun's godray/glow
                                        // contribution looks under- or over-powered relative to
                                        // satellites once seen in-app.

    // ── Long-exposure trail pipeline (fun side feature) ──────────────────────────────────────
    // Persistent, real-time-decayed accumulator for satellite/star/planet point splats — leaves
    // fading trails behind them, most dramatic at high timeScaleIdx. Same three-stage shape as the
    // flare pipeline above (fade/splat entirely inside recordCompute(), composite draw appended in
    // recordDraw()), reusing sat_point.vert/frag and star_point.vert/frag UNCHANGED against a new
    // render pass — no new pipeline LAYOUTS needed for the splat stage, only new VkPipeline objects
    // targeting trailAccumRenderPass instead of ctx.renderPass.
    //
    // Decay is REAL-TIME (wall-clock), not sim-time: a fixed real-world persistence window, so
    // cranking timeScaleIdx up automatically makes trails longer/showier (more simulated motion
    // fits inside the same real-world window) with no retuning. Runs every frame trails are
    // enabled, even while timePaused — matches a real camera shutter aging even when nothing new
    // is landing on it.
    //
    // Splats are drawn from the LIVE satVisibleBuf/starBuf/planetBuf this same frame's sat_flare.comp/
    // updateStars()/updatePlanets() already computed — one sample per real display frame. At very
    // high timeScaleIdx (where a satellite/star can move a large angle between two real frames)
    // this reads as short dashes rather than one continuous streak rather than smoothly interpolated
    // motion — accepted for this pass (temporal supersampling to resample positions several times
    // per real frame, reusing the fact that this sim's orbital/rotational math is a pure function of
    // absolute time, is a natural follow-up but not implemented here).
    VkExtent2D trailAccumExtent{};
    VkImage trailAccumImg = VK_NULL_HANDLE; // RGBA16F, COLOR_ATTACHMENT|STORAGE|SAMPLED, full ctx.swapExtent
                                            // (NOT quarter-res like flareExtent — downscaling would
                                            // blur thin satellite/star streaks; matches the
                                            // "satellites/stars/UI always render at native
                                            // resolution" Resolution Scaling design goal)
    VkDeviceMemory trailAccumMem = VK_NULL_HANDLE;
    VkImageView trailAccumView = VK_NULL_HANDLE;
    // trailSampler intentionally absent — reuses flareSampler (LINEAR/CLAMP_TO_EDGE, resolution-
    // independent) for the composite's sampled read.
    VkRenderPass trailAccumRenderPass = VK_NULL_HANDLE; // 1 color attachment, LOAD_OP_LOAD (never
                                                        // cleared by the render pass itself —
                                                        // persistence is the whole point),
                                                        // VK_IMAGE_LAYOUT_GENERAL throughout (same
                                                        // "sample directly in GENERAL" trick
                                                        // flareScratchImg already uses)
    VkFramebuffer trailAccumFramebuffer = VK_NULL_HANDLE;
    // Stage A (fade, compute): multiplies trailAccumImg in place by a real-time exponential decay.
    VkDescriptorSetLayout trailFadeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool trailFadeDescPool = VK_NULL_HANDLE;
    VkDescriptorSet trailFadeDescSet = VK_NULL_HANDLE;
    VkPipelineLayout trailFadePipeLayout = VK_NULL_HANDLE;
    VkPipeline trailFadePipeline = VK_NULL_HANDLE;
    // Stage B (splat, graphics): reuses drawPipeLayout/descSet and starPipeLayout/starDescSet/
    // planetDescSet completely unchanged (same SatDrawPC push-constant range, same bound buffers) —
    // only new VkPipeline objects, targeting trailAccumRenderPass with depthTestEnable=false (this
    // offscreen target has no depth attachment) instead of ctx.renderPass.
    VkPipeline trailSatPipeline = VK_NULL_HANDLE;
    VkPipeline trailStarPipeline = VK_NULL_HANDLE; // also used for the planet trail draw, exactly
                                                   // like the live starPipeline is
    // Stage C (composite, graphics): additive fullscreen-triangle draw into ctx.renderPass, appended
    // after the flare composite draw in recordDraw() (order between the two doesn't matter — both
    // are ONE/ONE additive blending, which is commutative).
    VkDescriptorSetLayout trailCompositeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool trailCompositeDescPool = VK_NULL_HANDLE;
    VkDescriptorSet trailCompositeDescSet = VK_NULL_HANDLE;
    VkPipelineLayout trailCompositePipeLayout = VK_NULL_HANDLE;
    VkPipeline trailCompositePipeline = VK_NULL_HANDLE;
    // User tunables (Settings > Display / Photometry), persisted in settings.json.
    bool trailEnabled = false;
    float trailDecaySeconds = 4.0f; // real-world exponential decay time constant
    float trailCompositeGain = 1.0f;
    // One-shot flag consumed at the top of the trail block in recordCompute(): set on toggle-on,
    // resize, and the manual "Clear Trail" button. NOT set on timescale/observer/pause changes —
    // trails are meant to persist through those.
    bool trailClearPending = true;

    // ── Per-beam cloud occlusion march (2026-08-09) ────────────────────────────
    // Replaces the per-target beam_cloud_block.comp pass (retired) — own small descriptor set/
    // pipeline, same shape (SSBO + 3 cloud textures + CloudParams UBO), but binding 0 is
    // reflectBeamsBuf itself (read+write) instead of a separate target buffer + output buffer, and
    // it's dispatched over BEAM_MAX_ACTIVE (2048) threads instead of 201 targets. See
    // beam_self_march.comp's header for the full design.
    VkDescriptorSetLayout beamSelfMarchDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool beamSelfMarchDescPool = VK_NULL_HANDLE;
    VkDescriptorSet beamSelfMarchDescSet = VK_NULL_HANDLE;
    VkPipelineLayout beamSelfMarchPipeLayout = VK_NULL_HANDLE;
    VkPipeline beamSelfMarchPipeline = VK_NULL_HANDLE;
    // Earth elevation texture (binding 5): 21600×10800 R8_UNORM land-elevation DEM.
    // Pixel p → elevation_m = p * 8848; ocean stored as 0. Terrain shell = R_EARTH + 9000 m.
    VkImage earthElevImg = VK_NULL_HANDLE;
    VkDeviceMemory earthElevMem = VK_NULL_HANDLE;
    VkImageView earthElevView = VK_NULL_HANDLE;
    VkSampler earthElevSampler = VK_NULL_HANDLE;
    uint32_t earthElevMips = 1;
    // CPU-side downsampled elevation for observer height lookup (2160×1080, ~18km/px)
    std::vector<uint8_t> earthElevCpu;
    int earthElevCpuW = 0, earthElevCpuH = 0;
    // CPU-side downsampled night-lights luminance for observer light-pollution lookup
    // (2160×1080, ~18km/px) — single byte per texel, precomputed Rec.709 luminance. Box-filtered
    // (not nearest-neighbor) so it doesn't itself alias before anything samples it.
    std::vector<uint8_t> earthNightCpu;
    int earthNightCpuW = 0, earthNightCpuH = 0;
    // Half-resolution box-blur of earthNightCpu (~37km/px) — updateLightPollutionDome() samples
    // this bilinearly instead of earthNightCpu directly, the CPU-array equivalent of picking a
    // coarser mip level, to smooth the dome's blocky per-sector transitions.
    std::vector<uint8_t> earthNightCpuBlur;
    int earthNightCpuBlurW = 0, earthNightCpuBlurH = 0;

    // ── UI visibility & settings ──────────────────────────────────────────────
    // UC3: persisted (see loadSettings/saveSettings) so this only auto-plays on first run —
    // defaults true (compiled-in default), and the "no settings.json at all" branch of
    // loadSettings() never touches it, so a genuine first run always shows it; any existing
    // settings.json (even pre-UC3) loads a persisted value that defaults to false so upgrading
    // users don't suddenly get a cinematic that didn't exist before. Replayable via the Display
    // tab's "Replay Intro" button, which just sets this back to true.
    bool showIntro = true;
    bool uiVisible = true;
    bool iconsLoaded = false;
    float uiScale = 1.5f;    // text/UI size multiplier (0.75 – 2.0)
    float masterVol_ = 0.8f; // mirrors AudioSystem default (display fallback)
    float musicVol_ = 0.6f;
    float sfxVol_ = 1.0f;
    // ── Photometry tuning (synced to SatFlarePC each frame) ───────────────────
    // Defaults below are the user-tuned values baked in from settings.json rather than placeholder
    // guesses — re-synced 2026-08-10 (extinctionCoeff, lightPollutionGain, and the Milky Way
    // pollution-response block moved noticeably; the rest were already current), then rounded to
    // 2 significant figures 2026-08-15 (see the cloud block's note below — same pass, same rule).
    float brightnessScale = 1.0f;
    float daySuppression = 570.0f;
    float mirrorBoost = 430.0f;
    float visThresh = 0.0001f;
    float highlightFlare = 0.17f;
    float moonSuppression = 6.6f;    // sky background suppression from moonlight (mirrors daySuppression,
                                     // user-tuned value — moon is ~14 magnitudes dimmer than the sun)
    float lightPollutionGain = 7.0f; // multiplies lightDomeAz[] at the source (updateLightPollutionDome),
                                     // so satellites + stars stay coherently scaled by construction
    float extinctionCoeff = 0.092f;  // atmospheric extinction, magnitudes per airmass (Kasten & Young
                                     // 1989); ~0.2-0.3 is typical clear-sky sea-level; shared formula
                                     // in both sat_flare.comp and updateStars() so a star and a
                                     // satellite at the same elevation dim identically
    // Ground-directed flare mitigation attitude control for space datacenters (AttitudeMode::
    // SunTrackingTilted). A single global operator-policy knob rather than a per-satellite-type
    // JSON constant — real operators would tune one mitigation posture across a fleet, and it
    // needs to be live-tunable to compare against the unmitigated (0°) baseline. Pitches the
    // sun-tracking normal away from nadir toward zenith by this many degrees; power output drops
    // by cos(tiltDeg) (Lambert's cosine law — see the enum comment and sat_orbit.comp's
    // computeNormal()), read back for the selection-panel "Power output" line in
    // formatSelectedSatInfo(). 0° = no mitigation, bit-identical to plain SunTracking.
    float flareMitigationTiltDeg = 0.0f;
    // ── Milky Way pollution response (own threshold + hysteresis, decoupled from
    //    lightPollutionGain/kMWPollutionMaxDim above — see updateLightPollutionDome() and
    //    SatDrawPC::mwSuppressEased) ────────────────────────────────────────────
    // Thresholds are against the RAW (pre-lightPollutionGain) local city-brightness signal, so
    // retuning lightPollutionGain for star/satellite realism never silently shifts where the Milky
    // Way cuts off — these two sliders are the only knob for that.
    float mwPollutionThresholdLo = 0.0022f; // below this: Milky Way at full brightness
    float mwPollutionThresholdHi = 0.036f;  // at/above this: fully suppressed (narrow band = steep cutoff)
    float mwFadeInTimeS = 3.2f;             // seconds to fade back IN once local pollution drops out of the
                                            // band above (bright area -> dark, or ascending into space)
    float mwFadeOutTimeS = 0.0f;            // seconds to fade back OUT once it rises back into the band
    float sunlitBgVisibility = 0.15f;       // Stars/Milky Way visibility fraction in space when the sun is
                                            // off-screen but the observer is still in direct sunlight — 0 =
                                            // fully hidden (like being fully day-suppressed), 1 = as visible as
                                            // true night. Sun-on-screen always forces 0 regardless of this
                                            // slider. See recordCompute()'s sky-glare gate and updateStars().
    // ── Reflect-Orbital ground beams (C12) ────────────────────────────────────
    // groundIrradiance * beamGain is NOT the same quantity as mirrorBoost/mirrorPeak (that's the
    // view-dependent specular term the OBSERVER sees the mirror glint by; this is the physical
    // irradiance the mirror delivers to its ground target, independent of view angle — see
    // sat_orbit.comp's beam-writer comment). Uploaded via SatOrbitPC.
    float beamGain = 0.0017f;
    // C12 follow-up #34: beamFootprintRadM (a flat, tunable constant) removed — the ground
    // footprint is now physically derived in sat_orbit.comp from mirror area + range to target.
    float beamMaxRangeM = 1100000.0f; // C12 follow-up #6 — render-time "is the observer close
                                        // enough to this site" cutoff (site-referenced beams have
                                        // no observer-side write gate any more, see sat_orbit.comp)
    // C12 follow-up #17: simple atmospheric-scattering beam sky glow (replaces the removed real
    // cloud-density march from follow-ups #14-#16, reverted per user request — no cloud lighting
    // yet). Own gain, separate from beamGain (that's the physical ground-irradiance term feeding
    // the ground spot; this purely scales the visual glow's brightness) — dim default, tunable.
    float beamSkyGlowGain = 0.0088f;
    // 2026-08-06 reversibility rework: replaced the old rate-limited mirror slew (deg/sec,
    // integrated frame-by-frame — inherently history-dependent, so it could not be made
    // reversible) with a fixed-width sim-time window a satellite commits to one target for. See
    // sat_orbit.comp's TargetedReflector block and CLAUDE.md for the full design. Same settings
    // slider slot (Settings → Beams), retitled "Target lock window (s)".
    float reflectorLockWindowS = 90.0f;
    // 2026-08-06 same-day follow-up: real angular-rate cap for the rate-limited ease that smooths
    // a TargetedReflector mirror's orientation across a target change — see sat_orbit.comp's
    // TargetedReflector block (nearFallbackIdeal/startAim/destAtStart) for how the ease duration
    // is derived from this and the actual angle to cover. The window-crossfade-only version
    // shipped earlier the same day only covered one of several transition cases and wasn't tied to
    // real angular distance, which read as satellites snapping to target.
    float mirrorMaxRateDegPerSec = 0.11f;
    // S1 follow-up (RELEASE_v1_1_PLAN.md): minimum acceptable local elevation angle of the
    // satellite as seen FROM a candidate ground target, in degrees. Below this, a target is
    // rejected outright by sat_orbit.comp's TargetedReflector selection (grazing beams suffer
    // heavy atmospheric extinction and aren't worth taking even as a last resort) rather than
    // being deprioritized — see that shader's own comment for the full rationale, including why
    // this replaced a pure "nearest target" rule. Sent as sin(radians(this)) via
    // SatOrbitPC::minBeamElevSin (recordCompute fills it; the sin conversion happens there, not
    // per-candidate in the shader). This is now the ONLY floor — the 2026-08-06 reversibility
    // rework removed the separate release floor/hysteresis margin along with the persisted lock it
    // existed to protect (temporal stability now comes from reflectorLockWindowS instead).
    float reflectorMinElevDeg = 10.0f;
    // (beamExtinctionMult lived here — user-tunable extra extinction for the deleted analytic
    // beam sky tube. Removed in C12 follow-up #44 along with the tube itself: the replacement
    // per-sample beam-cloud term is a real volumetric contribution composited through the cloud
    // march's own transmittance, with no separate closed-form extinction exponent to tune.)
    // C12 follow-up #30: gain for the near-field directional sky-glow bleed — the replacement for
    // the tube glow's near-field behavior (which has structural artifacts up close: "cut in half,"
    // a "hard shell," darkening in the middle — a single-point analytic approximation was never
    // designed for a camera near/inside the volume). The tube fades out approaching a beam
    // (crossfade in the shader) while this purely angular (no segment geometry) glow term fades
    // in — own gain per [[feedback_shared_gain_sliders]], not a reuse of beamSkyGlowGain.
    float beamGlowBleedGain = 0.0012f;
    // C12 follow-up #40: radius (meters) of the crossfade blend zone around a beam's own 3D line —
    // was a hardcoded kNearFieldCrossoverM constant in cloud_march.comp, now user-tunable.
    // Per-pixel cloud shadow fade distance. Was 80 km (matching the deleted 128x128 grid's
    // half-extent, so anything the grid shadowed still was). Measured 2026-08-06: pushing this to
    // the slider ceiling costs nothing, because the shadow march is per-terrain-pixel and gated on
    // `tScene < kNoSurfaceT` — raising the range does not add samples, it only stops the fade from
    // killing shadows the march already computed. So it now defaults maxed, and it is deliberately
    // NOT in applyGraphicsPreset's table; Planetarium still removes the whole pass via bit 256.
    float cloudShadowRangeM = 300000.0f;
    float beamNearFieldFadeM = 1000.0f;
    // 2026-08-09: exposed per explicit user request ("How can we control the thresholding on what
    // gets considered a group?"). Beams converged on the same real target must not all blend into
    // one cloud light when they arrive from very different parts of the sky — satellites on
    // different orbital passes genuinely illuminate that cloud from different directions, and with
    // hgG~0.99 averaging them together doesn't read as "slightly off", it reads as light arriving
    // from the wrong place entirely (in-app test #8, BEAM_CLOUD_PLAN.md).
    //
    // 2026-08-12 — SEMANTICS CHANGED, same slider, same settings key, same direction of effect.
    // Was: the maximum angle between a candidate beam and a cluster's RUNNING-AVERAGE direction for
    // the two to merge. That test depended on which beams had already joined, which is precisely
    // what made the partition order-dependent and discontinuous. Now: the angular SIZE of a fixed
    // direction bucket in the target site's own local frame. A beam's bucket is a pure function of
    // its own direction, so a cluster can no longer repartition when a member leaves. Lower =
    // finer buckets (more, more coherent clusters); higher = coarser (fewer, cheaper, more
    // blending). Stored in degrees for the UI; converted to bucket counts once per frame.
    float beamClusterDirThresholdDeg = 38.0f;
    // 2026-08-12: cross-frame fade for the cloud-light list. Now that every light has a stable
    // integer identity (see TrackedBeamLight), a light that appears, disappears or changes strength
    // can be eased instead of snapping. Deliberately asymmetric fast-in/slow-out, the same
    // convention mwFadeInTimeS/mwFadeOutTimeS and skyGlareEased already use here: a beam becoming
    // relevant should register promptly, but one dropping out should linger rather than blink off.
    // A pure appearance-smoothing knob — it does not change which lights exist, only how quickly
    // their contribution ramps.
    float beamClusterFadeInS = 0.35f;
    float beamClusterFadeOutS = 1.5f;
    // Geometry (position/direction/block altitude/opacity) eases on its own, much shorter constant,
    // deliberately NOT user-exposed. It exists to absorb the small residual steps that survive
    // stable keying — a beam entering or leaving a bucket shifts that bucket's intensity-weighted
    // mean geometry slightly — without letting a light visibly lag genuinely fast beam motion. If
    // this ever needs tuning it should become a slider rather than being folded into the fade
    // times above, which answer a different question.
    static constexpr float kTrackedLightGeomEaseS = 0.12f;
    // C12 follow-up #41: 0-1, how close the observer is to ANY active beam's actual 3D line —
    // smoothstepped from lastNearestBeamDistM/beamNearFieldFadeM each frame in recordCompute().
    // Drives the non-directional sky-glow wash in sat_sky.frag, replacing #39/#40's directional
    // dome-based approach (which read as a narrow pillar and faded incorrectly with altitude).
    float beamProximityGlow = 0.0f;
    // ── Milky Way skybox basis (session 27) ────────────────────────────────────
    // ENU->galactic rotation, recomputed each frame in updatePositions() and uploaded to
    // CloudParams. Orientation confirmed by eye against the real star field — no runtime
    // tuning knobs needed (see updatePositions() for the fixed longitude-mirror correction).
    glm::vec3 mwRow0{1.0f, 0.0f, 0.0f};
    glm::vec3 mwRow1{0.0f, 1.0f, 0.0f};
    glm::vec3 mwRow2{0.0f, 0.0f, 1.0f};
    // Cloud tunables (CPU-side; uploaded to cloudParamsBuf each frame)
    // Defaults below are the user-tuned values baked in from settings.json (most recently
    // 2026-08-10 — cloudDensity, cloudAmbientGain, airglowCoverageGain/PolarGain, and most of the
    // Reflect-Orbital beam gains/ranges moved; the rest of the block was already current) rather
    // than placeholder guesses. They are
    // a measured, self-consistent SET — the flat-2D scales calibrate against the volumetric path,
    // and the lighting gains against each other. Do not re-derive any one of them in isolation.
    //
    // 2026-08-15: every default here (and in the photometry/beam blocks above) is rounded to
    // 2 SIGNIFICANT FIGURES, deliberately. Baking a slider position straight out of settings.json
    // produces values like 0.838158 / 157.302979 / 151902.171875 whose extra digits are the
    // slider's pixel quantization, not tuning intent, and which make the block unreadable. Keep new
    // defaults to 2 s.f. as well; if a value genuinely needs more precision than that to behave
    // correctly, say so in a comment on that line rather than silently reintroducing noise digits.
    // Anything here that GraphicsPreset::High also lists must change in both places at once:
    // High's table row is documented as "the compiled-in class member defaults, verbatim."
    float cloudCoverage = 1.0f;
    float cloudDensity = 0.84f;
    float cloudBaseAltM = 6000.0f; // layer 0 shell altitude (low cloud / stratus)
    float cloudTopAltM = 15000.0f; // layer 1 shell altitude (high cirrus)
    float cloudDriftRate = 6.6e-06f;
    float cloudSunGain = 4.0f;       // near-horizon/sunset sun-gain endpoint — blended toward
                                     // cloudSunGainZenith by sun elevation (see cloud_march.comp)
    float cloudSunGainZenith = 1.0f; // sun-gain endpoint when the sun is near zenith (midday)
    float cloudAmbientGain = 1.0f;
    float cloudTwilightAmbientGain = 0.40f; // manual gain on sky-lit cloud during twilight (was piggybacking
                                            // on cloudAmbientGain, which also drives city-light
                                            // upwelling — see kNightSkyAmbientColor in cloud_march.comp)
    float cloudBaseVariance = 0.27f;        // noise-driven cloud base height undulation, hNorm units
                                            // (0 = old perfectly flat base) — see cloudMarchCS
    float cloudErosionEdge = 0.91f;         // cloudDensity() erosion strength at the silhouette edge
    float sunGainElevBand = 0.12f;          // ~1.1 deg elevation band — user-tuned 2026-08-04
    // Brought forward from the original hardcoded 0.15 so the sky term overlaps the tail of
    // direct sunlight instead of starting after it; 0.35 is ~20 deg of sun elevation.
    float twilightBandHi = 0.14f;
    float twilightBandLo = -0.25f; // user-tuned 2026-08-04
    // 1.0 rather than 0.0: a compromise starting point. Lower = more small-scale structure and
    // a closer match to the flat layer, at the cost of worse texture-cache behaviour (mip 0 of
    // the 8K map is ~33 MB and is sampled once per in-cloud march step).
    float coverageMipLod = 0.0f;
    // Measured against the volumetric at MIP 0: volumetric (coverage 1.00, sun gain 0.46)
    // matched flat (coverage 0.69, sun gain 1.84). Defaults encode those ratios so the shared
    // sliders now move both paths together instead of only ever suiting one of them.
    float flatCoverageScale = 0.69f;
    float flatSunGainScale = 2.4f;
    // Clouds at 11 km have a ground-level horizon of ~374 km, so this band puts the transition
    // near the horizon when standing on the surface, and makes everything 2D from orbit.
    float cloudDistFadeStartM = 150000.0f;
    float cloudDistFadeEndM = 400000.0f;
    // S4 (RELEASE_v1_1_PLAN.md, session 31): terrain-relief march distance fade. Below the start
    // distance the march gets its full step budget; between start and end the budget fades out;
    // past the end it is skipped entirely and the ray falls back to the sea-level sphere (exactly
    // what the "terrain off" knockout produces). So a LOWER start is CHEAPER.
    // The start was 300000 (chosen so ground views, whose grazing rays cap at ~250 km reach, were
    // untouched) until the 2026-08-06 pass took it to 50000 for a long, gradual roll-off instead of
    // a late abrupt one — distant relief now dissolves into the sphere over 50-900 km rather than
    // holding full detail to 300 km and then dropping. NOTE this is now lower than Medium's and
    // Low's start distances in applyGraphicsPreset; see the comment on that table.
    float terrainDistFadeStartM = 50000.0f;
    float terrainDistFadeEndM = 900000.0f;
    // Cloud opacity scale (see GpuCloudParams::cloudOpacityScale) — multiplies the volumetric
    // cloud march's extinction-per-metre constant directly (and, since this same value also
    // scales layer 0's flat-2D-crossfade alphaMax ceiling in recordCompute(), the flat layer used
    // at higher observer altitude too), since raising `cloudDensity` alone cannot push a
    // saturated column past full opacity. 1.0 reproduces the original hardcoded extinction/alphaMax
    // exactly. This briefly defaulted to 7.0 (2026-08-02) to force opacity through moderately dense
    // cloud; the 2026-08-06 tuning pass took it back to ~1.0 and got the opacity from `density`/
    // erosion instead, which does not harden thin/wispy edges the way a raw extinction multiplier
    // does. The 2026-08-15 2-s.f. rounding pass took the residual 1.014 to exactly neutral — the
    // knob is still live if it's ever needed.
    float cloudOpacityScale = 1.0f;
    // City-lights blur-through-cloud (see GpuCloudParams::cityLightBlurLod) — mip LOD
    // earthNightTex/cityNightDetailTex blend toward under full local cloud opacity, so light
    // diffused through haze reads as a soft glow instead of a sharp copy of the raw texture.
    float cityLightBlurLod = 8.1f;
    // Atmospheric scattering strength gains (see GpuCloudParams::atmosRayleighGain/atmosMieGain
    // and common.glsl's BETA_R_BASE/BETA_M_BASE) — 1.0 reproduces the original hardcoded physical
    // constants exactly. Rayleigh gain controls how much red/orange the sky and horizon clouds
    // pick up at grazing angles/low sun elevation (preserves the R:G:B ratio, just deepens or
    // shallows the effect); Mie gain controls how much wavelength-neutral haze dilutes that color
    // back toward white/grey.
    // Domain-warp shear controls (see GpuCloudParams::cloudWarpStrength and cloud_params.glsl's
    // folding-threshold note). Were hardcoded kWarpStrength/kWarpFreq in cloud_march.comp.
    // Frequency defaulted to 6.0 until the warp was measured against the noise domains it feeds:
    // that put the detail lookup at a shear ratio of 0.8 and the per-column lookup at 4.8, i.e.
    // badly folded — the reported pinching/banding/"wavy chips". 3.0 halves the shear while
    // leaving displacement magnitude (the large-scale structure movement) untouched.
    float cloudWarpStrength = 32.0f;
    float cloudWarpFreq = 3.7f;
    // Erosion redesign (see GpuCloudParams and cloud_params.glsl). cloudSurfaceCarve = 0 and
    // cloudErosionBillow = 0 together reproduce the previous erosion exactly, so the two of them
    // bisect this change; cloudErosionFreq = 1.5 was the old hardcoded coordinate scale (the
    // 2026-08-06 pass took it to 0.5, i.e. coarser erosion lumps than the original constant).
    // 1.0 (fully subtractive) per in-app review — it also happened to mask the density-clamp
    // plateau documented in cloudDensity(), which is now fixed at its source.
    float cloudSurfaceCarve = 1.0f;
    float cloudErosionBillow = 1.0f;
    float cloudErosionBillowH = 0.45f;
    float cloudErosionFreq = 0.5f;
    // Directional-shading contrast. Previous hardcoded equivalents were 1.0 / 0.05 / 0.35 / 1.0.
    // These were first pushed hard toward contrast (clouds were shading near-uniform at sunset),
    // then pulled back by the 2026-08-06 pass once the erosion redesign and the terminator gate
    // were supplying that contrast structurally: shadowFloorT/grazeShadow/coneLenScale now sit
    // near or below the original constants and multiScatter roughly halved. See cloud_params.glsl.
    float cloudMultiScatter = 0.55f;
    float cloudShadowFloorT = 0.11f;
    float cloudGrazeShadow = 0.33f;
    float cloudConeLenScale = 0.79f;
    // Height-only shading was the "lasagna" cause; halved and now sun-elevation weighted, with a
    // density-driven occlusion term supplying the horizontal variation it never had.
    float cloudVertShadeGain = 0.53f;
    float cloudDensityAO = 0.50f;
    float cloudAOPower = 0.05f;
    // 1.0 = previous coupled behaviour. Raise after lowering `density` for the volumetric path.
    float flatDensityScale = 2.0f;
    // Flat 2D layer's own Rayleigh multiplier, stacked on atmosRayleighGain (see
    // GpuCloudParams::flatRayleighGain). 1.0 = the previous fully-coupled behaviour; raise or
    // lower it to close the hue/depth step across the 3D->2D crossfade.
    float flatRayleighGain = 0.18f;
    // Flat 2D layer's twilight sky ambient, stacked on cloudTwilightAmbientGain (see
    // GpuCloudParams::flatTwilightAmbientGain). The flat path had no ambient term at all before
    // this, so there is no prior behaviour to preserve — 1.0 starts it at the same strength the
    // volumetric shell gets, which is the matched-crossfade starting point; 0 disables it.
    float flatTwilightAmbientGain = 2.1f;
    // Orbital terminator gate (see GpuCloudParams::atmosTermStrength). Defaulted ON at the "mid"
    // setting measured during design — SZA 92 about 23x down, day side untouched — because the
    // whole point is to look at it. Drag strength to 0 for an exact A/B against the old look;
    // nothing else in the frame changes when you do.
    float atmosTermStrength = 1.0f;
    float atmosTermWidth = 0.071f;
    float atmosRayleighGain = 0.98f;
    float atmosMieGain = 1.0f;
    // C11 ground fog layer — real per-sample volumetric march in cloud_march.comp's fogMarchCS.
    // (Originally also reused beamCloudLighting() for beam godrays through fog; that reuse was
    // removed with the function itself 2026-08-09 — fog no longer carries a beam term.) Retuned
    // in-app 2026-08-06: a much taller (1.4 km) but far thinner (density ~0.07)
    // layer than the first-pass 300 m / 1.0 guess — the thin tall version reads as real haze the
    // camera can fly up through, where the thick shallow one read as a hard ground-hugging slab.
    float fogTopAltM = 1400.0f;       // shell top altitude (m above sea level); sea level is the base
    float fogDensity = 0.068f;        // density scale, analogous to cloud.density
    float fogCoverage = 0.6f;         // global coverage gate for the patchiness noise, [0,1]
    float fogSunGain = 1.1f;          // sun-lit fog brightness gain, own slider (not cloud.sunGain)
    float cloudErosionCore = 1.0f;    // cloudDensity() erosion strength at the dense core
    float cloudHgG = 0.99f;
    float cloudMarchSteps = 220.0f;
    float cloudLightSteps = 13.0f;
    float cloudCirrusWindDeg = 40.0f;  // C13: cirrus streak wind azimuth (degrees, converted to radians for the UBO)
    float cloudCirrusStretch = 2.4f;   // C13: cirrus noise anisotropic elongation factor (1 = no stretch)
    float airglowGain = 0.066f;        // C15: master airglow brightness multiplier
    float airglowGreenGain = 0.053f;   // C15: green (557.7nm) band gain
    float airglowRedGain = 0.013f;     // C15: red (630.0nm) band gain — diffuse/broad, keep subtle
    float airglowSodiumGain = 0.08f;   // C15: sodium (589.3nm) band gain — kept dim relative to green
    float airglowCoverageGain = 0.32f; // patchy-coverage strength for all 3 airglow bands, [0,1]
    float airglowPolarGain = 2.4f;     // red band only: extra boost toward the geomagnetic pole
    // Sun self-shadow cone (N_CONE) fades out beyond this distance. Was 22 km, when the cone
    // marched a fixed stride and distance directly bought samples. The cone now absorbs distance
    // into its stride (see cloud_march.comp's shadowFade comment), so this became a reach knob
    // rather than a cost knob and the 2026-08-06 pass effectively unbounded it — self-shadowing
    // now survives all the way out to the cloud shell's own horizon instead of dropping off a
    // cliff at 22 km and leaving distant cloud flat-lit.
    float cloudShadowMaxDistM = 6000000.0f;
    float cloudMaxRenderDistM = 800000.0f; // cloudMarch tExit distance cap — raised to ~400km
                                           // (session 28 follow-up #10): the low-cloud shell's own
                                           // geometric horizon distance at 11km altitude is
                                           // ~sqrt(2*R_EARTH*11000)≈374km; the prior 165km default
                                           // cut the march off well short of that, letting
                                           // aurora/Milky Way/stars show straight through clouds
                                           // near the horizon instead of them thinning out naturally
    // Perf follow-up (session 24): main atmosphere loop + ocean wave quality, all previously
    // hardcoded compile-time constants.
    // N_VIEW is now adaptive per-ray (round 2): a fixed sample count badly serves a loop whose
    // path length (tEnd) varies from ~100km (straight up) to 2000+km (grazing/horizon/orbit) —
    // see the adaptive-N_VIEW comment in sat_sky.frag for the full reasoning. viewSamplesMin is
    // the user-validated "looks convincing" floor for short ground-level rays (4 showed visible
    // artifacts in testing; 6 was clean — round 3); viewSamplesMax is the ceiling for long/grazing
    // rays. That ceiling was the prior universal fixed value (124) until the 2026-08-06 tuning
    // pass raised it to ~157 (rounded to 160 by the 2026-08-15 2-s.f. pass): the orbital
    // terminator gate (atmosTermStrength) puts real structure in the deep-twilight tail of the
    // integral, where 124 samples banded on long limb rays.
    float viewSamplesMin = 6.5f;
    float viewSamplesMax = 160.0f;
    float lightSamples = 2.4f;               // N_LIGHT: optDepth sun-side sub-march count
    float oceanSeaOctaves = 3.0f;            // seaMap() octave count (height-trace geometry)
    float oceanDetailOctaves = 5.0f;         // seaMapDetail() octave count (wave normal)
    float oceanReflSamples = 6.0f;           // ocean sky-reflection loop sample count (N_REFL)
    float moonGain = 0.0053f;                // shared moonlight brightness: terrain direct term + cloud
                                             // moonContrib (default matches the prior hardcoded cloud value)
    float stormStrength = 0.33f;             // C16: aurora oval expansion/brightness/chaos [0,1]
    float auroraGain = 0.1f;                 // C16: master aurora brightness multiplier
    float auroraCloudGain = 0.0018f;         // C16: ambient aurora light on clouds only (no albedo term
                                             // in that formula, so it needs a much lower default than
                                             // terrain/ocean to land in the same plausible range)
    float auroraGroundGain = 0.0075f;        // C16: ambient aurora light on terrain/ocean only
    float auroraCoverageFreq = 0.43f;        // C16: coverage patch size (per-degree colat frequency)
    float auroraCoverageAzFreq = 4.3f;       // C16: coverage azimuthal wobble frequency
    float auroraCoverageDriftRate = 0.0012f; // C16: coverage evolution speed (wall-clock rad/s)
    float auroraShimmerRate = 0.0018f;       // C16: curtain fold noise evolution speed (wall-clock rad/s)
    VulkanContext *ctx_ = nullptr;           // set in init(), used for lazy icon loading
    AudioSystem *audio_ = nullptr;           // set via setAudio(), used in buildUI()
    std::string exeDir_;                     // directory containing the exe (read-only game data); set in init()
    std::string userDataDir_;                // per-user writable dir for settings/perf (see Paths.h); set in init()

    // ── NEW-3: crash-safe mode ──────────────────────────────────────────────
    // A sentinel file is created at the top of init() and deleted at the bottom of cleanup()
    // (the clean-exit path). If it's already present at the NEXT launch, the previous run never
    // reached cleanup() — crash, hang + force-kill, power loss — so this run forces the
    // Planetarium preset and shows a one-line notice, converting "launch -> crash -> uninstall"
    // into a recoverable outcome. See applySettings-adjacent logic in init()/cleanup().
    float crashRecoveryNoticeTimer = 0.0f; // seconds remaining to show the notice banner; see buildCrashRecoveryNotice
    bool crashRecoveryMode = false;        // mirrors the crashDetected local in init(); read by finishIntro()
                                           // so a crash-recovery launch never runs the UC1 benchmark promote/
                                           // demote (that launch already forced Planetarium for a different reason)

    // ── UC3: cinematic intro camera path (folds in UC1 mechanism 2, the first-run benchmark) ──
    // Does not move obsDir/lat-lon DURING playback — only obsHeightOffset (altitude, literally
    // what Q/E controls), camera.elDeg/fovYDeg, and a facing-azimuth rotation change across the
    // beat sheet (see kIntroKeyframes in SatelliteSim.cpp). It DOES force obsDir/lat-lon once, at
    // the very start of playback (see updateIntroCinematic's one-time init block) — to the fixed
    // kIntroObserverLatDeg/LonDeg vantage point, not whatever the player's last position happened
    // to be — so the intro (including a Display-tab replay) is reproducible regardless of where
    // the player has since wandered off to. Since obsDir is fixed for the rest of playback after
    // that, the East/North tangent basis below is computed once and stays valid the whole time —
    // no great-circle interpolation needed.
    bool introBasisValid = false;
    glm::vec3 introEastEF{1, 0, 0}, introNorthEF{0, 1, 0};
    float introElapsed = 0.0f;    // seconds since the intro cinematic started
    int introCaptionIndex = 0;    // index into kIntroKeyframes of the most recently reached caption
    bool introSkipped = false;    // true if dismissed early (Space/gamepad Start) — gates the benchmark below
    bool introIsReplay = false;   // set by the Display tab's "Replay Intro" button; suppresses the
                                  // benchmark regardless of how the replay ends (see finishIntro) —
                                  // it's a one-shot first-run decision, not something a replay should redo
    float introBenchMsSum = 0.0f; // accumulates gpuMsTotalSmoothed across the camera-motion beats (see updateIntroCinematic)
    int introBenchFrames = 0;
    char introControlsTextBuf[96] = {}; // member buffer for the final WASD/Q-E controls caption (built
                                        // from live keybindings — Clay stores raw string pointers read
                                        // after buildUI returns, so this can't be a stack local; see
                                        // CLAUDE.md's Clay runtime-string rule)

    // Post-intro "graphics set to X" notice (UC1 mechanism 3: always tell the user, never
    // silently re-decide) — same dismissible-banner pattern as buildCrashRecoveryNotice, separate
    // timer/text since the two can in principle be showing different things.
    float graphicsAutoNoticeTimer = 0.0f;
    char graphicsAutoNoticeText[128] = {};

    // ── UC6: screenshots ────────────────────────────────────────────────────────
    // See Simulation.h's wantsCleanScreenshot/recordScreenshotCopy/finalizeScreenshot doc
    // comments for the three-phase (request -> copy -> readback) protocol this drives.
    bool screenshotRequested = false;                     // set by dispatchKeyAction(KB_SCREENSHOT); consumed by
                                                          // recordScreenshotCopy (also gates wantsCleanScreenshot)
    bool screenshotCopyPending = false;                   // true between "copy recorded" and "readback finalized"
    VkBuffer screenshotStagingBuf = VK_NULL_HANDLE;       // host-visible; freed/recreated per capture —
    VkDeviceMemory screenshotStagingMem = VK_NULL_HANDLE; // screenshots are rare, not a hot path
    uint32_t screenshotW = 0, screenshotH = 0;
    VkFormat screenshotFormat = VK_FORMAT_UNDEFINED;
    std::string screenshotPath; // full output path, built at request time
    float screenshotToastTimer = 0.0f;
    char screenshotToastText[160] = {};
    // PNG encoding (stbi_write_png) is genuinely slow in an unoptimized Debug build — easily
    // tens of seconds at 1080p+, which reads as "the game froze" since finalizeScreenshot() used
    // to run it synchronously on the main thread. Moved to a detached background thread: the main
    // thread only maps the GPU buffer, swizzles into a plain std::vector it hands off by move, and
    // returns immediately. screenshotEncoding guards against starting a second capture while one
    // is still encoding (the vector handoff means the GPU staging buffer itself is free again
    // immediately, but re-entrant encodes would still race on the toast result below).
    // screenshotResultReady/screenshotResultMutex/screenshotResultText are the thread's one-shot
    // handoff back to the main thread (checked once per frame in buildUI) — the atomic gates
    // whether it's worth taking the mutex at all, avoiding any per-frame lock when idle.
    std::atomic<bool> screenshotEncoding{false};
    std::atomic<bool> screenshotResultReady{false};
    std::mutex screenshotResultMutex;
    std::string screenshotResultText;
    // Kept joinable (never .detach()ed) so cleanup() can join it before the object it captures
    // (`this`, for the mutex/atomics/string above) is destroyed — a detached thread still running
    // past that point would be a use-after-free. screenshotEncoding already prevents two threads
    // existing at once, so join() here is always fast (the previous one has either already
    // finished or is about to).
    std::thread screenshotThread;

    // ── Key bindings (editable in the settings window) ────────────────────────
    // All interactive keys go here — both event keys (pressed once) and held keys
    // (polled each frame).  Adding a new control is one line in the keybindings
    // initializer; the settings window and rebind UI are driven entirely from this
    // vector so no other plumbing is needed.
    //
    // held=false  → dispatched in onKey() via pressed(idx)
    // held=true   → polled in recordCompute() via glfwGetKey(win, keybindings[idx].key)
    //
    // gpButton mirrors key but for an Xbox-style gamepad (GLFW_GAMEPAD_BUTTON_*, -1 =
    // unbound) — either input fires the same action, so rebinding one never disturbs the
    // other. listening/listeningPad are mutually exclusive across the whole vector (the UI
    // clears every other flag before setting one), each capturing the next keyboard key or
    // gamepad button respectively — see onKey() and pollGamepad().
    struct KeyBinding
    {
        const char *action;
        int key;
        int gpButton = -1;
        bool held = false; // true = polled (held modifier), false = event (pressed once)
        bool listening = false;
        bool listeningPad = false;
    };
    std::vector<KeyBinding> keybindings;

    // Canonical index constants — keeps onKey / recordCompute in sync with the
    // keybindings array without magic numbers.
    enum KB
    {
        KB_TOGGLE_UI = 0,
        KB_PAUSE = 1,
        KB_SLOWER = 2,
        KB_FASTER = 3,
        KB_REVERSE = 4,
        KB_MOVE_BOOST = 5,     // held
        KB_MOVE_FINE = 6,      // held
        KB_CINEMATIC = 7,      // event — toggles camera drift mode while panning
        KB_RAISE_ELEV = 8,     // Q — held — raise observer above terrain (gamepad: analog right trigger, see gpElevRaise, not this binding's gpButton)
        KB_LOWER_ELEV = 9,     // E — held — lower observer toward terrain (gamepad: analog left trigger, see gpElevLower)
        KB_RESET_ELEV = 10,    // Z — event — snap observer back to terrain elevation
        KB_ZOOM_IN = 11,       // held — narrows FOV (zoom in)
        KB_ZOOM_OUT = 12,      // held — widens FOV (zoom out)
        KB_ZOOM_RESET = 13,    // event — snap FOV back to default
        KB_SELECT_SAT = 14,    // event — select the satellite nearest the center of the screen
        KB_SCREENSHOT = 15,    // event — UC6: capture screenshots/satlight_<timestamp>.png (clean, no UI)
        KB_TOGGLE_CURSOR = 16, // event — UC5: gamepad virtual-cursor mode toggle (default: Menu/Start)
        KB_TOGGLE_TRAILS = 17, // event — long-exposure trail on/off (default: F; Select Satellite
                               // moved off F to T to free this up — see keybindings init)
        KB_COUNT = 18,
    };

    // Dispatches the event-style action for keybindings[bindIdx] — shared by onKey()
    // (keyboard) and pollGamepad() (gamepad edge-detect) so the two input paths can never
    // drift apart. No-op for held bindings (MOVE_BOOST/FINE, RAISE/LOWER_ELEV, ZOOM_IN/OUT):
    // those are polled directly, not dispatched.
    void dispatchKeyAction(int bindIdx);

    // Polled once per frame from recordCompute(). Scans for a connected gamepad, edge-detects
    // event-style button presses (dispatchKeyAction) and rebind capture (listeningPad), and
    // fills gpMoveFwd/gpMoveRight/gpLookYawDeg/gpLookPitchDeg from the sticks for the
    // movement/look code in recordCompute()/buildUI() to consume.
    void pollGamepad(float dt);
    // True if the gamepad button bound to keybindings[bindIdx] is currently held down.
    bool gpHeld(int bindIdx) const;
    // UC4: reports the virtual cursor's screen position/click state (updated in pollGamepad);
    // see Simulation.h for the calling convention.
    bool virtualCursor(float &x, float &y, bool &lmb) const override;

    // ── ECI → ENU rotation (updated each frame in updatePositions) ────────────
    // Encodes the surface-fixed observer's local frame in ECI coordinates.
    glm::vec4 eci2enuX{1, 0, 0, 0}; // East  basis in ECI
    glm::vec4 eci2enuY{0, 1, 0, 0}; // North basis in ECI
    glm::vec4 eci2enuZ{0, 0, 1, 0}; // Up    basis in ECI

    // ── Sun + observer state (updated each frame in updatePositions) ──────────
    glm::vec3 sunDirECI{1, 0, 0};    // unit vector from Earth toward Sun in ECI
    glm::vec4 sunDirENU{0, 1, 0, 0}; // sun direction in ENU (xyz), w = sin(elevation)
    glm::vec3 obsECI{0, 0, 6371000}; // observer ECI position (meters)

    // ── Planets (updated each frame in updatePositions/updatePlanets) ─────────
    PlanetState planetStates[kPlanetCount]{};                                // ephemeris; direction/distance/phase
    bool planetEnabled[kPlanetCount] = {true, true, true, true, true, true}; // per-planet toggle
    bool showPlanets = true;                                                 // global toggle, settings-persisted
    int selectedPlanetIndex = -1;                                            // index into planetStates[]/kPlanetNames[], -1 = none selected;
                                                                             // mutually exclusive with selectedSatIndex (see its declaration)

    // ── Sky-background sun-glare gate (hysteresis state, not persisted) ───────
    // Eased toward its per-frame target in recordCompute(), right after updatePositions() and
    // before updateStars(). Consumed by updateStars() (folded into nightFactorEff) and pushed to
    // sat_sky.frag via SatDrawPC for the Milky Way. See recordCompute() for the target/easing
    // logic and rationale (asymmetric fast-dim/slow-recover rates).
    float skyGlareEased = 1.0f;

    // ── Milky Way pollution suppression (hysteresis state, not persisted) ─────
    // [0,1], 0 = fully visible, 1 = fully suppressed. Eased each frame in
    // updateLightPollutionDome() toward a target derived from the RAW (pre-lightPollutionGain)
    // local light-pollution level via mwPollutionThresholdLo/Hi, using mwFadeInTimeS/mwFadeOutTimeS
    // as asymmetric rates (same shape as skyGlareEased above, see that member/recordCompute() for
    // the pattern this mirrors). Pushed to sat_sky.frag via SatDrawPC::mwSuppressEased.
    float mwSuppressEased = 0.0f;

    // ── TargetedReflector ground targets ──────────────────────────────────────
    // S1 (RELEASE_v1_1_PLAN.md): real solar-farm sites loaded from reflector_targets.json (falls
    // back to random points — see loadReflectorTargets()), stored as unit ECEF vectors. Rotated to
    // ECI each frame in updatePositions; filtered to those on the night side.
    // kNumReflectorTargets is a CAPACITY (buffer sizing), not the real count — see
    // reflectorTargetCount below. Modders can supply anywhere up to this many sites.
    static constexpr int kNumReflectorTargets = 201;
    // Real number of loaded targets this run (<= kNumReflectorTargets); set by loadReflectorTargets()
    // /generateReflectorTargetsRandomFallback(). Only [0, reflectorTargetCount) of the arrays below
    // are meaningful — entries beyond it are zero-initialized and never read.
    int reflectorTargetCount = 0;
    // Index into the loaded array that's the observer-spawn pin (reflector_targets.json's
    // "observer_spawn": true entry, or index 0 in the random fallback) — -1 if somehow neither
    // path set one. Purely informational/logging today; the pin works simply by being a real,
    // correctly-populated entry like any other (see loadReflectorTargets()'s doc comment for the
    // bug this replaced: index 0 used to be silently left as a degenerate zero-vector).
    int reflectorObserverSpawnIdx = -1;
    // reflectorActiveCount (S1's per-frame night-side compaction count) was removed 2026-08-06 —
    // sat_orbit.comp now scans all reflectorTargetCount targets itself each frame, gating night-
    // side per-candidate from the live/eval-time sun direction instead of relying on a CPU-
    // precompacted subset. See sat_orbit.comp's TargetedReflector block and CLAUDE.md.
    glm::vec3 reflectorTargetsECEF[kNumReflectorTargets]{}; // unit ECEF, set by initConstellation
    // Real ground radius per target (C12 follow-up #18) — kEarthRadius + actual terrain elevation
    // at that target's lat/lon, looked up once via earthElevCpu when targets are generated
    // (buildOrbits() runs after createGlowResources() has loaded earthElevCpu — see init()'s call
    // order). Fixes targets on any elevated terrain (mountains, plateaus) being placed at the
    // sea-level sphere, which put the "ground" endpoint of every beam-related ray for that target
    // underground. Defaults to kEarthRadius (sea level) if earthElevCpu isn't available for
    // whatever reason. Consumed by updatePositions() in place of the bare kEarthRadius constant
    // when converting reflectorTargetsECEF to a real ECI position.
    float reflectorTargetsRadiusM[kNumReflectorTargets]{};
    // Per-site local ENU frame (2026-08-12), computed once alongside the radius above and never
    // changed after — a target site is fixed in ECEF, so its own local East/North/Up frame is too.
    // Used ONLY by the cloud-light build's direction bucketing: quantizing a beam's approach
    // direction in the SITE's frame (rather than the observer's) makes a beam's bucket a pure
    // function of its own geometry — observer-independent, so panning/flying the camera cannot
    // repartition a cluster. Purely an internal quantization frame, so it deliberately does NOT
    // have to match terrain.glsl's enuBasis() convention; unlike that one it guards the polar
    // degeneracy, since a NaN here would silently poison a bucket index.
    glm::vec3 reflectorSiteEnuX[kNumReflectorTargets]{};
    glm::vec3 reflectorSiteEnuY[kNumReflectorTargets]{};
    glm::vec3 reflectorSiteEnuZ[kNumReflectorTargets]{};

    // Mirror slew rate for TargetedReflector: maximum degrees the mirror normal
    // may rotate per real second.  Prevents instant snapping when the nearest
    // valid target changes (e.g. a target crosses into daylight and a different
    // one takes over).  The mirror physically slews toward the goal direction.
    static constexpr float kMirrorRotRateDegPerSec = 1.0f;

    // Maximum body roll for KnifeEdge attitude (degrees from nadir-pointing).
    // Real Starlink solar panels counter-rotate around the along-track axis to
    // compensate body roll.  At 80° the panels still receive ~17% of peak
    // irradiance; beyond this the gimbal runs out of range and power drops
    // sharply.  Limits knife-edge effectiveness when the geometry demands >80°.
    static constexpr float kKnifeMaxRollDeg = 80.0f;

    // Per-satellite current mirror normal in ECI (TargetedReflector only).
    // satMirrorNormals (CPU-era placeholder for GPU mirror slew state) and mirrorSnapFrames/
    // requestMirrorSnap() (forced every TargetedReflector mirror to its ideal attitude for a few
    // frames after an event that invalidated persisted GPU lock/slew state) were removed
    // 2026-08-06 — the reversibility rework deleted the persisted state itself, so there is
    // nothing left to snap out of. See sat_orbit.comp's TargetedReflector block.

    // ── Satellite type catalogue (defined once in initConstellation) ──────────
    std::vector<SatelliteType> satTypes;

    // ── Orbital parameters (fixed at init, positions computed by GPU) ─────────
    std::vector<ConstellationConfig> constellations;
    std::vector<SatOrbit> satOrbits;

    // Re-bake satOrbitBuf when the sim has advanced more than this many days from
    // the baked epoch, keeping float deltaT < 7×86400 = 604800 s (float ULP ≈ 0.07 s).
    static constexpr int64_t kOrbitRebakeDays = 7;
    // Epoch at which satOrbitBuf was last baked (two-part, matches simTime representation).
    // uploadSatOrbits() re-bakes if |simDayJ2000 - orbitEpochDay| > kOrbitRebakeDays.
    int64_t orbitEpochDay = 0;
    double orbitEpochSec = 0.0;

    // ── Gamepad state (Xbox controller support, works the same over Bluetooth or USB —
    //    Windows exposes both as an XInput device, which GLFW 3.4's joystick backend already
    //    talks to) ────────────────────────────────────────────────────────────────
    // GLFW joystick id of the active gamepad; -1 = none connected. Re-scanned in pollGamepad()
    // whenever it goes stale (disconnect), so plug-in/plug-out works without a restart.
    int gamepadId = -1;
    GLFWgamepadstate gpState{};                                     // last frame's full state (for held-button checks in recordCompute)
    unsigned char prevGpButtons[GLFW_GAMEPAD_BUTTON_LAST + 1] = {}; // previous frame's buttons, for edge detection

    // Gamepad rebind-capture anti-self-satisfy guard (see pollGamepad's rebind-capture block).
    // The "Bind Pad" click that sets some binding's listeningPad=true happens via the SAME A
    // press pollGamepad's own edge-detect loop would otherwise see as "a fresh button press" one
    // function call later in the same frame — without this, clicking "Bind Pad" with A instantly
    // self-captured A as the new binding, before the player ever got a chance to press anything
    // else. gpRebindListenIdx tracks which keybinding index the snapshot below belongs to (-1 =
    // none); gpRebindHeldAtStart snapshots which buttons were already down the moment that listen
    // session began, and each stays ineligible for capture until seen released at least once.
    int gpRebindListenIdx = -1;
    bool gpRebindHeldAtStart[GLFW_GAMEPAD_BUTTON_LAST + 1] = {};
    float gpMoveFwd = 0.0f, gpMoveRight = 0.0f;       // left stick, deadzoned, [-1,1] — combines additively with WASD
    float gpLookYawDeg = 0.0f, gpLookPitchDeg = 0.0f; // right stick, this-frame look delta in degrees (already dt-scaled)
    // Analog triggers for elevation — deliberately NOT part of the keybindings/gpButton
    // rebind system (triggers are axes, not digital buttons; same reasoning as WASD/sticks
    // not being rebindable). [0,1] pressure, combined via max() with the (still rebindable)
    // digital KB_RAISE_ELEV/KB_LOWER_ELEV state in recordCompute's elevation block, so
    // "pressure corresponds to vertical speed."
    float gpElevRaise = 0.0f, gpElevLower = 0.0f;

    // UC4: which input device produced the most recent activity — set true by pollGamepad() on
    // any stick deflection/trigger pressure/button press, set false by onKey() (any keypress) and
    // buildUI() (mouse click/drag/scroll). Not persisted (a per-session UI nicety, not a
    // preference). Read by buildViewControlsBody() to lead with whichever device the player is
    // actually holding, instead of always listing keyboard first — see RELEASE_v1_1_PLAN.md UC4.
    bool lastInputWasGamepad = false;

    // ── UC4/UC5: gamepad virtual cursor ("cheap 90%" UI navigation) ────────────
    // Explicitly toggled by KB_TOGGLE_CURSOR (default: Menu/Start), not merely "a UI window is
    // open" — the earlier automatic version hijacked the right stick the instant Settings/Controls
    // was open, which fought free-look and dropped new gamepad players straight into cursor mode
    // on first launch (see RELEASE_v1_1_PLAN.md UC5). While the toggle is off, the right stick
    // always drives camera look, UI visible or not. vCursorX/Y start at -1 as a "not yet
    // positioned" sentinel; first activation centers on screen.
    float vCursorX = -1.0f, vCursorY = -1.0f;
    bool vCursorToggled = false; // player-facing on/off state, flipped by KB_TOGGLE_CURSOR
    bool vCursorActive = false;  // this-frame effective state = vCursorToggled && uiVisible && !showIntro
    bool vCursorClick = false;   // A button currently held (level state, like the real lmb App.cpp
                                 // reads from GLFW) — UIRenderer::beginFrame() does its own frame-to-
                                 // frame edge detection into UIInput::lmbPressed, same as the mouse path

    // ── Mouse state / window handle ───────────────────────────────────────────
    GLFWwindow *win = nullptr;
    int windowedX = 100, windowedY = 100;  // saved windowed position (for restore)
    int windowedW = 1280, windowedH = 720; // saved windowed size (for restore)
    bool firstMouse = true;
    double prevX = 0, prevY = 0;
    float dmx = 0, dmy = 0; // accumulated delta for this frame
    // Cinematic drift mode (toggled by KB_CINEMATIC while RMB is held).
    // Mouse adds force to velocity instead of directly rotating; velocity coasts and decays.
    // Mode is cleared automatically when RMB is released.
    bool cinematicMode = false;     // toggle state
    float cinematicYawVel = 0.0f;   // pixels-equivalent/s driving Rodrigues yaw
    float cinematicPitchVel = 0.0f; // pixels-equivalent/s driving elDeg pitch
    bool cinematicActive = false;   // true last frame — used to detect transition-out

    // ── UI hover state (one-frame lag) ────────────────────────────────────────
    std::vector<bool> hovConst;           // one entry per constellation; sized in loadDefinitions()
    std::vector<bool> hovHighlightConst;  // highlight button hover state, parallel to hovConst
    bool hovShowPlanets = false;          // "Show planets" global toggle hover state
    bool hovPlanetBtn[kPlanetCount] = {}; // per-planet ON/OFF toggle hover state
    bool hovTimeSlower = false;
    bool hovTimePause = false;
    bool hovTimeFaster = false;
    bool hovTimeReverse = false;
    bool hovScreenshot = false; // UC6: left HUD panel camera button
    bool hovSettings = false;
    bool hovSettingsClose = false;
    bool hovAltModeToggle = false;
    bool hovViewControlsClose = false;
    bool hovUnitMetric = false;
    bool hovUnitImperial = false;
    bool hovOpenControlsWindow = false; // Controls tab's "Open Controls Reference" button
    bool hovTab[12] = {}; // one per settings-window tab (kSettingsTabNames)
    bool hovScaleMinus = false;
    bool hovScalePlus = false;
    bool hovRenderScaleMinus = false;
    bool hovRenderScalePlus = false;
    bool hovTrailsBtn = false; // "Star Trails" HUD icon-button hover state (buildLeftHudPanel)
    bool hovMasterVolMinus = false;
    bool hovMasterVolPlus = false;
    bool hovMusicVolMinus = false;
    bool hovMusicVolPlus = false;
    bool hovSfxVolMinus = false;
    bool hovSfxVolPlus = false;
    bool hovRebind[KB_COUNT] = {};    // per keybinding row — sized to match keybindings vector
    bool hovRebindPad[KB_COUNT] = {}; // per keybinding row, gamepad-button rebind button
    bool hovFullscreen = false;
    bool hovSaveSnapshot = false;
    float snapshotMsgTimer = 0.0f; // seconds remaining to show "Saved" confirmation on the perf snapshot button
    bool hovResetDefaults = false;
    float resetDefaultsMsgTimer = 0.0f;          // seconds remaining to show the "Restart to apply" confirmation (NEW-5)
    bool hovDebugToggle[kDebugToggleSlots] = {}; // one per knockout checkbox — see kDebugToggles
                                                 // (top of SatelliteSimUI.cpp) for the row list
    bool hovBeamDebugRaysToggle = false;         // hover state for the "Show beam pointing rays" checkbox (C12 follow-up #12)
    bool hovBeamDebugRaysToggleQuick = false;    // hover state for the Display-tab quick-access copy
                                                 // of the same checkbox, next to Graphics preset
                                                 // (2026-08-06) — separate bool because it's a second,
                                                 // simultaneously-visible Clay element bound to the
                                                 // same showBeamDebugRays bool; sharing one hover bool
                                                 // between two elements would fight over it.
    // Sized 11, not 9 — flare_glow_gain/flare_streak_gain (flare architecture overhaul) added two
    // more PhotoParam rows; per [[feedback_cloud_slider_arrays]], all three hover/dragging arrays
    // must grow together with any new slider id.
    bool hovPhotoMinus[18] = {}; // 15 existing photometry params + 2 trail sliders (Trail decay/gain)
                                 // + 1 flare-mitigation tilt (idx 17)
    bool hovPhotoPlus[18] = {};
    bool draggingPhoto[18] = {};
    bool hovCloudMinus[88] = {}; // was [86] — idx 86/87 are the beam light fade in/out sliders
    bool hovCloudPlus[88] = {};
    bool draggingCloud[88] = {}; // MUST stay sized to match hovCloudMinus/Plus — see
                                 // feedback_cloud_slider_arrays memory: this one was missed once
                                 // already and the out-of-bounds write corrupted the window-chrome
                                 // state declared right below, breaking the settings window.
    // Collapsible section state for the Clouds tab (see buildCloudSliderSections). Indexed by a
    // section's position in its tab's own section array, NOT by slider idx — sections are a pure
    // presentation grouping and own no slider state. Sized 24 (capacity) so adding a category
    // needs no array edit; kCloudSectionSlots is asserted against in buildCloudSliderSections.
    static constexpr int kCloudSectionSlots = 24;
    bool cloudSectionOpen[kCloudSectionSlots] = {}; // all collapsed on open — the point of the grouping
    bool hovCloudSection[kCloudSectionSlots] = {};

    // ── Window chrome (drag+resize; see UIRenderer::WindowChrome) ──────────────
    // x/y default to -1 (uninitialized, centers/places on first open); w/h are set
    // once by the owning builder before the first updateWindowChrome() call.
    // Only the two real windows (Settings, View Controls) have chrome — the left/right
    // HUD panels are fixed to their screen corner (see buildLeftHudPanel/buildRightHudPanel),
    // recomputed from screenW/H every frame so they track window resizes automatically.
    WindowChrome settingsChrome;
    WindowChrome viewControlsChrome;
    int settingsActiveTab = 0; // index into kSettingsTabNames (persisted; clamped on load)

    // ── Right HUD panel: altitude display mode + unit system ──────────────────
    bool altModeSeaLevel = true;                // true = MSL (sea level), false = AGL (above terrain)
    UnitSystem unitSystem = UnitSystem::Metric; // Display tab setting; affects altitude readout

    // ── NEW-7: frame limiter ────────────────────────────────────────────────
    FpsCapMode fpsCapMode = FpsCapMode::VSync;  // Display tab setting; see FpsCapMode comment
    bool fpsCapSwapchainRebuildPending = false; // set true when fpsCapMode changes; see
                                                // consumeSwapchainRebuildRequest() override
    bool hovFpsCap[5] = {};                     // one per FpsCapMode button, same order as the enum

    // ── UC1: graphics preset ─────────────────────────────────────────────────
    // Default High until init() either loads a persisted value or (first run only) seeds one from
    // VkPhysicalDeviceProperties::deviceType — see seedGraphicsPresetFromDevice() in init().
    GraphicsPreset graphicsPreset = GraphicsPreset::High;
    bool showAdvancedSettings = false; // Display tab "Show advanced settings" — reveals the
                                       // Clouds/Ocean/Terrain/Aurora tabs; off by default so a
                                       // new user's front door is Preset + the handful of
                                       // top-level controls, not 46 developer sliders.
    bool hovPreset[6] = {};            // one per clickable preset button (Custom has no button —
                                       // it's a read-only status, not a click target; Potato does)
    float fpsBadgeEma = 0.0f;          // EMA-smoothed 1/dt for the HUD fps badge, so a true low
                                       // frame rate reads as a steady number instead of flickering
                                       // between adjacent integers frame to frame
    bool hovAdvancedToggle = false;
    bool hovReplayIntro = false; // UC3 "Replay Intro" button, Display tab
    // UC3 follow-up: "Play intro on startup" checkbox, Display tab. Persisted (loadSettings/
    // saveSettings) and applied to showIntro at load time. Default true so a genuine first run
    // (no settings.json at all — loadSettings() returns before reaching this key) still opens with
    // the cinematic; an existing settings.json missing this key (i.e. upgrading from a build that
    // predates it) defaults to false at load time instead, so upgrading players don't suddenly get
    // a cinematic that didn't exist in their version — see loadSettings().
    bool playIntroOnStartup = true;
    bool hovPlayIntroStartup = false;

    // ── Private helpers ───────────────────────────────────────────────────────
    // NEW-7: pushes fpsCapMode's present-mode requirement into VulkanContext and flags App to
    // rebuild the swapchain with it (see consumeSwapchainRebuildRequest() above). Called both from
    // the Settings > Display button row and once after loadSettings() if the persisted mode isn't
    // the VulkanContext default (VSync/FIFO).
    void applyFpsCapMode()
    {
        if (!ctx_)
            return;
        switch (fpsCapMode)
        {
        case FpsCapMode::VSync:
            ctx_->presentModePreference = VK_PRESENT_MODE_FIFO_KHR;
            break;
        case FpsCapMode::Off:
            ctx_->presentModePreference = VK_PRESENT_MODE_IMMEDIATE_KHR;
            break;
        default:
            ctx_->presentModePreference = VK_PRESENT_MODE_MAILBOX_KHR;
            break; // Cap30/60/120
        }
        fpsCapSwapchainRebuildPending = true;
    }
    void createBuffers(VulkanContext &ctx);
    void createDescriptors(VulkanContext &ctx);
    void createOrbitDescriptors(VulkanContext &ctx);
    void createOrbitPipeline(VulkanContext &ctx);
    void uploadSatOrbits(VulkanContext &ctx);
    void createCloudNoisePipeline(VulkanContext &ctx);
    void createCloudWarpNoisePipeline(VulkanContext &ctx);
    void createAuroraNoisePipeline(VulkanContext &ctx);
    void createCloudMarchResources(VulkanContext &ctx);
    void createCloudMarchDescriptors(VulkanContext &ctx);
    void createCloudMarchPipeline(VulkanContext &ctx);
    void createSceneDepthResources(VulkanContext &ctx);
    void createSceneDepthDescriptors(VulkanContext &ctx);
    void createSceneDepthPipeline(VulkanContext &ctx);
    void createBeamSelfMarchDescriptors(VulkanContext &ctx);
    void createBeamSelfMarchPipeline(VulkanContext &ctx);
    void createGlowResources(VulkanContext &ctx);
    void createComputePipeline(VulkanContext &ctx);
    void createSkyBgPipeline(VulkanContext &ctx);
    void createSkyLowResResources(VulkanContext &ctx); // resolution scaling (session 29)
    void destroySkyLowResResources(VkDevice device);   // called from onResize (before recreate) and cleanup
    void createDrawPipeline(VulkanContext &ctx);
    // ── Flare/corona render-to-texture pipeline (flare architecture overhaul) ─────────────────
    void createFlareResources(VulkanContext &ctx);   // images, samplers, render pass, framebuffer
    void destroyFlareResources(VkDevice device);     // called from onResize (before recreate) and cleanup
    void createFlareDescriptors(VulkanContext &ctx); // new flareBlur (2 storage images) and
                                                     // flareComposite (1 sampler) descriptor sets.
                                                     // descLayout/descSet bindings 7/8 and
                                                     // skyDescSet binding 20 are added directly in
                                                     // createDescriptors()/createGlowResources()
                                                     // themselves (descriptor set layouts are
                                                     // immutable once created, so those two can't
                                                     // be "extended" afterward from here).
    void createFlarePipelines(VulkanContext &ctx);   // flareSource (graphics), flareBlur (compute),
                                                     // flareComposite (graphics) pipelines
    // ── Long-exposure trail pipeline ───────────────────────────────────────────────────────────
    // Must run after createFlarePipelines (needs drawPipeLayout/descSet) AND after initPlanets
    // (needs starPipeLayout/starDescSet/planetDescSet) — see init()'s call order.
    void createTrailResources(VulkanContext &ctx);   // image/view/mem, render pass, framebuffer
    void destroyTrailResources(VkDevice device);     // called from onResize (before recreate) and cleanup
    void createTrailDescriptors(VulkanContext &ctx); // trailFade (1 storage image) and
                                                     // trailComposite (1 sampler) descriptor sets
    void createTrailPipelines(VulkanContext &ctx);   // trailFade (compute), trailSat/trailStar
                                                     // (graphics, reuse existing pipeline layouts),
                                                     // trailComposite (graphics) pipelines
    void initStars(VulkanContext &ctx);
    void createStarPipeline(VulkanContext &ctx);
    void updateStars();
    void initPlanets(VulkanContext &ctx);                                       // planetBuf + planetDescSet, reusing starDescLayout
    void updatePlanets();                                                       // mirrors updateStars(); called right after it
    int pickPlanetAt(float clickX, float clickY, float screenW, float screenH); // mirrors
                                                                                // pickSatelliteAt but reads the already host-mapped
                                                                                // planetBuf directly — no device->host staging copy
    void formatSelectedPlanetInfo();                                            // mirrors formatSelectedSatInfo, fills planetInfoLine[]
    void updateLightPollutionDome();                                            // called each frame before updateStars(): fills lightDomeAz[]
                                                                                // + uploads to lightDomeBuf for sat_flare.comp
    void updateGpuTimingStats(VulkanContext &ctx);                              // called at top of recordCompute(): EMA-smooths
                                                                                // ctx.timestampMs into gpuMsSmoothed[]/gpuMsTotalSmoothed
    void initConstellation();                                                   // called once: loads definitions then builds orbits
    void loadDefinitions();                                                     // reads constellations.json; falls back to hardcoded defaults
    void loadHardcoded();                                                       // hardcoded satTypes + constellations (used as fallback)
    void buildOrbits();                                                         // populates satOrbits from satTypes + constellations
    // S1 (RELEASE_v1_1_PLAN.md): reads reflector_targets.json (real solar-farm sites, moddable
    // like constellations.json); falls back to generateReflectorTargetsRandomFallback() if
    // missing/malformed/empty. Called from buildOrbits(), same call-order constraint as the old
    // inline code it replaced (needs earthElevCpu, populated by createGlowResources() earlier in
    // init()). Sets reflectorTargetCount and reflectorObserverSpawnIdx.
    void loadReflectorTargets();
    // Fallback used only when reflector_targets.json is absent/unusable: kNumReflectorTargets-1
    // uniformly-random lat/lon points, plus a REAL fixed entry at index 0 for the observer spawn
    // point (67S 67W) — fixes the same "index 0 left as a degenerate zero-vector" bug the JSON
    // path fixes, so both paths guarantee a real reachable target near spawn on first launch.
    void generateReflectorTargetsRandomFallback();
    // Shared by both paths above: looks up real terrain elevation at reflectorTargetsECEF[ti]
    // (3x3-max texel neighborhood, C12 follow-up #23) and writes reflectorTargetsRadiusM[ti].
    // Defaults to sea level if earthElevCpu isn't populated for whatever reason.
    void computeReflectorTargetElevationRadius(int ti);
    void loadSettings(); // reads settings.json; silently uses defaults if missing
    void saveSettings(); // writes settings.json next to exe
    // UC1: overwrites debugDisableMask/renderScale/advanced sliders per the named preset's table
    // (no-op data-wise for Custom — see GraphicsPreset comment). Recreates the render-scale
    // offscreen target since presets can change renderScale.
    void applyGraphicsPreset(GraphicsPreset p);
    // UC1 first-run seed: VkPhysicalDeviceProperties::deviceType -> Low (integrated/CPU/virtual)
    // or Medium (discrete). Coarse on purpose — see RELEASE_v1_1_PLAN.md UC1, "do not build a
    // GPU-name lookup table." Only called once, from init(), when no persisted preset exists.
    GraphicsPreset seedGraphicsPresetFromDevice(VulkanContext &ctx) const;
    void savePerfSnapshot(float cpuDt);                // appends one profiling record to perf_profiles/profile_log.jsonl
    nlohmann::json buildPerfSnapshotJson(float cpuDt); // the shared body of the above and of the sweep record
    void appendPerfRecord(const nlohmann::json &j);    // one JSONL line into perf_profiles/profile_log.jsonl
    void startKnockoutSweep();                         // Display tab "Run knockout sweep" button
    void updateKnockoutSweep(float cpuDt);             // one step of the sweep state machine; called
                                                       // once per frame from recordCompute()
    void updatePositions(double t, float dt = 0.0f);   // called each frame: fills satInputData + eci2enu
    void toggleTimeDirection() { timeDir = -timeDir; } // shared by KB_REVERSE and the left-panel Reverse button

    // ── Satellite picking (see "Satellite picking / selection tracking" members above) ────
    // Pure camera geometry mirror of sat_point.vert's projection (shaders/sat_point.vert:34,47,60-62);
    // shared by the one-shot full-buffer scan below and the per-frame tracked-selection reprojection.
    bool projectSkyDirToScreen(const glm::vec3 &skyDir, float screenW, float screenH,
                               float &outX, float &outY) const;
    // One-shot: copies satVisibleBuf back to a transient host-visible staging buffer and scans it
    // for the nearest on-screen visible satellite to (clickX, clickY). Returns -1 if none hit.
    int pickSatelliteAt(float clickX, float clickY, float screenW, float screenH);
    void formatSelectedSatInfo(); // fills selectedSatInfoBuf from satOrbits[selectedSatIndex]; call when selection changes

    // ── Small UI helpers shared across every UI builder method ────────────────
    // (formerly local lambdas inside the single monolithic buildUI(); now member
    // functions since buildUI is split across buildLeftHudPanel/buildSettingsWindow/etc.)
    uint16_t fs(int base) const { return (uint16_t)std::max(8, (int)(base * uiScale + 0.5f)); }
    // Defined in SatelliteSimUI.cpp — need AudioSystem's complete type (only forward-declared here).
    void sndRollover(bool nowHov, bool prevHov) const;
    void sndClick(bool nowHov, bool lmbPressed) const;

    // ── UI builders (defined in SatelliteSimUI.cpp) ────────────────────────────
    void buildLeftHudPanel(const UIInput &inp, UIRenderer &ui);
    void buildRightHudPanel(const UIInput &inp, UIRenderer &ui);
    void buildSelectedSatPanel(const UIInput &inp, UIRenderer &ui); // floating panel that tracks selectedSatIndex

    // Shared resizable+draggable+(optionally) closable window frame — title bar,
    // 8-direction edge/corner resize, bevel border — used by both the settings
    // window and the view-controls window so there is one window implementation,
    // not two. `buildBody` declares whatever content goes inside (tabs+content for
    // settings, a plain scroll list for view-controls). Returns true the frame the
    // close button was clicked (closable windows only), so callers can react (e.g.
    // save settings).
    bool buildResizableWindow(const UIInput &inp, UIRenderer &ui, WindowChrome &chrome,
                              int winId, const char *title, bool closable, bool &hovCloseFlag,
                              float defaultX, float defaultY,
                              float minW, float minH, float maxW, float maxH,
                              const std::function<void()> &buildBody);

    void buildSettingsWindow(const UIInput &inp, UIRenderer &ui);
    void buildSettingsTabbedBody(const UIInput &inp, UIRenderer &ui);
    void buildSettingsConstellationsTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsSoundTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsControlsTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsCameraTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsDisplayTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsPhotometryTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsCloudsTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsOceanTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsTerrainTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsAuroraTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsBeamsTab(const UIInput &inp, UIRenderer &ui);
    void buildSettingsAttributionsTab(const UIInput &inp, UIRenderer &ui);
    // Shared slider-row struct/renderer for the Clouds/Ocean/Terrain/Aurora tabs (split from one
    // combined "Clouds" tab, session 28 follow-up #9) — `idx` indexes the shared draggingCloud/
    // hovCloudMinus/hovCloudPlus member arrays and a function-local static text-buffer array, so
    // each tab's slider subset keeps its ORIGINAL global index (no renumbering needed) even though
    // only a slice of the full 0-32 range is passed to any one call.
    struct CloudSlider
    {
        const char *label;
        float *val;
        float vmin, vmax, step;
        const char *fmt;
        int idx;
    };
    void buildCloudSliderRows(const UIInput &inp, UIRenderer &ui, CloudSlider *sliders, int count);
    // Collapsible category grouping on top of buildCloudSliderRows. A section owns no slider
    // state — it only decides whether its slice of the array is rendered this frame — so a
    // slider's global `idx` (and therefore its hover/drag/text-buffer slots) is untouched by
    // which section it lands in, and sliders can be regrouped freely without renumbering.
    struct CloudSliderSection
    {
        const char *title;
        CloudSlider *sliders;
        int count;
    };
    void buildCloudSliderSections(const UIInput &inp, UIRenderer &ui, CloudSliderSection *sections, int count);
    void buildViewControlsWindow(const UIInput &inp, UIRenderer &ui);
    void buildViewControlsBody(const UIInput &inp, UIRenderer &ui);
    void buildIntroOverlay(const UIInput &inp, UIRenderer &ui);
    void buildCrashRecoveryNotice(float dt, const UIInput &inp, UIRenderer &ui); // NEW-3
    void buildGraphicsAutoNotice(float dt, const UIInput &inp, UIRenderer &ui);  // UC1 mechanism 3
    void buildScreenshotToast(float dt, const UIInput &inp, UIRenderer &ui);     // UC6 confirmation toast
    // UC3: advances introElapsed and drives obsHeightOffset/camera.elDeg/fovYDeg/obsFacing from
    // kIntroKeyframes; called from recordCompute() in place of the normal WASD/zoom block while
    // showIntro is true. Also accumulates the UC1 benchmark and auto-ends the intro at the last
    // keyframe (calling finishIntro(false)).
    void updateIntroCinematic(float dt);
    // Ends the intro (showIntro=false). wasSkipped=true means the user dismissed early (click/key/
    // pad) — no representative frame-time average was collected, so the benchmark promote/demote
    // is skipped entirely and whatever preset was already active (the device-type seed, or a prior
    // session's saved preset) stands, per RELEASE_v1_1_PLAN.md UC3: "do not run the benchmark
    // during the skip path."
    void finishIntro(bool wasSkipped);
    void setLat(float newLatDeg);   // moves observer to a new latitude; used by the right panel's lat display scroll-adjust
    void adjustLon(float deltaDeg); // rotates observer around Earth's polar axis; right panel's lon display scroll-adjust
                                    // dt = simulated seconds elapsed this frame (0 when paused);
                                    // used for mirror slew rate so behaviour is consistent at all time scales
};

// Time scale options (simulated seconds per real second)
static constexpr float kTimeScales[] = {1.0f, 10.0f, 60.0f, 300.0f, 3600.0f,
                                        86400.0f, 86400.0f * 7.0f, 86400.0f * 30.0f, 86400.0f * 365.0f};
static constexpr const char *kTimeLabels[] = {"1x", "10x", "1m", "5m", "1h", "1d", "1w", "1mo", "1yr"};
static constexpr int kNumTimeScales = 9;

// ── UC3 intro cinematic fixed vantage (session follow-up) ─────────────────────
// The intro always opens from this exact real-world spot — the California coast at twilight,
// facing out toward the SpaceX AI-datacenter satellites and the Reflect Orbital mirrors aimed at
// the nearby Topaz solar farm — rather than wherever the player's last saved/current observer
// position happens to be. Values are the live camera/observer state the vantage was designed
// against (settings.json: observer.lat_deg/lon_deg, camera.az_deg/el_deg/fov_y_deg). Forced onto
// obsDir/obsLatDeg/obsLonDeg/camera.azDeg/elDeg/fovYDeg once, at the start of every intro playback
// (see updateIntroCinematic's one-time init block) — including a Display-tab replay — so the
// cinematic is reproducible no matter where the player has since wandered off to.
static constexpr float kIntroObserverLatDeg = 35.871456f;
static constexpr float kIntroObserverLonDeg = -121.400291f;
static constexpr float kIntroStartAzDeg = -61.32f;
static constexpr float kIntroStartElDeg = 20.8f;
static constexpr float kIntroStartFovDeg = 70.0f;
// Cloud drift rate forced for the duration of intro playback. This is a POSITION constant as much
// as a speed one: cloudPhase = fmod(cloudDriftRate * simTime, 2pi) and the intro always starts at
// the same fixed epoch, so this value picks which part of the cloud noise field sits over the
// vantage above on frame 1. Retuned when cloud-noise changes left the intro opening under solid
// overcast. Applied in updateIntroCinematic's one-time init block, overriding the settings.json
// value (see cloudDriftRate's own default for the non-intro case).
static constexpr float kIntroCloudDriftRate = 6.55e-6f;

// ── UC3 intro cinematic beat sheet (RELEASE_v1_1_PLAN.md) ─────────────────────
// Shared between SatelliteSim.cpp (updateIntroCinematic/finishIntro, the playback) and
// SatelliteSimUI.cpp (buildIntroOverlay, the caption text) — header-scope so both translation
// units see the identical table without a getter. `text == nullptr` means "no new caption at
// this keyframe" (introCaptionIndex just holds whatever the last non-null one was).
struct IntroKeyframe
{
    float t;    // seconds from intro start
    float altM; // obsHeightOffset target
    float azDeg, elDeg, fovDeg;
    const char *text;
};
static constexpr float songbeat = 7.61f; // beat timings below are synced to this (music tempo)
static constexpr IntroKeyframe kIntroKeyframes[] = {
    // Beat 0 — "2036" title/date card. Ground, twilight, facing the fixed vantage above.
    {0.0f, 0.0f, kIntroStartAzDeg, kIntroStartElDeg, kIntroStartFovDeg, "2036"},
    // Beat 1 — first narrative line; holds the same framing. The skip hint (buildIntroOverlay)
    // only appears once this beat is reached — see kIntroHintRevealIndex.
    {songbeat * 1.0, 0.0f, kIntroStartAzDeg, kIntroStartElDeg, kIntroStartFovDeg,
     "Satellite megaconstellations dominate the night sky"},
    // Beat 3 — level out in preparation for launch. Still ground level.
    {songbeat * 2.0, 00000.0f, kIntroStartAzDeg - 5, 25.0f, 64.0f, "From the ground, we watch sunlight glint off their solar arrays"},
    // Beat 4 — the pull to LEO begins (ascent happens across THIS beat's transition). Still
    // facing horizontally west, per the storyboard, even as altitude climbs.
    {songbeat * 3.0, 60000.0f, kIntroStartAzDeg - 25, 35.0f, 62.0f, "They power an orbital network of communications and AI compute."},
    // Beat 5 — pull continues; camera starts rotating away from due-west as we climb high enough
    // to reveal the Earth's curve.
    {songbeat * 4.0, 140000.0f, kIntroStartAzDeg - 35, 20.0f, 62.0f, "A promise to the markets above all else"},
    {songbeat * 5.0, 250000.0f, kIntroStartAzDeg - 45, 10.0, 70.0f, "We will come to miss the quiet sky"},
    // Beat 6 — arrival in LEO: pulled back and up enough to see the backlit AI sats against a
    // rising sun. Camera motion stops here; beats 7-8 hold this exact framing. Title reveal.
    {songbeat * 6.0, 300000.0f, kIntroStartAzDeg - 50, 0.0, 80.0f, "SAT LIGHT SIM"},
    // Beat 7 — controls hint. "WASD to move" is a fixed line (buildIntroOverlay); the Q/E line is
    // generated at render time from live keybindings, not this literal (kIntroControlsIndex marks
    // which entry to override).
    {songbeat * 7.0, 300000.0f, kIntroStartAzDeg - 50, 20.0, 120.0f, "Q / E to raise/lower height"},
    {songbeat * 8.0, 0000.0f, kIntroStartAzDeg - 50, 20.0, 80.0f, nullptr}, // hold, then auto-handoff (finishIntro(false))
};
static constexpr int kIntroKeyframeCount = sizeof(kIntroKeyframes) / sizeof(kIntroKeyframes[0]);
static constexpr int kIntroYearIndex = 0;       // "2036" title/date card
static constexpr int kIntroHintRevealIndex = 1; // skip hint doesn't show before this beat is reached
static constexpr int kIntroTitleIndex = 6;      // "SAT LIGHT SIM" reveal, arrival in LEO
static constexpr int kIntroControlsIndex = 7;   // WASD / Q-E controls hint
// Beats 0-6 (through arrival in LEO, where camera motion stops) feed the UC1 benchmark
// accumulator; the static hold over beats 7-8 isn't representative load, so it's excluded.
static constexpr float kIntroBenchEndT = songbeat * 8.0f;
static constexpr const char *kGraphicsPresetNames[] = {"Planetarium", "Low", "Medium", "High", "Ultra", "Custom", "Potato"};
