#include "SatelliteSim.h"
#include "../UIRenderer.h"
#include "../AudioSystem.h"
#include "../Paths.h"
#include "../Log.h"
#include "version.h"
#include "clay.h"
#include "star_catalog.h"
#include "stb_image.h"
#include "stb_image_write.h" // UC6 — implementation lives in UIRenderer.cpp (one TU only)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

// ── Earth + observer constants ─────────────────────────────────────────────────
static constexpr float kEarthRadius = 6'371'000.0f; // mean Earth radius (m)
static constexpr double kOmegaEarth = 7.2921150e-5; // sidereal rotation rate (rad/s)
static constexpr float kObsLatDefault = 37.0f;      // default observer latitude (°N, ~Bay Area)

// ── Orbital mechanics ─────────────────────────────────────────────────────────
static constexpr double kGM = 3.986004418e14;        // Earth gravitational parameter (m³/s²)
static constexpr double kJ2 = 1.08263e-3;            // Earth oblateness (J2) coefficient
static constexpr double kYearSec = 365.25 * 86400.0; // seconds per tropical year
// SSO nodal precession rate = Earth's mean orbital motion ≈ 0.9856°/day eastward.
// J2 causes a retrograde circular orbit (i > 90°) to precess its RAAN eastward at
// exactly this rate, keeping the nodal plane fixed relative to the sun.
static constexpr double kSSOPrecRate = 2.0 * 3.14159265358979323846 / kYearSec; // rad/s

// ── Photometry (must mirror sat_flare.comp constants) ────────────────────────
// kBrightnessScale MUST stay in sync with BRIGHTNESS_SCALE in sat_flare.comp.
// kMagRef and kMagRefFlare define the calibration anchor for the magnitude readout.
// kBrightnessScale / kDaySuppression removed — now runtime members on SatelliteSim,
// synced to SatFlarePC each frame so CPU magnitude readout matches GPU render.
static constexpr float kRefRange = 500'000.0f; // 500 km normalisation range (m)
static constexpr float kMagRef = 6.0f;         // apparent magnitude at kMagRefFlare
static constexpr float kMagRefFlare = 0.008f;  // effectFlare corresponding to kMagRef
// Virtual diffuse floor for the magnitude readout only (not sent to GPU).
// Zero-diffuse satellites (Starlink) are only visible via transient specular flares;
// this floor lets them appear in the readout as a meaningful steady-state estimate.
static constexpr float kMagDiffuseFloor = 0.003f;

static inline float computeMeanMotion(float altM)
{
    double a = (double)kEarthRadius + (double)altM;
    return (float)sqrt(kGM / (a * a * a)); // rad/s
}

// SSO inclination from J2 nodal precession: solves dΩ/dt = kSSOPrecRate.
// dΩ/dt = -1.5 * n * J2 * (Re/a)² * cos(i)   →   cos(i) = -kSSOPrecRate / (1.5*n*J2*(Re/a)²)
// Result is in the retrograde range (~97–107° for typical LEO/MEO SSO altitudes).
static inline float computeSSOInclination(float altM)
{
    double a = (double)kEarthRadius + (double)altM;
    double n = sqrt(kGM / (a * a * a));
    double rat = (double)kEarthRadius / a;
    double cosI = -kSSOPrecRate / (1.5 * n * kJ2 * rat * rat);
    return (float)acos(glm::clamp(cosI, -1.0, 1.0));
}

// ── Planetary ephemeris (low-precision Keplerian) ────────────────────────────
// Planets feature (RELEASE_v1_1_PLAN follow-up, session 30). Deliberately NOT the satellite
// orbital-compute pipeline (GpuSatOrbit/sat_orbit.comp) — that solves near-field Earth-relative
// geometry (shadow, attitude, specular surfaces) planets don't have. This is the same shape of
// closed-form CPU math as the sun/moon block above, just parameterised per body.
//
// Elements + rates (a, e, i, L, longitude of perihelion, longitude of ascending node; centurial
// rates) are JPL/Standish "Keplerian Elements for Approximate Positions of the Major Planets",
// valid 1800-2050 AD (https://ssd.jpl.nasa.gov/planets/approx_pos.html) — comfortably covers the
// sim's fixed 2036-06-21 epoch. Earth uses the table's EM Bary row (standard for this purpose).
// The Moon uses a two-body Keplerian ellipse fit to the linear terms of its ELP2000-82B mean
// elements (Meeus, "Astronomical Algorithms" ch. 47) — a real, if approximate (no evection/
// variation/other periodic perturbations — "Kepler approximation is fine" per the plan this
// implements), improvement over the previous circular-equatorial orbit with a phase constant
// hand-calibrated for a single epoch (2026-03-30) that drifted for any other date.
struct KeplerElements
{
    double a0, aDot;       // semi-major axis (AU), per Julian century
    double e0, eDot;       // eccentricity (dimensionless), per Julian century
    double i0, iDot;       // inclination (deg), per Julian century
    double L0, LDot;       // mean longitude (deg), per Julian century
    double peri0, periDot; // longitude of perihelion, ϖ = ω+Ω (deg), per Julian century
    double node0, nodeDot; // longitude of ascending node, Ω (deg), per Julian century
};

static constexpr KeplerElements kEarthElements{
    1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
    100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0};

// PlanetId/kPlanetCount declared in SatelliteSim.h (shared with UI code).
const char *const kPlanetNames[kPlanetCount] = {
    "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus"};

// Approximate true-color tint per planet (session 30 follow-up) — same role B-V color plays for
// stars, but planets don't have a spectral index to derive one from, so these are hand-picked to
// the real, commonly-cited visual colors (matches planetarium-software convention) rather than
// computed. Normalized so the brightest channel is ~1.0 (matches how star colors are scaled),
// not physically calibrated albedo/reflectance — this is a tint on top of the magnitude-driven
// intensity, not a separate brightness source. Mars is the one that actually reads as colored to
// the naked eye at this scale; the others are subtle by design (real gas-giant cloud tops are
// pale, not vividly colored) — a strong gray floor stays close to the previous flat near-white
// on Mercury/Jupiter/Saturn, with Venus/Uranus getting a faint warm/cool cast.
static constexpr glm::vec3 kPlanetColor[kPlanetCount] = {
    {0.92f, 0.87f, 0.80f}, // Mercury — grayish tan, faintly warm (airless, dusty regolith)
    {1.00f, 0.94f, 0.78f}, // Venus — pale cream (sulfuric acid cloud tops)
    {1.00f, 0.2f, 0.2f},   // Mars — the one that visibly reads as colored: rust/salmon
    {0.96f, 0.89f, 0.76f}, // Jupiter — pale tan (ammonia cloud bands, subtle at this scale)
    {0.95f, 0.89f, 0.68f}, // Saturn — pale gold
    {0.72f, 0.90f, 0.93f}, // Uranus — pale cyan (methane absorption)
};

static constexpr KeplerElements kPlanetElements[kPlanetCount] = {
    // Mercury
    {0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
     252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081},
    // Venus
    {0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
     181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418},
    // Mars
    {1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
     -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343},
    // Jupiter
    {5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
     34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106},
    // Saturn
    {9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
     49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794},
    // Uranus
    {19.18916464, -0.00196176, 0.04725744, -0.00004397, 0.77263783, -0.00242939,
     313.23810451, 428.48202785, 170.95427630, 0.40805281, 74.01692503, 0.04240589},
};

// Moon: geocentric two-body fit. a/e/i held constant (negligible drift at this precision); L and
// mean anomaly M' linear terms are ELP2000-82B's (Meeus ch.47); peri = L - M' (both linear in T,
// so peri is too) recovers the "longitude of perihelion" shape kEarthElements/kPlanetElements use,
// letting keplerEclipticPos() serve the Moon with no special-casing.
static constexpr double kMoonL0 = 218.3164477, kMoonLDot = 481267.88123421;
static constexpr double kMoonM0 = 134.9633964, kMoonMDot = 477198.8675055;
static constexpr KeplerElements kMoonElements{
    0.00256955, 0.0, 0.0549, 0.0, 5.145, 0.0,
    kMoonL0, kMoonLDot,
    kMoonL0 - kMoonM0, kMoonLDot - kMoonMDot,
    125.0445479, -1934.1362891};

// Solves Kepler's equation and returns position in the J2000 mean-ecliptic frame, AU (heliocentric
// for kEarthElements/kPlanetElements, geocentric for kMoonElements — the math doesn't care which,
// only the caller's interpretation of the origin does). T = Julian centuries since J2000 TT.
static glm::dvec3 keplerEclipticPos(const KeplerElements &el, double T)
{
    double a = el.a0 + el.aDot * T;
    double e = el.e0 + el.eDot * T;
    double iR = glm::radians(el.i0 + el.iDot * T);
    double L = el.L0 + el.LDot * T;
    double peri = el.peri0 + el.periDot * T;
    double node = el.node0 + el.nodeDot * T;

    double M = fmod(L - peri, 360.0);
    if (M > 180.0)
        M -= 360.0;
    if (M < -180.0)
        M += 360.0;
    double MR = glm::radians(M);

    // Newton-Raphson solve of Kepler's equation M = E - e*sin(E). Converges in ~4-5 iterations
    // even at Mercury's eccentricity (0.206); 8 is cheap insurance, this runs 7x once per frame.
    double E = MR;
    for (int iter = 0; iter < 8; ++iter)
    {
        double dE = (E - e * sin(E) - MR) / (1.0 - e * cos(E));
        E -= dE;
        if (fabs(dE) < 1e-9)
            break;
    }

    double xOrb = a * (cos(E) - e);
    double yOrb = a * sqrt(1.0 - e * e) * sin(E);

    double wR = glm::radians(peri - node); // argument of perihelion ω = ϖ - Ω
    double nodeR = glm::radians(node);
    double cw = cos(wR), sw = sin(wR);
    double cn = cos(nodeR), sn = sin(nodeR);
    double ci = cos(iR), si = sin(iR);

    // Standard 3-1-3 (ω, i, Ω) rotation from the orbital plane into J2000 mean-ecliptic XYZ.
    double xEcl = (cw * cn - sw * sn * ci) * xOrb + (-sw * cn - cw * sn * ci) * yOrb;
    double yEcl = (cw * sn + sw * cn * ci) * xOrb + (-sw * sn + cw * cn * ci) * yOrb;
    double zEcl = (sw * si) * xOrb + (cw * si) * yOrb;

    return glm::dvec3(xEcl, yEcl, zEcl);
}

// Apparent visual magnitude, V = V0 + 5*log10(r*delta) + phase-angle polynomial(alphaDeg).
// Source: Paul Schlyter, "How to compute planetary positions" (stjarnhimlen.se/comp/ppcomp.html)
// — a standard, widely-cited amateur-astronomy reference. Saturn's ring-brightness contribution
// (which needs Saturnicentric ring-plane geometry, not just phase angle) is deliberately omitted —
// accepted simplification, Saturn will read slightly dim near ring-plane-open oppositions.
static float planetApparentMagnitude(PlanetId id, double rAU, double deltaAU, double alphaDeg)
{
    double base = 5.0 * log10(rAU * deltaAU);
    switch (id)
    {
    case kMercury:
        return (float)(-0.36 + base + 0.027 * alphaDeg + 2.2e-13 * pow(alphaDeg, 6.0));
    case kVenus:
        return (float)(-4.34 + base + 0.013 * alphaDeg + 4.2e-7 * pow(alphaDeg, 3.0));
    case kMars:
        return (float)(-1.51 + base + 0.016 * alphaDeg);
    case kJupiter:
        return (float)(-9.25 + base + 0.014 * alphaDeg);
    case kSaturn:
        return (float)(-9.00 + base + 0.044 * alphaDeg);
    case kUranus:
        return (float)(-7.15 + base + 0.001 * alphaDeg);
    default:
        return 99.0f;
    }
}

// ─── init ─────────────────────────────────────────────────────────────────────
void SatelliteSim::init(VulkanContext &ctx)
{
    // exeDir_: read-only game data (constellations.json, assets/, shaders/) — always next to
    // the exe. userDataDir_: settings/perf writes — exeDir_ itself when writable (the common
    // case), falling back to %APPDATA%/SatLightSim etc. only on a read-only install
    // (see Paths.h / NEW-4 in RELEASE_v1_1_PLAN.md).
    exeDir_ = Paths::exeDir();
    userDataDir_ = Paths::userDataDir();

    // NEW-3 (RELEASE_v1_1_PLAN.md): crash-safe mode. If the sentinel from a PREVIOUS run is
    // still here, that run never reached cleanup()'s clean-exit path. Recreate it now (empty
    // file, presence is the only signal) so this run is tracked from here on; the effect
    // (force Planetarium + notice) is applied after loadSettings() runs below, so it overrides
    // whatever preset that load would otherwise have chosen.
    auto crashSentinelPath = std::filesystem::path(userDataDir_) / "session.lock";
    bool crashDetected = std::filesystem::exists(crashSentinelPath);
    {
        std::ofstream sentinelOut(crashSentinelPath, std::ios::trunc);
    }

    // Fixed start time: 2036-06-21 00:00:00 UTC
    // J2000.0 = 2000-01-01 12:00:00 UTC = Unix 946728000
    // 2036-06-21 00:00:00 UTC = Unix 2097619200
    // Fixed start time: 1150891200 seconds from J2000 = 13320 days + 43200 s.
    // Stored split so float deltaT stays small regardless of time-warp distance.
    constexpr int64_t kInitWholeSec = 1150891200LL + 6 * 30 * 24 * 60 * 60 + 20.9 * 60 * 60;
    simDayJ2000 = kInitWholeSec / 86400LL;           // 13320
    simSecInDay = (double)(kInitWholeSec % 86400LL); // 43200.0
    simInitDayJ2000 = simDayJ2000;
    simInitSecInDay = simSecInDay;

    ctx_ = &ctx;

    // Order must match the KB_* enum in SatelliteSim.h.
    // held=true  → polled every frame in recordCompute (modifier/held keys)
    // held=false → fired once in onKey/pollGamepad (toggle/event keys)
    // gpButton   → default Xbox-controller binding; -1 = unbound (user can still bind one
    //              in the settings window). KB_CINEMATIC is left unbound by default since
    //              it's a mouse-drag-flavored feature (see dispatchKeyAction).
    keybindings = {
        {"Toggle UI", GLFW_KEY_TAB, GLFW_GAMEPAD_BUTTON_BACK, false, false},                // KB_TOGGLE_UI
        {"Pause/Resume", GLFW_KEY_SPACE, GLFW_GAMEPAD_BUTTON_B, false, false},              // KB_PAUSE — moved off Start (session follow-up) to free it for KB_TOGGLE_CURSOR below
        {"Slow Down", GLFW_KEY_COMMA, GLFW_GAMEPAD_BUTTON_DPAD_LEFT, false, false},         // KB_SLOWER
        {"Speed Up", GLFW_KEY_PERIOD, GLFW_GAMEPAD_BUTTON_DPAD_RIGHT, false, false},        // KB_FASTER
        {"Reverse Time", GLFW_KEY_R, GLFW_GAMEPAD_BUTTON_DPAD_UP, false, false},            // KB_REVERSE
        {"Move Fast", GLFW_KEY_LEFT_SHIFT, GLFW_GAMEPAD_BUTTON_LEFT_THUMB, true, false},    // KB_MOVE_BOOST (held)
        {"Move Fine", GLFW_KEY_LEFT_CONTROL, GLFW_GAMEPAD_BUTTON_RIGHT_THUMB, true, false}, // KB_MOVE_FINE  (held)
        {"Cinematic Pan", GLFW_KEY_LEFT_ALT, -1, false, false},                             // KB_CINEMATIC  (event, toggle)
        {"Raise Elevation", GLFW_KEY_Q, -1, true, false},                                   // KB_RAISE_ELEV (held) — gamepad is the analog right trigger, see gpElevRaise
        {"Lower Elevation", GLFW_KEY_E, -1, true, false},                                   // KB_LOWER_ELEV (held) — gamepad is the analog left trigger, see gpElevLower
        {"Reset Elevation", GLFW_KEY_Z, -1, false, false},                                  // KB_RESET_ELEV (event) — Y reassigned to Reset Zoom below
        {"Zoom In", GLFW_KEY_EQUAL, GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER, true, false},         // KB_ZOOM_IN    (held)
        {"Zoom Out", GLFW_KEY_MINUS, GLFW_GAMEPAD_BUTTON_LEFT_BUMPER, true, false},         // KB_ZOOM_OUT   (held)
        {"Reset Zoom", GLFW_KEY_0, GLFW_GAMEPAD_BUTTON_Y, false, false},                    // KB_ZOOM_RESET (event)
        {"Select Satellite", GLFW_KEY_T, GLFW_GAMEPAD_BUTTON_A, false, false},              // KB_SELECT_SAT (event) — center-of-screen pick; moved off F (session follow-up) to free F for KB_TOGGLE_TRAILS below
        {"Screenshot", GLFW_KEY_F12, -1, false, false},                                     // KB_SCREENSHOT (event) — no standard gamepad "capture" button to default to
        {"Toggle Cursor", GLFW_KEY_C, GLFW_GAMEPAD_BUTTON_START, false, false},             // KB_TOGGLE_CURSOR (event) — UC5: gamepad virtual-cursor mode; no meaningful effect for KBM (mouse is always a free cursor), kept rebindable/listed for consistency
        {"Star Trails", GLFW_KEY_F, GLFW_GAMEPAD_BUTTON_X, false, false},                   // KB_TOGGLE_TRAILS (event) — long-exposure trail on/off
    };
    static_assert(KB_COUNT == 18, "KB enum and keybindings initializer are out of sync");

    createBuffers(ctx);
    createCloudNoisePipeline(ctx);
    createCloudWarpNoisePipeline(ctx); // must run before createCloudMarchDescriptors (binding 9)
    createAuroraNoisePipeline(ctx);    // must run before createGlowResources' writes (binding 16)
    createCloudMarchResources(ctx);    // images must exist before createGlowResources' writes (bindings 10/11)
    createSceneDepthResources(ctx);    // image must exist before createGlowResources' writes (binding 19)
    createGlowResources(ctx);
    createDescriptors(ctx);
    createComputePipeline(ctx);
    createOrbitDescriptors(ctx);
    createOrbitPipeline(ctx);
    createCloudMarchDescriptors(ctx); // needs cloudParamsBuf from createGlowResources above
    createCloudMarchPipeline(ctx);
    createSceneDepthDescriptors(ctx); // needs earthElev/earthSpec from createGlowResources above
    createSceneDepthPipeline(ctx);
    createBeamSelfMarchDescriptors(ctx); // needs cloudParamsBuf/earthClouds/cloudNoise/cloudWarpNoise
                                          // (createGlowResources above) and reflectBeamsBuf (createBuffers)
    createBeamSelfMarchPipeline(ctx);
    createSkyBgPipeline(ctx);
    createSkyLowResResources(ctx); // resolution scaling — needs skyBgPipeLayout from just above
    createDrawPipeline(ctx);
    // Flare/corona render-to-texture pipeline (flare architecture overhaul) — needs descLayout
    // (createDescriptors above, for flareSourcePipeLayout's reused descriptor set) and
    // ctx.renderPass (already valid — createSkyBgPipeline/createDrawPipeline above already use it).
    createFlareResources(ctx);
    createFlareDescriptors(ctx);
    createFlarePipelines(ctx);
    updatePositions((double)simDayJ2000 * 86400.0 + simSecInDay); // must run first — initConstellation reads sunDirECI
    initConstellation();
    // C12 follow-up #33: one-time upload of reflectorTargetsECEF[]/RadiusM[] (fixed for the
    // simulation's lifetime once initConstellation() generates them) into their GPU-visible
    // companion buffer — sat_orbit.comp reads this every frame (TargetedReflector target search),
    // but it never needs refreshing since the CPU arrays themselves never change after this point.
    {
        std::vector<glm::vec4> targetsECEF(kNumReflectorTargets);
        for (int ti = 0; ti < kNumReflectorTargets; ++ti)
            targetsECEF[ti] = glm::vec4(reflectorTargetsECEF[ti], reflectorTargetsRadiusM[ti]);
        memcpy(reflectorTargetsECEFMapped, targetsECEF.data(), sizeof(glm::vec4) * kNumReflectorTargets);
    }
    uploadSatOrbits(ctx); // bake + upload GpuSatOrbit data after orbits are built
    initStars(ctx);
    initPlanets(ctx); // must run after initStars() — reuses starDescLayout/starPipeline
    // Long-exposure trail pipeline — needs drawPipeLayout/descSet (createDrawPipeline/
    // createDescriptors above) AND starPipeLayout/starDescSet/planetDescSet (initStars/initPlanets
    // just above) for its splat-stage pipelines, plus flareSampler (createFlareResources above)
    // for its composite stage.
    createTrailResources(ctx);
    createTrailDescriptors(ctx);
    createTrailPipelines(ctx);

    // Default window chrome sizes — must be set before the first updateWindowChrome()
    // call (buildUI); loadSettings() below may override x/y/w/h with persisted values.
    // settingsChrome defaults above its own 680x420 min (buildSettingsWindow) so it
    // never opens already-clamped-smaller-than-its-own-content on a fresh install.
    settingsChrome.w = 720.0f;
    settingsChrome.h = 480.0f;
    viewControlsChrome.w = 300.0f;
    viewControlsChrome.h = 340.0f;

    loadSettings(); // override defaults with any previously saved values

    // renderScale (just loaded above) may differ from the 1.0 default createSkyLowResResources
    // used a few lines earlier in this function — recreate at the correct persisted size now.
    // Cheap and harmless when unchanged (same one-time startup cost either way).
    destroySkyLowResResources(ctx.device);
    createSkyLowResResources(ctx);

    // Shown by default on first run per showControlsOnStartup (itself persisted); applied
    // after loadSettings() so a saved false sticks. viewControlsChrome.open is intentionally
    // NOT persisted — closing it only lasts for the current run (see buildViewControlsWindow).
    viewControlsChrome.open = showControlsOnStartup;

    // NEW-7: fpsCapMode (just loaded above) may differ from the VulkanContext default the
    // startup swapchain was already created with (FIFO/VSync) — push it through now. Idempotent
    // and cheap when the loaded value already matches, same reasoning as the render-scale
    // recreate just above.
    applyFpsCapMode();

    // NEW-3: apply the crash-recovery override AFTER loadSettings() (and the preset re-derivation
    // inside it) so it wins over whatever preset the previous session had — the whole point is
    // that the very next launch after a bad exit comes up in the cheapest, least-likely-to-repeat-
    // the-crash configuration, not back in the settings that may have caused it.
    if (crashDetected)
    {
        crashRecoveryMode = true;
        applyGraphicsPreset(GraphicsPreset::Planetarium);
        crashRecoveryNoticeTimer = 8.0f;
        fprintf(stderr, "[SatelliteSim] Previous session did not exit cleanly — forcing "
                        "Planetarium preset.\n");
    }
}

// ─── onResize ─────────────────────────────────────────────────────────────────
void SatelliteSim::onResize(VulkanContext &ctx)
{
    vkDestroyPipeline(ctx.device, skyBgPipeline, nullptr);
    skyBgPipeline = VK_NULL_HANDLE;
    vkDestroyPipeline(ctx.device, skyBgMinimalPipeline, nullptr);
    skyBgMinimalPipeline = VK_NULL_HANDLE;
    createSkyBgPipeline(ctx); // recreates both skyBgPipeline and skyBgMinimalPipeline

    // Resolution scaling: low-res target is sized off ctx.swapExtent too, so it needs the same
    // destroy+recreate treatment as skyBgPipeline just above.
    destroySkyLowResResources(ctx.device);
    createSkyLowResResources(ctx);

    vkDestroyPipeline(ctx.device, drawPipeline, nullptr);
    drawPipeline = VK_NULL_HANDLE;
    createDrawPipeline(ctx);

    vkDestroyPipeline(ctx.device, starPipeline, nullptr);
    starPipeline = VK_NULL_HANDLE;
    createStarPipeline(ctx);

    // ── Half-res cloud march targets (C15-perf) — the only swapchain-size-dependent images this
    // class owns; recreate at the new half-extent, then patch the two descriptor sets that point
    // at their views (the sampler is resolution-independent and kept as-is). Safe with no extra
    // synchronization: this app has exactly one frame in flight (single fence, waited on at the
    // top of every drawFrame), matching the unsynchronized pipeline recreation just above.
    vkDestroyImageView(ctx.device, cloudMarchTargetAView, nullptr);
    vkDestroyImage(ctx.device, cloudMarchTargetAImg, nullptr);
    vkFreeMemory(ctx.device, cloudMarchTargetAMem, nullptr);
    vkDestroyImageView(ctx.device, cloudMarchTargetBView, nullptr);
    vkDestroyImage(ctx.device, cloudMarchTargetBImg, nullptr);
    vkFreeMemory(ctx.device, cloudMarchTargetBMem, nullptr);
    cloudMarchTargetAImg = cloudMarchTargetBImg = VK_NULL_HANDLE;
    cloudMarchTargetAMem = cloudMarchTargetBMem = VK_NULL_HANDLE;
    cloudMarchTargetAView = cloudMarchTargetBView = VK_NULL_HANDLE;
    createCloudMarchResources(ctx); // recreates images/views; leaves both in SHADER_READ_ONLY_OPTIMAL

    VkDescriptorImageInfo skyAInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo skyBInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet skyWrites[2] = {};
    skyWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, skyDescSet, 10, 0, 1,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &skyAInfo, nullptr, nullptr};
    skyWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, skyDescSet, 11, 0, 1,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &skyBInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, skyWrites, 0, nullptr);

    VkDescriptorImageInfo storageAInfo{VK_NULL_HANDLE, cloudMarchTargetAView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo storageBInfo{VK_NULL_HANDLE, cloudMarchTargetBView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet computeWrites[2] = {};
    computeWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 5, 0, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storageAInfo, nullptr, nullptr};
    computeWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 6, 0, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storageBInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, computeWrites, 0, nullptr);

    // C12 follow-up #33: descSet (satellite draw pipeline) also holds bindings 5/6 pointing at
    // these same views (sat_point.frag's cloud occlusion) — needs the same refresh as skyDescSet
    // above, or it would keep pointing at the image views just destroyed.
    VkDescriptorImageInfo satCloudAInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo satCloudBInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet satCloudWrites[2] = {};
    satCloudWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 5, 0, 1,
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &satCloudAInfo, nullptr, nullptr};
    satCloudWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 6, 0, 1,
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &satCloudBInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, satCloudWrites, 0, nullptr);

    // Session 30 bug fix: starDescSet/planetDescSet (bindings 2/3 of starDescLayout, shared by
    // both) also point at these same views for star_point.frag's cloud occlusion — same refresh
    // as descSet just above, or they'd keep pointing at the image views just destroyed.
    VkDescriptorImageInfo starCloudAInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo starCloudBInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet starCloudWrites[4] = {};
    starCloudWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, starDescSet, 2, 0, 1,
                          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &starCloudAInfo, nullptr, nullptr};
    starCloudWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, starDescSet, 3, 0, 1,
                          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &starCloudBInfo, nullptr, nullptr};
    starCloudWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, planetDescSet, 2, 0, 1,
                          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &starCloudAInfo, nullptr, nullptr};
    starCloudWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, planetDescSet, 3, 0, 1,
                          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &starCloudBInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 4, starCloudWrites, 0, nullptr);

    // ── Shared scene depth — same swapchain-size dependency, same destroy/recreate/patch dance.
    // Two sets reference it: its own (as a storage image, for writing) and skyDescSet binding 20
    // (as a sampled image, for reading). Miss either and the next frame samples a destroyed view.
    vkDestroyImageView(ctx.device, sceneDepthView, nullptr);
    vkDestroyImage(ctx.device, sceneDepthImg, nullptr);
    vkFreeMemory(ctx.device, sceneDepthMem, nullptr);
    sceneDepthImg = VK_NULL_HANDLE;
    sceneDepthMem = VK_NULL_HANDLE;
    sceneDepthView = VK_NULL_HANDLE;
    createSceneDepthResources(ctx); // recreates image/view; leaves it in SHADER_READ_ONLY_OPTIMAL

    VkDescriptorImageInfo depthStorageInfo{VK_NULL_HANDLE, sceneDepthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo depthSampledInfo{sceneDepthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet depthWrites[6] = {};
    depthWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDepthDescSet, 2, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthStorageInfo, nullptr, nullptr};
    depthWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 13, 0, 1,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthSampledInfo, nullptr, nullptr};
    depthWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, skyDescSet, 19, 0, 1,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthSampledInfo, nullptr, nullptr};
    // descSet binding 7 (sceneDepthTex, flare architecture overhaul) — flare_source.frag's own
    // terrain occlusion test needs the same repatch every other sceneDepthView consumer gets here.
    depthWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 7, 0, 1,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthSampledInfo, nullptr, nullptr};
    // starDescSet/planetDescSet binding 4 (sceneDepthTex, long-exposure trail terrain occlusion) —
    // same repatch, both share starDescLayout's shape.
    depthWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, starDescSet, 4, 0, 1,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthSampledInfo, nullptr, nullptr};
    depthWrites[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, planetDescSet, 4, 0, 1,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthSampledInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 6, depthWrites, 0, nullptr);

    // ── Flare/corona render-to-texture pipeline (flare architecture overhaul) — flareExtent
    // derives from ctx.swapExtent, same destroy/recreate/patch dance as the targets above.
    // flareSourcePipeline/flareCompositePipeline bake their viewport size at creation (same
    // convention as drawPipeline/skyBgPipeline) so both need destroying first; flareBlurPipeline
    // is swapchain-size-independent (compute, no viewport) and createFlarePipelines() guards its
    // own recreation, so it's left alone here.
    vkDestroyPipeline(ctx.device, flareSourcePipeline, nullptr);
    flareSourcePipeline = VK_NULL_HANDLE;
    vkDestroyPipeline(ctx.device, flareCompositePipeline, nullptr);
    flareCompositePipeline = VK_NULL_HANDLE;
    destroyFlareResources(ctx.device);
    createFlareResources(ctx);
    createFlarePipelines(ctx);

    VkDescriptorImageInfo flareAInfo{VK_NULL_HANDLE, flareSourceView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo flareBInfo{VK_NULL_HANDLE, flareScratchView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet flareBlurWrites[2] = {};
    flareBlurWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, flareBlurDescSet, 0, 0, 1,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &flareAInfo, nullptr, nullptr};
    flareBlurWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, flareBlurDescSet, 1, 0, 1,
                          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &flareBInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, flareBlurWrites, 0, nullptr);

    VkDescriptorImageInfo flareFinalInfo{flareSampler, flareScratchView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet flareCompWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                        flareCompositeDescSet, 0, 0, 1,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &flareFinalInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &flareCompWrite, 0, nullptr);

    // ── Long-exposure trail pipeline — trailAccumExtent derives from ctx.swapExtent, same
    // destroy/recreate/patch dance as the flare block above. trailSatPipeline/trailStarPipeline/
    // trailCompositePipeline bake their viewport size at creation so all three need destroying
    // first; trailFadePipeline is swapchain-size-independent (compute, no viewport) and
    // createTrailPipelines() guards its own recreation, so it's left alone here.
    vkDestroyPipeline(ctx.device, trailSatPipeline, nullptr);
    trailSatPipeline = VK_NULL_HANDLE;
    vkDestroyPipeline(ctx.device, trailStarPipeline, nullptr);
    trailStarPipeline = VK_NULL_HANDLE;
    vkDestroyPipeline(ctx.device, trailCompositePipeline, nullptr);
    trailCompositePipeline = VK_NULL_HANDLE;
    destroyTrailResources(ctx.device);
    createTrailResources(ctx);
    createTrailPipelines(ctx);
    // Resize policy: clear trail contents — createTrailResources() already zeroed the freshly
    // recreated image, this just documents that intent defensively (see the trailClearPending
    // member comment).
    trailClearPending = true;

    VkDescriptorImageInfo trailFadeImgInfo{VK_NULL_HANDLE, trailAccumView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet trailFadeWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                        trailFadeDescSet, 0, 0, 1,
                                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &trailFadeImgInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &trailFadeWrite, 0, nullptr);

    VkDescriptorImageInfo trailCompImgInfo{flareSampler, trailAccumView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet trailCompWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                        trailCompositeDescSet, 0, 0, 1,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &trailCompImgInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &trailCompWrite, 0, nullptr);
}

// ─── recordCompute ────────────────────────────────────────────────────────────
// Reads ctx.timestampMs (resolved by App::drawFrame right after this frame's fence wait,
// i.e. before this call — see VulkanContext::resolveTimestamps) and EMA-smooths the eight
// pass-duration buckets into gpuMsSmoothed[].  VulkanContext::kTimestampCount carries the
// authoritative slot table; this function, kPerfLabels[] in SatelliteSimUI.cpp, and the JSON
// keys in savePerfSnapshot() must all stay in sync with it and with each other.
// Diagnostic env-var gate (checked once, cached). Used to bisect the macOS/MoltenVK per-frame
// GPU stall by removing whole passes from the command buffer. All default off — normal builds
// are unaffected.
static bool dbgEnv(const char *name)
{
    // A tiny cache keyed by pointer identity is enough — call sites pass string literals.
    struct E { const char *k; bool v; };
    static E cache[8];
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (cache[i].k == name)
            return cache[i].v;
    const char *e = std::getenv(name);
    bool v = e && e[0] == '1';
    if (n < 8)
        cache[n++] = {name, v};
    return v;
}

void SatelliteSim::updateGpuTimingStats(VulkanContext &ctx)
{
    if (!ctx.timestampsReady)
        return;
    const double *t = ctx.timestampMs;
    // The pipeline-unification pass added two buckets and removed one: beam cloud block (whose
    // cost used to be folded silently into orbit compute, which is why that bucket always read a
    // suspiciously flat 0.37-0.59 ms) and scene depth, the new shared terrain-depth pass; cloud
    // shadow map went away when its dispatch was folded into cloud_march.comp.
    float raw[8] = {
        (float)(t[1] - t[0]), // scene depth compute (shared terrain/ocean depth)
        (float)(t[2] - t[1]), // beam cloud block compute (C12 follow-up #33)
        (float)(t[3] - t[2]), // orbit compute
        (float)(t[4] - t[3]), // cloud march compute (incl. the per-pixel cloud shadow)
        (float)(t[5] - t[4]), // flare compute
        (float)(t[6] - t[5]), // sky/terrain/ocean bg + cloud composite fragment shader
        (float)(t[7] - t[6]), // satellite points + star draw
        (float)(t[8] - t[7]), // UI overlay
    };
    const float kAlpha = 0.1f; // low-pass so the HUD numbers don't flicker frame to frame
    for (int i = 0; i < 8; ++i)
    {
        gpuMsRaw[i] = raw[i];
        gpuMsSmoothed[i] = glm::mix(gpuMsSmoothed[i], raw[i], kAlpha);
    }
    gpuMsRawTotal = (float)(t[8] - t[0]);
    gpuMsTotalSmoothed = glm::mix(gpuMsTotalSmoothed, gpuMsRawTotal, kAlpha);
}

// ─── beginCpuFrameTiming ──────────────────────────────────────────────────────
// Publishes the previous frame's accumulated CPU bucket times and clears the accumulator for the
// frame about to run. Called first thing in buildUI() — the first sim entry point of a frame — so
// what it publishes is always a COMPLETE frame's worth of timers, never a partially-filled one.
// The resulting one-frame staleness matches gpuMsRaw[]'s (App resolves the query pool for the
// previous frame just before this), which is what lets a sweep step sample both from the same
// moment. Same EMA constant as the GPU side, for the same reason: readable HUD numbers.
void SatelliteSim::beginCpuFrameTiming()
{
    const float kAlpha = 0.1f;
    for (int i = 0; i < CPU_COUNT; ++i)
    {
        cpuMsRaw[i] = cpuAccumMs[i];
        cpuMsSmoothed[i] = glm::mix(cpuMsSmoothed[i], cpuMsRaw[i], kAlpha);
        cpuAccumMs[i] = 0.0f;
    }
}

bool SatelliteSim::gpHeld(int bindIdx) const
{
    if (gamepadId < 0 || bindIdx < 0 || bindIdx >= (int)keybindings.size())
        return false;
    int b = keybindings[bindIdx].gpButton;
    return b >= 0 && b <= GLFW_GAMEPAD_BUTTON_LAST && gpState.buttons[b] == GLFW_PRESS;
}

// Polled once per frame, before anything else in recordCompute reads gamepad state.
// Mirrors onKey()'s dispatch/rebind-capture for the gamepad's digital buttons, and fills
// gpMoveFwd/gpMoveRight/gpLookYawDeg/gpLookPitchDeg from the sticks — consumed later this
// same recordCompute call (movement) and by buildUI() next frame (look; buildUI runs before
// recordCompute in the frame loop, so the stick's contribution is one frame behind the
// mouse's, same as any other cross-frame state here — imperceptible for continuous input).
void SatelliteSim::pollGamepad(float dt)
{
    gpMoveFwd = gpMoveRight = 0.0f;
    gpLookYawDeg = gpLookPitchDeg = 0.0f;
    gpElevRaise = gpElevLower = 0.0f;

    if (gamepadId < 0 || !glfwJoystickIsGamepad(gamepadId))
    {
        // Rescan — handles both first connection and hot-swap after a disconnect.
        gamepadId = -1;
        for (int j = GLFW_JOYSTICK_1; j <= GLFW_JOYSTICK_LAST; ++j)
        {
            if (glfwJoystickIsGamepad(j))
            {
                gamepadId = j;
                break;
            }
        }
        if (gamepadId < 0)
            return;
        memset(prevGpButtons, GLFW_RELEASE, sizeof(prevGpButtons));
        memset(&gpState, 0, sizeof(gpState));
    }

    GLFWgamepadstate state;
    if (!glfwGetGamepadState(gamepadId, &state))
        return;

    // UC3/UC4: a controller has no Space bar, so Start is its one defined skip button — mirrors
    // onKey()'s single-key Space rule (see its comment) rather than the old "any newly-pressed
    // button" behavior, which was just as easy to trigger by accident as a stray keypress/click.
    // Kept live for the WHOLE intro (both before and after controls unlock below), same as Space.
    if (showIntro)
    {
        if (state.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS &&
            prevGpButtons[GLFW_GAMEPAD_BUTTON_START] != GLFW_PRESS)
            finishIntro(true);

        // UC3 follow-up: movement/look go live once the controls-hint beat is showing, same
        // handoff point recordCompute's WASD block and buildUI's mouse-look block already use
        // (introCaptionIndex >= kIntroControlsIndex) — this used to be missing here, so gamepad
        // movement/look silently stayed dead for the rest of the intro even after the on-screen
        // text said otherwise (keyboard/mouse already worked via those two call sites; only this
        // early-return was gamepad-specific). Rebind capture, edge-triggered button actions, and
        // the virtual cursor stay fully cinematic-locked for the whole intro though — mirroring
        // onKey()'s Space-only gate, which blocks everything else regardless of caption index.
        if (introCaptionIndex < kIntroControlsIndex)
        {
            memcpy(prevGpButtons, state.buttons, sizeof(prevGpButtons));
            gpState = state;
            return;
        }
    }

    if (!showIntro)
    {
        // Rebind capture: if a binding is waiting for a pad button, claim the first newly-pressed
        // one and stop — mirrors onKey()'s keyboard capture, including consuming this poll so the
        // same button press can't also fire as a normal action below.
        //
        // Anti-self-satisfy guard: the settings UI's "Bind Pad" button is itself clicked with a
        // gamepad button (A, via the virtual cursor's click) — WITHOUT this guard, that exact same
        // still-held A press looked, one function call later in this same frame, like "a fresh
        // button press" to the capture loop below, instantly binding A to whatever row was just
        // clicked before the player ever got a chance to press anything else. (This is how
        // "Raise Elevation" ended up silently bound to A in a real run — self-captured by the
        // click that opened its own "Bind Pad" listening state.) Snapshot whichever buttons are
        // already held the moment a NEW listen session starts (tracked by keybinding index, so
        // switching which row is listening re-snapshots too); each stays ineligible until seen
        // released at least once.
        int listeningIdx = -1;
        for (size_t i = 0; i < keybindings.size(); ++i)
            if (keybindings[i].listeningPad)
            {
                listeningIdx = (int)i;
                break; // only one binding can listen at a time (enforced by the settings UI)
            }

        if (listeningIdx != gpRebindListenIdx)
        {
            gpRebindListenIdx = listeningIdx;
            for (int b = 0; b <= GLFW_GAMEPAD_BUTTON_LAST; ++b)
                gpRebindHeldAtStart[b] = (state.buttons[b] == GLFW_PRESS);
        }

        if (listeningIdx >= 0)
        {
            KeyBinding &kb = keybindings[listeningIdx];
            for (int b = 0; b <= GLFW_GAMEPAD_BUTTON_LAST; ++b)
            {
                bool down = state.buttons[b] == GLFW_PRESS;
                if (gpRebindHeldAtStart[b])
                {
                    if (!down)
                        gpRebindHeldAtStart[b] = false; // released — eligible from next press
                    continue;
                }
                bool wasDown = prevGpButtons[b] == GLFW_PRESS;
                if (down && !wasDown)
                {
                    kb.gpButton = b;
                    kb.listeningPad = false;
                    gpRebindListenIdx = -1;
                    memcpy(prevGpButtons, state.buttons, sizeof(prevGpButtons));
                    gpState = state;
                    return;
                }
            }
        }

        // UC5: whether the virtual cursor will be active this frame, computed ahead of the
        // edge-triggered loop below purely so KB_SELECT_SAT (also bound to A) can be skipped for
        // the same press that clicks the cursor — see that skip's comment.
        bool cursorWillBeActive = vCursorToggled && uiVisible;

        // Edge-triggered (event) actions.
        for (size_t i = 0; i < keybindings.size(); ++i)
        {
            const KeyBinding &kb = keybindings[i];
            if (kb.held || kb.gpButton < 0 || kb.gpButton > GLFW_GAMEPAD_BUTTON_LAST)
                continue;
            // UC5: A double-books as both "select nearest satellite to screen center" and the
            // virtual cursor's click. While the cursor is up, A should only click whatever it's
            // pointed at (handled by vCursorClick below, consumed as the UI/pick lmb in App.cpp)
            // — firing the center-screen pick on the same press used to also silently reselect
            // whatever satellite sat behind the menu, fighting the cursor click.
            if ((int)i == KB_SELECT_SAT && cursorWillBeActive)
                continue;
            bool down = state.buttons[kb.gpButton] == GLFW_PRESS;
            bool wasDown = prevGpButtons[kb.gpButton] == GLFW_PRESS;
            if (down && !wasDown)
                dispatchKeyAction((int)i);
        }
    }

    memcpy(prevGpButtons, state.buttons, sizeof(prevGpButtons));
    gpState = state; // held-button checks (gpHeld) read this later in recordCompute

    auto deadzone = [](float v, float dz)
    {
        float a = fabsf(v);
        if (a < dz)
            return 0.0f;
        return copysignf((a - dz) / (1.0f - dz), v);
    };

    // Left stick: forward/right, additive with WASD in recordCompute's movement block.
    gpMoveFwd = -deadzone(state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], 0.2f); // stick forward (up) = negative axis Y
    gpMoveRight = deadzone(state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], 0.2f);

    // Right stick: look, applied unconditionally in buildUI (no RMB-capture concept for a
    // controller — there's no cursor to hide). Rate-based, so scale by dt here; mouse deltas
    // are already frame-rate independent (raw pixel deltas), gamepad deflection is not.
    constexpr float kGamepadLookDegPerSec = 90.0f;
    gpLookYawDeg = deadzone(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], 0.15f) * kGamepadLookDegPerSec * dt;
    gpLookPitchDeg = deadzone(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], 0.15f) * kGamepadLookDegPerSec * dt;

    // Triggers: elevation pressure. GLFW's standardized gamepad axes are documented as -1
    // (released) to +1 (fully pressed), so remap to [0,1]; small deadzone at the released end
    // only (not a symmetric deadzone — a trigger has no "center" to null out).
    auto triggerPressure = [](float axis)
    {
        float t = glm::clamp((axis + 1.0f) * 0.5f, 0.0f, 1.0f);
        return t < 0.02f ? 0.0f : t;
    };
    gpElevRaise = triggerPressure(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER]);
    gpElevLower = triggerPressure(state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]);

    // UC4/UC5: gamepad virtual cursor ("cheap 90%" UI navigation, RELEASE_v1_1_PLAN.md). Gated on
    // the explicit KB_TOGGLE_CURSOR toggle (default: Menu/Start) rather than "a UI window is
    // open" — see vCursorToggled's comment in SatelliteSim.h for why. While off, the right stick
    // always drives camera look, UI visible or not. A = click.
    bool cursorRawX = false, cursorRawY = false; // did the stick actually move the cursor this frame?
    bool cursorActive = vCursorToggled && uiVisible;
    if (cursorActive && win)
    {
        // Cursor bounds/position must match the coordinate space buildUI() feeds Clay (window
        // logical size, from glfwGetWindowSize — same space glfwGetCursorPos's real mouse uses),
        // NOT ctx_->swapExtent (framebuffer/physical-pixel size). The two differ under Windows
        // display scaling (125%/150%/etc.), which used to put the drawn cursor dot and Clay's
        // actual hit-tested position at different points on screen — the root cause of hover/
        // click landing on the wrong element ("finicky" button selection).
        int ww = 0, wh = 0;
        glfwGetWindowSize(win, &ww, &wh);
        if (vCursorX < 0.0f) // "not yet positioned" sentinel — first activation centers on screen
        {
            vCursorX = (float)ww * 0.5f;
            vCursorY = (float)wh * 0.5f;
        }
        // Wider deadzone than the look-stick's 0.15 (used above for camera look/gpLookYawDeg):
        // a worn/imperfectly-centered stick resting a few % past 0.15 is imperceptible for
        // camera look (which just drifts a hair, self-correcting), but for a rate-based pointer
        // aimed at a small button it reads as the cursor never actually stopping — it creeps out
        // of the button's hit-box a moment after arriving, which looks exactly like "hover blips
        // on contact but never sticks, and doesn't register at all when the player thinks they've
        // let go." 0.30 costs some fine-aim range at the low end but removes that failure mode for
        // any stick whose true rest position falls short of it.
        constexpr float kCursorSpeedPxPerSec = 1000.0f;
        constexpr float kCursorDeadzone = 0.30f;
        float rawX = deadzone(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], kCursorDeadzone);
        float rawY = deadzone(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], kCursorDeadzone);
        cursorRawX = rawX != 0.0f;
        cursorRawY = rawY != 0.0f;
        vCursorX = glm::clamp(vCursorX + rawX * kCursorSpeedPxPerSec * dt, 0.0f, (float)ww);
        vCursorY = glm::clamp(vCursorY + rawY * kCursorSpeedPxPerSec * dt, 0.0f, (float)wh);
        vCursorActive = true;
        vCursorClick = state.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        // Suppress the normal look-stick response so the camera doesn't also spin while the
        // player is aiming the cursor at a button.
        gpLookYawDeg = 0.0f;
        gpLookPitchDeg = 0.0f;
    }
    else
    {
        vCursorActive = false;
        vCursorClick = false;
    }

    // UC4: any real deflection/pressure this frame counts as "the player is using a gamepad
    // right now" — button presses are already covered by the edge-triggered loop above via
    // dispatchKeyAction, but that doesn't distinguish input source, so check the raw buttons too.
    if (gpMoveFwd != 0.0f || gpMoveRight != 0.0f || gpLookYawDeg != 0.0f || gpLookPitchDeg != 0.0f ||
        gpElevRaise != 0.0f || gpElevLower != 0.0f || cursorRawX || cursorRawY)
        lastInputWasGamepad = true;
    else
        for (int b = 0; b <= GLFW_GAMEPAD_BUTTON_LAST; ++b)
            if (state.buttons[b] == GLFW_PRESS)
            {
                lastInputWasGamepad = true;
                break;
            }
}

// ─── virtualCursor ───────────────────────────────────────────────────────────
// UC4: reports the current virtual-cursor state to App (see Simulation.h's calling convention
// and pollGamepad's comment for how vCursorX/Y/Active/Click are maintained).
bool SatelliteSim::virtualCursor(float &x, float &y, bool &lmb) const
{
    if (!vCursorActive)
        return false;
    x = vCursorX;
    y = vCursorY;
    lmb = vCursorClick;
    return true;
}

// ── Bounded top-K helper (2026-08-10 perf) ────────────────────────────────────────────
// Three separate places below used to keep a fixed-capacity "strongest N by intensity" table and,
// once full, found the weakest entry with a FULL LINEAR SCAN for every remaining candidate:
// ground beams (256 slots), cloud clusters (256) and individual cloud lights (256), against
// up to kMaxActiveBeams=2048 candidates — worst case ~1.5M comparisons per frame, all in
// the block that measured 1.89 ms.
//
// 2026-08-12: only the ground-beam table still uses this. The two cloud-light tables were replaced
// by persistent key-addressed pools (see TrackedBeamLight in SatelliteSim.h) whose eviction is a
// genuinely rare path rather than a per-candidate one, so they scan directly instead of needing a
// cached minimum. Kept as-is for ground beams, where the per-candidate reject is still the hot path.
//
// This caches the current minimum instead. A candidate that cannot beat the cached minimum
// is rejected without touching the table at all (the overwhelmingly common case once the
// table is full of strong entries); only an actual insertion pays a rescan. The result is
// BIT-IDENTICAL to the old full scan, including tie-breaking: both pick the first index
// holding the minimum value, and the reject path was already a no-op.
struct TopK
{
    uint32_t weakestIdx = 0;
    float weakestVal = 0.0f;
    bool dirty = true; // recompute before the next comparison

    // Returns the slot to write, or ~0u to reject. `values` is a stride-agnostic accessor
    // so the three call sites can keep their own differently-shaped arrays.
    template <typename GetVal>
    uint32_t claim(uint32_t &count, uint32_t capacity, float candidate, GetVal getVal)
    {
        if (count < capacity)
        {
            dirty = true; // a fresh slot changes the minimum
            return count++;
        }
        if (dirty)
        {
            weakestIdx = 0;
            weakestVal = getVal(0u);
            for (uint32_t i = 1u; i < capacity; ++i)
            {
                float v = getVal(i);
                if (v < weakestVal)
                {
                    weakestVal = v;
                    weakestIdx = i;
                }
            }
            dirty = false;
        }
        if (candidate <= weakestVal)
            return ~0u;
        dirty = true; // we are about to overwrite the minimum
        return weakestIdx;
    }
};

void SatelliteSim::recordCompute(VkCommandBuffer cmd, VulkanContext &ctx, float dt)
{
    updateGpuTimingStats(ctx);
    // Immediately after, so the sweep accumulates THIS frame's freshly-resolved gpuMsRaw[] and any
    // mask change it makes takes effect in the push constants filled later in this same call.
    updateKnockoutSweep(dt);
    pollGamepad(dt);

    // UC6: pick up a finished background screenshot encode, if any (see finalizeScreenshot()'s
    // comment for why the encode runs on a thread). Polled here (always runs, every frame)
    // rather than inside buildScreenshotToast (gated on uiVisible) so a screenshot taken with the
    // UI hidden still gets its toast queued up and ready the moment the UI is shown again.
    if (screenshotResultReady.exchange(false))
    {
        std::lock_guard<std::mutex> lock(screenshotResultMutex);
        snprintf(screenshotToastText, sizeof(screenshotToastText), "%s", screenshotResultText.c_str());
        screenshotToastTimer = 4.0f;
    }

    // ── WASD surface navigation ───────────────────────────────────────────────
    // Pure 3D ECEF — no lat/lon arithmetic, no gimbal lock, works at any latitude.
    //
    // obsDir    : unit position vector on the Earth-fixed sphere.
    // obsFacing : unit tangent vector (forward), always ⊥ obsDir.
    //
    // W/S move along obsFacing; A/D move along cross(obsFacing, obsDir) (right).
    // After each step obsFacing is parallel-transported to stay tangent at newPos.
    if (showIntro)
    {
        // UC3: the cinematic camera path drives obsHeightOffset/camera.elDeg/fovYDeg/obsFacing
        // directly this frame — skip normal WASD/zoom input entirely so they can't fight it. Also
        // advances introElapsed/introCaptionIndex and eventually calls finishIntro().
        updateIntroCinematic(dt);
    }
    // UC3 follow-up: once the controls-hint beat is showing ("WASD to move" / "Q / E to
    // raise/lower height" — see buildIntroOverlay), real input starts responding immediately
    // rather than waiting for the intro to fully end. It felt wrong to display those instructions
    // while the keys visibly did nothing. updateIntroCinematic stops forcing the camera from that
    // beat onward (its camera-live check above), so this can run unopposed; !showIntro covers the
    // normal post-intro case the same way the old "else" branch did.
    if ((!showIntro || introCaptionIndex >= kIntroControlsIndex) && win)
    {
        bool boost = (win && glfwGetKey(win, keybindings[KB_MOVE_BOOST].key) == GLFW_PRESS) || gpHeld(KB_MOVE_BOOST);
        bool fine = (win && glfwGetKey(win, keybindings[KB_MOVE_FINE].key) == GLFW_PRESS) || gpHeld(KB_MOVE_FINE);
        float speed = boost ? 0.5f : fine ? 0.005f
                                          : 0.08f; // boost = fast, fine = slow, default = normal

        float fwd = (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS ? 1.0f : 0.0f) - (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS ? 1.0f : 0.0f);
        float right = (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS ? 1.0f : 0.0f) - (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS ? 1.0f : 0.0f);
        // Left stick augments WASD rather than replacing it, so both can be used together;
        // clamp keeps combined input from exceeding the analog range WASD alone already implied.
        fwd = glm::clamp(fwd + gpMoveFwd, -1.0f, 1.0f);
        right = glm::clamp(right + gpMoveRight, -1.0f, 1.0f);

        if (fwd != 0.0f || right != 0.0f)
        {
            // right tangent = cross(obsFacing, obsDir)  (right-hand rule: forward × up = right)
            glm::vec3 rightDir = glm::normalize(glm::cross(obsFacing, obsDir));
            glm::vec3 newPos = glm::normalize(
                obsDir + speed * dt * (fwd * obsFacing + right * rightDir));

            // Parallel-transport obsFacing: project out any radial component at newPos.
            obsFacing = glm::normalize(obsFacing - glm::dot(obsFacing, newPos) * newPos);
            obsDir = newPos;

            // Refresh display caches (atan2(0,0)==0 at poles — fine for display only)
            obsLatDeg = glm::degrees(asinf(glm::clamp(obsDir.z, -1.0f, 1.0f)));
            obsLonDeg = glm::degrees(atan2f(obsDir.y, obsDir.x));
        }

        // Q/E: raise/lower observer relative to terrain; rate scales with height offset
        // (faster when high up, 10m/s minimum near the surface). Digital sources (keyboard,
        // or a gamepad button if the user rebinds one) contribute full-speed (1.0); the
        // gamepad's default control is the analog trigger pressure instead — combined via
        // max() so pressure directly scales vertical speed without needing its own rate curve.
        float raiseAmt = std::max((glfwGetKey(win, keybindings[KB_RAISE_ELEV].key) == GLFW_PRESS || gpHeld(KB_RAISE_ELEV)) ? 1.0f : 0.0f, gpElevRaise);
        float lowerAmt = std::max((glfwGetKey(win, keybindings[KB_LOWER_ELEV].key) == GLFW_PRESS || gpHeld(KB_LOWER_ELEV)) ? 1.0f : 0.0f, gpElevLower);
        if (raiseAmt > 0.0f || lowerAmt > 0.0f)
        {
            float rate = std::max(10.0f, obsHeightOffset * 0.5f);
            if (boost)
                rate *= 10.0f;
            if (fine)
                rate *= 0.1f;
            obsHeightOffset += (raiseAmt - lowerAmt) * rate * dt;
            // Clamp so observer never sinks below the terrain surface (only reset via Z)
            obsHeightOffset = std::max(0.0f, obsHeightOffset);
        }

        // Zoom in/out (held): narrows/widens FOV at a fixed rate. Independent of boost/fine —
        // those are movement-speed modifiers, not a zoom concept.
        bool zoomIn = (glfwGetKey(win, keybindings[KB_ZOOM_IN].key) == GLFW_PRESS) || gpHeld(KB_ZOOM_IN);
        bool zoomOut = (glfwGetKey(win, keybindings[KB_ZOOM_OUT].key) == GLFW_PRESS) || gpHeld(KB_ZOOM_OUT);
        if (zoomIn || zoomOut)
        {
            constexpr float kZoomRateDegPerSec = 40.0f;
            camera.fovYDeg += (zoomOut ? 1.0f : -1.0f) * kZoomRateDegPerSec * dt;
            camera.fovYDeg = glm::clamp(camera.fovYDeg, 10.0f, 120.0f);
        }
    }

    float simDt = timePaused ? 0.0f : fabsf(dt * kTimeScales[timeScaleIdx]);
    if (!timePaused)
    {
        simSecInDay += (double)dt * kTimeScales[timeScaleIdx] * timeDir;
        // Re-base to [0, 86400) and carry whole-day overflow into simDayJ2000.
        // Using a loop (not fmod) so timeDir reversal is handled cleanly.
        while (simSecInDay >= 86400.0)
        {
            simSecInDay -= 86400.0;
            ++simDayJ2000;
        }
        while (simSecInDay < 0.0)
        {
            simSecInDay += 86400.0;
            --simDayJ2000;
        }
    }

    // Auto-rebake orbit buffer if the epoch has drifted more than kOrbitRebakeDays.
    // Keeps float deltaT < kOrbitRebakeDays*86400 s (float ULP ≈ 0.07 s at 7 days).
    if (std::abs(simDayJ2000 - orbitEpochDay) >= kOrbitRebakeDays)
        uploadSatOrbits(ctx);

    {
        CpuTimer _t(cpuAccumMs[CPU_UPDATE_POSITIONS]);
        updatePositions((double)simDayJ2000 * 86400.0 + simSecInDay, simDt);
    }

    // ── Sky-background sun-glare gate (stars / Milky Way, space only) ──────────
    // updateStars()'s atmFrac fade (see that function) lets the day/night sky-brightness gate
    // relax toward "always visible" once truly clear of the atmosphere — correct in principle
    // (no air left to scatter sunlight into a blue daytime sky) but previously relaxed all the
    // way to a flat 1.0 regardless of the sun's position, so stars/Milky Way stayed fully visible
    // in space even staring straight at the sun, or in full unshielded sunlight. Real glare still
    // applies: this computes a single per-frame, whole-screen target — not per-pixel, since it's
    // meant to blank the ENTIRE sky background, not just fade near the sun the way the existing
    // localized sunGlareSuppress halo in sat_sky.frag's Milky Way section does — and eases toward
    // it so a quick look-away doesn't snap the sky instantly back.
    {
        glm::vec3 sunCam = glm::mat3(camera.viewMatrix()) * glm::vec3(sunDirENU);
        float tanHalfFov = tanf(glm::radians(camera.fovYDeg) * 0.5f);
        float aspect = (float)ctx.swapExtent.width / (float)ctx.swapExtent.height;
        bool sunOnScreen = false;
        if (sunCam.z < -0.001f)
        {
            float ndcX = sunCam.x / (-sunCam.z) / (tanHalfFov * aspect);
            float ndcY = -sunCam.y / (-sunCam.z) / tanHalfFov;
            sunOnScreen = (fabsf(ndcX) <= 1.0f && fabsf(ndcY) <= 1.0f);
        }

        // Observer's own local sun elevation — same day/night test updateStars()'s nightFactor
        // already uses below, valid at any altitude this sim reaches (ENU is defined at the
        // observer's actual position, so this tracks a real eclipse/shadow crossing reasonably
        // well without a separate Earth-shadow ray test).
        bool sunlit = sunDirENU.w > 0.0f;
        float glareTarget = sunOnScreen ? 0.0f : (sunlit ? sunlitBgVisibility : 1.0f);

        // Asymmetric hysteresis: glare hits fast (sensor/eyes overwhelmed almost immediately),
        // recovery is slow (night-vision-style readaptation) — avoids an instant on/off pop
        // either direction while still feeling responsive when the sun swings into view.
        const float kSkyGlareOnRate = 3.0f;  // ~0.3s to mostly reach target when dimming
        const float kSkyGlareOffRate = 0.4f; // ~2.5s to mostly reach target when recovering
        float rate = (glareTarget < skyGlareEased) ? kSkyGlareOnRate : kSkyGlareOffRate;
        skyGlareEased = glm::mix(skyGlareEased, glareTarget, 1.0f - expf(-dt * rate));
    }

    {
        CpuTimer _t(cpuAccumMs[CPU_LIGHT_DOME]);
        updateLightPollutionDome();

        // Milky Way pollution hysteresis — same asymmetric-rate pattern as skyGlareEased above
        // (target computed instantly, eased toward it at a rate that differs by direction), but
        // driven by mwPollutionRaw (updateLightPollutionDome()'s pre-gain local pollution level)
        // against its own mwPollutionThresholdLo/Hi band, not the shared domeVal/lightPollutionGain
        // stars and satellites use. Hand-rolled smoothstep — no glm::smoothstep used elsewhere in
        // this file (see the beamProximityGlow comment above for the same convention).
        float mwX = glm::clamp((mwPollutionRaw - mwPollutionThresholdLo)
                                    / std::max(mwPollutionThresholdHi - mwPollutionThresholdLo, 1e-5f),
                                0.0f, 1.0f);
        float mwTarget = mwX * mwX * (3.0f - 2.0f * mwX); // 0 = fully visible, 1 = fully suppressed
        float mwRate = (mwTarget > mwSuppressEased) ? (1.0f / std::max(mwFadeOutTimeS, 0.01f))
                                                      : (1.0f / std::max(mwFadeInTimeS, 0.01f));
        mwSuppressEased = glm::mix(mwSuppressEased, mwTarget, 1.0f - expf(-dt * mwRate));
    }
    {
        CpuTimer _t(cpuAccumMs[CPU_UPDATE_STARS]);
        updateStars();
    }
    {
        CpuTimer _t(cpuAccumMs[CPU_UPDATE_PLANETS]);
        updatePlanets();
    }
    // Unlike formatSelectedSatInfo() (called only when the selection changes, since a satellite's
    // orbital elements are static), a planet's distance/phase/magnitude changes every frame — so
    // this re-derives planetInfoLine[] every frame the selection is active, from the planetStates[]
    // updatePositions() just refreshed this frame.
    if (selectedPlanetIndex >= 0)
        formatSelectedPlanetInfo();

    // Read previous frame's GPU glow results for the magnitude UI.
    // glowBuf is HOST_COHERENT; by the time recordCompute is called the previous
    // frame's queue work is complete, so the atomicMax writes from sat_flare.comp are visible.
    {
        const GpuGlowBuf *gb = static_cast<const GpuGlowBuf *>(glowMapped);
        float maxFlare = 0.0f;
        for (int i = 0; i < kGlowBins; ++i)
        {
            if (gb->bins[i] != 0u)
            {
                float f;
                memcpy(&f, &gb->bins[i], sizeof(float));
                maxFlare = std::max(maxFlare, f);
            }
        }
        peakMagnitude = (maxFlare > 0.0f)
                            ? kMagRef - 2.5f * std::log10(maxFlare / kMagRefFlare)
                            : 99.0f;
    }

    // Read previous frame's reflectBeamsBuf — diagnostic for C12 (is anything actually being
    // written, and how far is the nearest one) — AND (C12 follow-up #41) the source signal for
    // the beam-proximity sky-glow wash below — AND (2026-08-09) the small capped per-BEAM light
    // list that feeds cloud_march.comp's real beam->cloud illumination term (no per-target
    // aggregation any more — see GpuBeamCloudLights' own comment). Same one-frame-stale,
    // HOST_COHERENT idiom as peakMagnitude above.
    {
        // CPU timing (2026-08-10): this whole block is the prime suspect for the ~2.8 ms non-GPU
        // remainder the Release Anchorage capture exposed — it scans up to kMaxActiveBeams (2048)
        // entries, std::sorts them, and for each one runs LINEAR scans over the 256-slot
        // ground-beam top-K and the 256-slot cluster table, so its worst case is O(n*k) with
        // n*k ~= 1e6 per frame. Measured rather than assumed, same discipline as the GPU side.
        CpuTimer _t(cpuAccumMs[CPU_BEAM_READBACK]);
        const GpuReflectBeams *rbMapped = static_cast<const GpuReflectBeams *>(reflectBeamsMapped);
        int count = std::min((int)rbMapped->beamCount, kMaxActiveBeams);

        // ── Staging copy (2026-08-10 perf) ────────────────────────────────────────────────────
        // reflectBeamsBuf is HOST_VISIBLE|HOST_COHERENT device memory, which on a discrete GPU is
        // typically uncached / write-combined: sequential CPU reads are fine, RANDOM reads are
        // brutally slow (no cache line reuse, every access a fresh uncached fetch). This block did
        // exactly the pathological thing — a std::sort whose comparator dereferences two mapped
        // entries per comparison, followed by a main loop that visits entries in that SORTED order
        // (so, randomly with respect to memory layout) and reads eight-plus fields from each.
        //
        // One sequential memcpy of just the active region into ordinary RAM turns all of that into
        // cached reads. This is the same "read mapped memory once, linearly" discipline the glow
        // readback above already follows by accident of being a single tight scan.
        //
        // Measured at 1.89 ms (9.7% of the frame, 65% of the entire non-GPU remainder) in the
        // 2026-08-10 Release Anchorage sweep, which is what prompted this.
        static GpuReflectBeam beamsLocal[kMaxActiveBeams]; // 128 KB — static, not stack
        if (count > 0)
            std::memcpy(beamsLocal, rbMapped->entries, (size_t)count * sizeof(GpuReflectBeam));
        const GpuReflectBeam *beamsIn = beamsLocal;
        float nearest = -1.0f;
        float nearestBlockOpacity = 0.0f; // C11/C12 follow-up #47: blockOpacity of whichever
                                          // entry produced `nearest`, so beamProximityGlow below
                                          // can be dampened when cloud is actually blocking the
                                          // nearest beam rather than brightening the sky through it.

        // 2026-08-09 debug instrumentation: raw distribution of beam_self_march.comp's own output
        // across every active beam THIS frame, with no aggregation/arbitration in the way — lets
        // the Beams settings tab answer "is the march producing any signal at all" numerically,
        // instead of inferring it from the rendered ray/ground-spot/cloud-lighting, which also
        // depend on several consumer-side terms (fade, cloudGate, CPU per-target argmax) that can
        // mask or distort a working march. See BEAM_CLOUD_PLAN.md.
        dbgBeamSampleCount = count;
        dbgBeamOpacityMin = count > 0 ? 1.0f : 0.0f;
        dbgBeamOpacityMax = 0.0f;
        float opacitySum = 0.0f;
        int occludedCount = 0;
        // Perf follow-up: compacted raw-entry list for sat_sky.frag's ground-spot term, filtered
        // to the same beamMaxRangeM cutoff computed per-entry below — see GpuGroundBeams comment.
        GpuGroundBeams groundBeams{};
        // Reflect-Orbital beam->cloud light sources (2026-08-09 design — see GpuBeamCloudLights'
        // own comment for the full history: this replaced first a per-target aggregation, then a
        // screen-space glow that produced visible ring artifacts, then a flat per-beam top-K that
        // needed 512 slots for full coverage and tanked frame rate). A bounded list where a
        // CONVERGED beam (locked onto its target) is folded into a shared per-target cluster, and
        // a TRANSITING beam (still slewing) keeps its own individual slot — see the clustering
        // decision inline below. Ground intersections computed via the same rotation-invariant
        // local-frame raySphere trick the GPU side already uses (obsPos at local "north pole",
        // ENU offsets added directly, R_EARTH sphere at true origin) — no ECEF conversion needed.
        // Uses whatever obsTerrainH/obsHeightOffset currently hold (one-frame-stale is fine, same
        // tolerance every CPU aggregation in this loop already has); beam_self_march.comp's own
        // push-constant fill later this frame computes the identical obsEffH the same way.
        GpuBeamCloudLights cloudLights{};
        float obsEffHForLights = std::max(obsTerrainH, obsHeightOffset);
        glm::vec3 obsPosLocalForLights(0.0f, 0.0f, kEarthRadius + obsEffHForLights + 2.0f);

        TopK groundTopK;

        // 2026-08-09 (in-app finding: beams/ground spots visibly drag behind the observer while
        // moving): satENU/targetENU/reflectDirENU in reflectBeamsBuf are true East/North/Up
        // physical measurements at whatever obsDir was in effect when sat_orbit.comp/
        // beam_self_march.comp wrote them (lastBeamObsDir, cached one frame ago — see its own
        // comment in SatelliteSim.h). That basis rotates with the observer's lat/lon, so reading
        // last frame's numbers and combining them with THIS frame's obsPos (as every downstream
        // consumer does) without re-projecting is a real rotation error proportional to how far
        // the observer moved. Rebase once here, before anything below uses these vectors — a
        // small CPU port of terrain.glsl's enuBasis() (normalize + 2 cross products), applied to
        // the OLD and NEW bases once per frame, not per beam.
        auto enuBasisCPU = [](const glm::vec3 &dir, glm::vec3 &x, glm::vec3 &y, glm::vec3 &z) {
            z = glm::normalize(dir);
            x = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), z));
            y = glm::cross(z, x);
        };
        glm::vec3 oldEnuX, oldEnuY, oldEnuZ, newEnuX, newEnuY, newEnuZ;
        enuBasisCPU(lastBeamObsDir, oldEnuX, oldEnuY, oldEnuZ);
        enuBasisCPU(obsDir, newEnuX, newEnuY, newEnuZ);
        auto rebase = [&](const glm::vec3 &v) {
            glm::vec3 worldish = v.x * oldEnuX + v.y * oldEnuY + v.z * oldEnuZ;
            return glm::vec3(glm::dot(worldish, newEnuX), glm::dot(worldish, newEnuY), glm::dot(worldish, newEnuZ));
        };

        // ── Cloud-light identity (2026-08-12 rewrite) ─────────────────────────────────────────
        // See TrackedBeamLight in SatelliteSim.h for the full rationale. Summary: a light's
        // identity is now DECLARED by an exact integer key — (targetIdx, direction bucket) for a
        // converged cluster, originating satellite index for a transiting beam — instead of being
        // derived from an epsilon match against whichever beam the scan reached first. That makes
        // the partition a pure function of each beam's own geometry, so it is independent of scan
        // order, of what else is in the cluster, and of where the observer is.
        //
        // A direct consequence: the std::sort by debugPad that used to sit here is GONE. It existed
        // only to make the old order-dependent merge deterministic frame to frame (2026-08-09 test
        // #9). Nothing below depends on scan order any more — groundTopK ranks on intensity, the
        // opacity diagnostics and `nearest` are commutative, and the tracked-pool accumulation is a
        // sum. That removes a per-frame sort of up to kMaxActiveBeams=2048 entries whose comparator
        // touched two records each, from a block that measured 1.89 ms.

        // ENU <-> Earth-fixed ECEF, in double precision. The tracked pools store geometry in ECEF
        // specifically so an entry that goes unmatched for its whole fade-out cannot drift as the
        // observer moves — a ground site is stationary in ECEF, full stop. This is the lesson the
        // 2026-08-11 attempt spent three rounds relearning; do not store observer-relative ENU in
        // anything that outlives a frame.
        //
        // The "local" frame here is the same one obsPosLocalForLights/raySphere already use above:
        // ENU axes, origin at Earth's centre, observer at (0, 0, R+h). newEnuX/Y/Z (built from THIS
        // frame's obsDir by enuBasisCPU) are those axes expressed in ECEF, so the conversion is a
        // pure rotation plus that one fixed offset, and round-trips exactly within a frame.
        // Doubles because ECEF magnitudes are ~6.4e6 m and these get differenced at emit time —
        // float would quantize a light's position to ~0.4 m and lose more to cancellation.
        const glm::dvec3 dEnuX(newEnuX), dEnuY(newEnuY), dEnuZ(newEnuZ);
        const glm::dvec3 dObsLocal(0.0, 0.0, (double)obsPosLocalForLights.z);
        auto enuPosToEcef = [&](const glm::vec3 &e) -> glm::dvec3 {
            glm::dvec3 l = dObsLocal + glm::dvec3(e);
            return dEnuX * l.x + dEnuY * l.y + dEnuZ * l.z;
        };
        auto ecefPosToEnu = [&](const glm::dvec3 &p) -> glm::vec3 {
            glm::dvec3 l(glm::dot(p, dEnuX), glm::dot(p, dEnuY), glm::dot(p, dEnuZ));
            return glm::vec3(l - dObsLocal);
        };
        auto enuDirToEcef = [&](const glm::vec3 &d) -> glm::dvec3 {
            return dEnuX * (double)d.x + dEnuY * (double)d.y + dEnuZ * (double)d.z;
        };
        auto ecefDirToEnu = [&](const glm::dvec3 &d) -> glm::vec3 {
            return glm::vec3(glm::dot(d, dEnuX), glm::dot(d, dEnuY), glm::dot(d, dEnuZ));
        };

        // Direction bucketing. beamClusterDirThresholdDeg is now the angular SIZE of a bucket (see
        // its comment in SatelliteSim.h for the semantics change). Elevation rings of that size,
        // with each ring's azimuth sector count scaled by cos(elevation) so cells stay roughly
        // equal angular size instead of collapsing to slivers near the site's zenith. Floored at
        // 2 deg: finer than that just thrashes the 256-slot pool without producing visually
        // distinct lights, since the pool cap — not the bucket count — is the real limit.
        const float kBucketMinDeg = 2.0f;
        const float bucketRad = glm::radians(glm::clamp(beamClusterDirThresholdDeg, kBucketMinDeg, 90.0f));
        // Flat bucket index is iEl * kMaxAzSectors + iAz, and the cluster key packs it into 16 bits
        // alongside targetIdx — so these two bounds must satisfy
        // (kMaxElRings-1)*kMaxAzSectors + (kMaxAzSectors-1) <= 65535. 127*512+511 = 65535 exactly.
        // The 2 deg floor above already caps the real ring count at 45, well inside this.
        const int kMaxAzSectors = 512;
        const int kMaxElRings = 127;
        auto dirBucketFor = [&](int ti, const glm::dvec3 &dirEcef) -> uint32_t {
            // Project into the TARGET SITE's own local frame — observer-independent by
            // construction, which is what stops camera motion from repartitioning a cluster.
            glm::dvec3 sx(reflectorSiteEnuX[ti]), sy(reflectorSiteEnuY[ti]), sz(reflectorSiteEnuZ[ti]);
            float dx = (float)glm::dot(dirEcef, sx);
            float dy = (float)glm::dot(dirEcef, sy);
            float dz = (float)glm::dot(dirEcef, sz);
            // dirToSource points ground -> satellite, so elevation is >= 0 in practice (the
            // satellite has to clear the site's horizon to be selected at all). Clamped anyway.
            float el = std::asin(glm::clamp(dz, -1.0f, 1.0f));
            el = glm::clamp(el, 0.0f, glm::half_pi<float>() - 1e-4f);
            int iEl = std::clamp((int)(el / bucketRad), 0, kMaxElRings - 1);
            float elCentre = std::min((iEl + 0.5f) * bucketRad, glm::half_pi<float>());
            int nAz = std::clamp((int)std::lround(glm::two_pi<float>() * std::cos(elCentre) / bucketRad),
                                  1, kMaxAzSectors);
            float az = std::atan2(dy, dx);
            if (az < 0.0f) az += glm::two_pi<float>();
            int iAz = std::clamp((int)(az / (glm::two_pi<float>() / (float)nAz)), 0, nAz - 1);
            return (uint32_t)(iEl * kMaxAzSectors + iAz);
        };

        // lowbias32 (Chris Wellons) — cheap, well-distributed integer mix for the open-addressed
        // index below. Keys are dense-ish (targetIdx<<16 | bucket), so a raw modulo would cluster.
        auto hashKey32 = [](uint32_t k) -> uint32_t {
            k ^= k >> 16; k *= 0x7feb352du;
            k ^= k >> 15; k *= 0x846ca68bu;
            k ^= k >> 16;
            return k;
        };

        // Rebuild each pool's key->slot index from its live slots, and clear this frame's
        // accumulators. Rebuilding (rather than maintaining it incrementally) is O(live) <= 256 and
        // sidesteps tombstones entirely, which is the only genuinely error-prone part of open
        // addressing with deletion.
        auto resetPool = [&](TrackedBeamLight *pool, uint32_t *hash, int cap) {
            std::memset(hash, 0, sizeof(uint32_t) * kTrackedLightHashSize);
            for (int i = 0; i < cap; ++i)
            {
                TrackedBeamLight &L = pool[i];
                L.tgtIntensity = 0.0f;
                L.tgtFootprintRadM = 0.0f;
                L.tgtPosSum = glm::dvec3(0.0);
                L.tgtDirSum = glm::dvec3(0.0);
                L.tgtAltSum = 0.0f;
                L.tgtOpacitySum = 0.0f;
                if (!L.key) continue;
                uint32_t h = hashKey32(L.key) & (uint32_t)(kTrackedLightHashSize - 1);
                while (hash[h] != 0u) h = (h + 1u) & (uint32_t)(kTrackedLightHashSize - 1);
                hash[h] = (uint32_t)i + 1u;
            }
        };
        resetPool(trackedClusters, trackedClusterHash, kMaxClusterCloudLights);
        resetPool(trackedIndividuals, trackedIndividualHash, kMaxIndividualCloudLights);

        // Find this key's slot, allocating (or, only under real capacity pressure, evicting the
        // weakest) if it isn't live yet. Returns -1 if the beam can't be placed this frame.
        //
        // Entries are only ever INSERTED into the hash, never deleted mid-frame — an evicted slot's
        // stale entry is left in place and rejected by the `pool[slot].key == key` verification
        // below, then cleared by next frame's rebuild. Deleting from an open-addressed table
        // without tombstones would break the probe chains of unrelated keys.
        auto findOrAlloc = [&](TrackedBeamLight *pool, uint32_t *hash, int cap,
                                uint32_t key, float candidate) -> int {
            const uint32_t mask = (uint32_t)(kTrackedLightHashSize - 1);
            uint32_t h = hashKey32(key) & mask;
            uint32_t firstFreeH = 0xFFFFFFFFu;
            for (int probe = 0; probe < 128; ++probe)
            {
                uint32_t e = hash[h];
                if (e == 0u) { firstFreeH = h; break; }
                int slot = (int)e - 1;
                if (pool[slot].key == key) return slot;  // live match — the common case
                h = (h + 1u) & mask;                     // occupied by another key (or a stale
                                                          // post-eviction entry) — keep probing
            }
            if (firstFreeH == 0xFFFFFFFFu) return -1; // pathological probe chain; skip this beam

            // Allocate: first free pool slot, else evict the weakest. Ranked on
            // max(easedIntensity, tgtIntensity) so a slot already accumulating THIS frame is
            // protected — evicting one would silently discard contributions already folded in.
            int slot = -1;
            for (int i = 0; i < cap; ++i)
                if (!pool[i].key) { slot = i; break; }
            if (slot < 0)
            {
                int weakest = 0;
                float weakestVal = std::max(pool[0].easedIntensity, pool[0].tgtIntensity);
                for (int i = 1; i < cap; ++i)
                {
                    float v = std::max(pool[i].easedIntensity, pool[i].tgtIntensity);
                    if (v < weakestVal) { weakestVal = v; weakest = i; }
                }
                if (candidate <= weakestVal) return -1; // not worth displacing anything
                slot = weakest;
            }
            pool[slot] = TrackedBeamLight{};
            pool[slot].key = key;
            hash[firstFreeH] = (uint32_t)slot + 1u;
            return slot;
        };

        for (int s = 0; s < count; ++s)
        {
            // C12 follow-up #41: point-to-segment distance to the beam's actual 3D LINE (target
            // to satellite), not just its ground endpoint (`length(targetENU)`, the old formula)
            // — climbing up alongside a long beam away from the ground previously read as
            // "getting farther from the beam" even while staying right next to its line. Same
            // formula as cloud_march.comp's own obsToBeamDist, simplified: these vectors are
            // already observer-relative (origin = observer), so no obsPos subtraction is needed.
            glm::vec3 tE = rebase(beamsIn[s].targetENU);
            glm::vec3 sE = rebase(beamsIn[s].satENU);
            float slantRangeM = glm::length(sE - tE);
            glm::vec3 dirUp = (slantRangeM > 1.0f) ? (sE - tE) / slantRangeM : glm::vec3(0, 0, 1);
            float t = glm::clamp(-glm::dot(tE, dirUp), 0.0f, slantRangeM);
            float d = glm::length(tE + dirUp * t);

            float bOpacity = beamsIn[s].blockOpacity;
            dbgBeamOpacityMin = std::min(dbgBeamOpacityMin, bOpacity);
            dbgBeamOpacityMax = std::max(dbgBeamOpacityMax, bOpacity);
            opacitySum += bOpacity;
            if (bOpacity > 0.1f)
                ++occludedCount;

            if (nearest < 0.0f || d < nearest)
            {
                nearest = d;
                nearestBlockOpacity = beamsIn[s].blockOpacity;
            }

            float intensity = beamsIn[s].intensity;
            float targetDistM = glm::length(tE);
            glm::vec3 rDir = rebase(beamsIn[s].reflectDirENU);

            // Ground-spot compaction: same range cutoff sat_sky.frag's loop applies, done ONCE
            // here per beam instead of unconditionally per ground-hit pixel. Widened by
            // kGroundBeamFadeM (2026-08-07 user report: beams "pop in" at full brightness the
            // instant they cross beamMaxRangeM) so the fade band itself is still included in the
            // compacted list — sat_sky.frag applies the actual smooth fade per-pixel using this
            // same width, since the intensity used there needs to ramp, not just the membership
            // test here. Kept in sync with kSkyBeamFadeM in cloud_march.comp (same visual beam,
            // same fade feel) and sat_sky.frag's own copy of this constant.
            //
            // 2026-08-09 (in-app finding: "enormous flicker" on both cloud lighting and ground
            // spots): both this list and cloudLights below used to just take the first N beams
            // encountered IN SCAN ORDER (`s` = GPU dispatch slot index). Slot index is explicitly
            // NOT stable frame to frame — sat_orbit.comp's own header comment documents it as a
            // GPU atomicAdd race (see also cloud_march.comp's sky-glow downsampling, which hit and
            // fixed this exact failure mode once already, differently, for a different consumer).
            // Whenever the number of ELIGIBLE beams exceeded either cap, a near-fully-different
            // RANDOM subset got selected every single frame — the real cause of the reported
            // flicker, not "the cap is too small" per se (raising either cap only reduces how
            // often the overflow condition triggers, it doesn't fix the instability itself).
            // Fixed by ranking on intensity (a smooth per-satellite physical quantity, independent
            // of iteration order) via a bounded top-K: once full, only replace the CURRENT weakest
            // entry, and only if this candidate is actually stronger. The selected set now changes
            // only as real relevance changes, not randomly every frame.
            const float kGroundBeamFadeM = 200000.0f;
            if (intensity > 0.0f && targetDistM <= beamMaxRangeM + kGroundBeamFadeM)
            {
                uint32_t insertIdx = groundTopK.claim(
                    groundBeams.count, (uint32_t)kMaxGroundBeams, intensity,
                    [&](uint32_t i) { return groundBeams.entries[i].intensity; });
                if (insertIdx != ~0u)
                {
                    // 2026-08-10: solve the whole view-independent half of sat_sky.frag's
                    // ground-spot loop here, once per beam, instead of once per ground-hit pixel.
                    // See GpuGroundBeam's comment for the measurement that motivated this. Every
                    // line below is a direct port of what that shader loop used to do per pixel;
                    // the rebased (current-frame-basis) tE/sE/rDir are used throughout, since the
                    // raw entry still carries last frame's basis for those three vectors.
                    GpuGroundBeam gb{};

                    // Smooth range fade, keyed to the CHOSEN target's fixed site position rather
                    // than the transient ray-ground point below — "is the observer close enough to
                    // this SITE", which shouldn't flicker as a mid-slew ray briefly lands
                    // elsewhere. Same widened cutoff/shape the membership test above uses.
                    float rangeX = glm::clamp(
                        (targetDistM - (beamMaxRangeM - kGroundBeamFadeM)) / kGroundBeamFadeM, 0.0f, 1.0f);
                    float rangeFade = 1.0f - (rangeX * rangeX * (3.0f - 2.0f * rangeX));

                    // Elevation fade — sin(5 deg) cutoff, matching the sky beam's own. rDir points
                    // satellite->ground, so -rDir.z is the sine of the beam's elevation.
                    float sinElev = -rDir.z;
                    float ex = glm::clamp(sinElev / 0.08716f, 0.0f, 1.0f);
                    float elevFade = ex * ex * (3.0f - 2.0f * ex);

                    float shadowAtten = 1.0f - beamsIn[s].blockOpacity;

                    // Real ray/ground intersection (NOT the idealized target site) via the same
                    // rotation-invariant local-frame raySphere the cloud-light block below uses.
                    glm::vec3 satPosLocalGB = obsPosLocalForLights + sE;
                    float bGB = glm::dot(satPosLocalGB, rDir);
                    float roLenGB = glm::length(satPosLocalGB);
                    float cGB = (roLenGB - kEarthRadius) * (roLenGB + kEarthRadius);
                    float discGB = bGB * bGB - cGB;
                    float tHitGB = (discGB >= 0.0f) ? (-bGB - std::sqrt(discGB)) : -1.0f;

                    if (tHitGB > 0.0f && elevFade > 0.0f && rangeFade > 0.0f)
                    {
                        glm::vec3 hitENU = sE + rDir * tHitGB;
                        float footprintR = std::max(beamsIn[s].footprintRadM, 1.0f);
                        float coreR = std::max(beamsIn[s].mirrorRadiusM, 1.0f);
                        gb.groundHitX = hitENU.x;
                        gb.groundHitY = hitENU.y;
                        gb.invFootprintSq = 1.0f / (footprintR * footprintR);
                        gb.invCoreSq = 1.0f / (coreR * coreR);
                        gb.cutoffSq = (footprintR * 4.0f) * (footprintR * 4.0f);
                        gb.weight = intensity * rangeFade * elevFade * shadowAtten;
                    }
                    // else: weight/cutoffSq stay 0, so the shader's own reject drops it. This is
                    // the same outcome as the shader's old `continue` on those conditions —
                    // deliberately still occupying its slot rather than being skipped, so top-K
                    // membership stays a function of intensity alone and doesn't churn.
                    gb.intensity = intensity; // ranking key only — see GpuGroundBeam's comment
                    groundBeams.entries[insertIdx] = gb;
                }
            }

            if (intensity > 0.0f)
            {
                glm::vec3 satPosLocal = obsPosLocalForLights + sE;
                float b = glm::dot(satPosLocal, rDir);
                float roLen = glm::length(satPosLocal);
                float c = (roLen - kEarthRadius) * (roLen + kEarthRadius);
                float disc = b * b - c;
                if (disc >= 0.0f)
                {
                    float tHit = -b - std::sqrt(disc); // near root — satellite starts outside the sphere
                    if (tHit > 0.0f)
                    {
                        glm::vec3 groundPosENU = sE + rDir * tHit;
                        glm::vec3 dirToSourceCand = -rDir; // ground -> satellite, the beam's real direction

                        // 2026-08-09 (user direction: "beam cloud effects really only need to be
                        // considered for beams near the observer, and beams that aren't exactly
                        // over target — for those [converged, redundant with each other] we can
                        // simplify and combine"). A beam locked onto its target (small aimErrorRad
                        // — same convergence signal cloud_march.comp's own aimFade already uses)
                        // shares a per-target cluster with every other beam converged on that site
                        // from a similar direction; a beam still slewing gets its own light, since
                        // its geometry is genuinely unique right now.
                        //
                        // 2026-08-12: this is no longer a discontinuity. A beam crossing the
                        // threshold moves between two entries that are BOTH temporally eased, so
                        // its contribution crossfades out of one and into the other rather than
                        // teleporting. That is why no hysteresis is needed here despite the hard
                        // comparison — the old design needed it precisely because both sides
                        // snapped.
                        const float kConvergedAimErrorRad = glm::radians(10.0f); // matches
                            // cloud_march.comp's kDebugRayAimMaxRad, same underlying question
                        bool converged = beamsIn[s].aimErrorRad <= kConvergedAimErrorRad;

                        // Geometry into Earth-fixed ECEF once, here — everything downstream (the
                        // direction bucket, the accumulators, the stored state) works in that
                        // frame. See the conversion helpers above for why.
                        glm::dvec3 gEcef = enuPosToEcef(groundPosENU);
                        glm::dvec3 dEcef = enuDirToEcef(dirToSourceCand);

                        // Pick the pool and the KEY. This replaces the old epsilon-match scan over
                        // every existing cluster entirely: identity is declared, not searched for.
                        int ti = (int)beamsIn[s].targetIdx;
                        bool asCluster = converged && ti >= 0 && ti < reflectorTargetCount;
                        TrackedBeamLight *pool;
                        int slot;
                        if (asCluster)
                        {
                            uint32_t key = 0x80000000u | ((uint32_t)ti << 16) |
                                            (dirBucketFor(ti, dEcef) & 0xFFFFu);
                            pool = trackedClusters;
                            slot = findOrAlloc(trackedClusters, trackedClusterHash,
                                                kMaxClusterCloudLights, key, intensity);
                        }
                        else
                        {
                            // Transiting (or, defensively, a beam whose targetIdx somehow isn't a
                            // loaded target). Keyed by the ORIGINATING SATELLITE's dispatch index,
                            // which sat_orbit.comp guarantees is stable frame to frame — unlike the
                            // atomic-append slot index `s`. The two key namespaces live in separate
                            // pools, so the high bits are documentation rather than necessity.
                            uint32_t satIdx = (uint32_t)std::max(0.0f, beamsIn[s].debugPad);
                            uint32_t key = 0x40000000u | (satIdx & 0x3FFFFFFFu);
                            pool = trackedIndividuals;
                            slot = findOrAlloc(trackedIndividuals, trackedIndividualHash,
                                                kMaxIndividualCloudLights, key, intensity);
                        }

                        if (slot >= 0)
                        {
                            // Accumulate into THIS frame's target values. Order-independent by
                            // construction — these are sums and a max, so the same set of beams
                            // produces the same result no matter what order they arrive in.
                            TrackedBeamLight &L = pool[slot];
                            L.tgtIntensity += intensity;
                            L.tgtFootprintRadM = std::max(L.tgtFootprintRadM, beamsIn[s].footprintRadM);
                            L.tgtPosSum += gEcef * (double)intensity;
                            L.tgtDirSum += dEcef * (double)intensity;
                            L.tgtAltSum += beamsIn[s].blockAltM * intensity;
                            L.tgtOpacitySum += beamsIn[s].blockOpacity * intensity;
                        }
                    }
                }
            }
        }

        // ── Finalize, ease, emit ───────────────────────────────────────────────────────────────
        // Same 1 - exp(-dt/timeConstant) idiom as mwSuppressEased/skyGlareEased. Asymmetric on
        // intensity (fast in, slow out) so a light registers promptly but lingers on the way out;
        // geometry runs on its own much shorter constant so a light can't visibly lag real beam
        // motion. Clamped because dt can be 0 (paused presentation) or large after a hitch.
        auto blendFactor = [&](float timeConstS) {
            return glm::clamp(1.0f - std::exp(-dt / std::max(timeConstS, 1e-3f)), 0.0f, 1.0f);
        };
        const float kFadeIn = blendFactor(beamClusterFadeInS);
        const float kFadeOut = blendFactor(beamClusterFadeOutS);
        const float kGeom = blendFactor(kTrackedLightGeomEaseS);

        auto easeAndEmit = [&](TrackedBeamLight *pool, int cap) {
            int live = 0;
            for (int i = 0; i < cap; ++i)
            {
                TrackedBeamLight &L = pool[i];
                if (!L.key) continue;
                float w = L.tgtIntensity;
                if (w > 0.0f)
                {
                    glm::dvec3 p = L.tgtPosSum / (double)w;
                    glm::dvec3 d = L.tgtDirSum;
                    double dl = glm::length(d);
                    // Fallback is the local vertical at this light's own ground position — a real,
                    // physically sensible "light from straight overhead" rather than a zero vector.
                    // Cannot trigger in practice (a bucket spans well under 90 deg, so its member
                    // directions can't sum to zero) but a fresh slot's stored dirECEF IS zero, so
                    // falling back to it would be a live NaN source rather than a theoretical one.
                    d = (dl > 1e-9) ? d / dl : glm::normalize(p);
                    float alt = L.tgtAltSum / w;
                    float opa = L.tgtOpacitySum / w;
                    if (L.easedIntensity <= 0.0f)
                    {
                        // First frame alive: SNAP geometry. There is no meaningful previous value
                        // to ease from, and easing from a zeroed slot would drag the light in from
                        // the centre of the Earth.
                        L.posECEF = p;
                        L.dirECEF = d;
                        L.easedFootprintRadM = L.tgtFootprintRadM;
                        L.easedBlockAltM = alt;
                        L.easedBlockOpacity = opa;
                    }
                    else
                    {
                        L.posECEF += (p - L.posECEF) * (double)kGeom;
                        L.dirECEF += (d - L.dirECEF) * (double)kGeom;
                        double nl = glm::length(L.dirECEF); // nlerp — renormalize after the blend
                        L.dirECEF = (nl > 1e-9) ? L.dirECEF / nl : d;
                        L.easedFootprintRadM += (L.tgtFootprintRadM - L.easedFootprintRadM) * kGeom;
                        L.easedBlockAltM += (alt - L.easedBlockAltM) * kGeom;
                        L.easedBlockOpacity += (opa - L.easedBlockOpacity) * kGeom;
                    }
                }
                // else: nothing fed this slot — it fades out while HOLDING its geometry, which is
                // only safe because that geometry is Earth-fixed and cannot drift with the observer.
                L.easedIntensity += (w - L.easedIntensity) * ((w > L.easedIntensity) ? kFadeIn : kFadeOut);
                if (w <= 0.0f && L.easedIntensity < kTrackedLightEpsilon)
                {
                    L.key = 0; // fully faded and unfed — retire the slot
                    L.easedIntensity = 0.0f;
                    continue;
                }
                ++live;
                if (cloudLights.count < (uint32_t)kMaxCloudBeamLights)
                {
                    GpuBeamCloudLight &e = cloudLights.entries[cloudLights.count++];
                    e.posENU = ecefPosToEnu(L.posECEF);
                    e.intensity = L.easedIntensity;
                    // Guarded normalize rather than glm::normalize: a bucket spans well under 90
                    // deg so its member directions physically cannot cancel to zero, but this is
                    // the one place a degenerate value would escape to the GPU, and a single NaN
                    // dirToSource poisons every cloud sample that light reaches.
                    glm::vec3 dEnu = ecefDirToEnu(L.dirECEF);
                    float dEnuLen = glm::length(dEnu);
                    e.dirToSource = (dEnuLen > 1e-6f) ? dEnu / dEnuLen : glm::vec3(0.0f, 0.0f, 1.0f);
                    e.footprintRadM = L.easedFootprintRadM;
                    e.blockAltM = L.easedBlockAltM;
                    e.blockOpacity = L.easedBlockOpacity;
                }
            }
            return live;
        };
        // Clusters first, then individuals — the same ordering the previous build produced. The two
        // pools sum to exactly kMaxCloudBeamLights, so the emit above can never truncate.
        lastClusterLightCount = easeAndEmit(trackedClusters, kMaxClusterCloudLights);
        lastIndividualLightCount = easeAndEmit(trackedIndividuals, kMaxIndividualCloudLights);

        std::memcpy(groundBeamsMapped, &groundBeams, sizeof(GpuGroundBeams));
        std::memcpy(beamCloudLightMapped, &cloudLights, sizeof(GpuBeamCloudLights));

        dbgBeamOpacityAvg = count > 0 ? opacitySum / (float)count : 0.0f;
        dbgBeamOccludedCount = occludedCount;

        lastActiveBeamCount = count;
        lastGroundBeamCount = (int)groundBeams.count;
        lastNearestBeamDistM = nearest;

        // C12 follow-up #41: ready-to-use [0,1] sky-glow wash value — smoothstepped from the
        // corrected nearest-beam-line distance above, using the beamNearFieldFadeM radius slider.
        // (Prior to C12 follow-up #44 this same slider also drove the now-deleted analytic tube's
        // own near-field crossfade in cloud_march.comp; it now only feeds this wash.) Hand-rolled
        // smoothstep (no glm::smoothstep used elsewhere in this file).
        float x = glm::clamp(lastNearestBeamDistM / std::max(beamNearFieldFadeM, 1.0f), 0.0f, 1.0f);
        float sstep = x * x * (3.0f - 2.0f * x);
        // C11/C12 follow-up #47: dampened by the nearest beam's own blockOpacity — this wash used
        // to ignore cloud entirely, so it would brighten the sky near a beam even standing under a
        // solid cloud deck that's actually blocking it from view. Multiplicative, not a hard cutoff:
        // a partially-opaque column dims the wash proportionally rather than snapping it off.
        beamProximityGlow = (lastNearestBeamDistM >= 0.0f) ? (1.0f - sstep) * (1.0f - nearestBlockOpacity) : 0.0f;
    }

    // Read previous frame's beamGlowDomeBuf (C12 follow-up #31) — one-frame-stale, same idiom as
    // glowBuf/reflectBeamsBuf above. sat_orbit.comp stores raw atomicMax'd uint bit-patterns
    // (floatBitsToUint on the GPU side), so reinterpret via memcpy rather than a direct float
    // cast, matching glowBuf's own pattern.
    {
        const uint32_t *bgd = static_cast<const uint32_t *>(beamGlowDomeMapped);
        for (int i = 0; i < kNumBeamGlowSectors; ++i)
        {
            memcpy(&beamGlowDomeAz[i], &bgd[i], sizeof(float));
        }
    }

    // Read previous frame's tracked-selection position, same one-frame-stale idiom as
    // peakMagnitude above. On the frame a selection is first made (or changed), this still
    // holds the prior (possibly default/zero) value — the copy in the dispatch section below
    // captures the freshly-selected satellite's real data for the FIRST time this frame, so the
    // panel settles onto the correct tracked position within ~2 frames of the click, not instantly.
    if (selectedSatIndex >= 0)
    {
        const GpuSatVisible *pv = static_cast<const GpuSatVisible *>(pickedVisibleMapped);
        lastPickedSkyDir = pv->skyDir;
        lastPickedFlare = pv->flareIntensity;
    }

    // ── Observer terrain height + cloud params UBO fill (relocated from recordDraw) ─────────
    // cloud_march.comp's dispatch below needs fresh CloudParams/obsEffH data, but it must run
    // before the render pass begins (recordCompute), which is BEFORE recordDraw used to compute
    // either of these. Both are pure CPU/mapped-memory state with no dependency on anything else
    // in recordDraw, so relocating them here is a straightforward move — and both must run before
    // the `activeSatCount == 0` early-out above's return would otherwise skip clouds entirely
    // whenever no satellites are active. Note: pc.obsECEFDir.w (used by both this dispatch and
    // recordDraw's SatDrawPC) is obsHeightOffset ONLY, not obsTerrainH+obsHeightOffset — obsEffH
    // below computes the max explicitly rather than trusting that combination.
    if (!earthElevCpu.empty())
    {
        float latRad = glm::radians(obsLatDeg);
        float lonRad = glm::radians(obsLonDeg);
        float u = (lonRad + glm::pi<float>()) / (2.0f * glm::pi<float>());
        float v = (0.5f * glm::pi<float>() - latRad) / glm::pi<float>();
        int px = (int)(u * (float)earthElevCpuW) % earthElevCpuW;
        int py = std::min((int)(v * (float)earthElevCpuH), earthElevCpuH - 1);
        float pixVal = earthElevCpu[py * earthElevCpuW + px] / 255.0f;
        // DEM ocean baseline is 15/255; subtract it so sea-level land maps to 0 m.
        const float kSeaLevel = 15.0f / 255.0f;
        obsTerrainH = (pixVal <= kSeaLevel) ? 0.0f : std::max(0.0f, (pixVal - kSeaLevel) * 8848.0f);
    }

    // ── City-detail world-fixed offset ────────────────────────────────────────────────────────
    // sat_sky.frag's "City detail texture blend" adds this (cityOffsetEastM/NorthM, packed into
    // CloudParams pad1/pad2) straight onto hitPt.xy to cancel that coordinate's observer-relative
    // drift with a plain translation. hitPt.xy's drift for any point near the observer is, to
    // leading order, just a uniform shift equal to the observer's own north/east motion — moving
    // the reference frame doesn't rotate nearby points relative to each other, it shifts them all
    // together — so tracking the observer's own cumulative displacement is sufficient; no basis
    // reconstruction or grid-snapping needed (an earlier version tried exactly-fixed local ENU
    // bases snapped to a grid, but re-deriving the basis at each snap silently rotated the axes a
    // little, not just translated them, causing a visible pop at every snap instead of the
    // intended seamless tile-period jump).
    {
        double latRad = (double)glm::radians(obsLatDeg);
        double lonRad = (double)glm::radians(obsLonDeg);
        if (!cityOffsetInit)
        {
            cityPrevObsLatRad = latRad;
            cityPrevObsLonRad = lonRad;
            cityOffsetInit = true;
        }
        double dLat = latRad - cityPrevObsLatRad;
        double dLon = lonRad - cityPrevObsLonRad;
        if (dLon > glm::pi<double>())
            dLon -= glm::two_pi<double>(); // antimeridian wrap guard
        if (dLon < -glm::pi<double>())
            dLon += glm::two_pi<double>();
        double cosLat = std::max(0.05, cos(latRad)); // guards the /cosLat below near the poles
        cityOffsetNorthM += dLat * (double)kEarthRadius;
        cityOffsetEastM += dLon * (double)kEarthRadius * cosLat;
        cityPrevObsLatRad = latRad;
        cityPrevObsLonRad = lonRad;
    }

    if (cloudParamsMapped)
    {
        GpuCloudParams cp{};
        cp.coverage = cloudCoverage;
        cp.density = cloudDensity;
        cp.driftRate = cloudDriftRate;
        cp.sunGain = cloudSunGain;
        cp.sunGainZenith = cloudSunGainZenith;
        cp.ambientGain = cloudAmbientGain;
        cp.hgG = cloudHgG;
        cp.marchSteps = cloudMarchSteps;
        cp.lightSteps = cloudLightSteps;
        cp.extinctionCoeff = extinctionCoeff;
        cp.cirrusWindAngle = glm::radians(cloudCirrusWindDeg);
        cp.cirrusStretch = cloudCirrusStretch;
        cp.airglowGain = airglowGain;
        cp.airglowGreenGain = airglowGreenGain;
        cp.airglowRedGain = airglowRedGain;
        cp.airglowSodiumGain = airglowSodiumGain;
        cp.airglowCoverageGain = airglowCoverageGain;
        cp.airglowPolarGain = airglowPolarGain;
        cp.shadowMaxDistM = cloudShadowMaxDistM;
        cp.maxRenderDistM = cloudMaxRenderDistM;
        cp.viewSamplesMin = viewSamplesMin;
        cp.viewSamplesMax = viewSamplesMax;
        cp.lightSamples = lightSamples;
        cp.oceanSeaOctaves = oceanSeaOctaves;
        cp.oceanDetailOctaves = oceanDetailOctaves;
        cp.oceanReflSamples = oceanReflSamples;
        cp.moonGain = moonGain;
        cp.pad1 = (float)cityOffsetEastM;  // repurposed: city-detail world-fixed east offset (m)
        cp.pad2 = (float)cityOffsetNorthM; // repurposed: city-detail world-fixed north offset (m)
        cp.cloudTwilightAmbientGain = cloudTwilightAmbientGain;
        cp.cloudBaseVariance = cloudBaseVariance;
        cp.cloudErosionEdge = cloudErosionEdge;
        cp.cloudErosionCore = cloudErosionCore;
        cp.sunGainElevBand = sunGainElevBand;
        cp.twilightBandHi = twilightBandHi;
        cp.twilightBandLo = twilightBandLo;
        cp.coverageMipLod = coverageMipLod;
        cp.flatCoverageScale = flatCoverageScale;
        cp.flatSunGainScale = flatSunGainScale;
        cp.cloudDistFadeStartM = cloudDistFadeStartM;
        cp.cloudDistFadeEndM = cloudDistFadeEndM;
        cp.fogTopAltM = fogTopAltM; // C11
        cp.fogDensity = fogDensity;
        cp.fogCoverage = fogCoverage;
        cp.fogSunGain = fogSunGain;
        cp.terrainDistFadeStartM = terrainDistFadeStartM;
        cp.terrainDistFadeEndM = terrainDistFadeEndM;
        cp.cloudOpacityScale = cloudOpacityScale;
        cp.cityLightBlurLod = cityLightBlurLod;
        cp.cloudWarpStrength = cloudWarpStrength;
        cp.cloudWarpFreq = cloudWarpFreq;
        cp.cloudSurfaceCarve = cloudSurfaceCarve;
        cp.cloudErosionBillow = cloudErosionBillow;
        cp.cloudErosionBillowH = cloudErosionBillowH;
        cp.cloudErosionFreq = cloudErosionFreq;
        cp.cloudMultiScatter = cloudMultiScatter;
        cp.cloudShadowFloorT = cloudShadowFloorT;
        cp.cloudGrazeShadow = cloudGrazeShadow;
        cp.cloudConeLenScale = cloudConeLenScale;
        cp.cloudVertShadeGain = cloudVertShadeGain;
        cp.cloudDensityAO = cloudDensityAO;
        cp.cloudAOPower = cloudAOPower;
        cp.flatDensityScale = flatDensityScale;
        cp.flatRayleighGain = flatRayleighGain;
        cp.flatTwilightAmbientGain = flatTwilightAmbientGain;
        cp.atmosTermStrength = atmosTermStrength;
        cp.atmosTermWidth = atmosTermWidth;
        cp.atmosRayleighGain = atmosRayleighGain;
        cp.atmosMieGain = atmosMieGain;
        cp.stormStrength = stormStrength;
        cp.auroraGain = auroraGain;
        cp.auroraCloudGain = auroraCloudGain;
        cp.auroraGroundGain = auroraGroundGain;
        cp.auroraCoverageFreq = auroraCoverageFreq;
        cp.auroraCoverageAzFreq = auroraCoverageAzFreq;
        cp.auroraCoverageDriftRate = auroraCoverageDriftRate;
        cp.auroraShimmerRate = auroraShimmerRate;
        cp.mwBasisRow0 = glm::vec4(mwRow0, 1.0f); // .w = milky way gain (fixed; no longer user-tunable)
        cp.mwBasisRow1 = glm::vec4(mwRow1, 0.0f);
        cp.mwBasisRow2 = glm::vec4(mwRow2, 0.0f);
        cp.cloudPhase = (float)fmod((double)cloudDriftRate * (simDayJ2000 * 86400.0 + simSecInDay),
                                    glm::two_pi<double>());
        // Layer 0: low cloud / stratus shell. alphaMax was a flat, hardcoded 0.80 ceiling from the
        // layer system's original session-14 introduction — completely independent of
        // cloudOpacityScale (that only reaches the VOLUMETRIC march in cloud_march.comp), and this
        // flat layer is what's actually active from ~800km+ observer altitude (kCloud3DFadeStart
        // in both shaders) — i.e. most orbital viewing in this sim. A flat 20% floor with no depth
        // falloff at all is what was still letting city lights through "2D clouds" even after the
        // volumetric fix. Reuses the same already-tuned cloudOpacityScale rather than adding a
        // second opacity slider users would have to discover; layer 1 (cirrus) deliberately does
        // NOT scale by it — cirrus is meant to stay thin/translucent, not become a solid deck.
        cp.layers[0] = {cloudBaseAltM, 1.0f, glm::min(1.0f, 0.80f * cloudOpacityScale), 0.0f, 1.0f, 1.0f, 1.0f, 0.0f};
        // Layer 1: high cirrus shell
        cp.layers[1] = {cloudTopAltM, 2.0f, 0.15f, 2.0f, 0.5f, 0.4f, 1.0f, 0.0f};
        // Layers 2-3: unused
        cp.layers[2] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        cp.layers[3] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(cloudParamsMapped, &cp, sizeof(cp));
    }

    // ── Dispatch: sat_orbit.comp — orbital mechanics + attitude ───────────────────────────────
    // Moved to run BEFORE cloud_march.comp (C12 follow-up #22) — it used to run after cloud_march/
    // cloud_shadow, which meant cloud_march.comp's Reflect-Orbital beam sky glow always read
    // ReflectBeamsBuf as written by the PREVIOUS frame's sat_orbit.comp (one frame stale). Harmless
    // for observer-independent data, but satENU/targetENU are METERS offsets in the observer's ENU
    // basis AT WRITE TIME — when the observer moves, a stale offset no longer matches this frame's
    // fresh obsPos/basis when cloud_march.comp uses it, producing a visible lag proportional to how
    // far the observer moved that frame (imperceptible at walking speed, clearly visible at "boost"
    // movement). Running unconditionally here (even when activeSatCount==0, a legal 0-workgroup
    // no-op dispatch) — before the `if (activeSatCount==0) return` check below — means
    // cloud_march.comp always reads THIS frame's fresh beam data instead.
    // Build enabled / highlight masks from constellation config (one bit per constellation).
    uint32_t enabledMask = 0, highlightMask = 0;
    for (uint32_t ci = 0; ci < (uint32_t)constellations.size() && ci < 32; ++ci)
    {
        if (constellations[ci].enabled)
            enabledMask |= (1u << ci);
        if (constellations[ci].highlight)
            highlightMask |= (1u << ci);
    }

    SatOrbitPC orbitPc{};
    orbitPc.enuX = eci2enuX;
    orbitPc.enuY = eci2enuY;
    orbitPc.enuZ = eci2enuZ;
    orbitPc.sunDirECI = sunDirECI;
    // Two-part subtraction: integer day difference (exact) + double seconds (precise).
    // After auto-rebake, dDays < kOrbitRebakeDays so the float cast loses < 0.07 s.
    int64_t dDays = simDayJ2000 - orbitEpochDay;
    double dSec = simSecInDay - orbitEpochSec;
    if (dSec < 0.0)
    {
        --dDays;
        dSec += 86400.0;
    } // borrow from day if frac is negative
    orbitPc.deltaT = (float)((double)dDays * 86400.0 + dSec);
    orbitPc.obsECI = obsECI;
    orbitPc.satCount = activeSatCount;
    orbitPc.highlightMask = highlightMask;
    orbitPc.enabledMask = enabledMask;
    orbitPc.simDt = simDt;
    // Horizon cull threshold: open up to Earth limb for elevated observers.
    // limbSin = -sqrt(1 - (R_EARTH/obsR)²); always clamped to at most -0.01.
    {
        float obsR = glm::length(obsECI);
        float r = kEarthRadius / obsR;
        float limbSin = -sqrtf(std::max(0.0f, 1.0f - r * r));
        orbitPc.elevCutoff = std::min(-0.01f, limbSin);
    }
    orbitPc.beamGain = beamGain;
    orbitPc.reflectorLockWindowS = reflectorLockWindowS;
    orbitPc.mirrorMaxRateDegPerSec = mirrorMaxRateDegPerSec;
    orbitPc.flareMitigationTiltRad = glm::radians(flareMitigationTiltDeg);
    orbitPc.targetCount = (uint32_t)reflectorTargetCount;
    orbitPc.minBeamElevSin = sinf(glm::radians(reflectorMinElevDeg)); // S1 follow-up
    // 2026-08-06 reversibility rework: gmstNow/windowFrac are pure functions of absolute sim time
    // (computed here in double precision, narrowed only after the periodic reduction — same
    // "epoch-delta trick" spirit as deltaT above) so sat_orbit.comp can extrapolate exactly to its
    // lock-window boundaries without any persisted GPU state. See that shader's TargetedReflector
    // block and CLAUDE.md for the full design.
    {
        double simTimeAbs = (double)simDayJ2000 * 86400.0 + simSecInDay;
        orbitPc.gmstNow = (float)fmod(kOmegaEarth * simTimeAbs, glm::two_pi<double>());
        double windowS = std::max(1.0f, reflectorLockWindowS);
        double windowRatio = simTimeAbs / windowS;
        orbitPc.windowFrac = (float)(windowRatio - floor(windowRatio));
    }

    // ── Dispatch: scene_depth.comp — shared terrain/ocean depth (pipeline unification) ──────────
    // Runs FIRST. Everything downstream that needs to know "is this pixel's view blocked by the
    // ground" reads the result instead of re-deriving it: cloud_march.comp's beam occlusion (which
    // used to march the DEM per beam per pixel), and — from the next step — every volumetric
    // layer's own far bound. Depends on nothing else this frame, only the camera.
    //
    // Knockout bit 1024 skips the ENTIRE block — dispatch AND both per-frame layout barriers.
    // createSceneDepthResources() cleared the image to kNoSurfaceT once, and nothing else writes
    // it, so a skipped frame leaves a valid "nothing occludes anywhere" buffer in
    // SHADER_READ_ONLY_OPTIMAL (exactly what every consumer's descriptor expects). Removing the
    // two barriers matters on MoltenVK/older-Metal, where each barrier forces a command-encoder
    // restart that can dominate the frame — see Potato preset. (Previously bit 1024 only made the
    // shader early-return; the dispatch + barriers still ran every frame.)
    if ((debugDisableMask & 1024u) == 0u)
    {
        SceneDepthPC dpc{};
        dpc.skyView = camera.viewMatrix();
        dpc.fovYRad = glm::radians(camera.fovYDeg);
        // ALWAYS the true swapchain aspect, never a render-scaled one — this buffer is consumed
        // at several different resolutions and must be the same function of normalized screen UV
        // at all of them.
        dpc.aspect = (float)ctx.swapExtent.width / (float)ctx.swapExtent.height;
        dpc.debugDisableMask = debugDisableMask;
        dpc.obsECEFDir = glm::vec4(obsDir, obsHeightOffset);

        uint32_t halfW = (ctx.swapExtent.width + 1) / 2;
        uint32_t halfH = (ctx.swapExtent.height + 1) / 2;

        // Pre-dispatch: SHADER_READ_ONLY_OPTIMAL → GENERAL.
        // srcStage includes COMPUTE as well as FRAGMENT — unlike cloudMarchTargetA/B (read only by
        // fragment shaders), this image is also read by cloud_march.comp, so the write-after-read
        // hazard against the PREVIOUS frame's compute read has to be covered. Benign in practice
        // with one frame in flight plus the fence wait, but sync-validation flags its absence.
        ctx.imageBarrier(cmd, sceneDepthImg,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sceneDepthPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sceneDepthPipeLayout, 0, 1, &sceneDepthDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, sceneDepthPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(dpc), &dpc);
        vkCmdDispatch(cmd, (halfW + 15) / 16, (halfH + 15) / 16, 1);

        // Post-dispatch: GENERAL → SHADER_READ_ONLY_OPTIMAL. dstStage covers BOTH consumer kinds —
        // cloud_march.comp (compute, later this call) and sat_sky.frag / the point draws
        // (fragment, later this frame in the render pass).
        ctx.imageBarrier(cmd, sceneDepthImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
    ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 1);

    // Zero reflectBeamsBuf so this frame's orbit dispatch starts with an empty sector
    // selection (same rationale as the glowBuf fill below — atomicMax needs a known-zero start).
    vkCmdFillBuffer(cmd, reflectBeamsBuf, 0, sizeof(GpuReflectBeams), 0);
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = reflectBeamsBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }

    // Zero beamGlowDomeBuf (C12 follow-up #31) — same rationale as reflectBeamsBuf above,
    // atomicMax needs a known-zero start each frame.
    vkCmdFillBuffer(cmd, beamGlowDomeBuf, 0, sizeof(float) * kNumBeamGlowSectors, 0);
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = beamGlowDomeBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }

    // beam_cloud_block.comp (per-target vertical cloud occlusion, C12 follow-up #33) retired
    // 2026-08-09 — replaced by beam_self_march.comp, a per-BEAM slant march dispatched AFTER
    // sat_orbit.comp below (it needs each beam's real satENU/targetENU, which that shader writes
    // this same frame) instead of before it. See that shader's own header and BEAM_CLOUD_PLAN.md.
    // Timestamp slot 2 (this used to bound beam_cloud_block.comp's own dispatch) now reads ~0 every
    // frame — harmless, same convention every knockout-skipped bucket already uses; not worth
    // rewiring the timestamp pool for a bucket that no longer has a dispatch to measure.
    ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 2);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, orbitPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            orbitPipeLayout, 0, 1, &orbitDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, orbitPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(orbitPc), &orbitPc);
    vkCmdDispatch(cmd, (activeSatCount + 63) / 64, 1, 1);

    // Barrier: sat_orbit.comp writes satInputBuf → sat_flare.comp reads it.
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = satInputBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }
    // Barrier: sat_orbit.comp writes reflectBeamsBuf → beam_self_march.comp (below) reads
    // satENU/targetENU and overwrites blockAltM/blockOpacity for the same [0, beamCount) range.
    // Compute-only dependency here — the fragment-stage consumers (sat_sky.frag) wait on
    // beam_self_march's OWN barrier further below instead, since blockOpacity isn't valid until
    // that pass has run.
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = reflectBeamsBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }

    // ── Dispatch: beam_self_march.comp — per-beam cloud occlusion (2026-08-09) ──────────────────
    // Fixed-size dispatch (BEAM_MAX_ACTIVE/64 workgroups, always) since beamCount is a GPU atomic
    // counter written by sat_orbit.comp just above, not known at command-buffer record time —
    // inactive slots (i >= beamCount) return immediately inside the shader. See that shader's own
    // header for the full design, and why this is NOT the same shape as the historically-reverted
    // per-satellite attempts (TERRAIN_PLAN.md follow-ups #14-16).
    //
    // Knockout bit 512 skips the dispatch itself — repurposed from beam_cloud_block.comp's own
    // producer-side skip bit, now retired along with that pass. Skipping leaves blockAltM/
    // blockOpacity at 0.0 (reflectBeamsBuf is vkCmdFillBuffer-zeroed every frame above, and
    // sat_orbit.comp no longer writes these two fields at all — see that shader's own comment) —
    // blockOpacity=0 reads as "fully unoccluded," i.e. every beam renders as if no cloud exists,
    // the same reproduces-pre-feature-behavior convention bit 1024's scene-depth skip uses.
    // Cache the observer basis THIS frame's dispatches used — sat_orbit.comp (just above, same
    // unchanged obsDir) writes satENU/targetENU/reflectDirENU in this exact East/North/Up basis
    // regardless of whether the bit-512 knockout below skips beam_self_march.comp itself. Next
    // frame's CPU readback needs this to un-rotate those vectors before reuse — see
    // lastBeamObsDir's own comment in SatelliteSim.h. Kept unconditional (outside the knockout
    // gate) so toggling that debug bit can never leave this cache stale.
    lastBeamObsDir = obsDir;
    lastBeamObsEffH = std::max(obsTerrainH, obsHeightOffset);

    if ((debugDisableMask & 512u) == 0u)
    {
        BeamSelfMarchPC bmPc{};
        bmPc.obsECEFDir = glm::vec4(obsDir, obsHeightOffset);
        bmPc.obsEffH = std::max(obsTerrainH, obsHeightOffset);
        bmPc.waveTime = (float)(simSecInDay * 1.0);
        bmPc.cloudPhase = (float)fmod((double)cloudDriftRate * (simDayJ2000 * 86400.0 + simSecInDay),
                                     glm::two_pi<double>());

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, beamSelfMarchPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                beamSelfMarchPipeLayout, 0, 1, &beamSelfMarchDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, beamSelfMarchPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(bmPc), &bmPc);
        vkCmdDispatch(cmd, (kMaxActiveBeams + 63) / 64, 1, 1);
    }

    // Barrier: beam_self_march.comp writes reflectBeamsBuf (blockAltM/blockOpacity) → read THIS
    // frame by cloud_march.comp (compute, right below) and sat_sky.frag (fragment, later in the
    // render pass). Also covers every OTHER ReflectBeam field sat_orbit.comp wrote (targetENU,
    // satENU, reflectDirENU, ...) for those same two consumers — strictly safe to fold into this
    // later barrier since it's a superset of what the earlier one already guaranteed.
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = reflectBeamsBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }
    // Barrier: sat_orbit.comp writes beamGlowDomeBuf (C12 follow-up #31) → read THIS frame by
    // sat_flare.comp (compute) and sat_sky.frag's Milky Way section (fragment) — same scope as
    // reflectBeamsBuf's barrier above, same two consumer stage types.
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = beamGlowDomeBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }
    ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 3);

    // ── Dispatch: cloud_march.comp — half-resolution cloud/cirrus march (C15-perf) ──────────
    // Runs at half ctx.swapExtent, writing cloudMarchTargetA/B; sat_sky.frag samples them
    // (skyDescSet bindings 10/11) in place of the old inline cirrusMarch()/cloudMarch() calls.
    if (!dbgEnv("SATLIGHTSIM_SKIP_CLOUDMARCH"))
    {
        CloudMarchPC cpc{};
        cpc.skyView = camera.viewMatrix();
        cpc.fovYRad = glm::radians(camera.fovYDeg);
        cpc.aspect = (float)ctx.swapExtent.width / (float)ctx.swapExtent.height;
        cpc.waveTime = (float)(simSecInDay * 1.0);
        cpc.obsEffH = std::max(obsTerrainH, obsHeightOffset);
        cpc.sunDirENU = sunDirENU;
        cpc.moonDirENU = moonDirENU;
        cpc.obsECEFDir = glm::vec4(obsDir, obsHeightOffset);
        cpc.debugDisableMask = debugDisableMask;             // aurora knockout toggle now lives here too
        cpc.beamMaxRangeM = beamMaxRangeM;                   // C12 follow-up #6
        cpc.showBeamDebugRays = showBeamDebugRays ? 1u : 0u; // C12 follow-up #12
        cpc.beamSkyGlowGain = beamSkyGlowGain;               // C12 follow-up #17; #44: now gains the real
                                                             // per-sample beam->cloud term instead of the
                                                             // deleted analytic tube
        cpc.cloudShadowRangeM = cloudShadowRangeM;
        // C12 follow-up #39: cpc.beamGlowBleedGain removed — the near-field bleed/march it drove
        // in this shader was removed entirely; see buildSatDrawPC() for its new home.
        // C12 follow-up #44: cpc.daySuppression/beamExtinctionMult/beamNearFieldFadeM removed —
        // all three existed only for the analytic beam sky-glow block deleted this round.

        uint32_t halfW = (ctx.swapExtent.width + 1) / 2;
        uint32_t halfH = (ctx.swapExtent.height + 1) / 2;

        // Pre-dispatch: both targets are left in SHADER_READ_ONLY_OPTIMAL after the previous
        // frame's post-dispatch barrier below (or by createCloudMarchResources on the first frame
        // / after an onResize recreation) — transition back to GENERAL, required for storage-image
        // writes (imageStore).
        ctx.imageBarrier(cmd, cloudMarchTargetAImg,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        ctx.imageBarrier(cmd, cloudMarchTargetBImg,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudMarchPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                cloudMarchPipeLayout, 0, 1, &cloudMarchDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, cloudMarchPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(cpc), &cpc);
        vkCmdDispatch(cmd, (halfW + 15) / 16, (halfH + 15) / 16, 1);

        // Post-dispatch: transition both targets back to SHADER_READ_ONLY_OPTIMAL for
        // sat_sky.frag to sample. Explicit layout-transition barriers — the render pass's
        // existing VK_SUBPASS_EXTERNAL dependency (used for glowBuf) has no layout fields and
        // cannot perform a layout transition on its own.
        ctx.imageBarrier(cmd, cloudMarchTargetAImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        ctx.imageBarrier(cmd, cloudMarchTargetBImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
    ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 4);

    // (cloud_shadow.comp dispatched here — a fixed 128x128 observer-centred tangent-plane grid,
    //  plus the texel-snapping machinery it needed to stop shadows swimming as the observer
    //  moved. Deleted in the pipeline-unification pass: cloud_march.comp now marches the shadow
    //  per pixel from the terrain hit point the scene-depth pass already found, which is sharper
    //  near the camera, correct from any altitude, unbounded in range, and has nothing to snap.
    //  Its timestamp bucket went away with it.)

    if (activeSatCount == 0)
    {
        // sat_flare.comp below is skipped this frame — write the same timestamp into its slot so
        // updateGpuTimingStats() sees a zero-duration bucket next frame instead of stale or
        // unavailable query data. scene_depth/sat_orbit/beam_self_march/cloud_march above already
        // ran unconditionally this frame (sat_orbit.comp with 0 satellite workgroups when
        // applicable — a legal no-op dispatch) and got their own real timestamps, so only the
        // flare slot needs a placeholder here. See C12 follow-up #22 for why sat_orbit.comp now
        // runs before this check at all (it used to run after it, alongside flare).
        ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 5);
        return;
    }

    // Zero all glow bins so this frame's flare shader starts with an empty histogram.
    // floatBitsToUint(0.0) == 0u, so filling with 0 correctly marks every bin empty.
    vkCmdFillBuffer(cmd, glowBuf, 0, sizeof(GpuGlowBuf), 0);
    // Zero the ocean-glint list too (flare architecture overhaul) — same idiom, own small buffer,
    // own atomicAdd counter that must start at 0 each frame.
    vkCmdFillBuffer(cmd, oceanGlintBuf, 0, sizeof(GpuOceanGlintBuf), 0);
    {
        VkBufferMemoryBarrier bmb[2] = {};
        bmb[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bmb[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        bmb[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb[0].buffer = glowBuf;
        bmb[0].offset = 0;
        bmb[0].size = VK_WHOLE_SIZE;
        bmb[1] = bmb[0];
        bmb[1].buffer = oceanGlintBuf;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 2, bmb, 0, nullptr);
    }

    // ── Dispatch: sat_flare.comp — lighting + visibility ──────────────────────
    SatFlarePC pc{};
    pc.enuX = eci2enuX;
    pc.enuY = eci2enuY;
    pc.enuZ = eci2enuZ;
    pc.sunDirECI = sunDirECI;
    pc.satCount = activeSatCount;
    pc.obsECI = obsECI;
    pc.elevCutoff = orbitPc.elevCutoff; // same threshold computed above
    pc.brightnessScale = brightnessScale;
    pc.daySuppression = daySuppression;
    pc.mirrorBoost = mirrorBoost;
    pc.visThresh = visThresh;
    pc.highlightFlare = highlightFlare;
    pc.moonSuppression = moonSuppression;
    pc.moonDirECI = moonDirECI; // computed in updatePositions(), called earlier this frame
    pc.extinctionCoeff = extinctionCoeff;
    pc.sunRefIntensity = sunFlareRefIntensity; // S3: soft ceiling reference, see struct comment

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compPipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (activeSatCount + 63) / 64, 1, 1);

    // Barrier: sat_flare.comp writes satVisibleBuf → vertex shader reads it (and, when a satellite
    // is selected, the tiny per-frame pick-tracking copy just below also reads it via transfer).
    {
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = satVisibleBuf;
        bmb.offset = 0;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
    }

    // ── Flare/corona render-to-texture pipeline (flare architecture overhaul) ────────────────
    // Stage 1: render every visible satellite (satVisibleBuf, just barriered above) plus one
    // virtual point for the sun into flareSourceImg. Needs cloudTargetA/B + sceneDepthTex, both
    // already valid from earlier in this same recordCompute() call. See FlareSourcePC's comment
    // in SatelliteSim.h for the full three-stage design this replaces the old per-pixel
    // flareEntries loop with.
    {
        // Computed here (rather than after the draw, where this used to live) so fpc can carry
        // the sun's compensating factor below — see FlareSourcePC::sunDayCompensation's comment.
        const float kFlareDayFloor = 0.35f;
        float flareDarkness = glm::clamp(-sunDirENU.w * 5.0f, 0.0f, 1.0f);
        float flareEyeAdaptGain = glm::mix(kFlareDayFloor, 1.0f, flareDarkness);

        FlareSourcePC fpc{};
        fpc.skyView = camera.viewMatrix();
        fpc.fovYRad = glm::radians(camera.fovYDeg);
        fpc.aspect = (float)ctx.swapExtent.width / (float)ctx.swapExtent.height;
        fpc.satCount = activeSatCount;
        fpc.sunRefIntensity = sunFlareRefIntensity;
        fpc.sunDirENU = sunDirENU;
        fpc.screenSizePx = glm::vec2((float)flareExtent.width, (float)flareExtent.height);
        fpc.resScale = (float)flareExtent.width / (float)ctx.swapExtent.width;
        fpc.sunDayCompensation = 1.0f / glm::max(flareEyeAdaptGain, 0.05f);

        VkClearValue flareClear{};
        flareClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = flareSourceRenderPass;
        rbi.framebuffer = flareSourceFramebuffer;
        rbi.renderArea = {{0, 0}, flareExtent};
        rbi.clearValueCount = 1;
        rbi.pClearValues = &flareClear;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, flareSourcePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                flareSourcePipeLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, flareSourcePipeLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(fpc), &fpc);
        vkCmdDraw(cmd, activeSatCount + 1, 1, 0, 0); // +1 = the sun's virtual point
        vkCmdEndRenderPass(cmd);                     // finalLayout=GENERAL — ready for the compute blur below, no
                                                     // extra barrier (same convention skyLowResRenderPass established)

        // Stage 2: blur/streak — one pipeline, three dispatches ping-ponging flareSourceImg <->
        // flareScratchImg (see FlareBlurPC's comment for the direction/mode scheme). Each
        // dispatch's write must be complete before the next dispatch's read/write — a full
        // compute-stage barrier serializes them (the named image in each barrier only adds the
        // memory-visibility guarantee; the COMPUTE_SHADER_BIT->COMPUTE_SHADER_BIT stage scope is
        // what actually orders the two dispatches, same convention already used for
        // cloudMarchTargetA/B's own two-image compute dependency).
        uint32_t gx = (flareExtent.width + 15) / 16;
        uint32_t gy = (flareExtent.height + 15) / 16;
        VkImageMemoryBarrier flareBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        flareBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        flareBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        flareBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        flareBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        flareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        flareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        flareBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, flareBlurPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                flareBlurPipeLayout, 0, 1, &flareBlurDescSet, 0, nullptr);

        // S3 (RELEASE_v1_1_PLAN.md): scale the glow/streak gain with darkness (eye adaptation)
        // rather than letting a bright flare's blown-out core keep growing — the core brightness
        // ceiling is handled above (sat_flare.comp's Reinhard rolloff) and its size is already
        // logarithmic in effectFlare; this is what keeps the DRAMA at night rather than in daylight,
        // where the same bloom around a point source would look wrong against a bright sky. Same
        // formula as updateStars()'s nightFactor (sin(elevation) = sunDirENU.w) — a fixed day floor
        // rather than zero, so twilight doesn't pop the glow on/off. (flareDarkness/flareEyeAdaptGain
        // now computed earlier, above fpc — the sun's own sunDayCompensation needs it too.)

        FlareBlurPC bpc{};
        bpc.direction = 0;
        bpc.mode = 0;
        bpc.streakGain = flareStreakGain * flareEyeAdaptGain; // horizontal gaussian: source->scratch
        vkCmdPushConstants(cmd, flareBlurPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bpc), &bpc);
        vkCmdDispatch(cmd, gx, gy, 1);

        flareBarrier.image = flareScratchImg;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &flareBarrier);

        bpc.direction = 1;
        bpc.mode = 1; // vertical gaussian: scratch->source
        vkCmdPushConstants(cmd, flareBlurPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bpc), &bpc);
        vkCmdDispatch(cmd, gx, gy, 1);

        flareBarrier.image = flareSourceImg;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &flareBarrier);

        bpc.direction = 0;
        bpc.mode = 2; // streak: source->scratch (final result)
        vkCmdPushConstants(cmd, flareBlurPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bpc), &bpc);
        vkCmdDispatch(cmd, gx, gy, 1);

        // Final result (flareScratchImg) is read by the composite draw's FRAGMENT shader later
        // this frame, in recordDraw().
        flareBarrier.image = flareScratchImg;
        flareBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &flareBarrier);
    }

    // ── Long-exposure trail pipeline (fun side feature) ───────────────────────────────────────
    // See the trailAccumImg/trailEnabled member block comment in SatelliteSim.h for the full
    // design. Splats are drawn from THIS frame's live satVisibleBuf/starBuf/planetBuf (already
    // fully computed above by sat_flare.comp/updateStars()/updatePlanets()) — one sample per real
    // display frame, no separate orbital/rotational recompute (see that comment for the accepted
    // limitation this implies at very high timeScaleIdx). Real-time decay (dt, never simDt) runs
    // every frame trails are enabled, even while timePaused — matches a real camera shutter aging
    // even when nothing new is landing on it.
    if (trailEnabled)
    {
        if (trailClearPending)
        {
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, trailAccumImg, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
            trailClearPending = false;

            ctx.imageBarrier(cmd, trailAccumImg,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        // Barrier: previous frame's splat draws (COLOR_ATTACHMENT_OUTPUT/WRITE) -> this frame's
        // fade compute READ|WRITE. Needed because trailAccumRenderPass uses LOAD_OP_LOAD (unlike
        // flareSourceImg above, which is CLEARed every frame and has no such prior-content
        // dependency to protect).
        ctx.imageBarrier(cmd, trailAccumImg,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        TrailFadePC tfpc{};
        tfpc.decayFactor = expf(-dt / std::max(trailDecaySeconds, 0.05f));
        // Deliberately far above anything normal accumulation reaches (worst case: a near-static
        // bright point at the slowest decay the slider allows, trailDecaySeconds=30s at 60fps, has
        // decayFactor~0.9994 -> steady-state amplification ~1800x a single splat's peak per-frame
        // contribution, itself bounded well under 4.0 by sat_point.frag's/star_point.frag's own
        // coreScale caps — comfortably under this ceiling, and this ceiling itself stays comfortably
        // under RGBA16F's ~65504 max). Only meant to bound truly unbounded drift over an extremely
        // long unattended/paused session — see trail_fade.comp's own comment for why raising this
        // from its previous 4.0f (which routinely saturated and flattened real brightness
        // differences, defeating trail_composite.frag's tonemap below it) was necessary, not optional.
        tfpc.ceiling = 50000.0f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trailFadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                trailFadePipeLayout, 0, 1, &trailFadeDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, trailFadePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tfpc), &tfpc);
        vkCmdDispatch(cmd, (trailAccumExtent.width + 15) / 16, (trailAccumExtent.height + 15) / 16, 1);

        // Barrier: fade compute WRITE -> the splat render pass's implicit GENERAL->COLOR_ATTACHMENT
        // transition + LOAD.
        ctx.imageBarrier(cmd, trailAccumImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        SatDrawPC tpc = buildSatDrawPC(ctx, trailAccumExtent); // trailAccumExtent == ctx.swapExtent
        // This offscreen render pass has no depth attachment — see sat_point.frag/star_point.frag's
        // own terrain-occlusion comment. ppc below inherits this via its `= tpc` copy.
        tpc.manualTerrainTest = 1.0f;

        VkRenderPassBeginInfo trbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        trbi.renderPass = trailAccumRenderPass;
        trbi.framebuffer = trailAccumFramebuffer;
        trbi.renderArea = {{0, 0}, trailAccumExtent};
        trbi.clearValueCount = 0; // LOAD_OP_LOAD — nothing to clear
        vkCmdBeginRenderPass(cmd, &trbi, VK_SUBPASS_CONTENTS_INLINE);

        if (activeSatCount > 0)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trailSatPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    drawPipeLayout, 0, 1, &descSet, 0, nullptr);
            vkCmdPushConstants(cmd, drawPipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(tpc), &tpc);
            vkCmdDraw(cmd, activeSatCount, 1, 0, 0);
        }
        if (starCount > 0 && trailStarPipeline != VK_NULL_HANDLE)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trailStarPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    starPipeLayout, 0, 1, &starDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, starPipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(tpc), &tpc);
            vkCmdDraw(cmd, starCount, 1, 0, 0);
        }
        if (showPlanets && planetDescSet != VK_NULL_HANDLE && trailStarPipeline != VK_NULL_HANDLE)
        {
            SatDrawPC ppc = tpc;
            ppc.noTwinkle = 1.0f;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trailStarPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    starPipeLayout, 0, 1, &planetDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, starPipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ppc), &ppc);
            vkCmdDraw(cmd, kPlanetCount, 1, 0, 0);
        }

        vkCmdEndRenderPass(cmd); // finalLayout=GENERAL — ready for next frame's fade compute /
                                 // this frame's composite sample (recordDraw), no extra barrier
    }

    // Selected-satellite tracking: mirror just that one 32-byte entry into pickedVisibleBuf so
    // next frame's buildUI can reproject it (see the one-frame-stale read near peakMagnitude
    // above). No-op — no command recorded at all — when nothing is selected.
    if (selectedSatIndex >= 0 && selectedSatIndex < (int)activeSatCount)
    {
        VkBufferCopy pickRegion{};
        pickRegion.srcOffset = (VkDeviceSize)selectedSatIndex * sizeof(GpuSatVisible);
        pickRegion.dstOffset = 0;
        pickRegion.size = sizeof(GpuSatVisible);
        vkCmdCopyBuffer(cmd, satVisibleBuf, pickedVisibleBuf, 1, &pickRegion);
    }
    ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 5);
}

// ─── projectSkyDirToScreen ────────────────────────────────────────────────────
// Pure camera geometry — mirrors sat_point.vert's projection exactly
// (shaders/sat_point.vert:34,47,60-62) so CPU-side picking/tracking agrees pixel-for-pixel
// with what's actually rendered. No orbital mechanics here, so unlike the GPU orbit/attitude
// math this carries little drift risk from being hand-duplicated in C++.
bool SatelliteSim::projectSkyDirToScreen(const glm::vec3 &skyDir, float screenW, float screenH,
                                         float &outX, float &outY) const
{
    glm::vec3 cam = glm::vec3(camera.viewMatrix() * glm::vec4(skyDir, 0.0f));
    if (cam.z >= -0.001f)
        return false; // behind camera — same threshold sat_point.vert uses

    float tanHalfFov = tanf(glm::radians(camera.fovYDeg) * 0.5f);
    float aspect = screenW / screenH;
    float ndcX = cam.x / -cam.z / (tanHalfFov * aspect);
    float ndcY = -cam.y / -cam.z / tanHalfFov;

    outX = (ndcX * 0.5f + 0.5f) * screenW;
    outY = (ndcY * 0.5f + 0.5f) * screenH;
    return true;
}

// ─── pickSatelliteAt ───────────────────────────────────────────────────────────
// One-shot click hit-test. Copies satVisibleBuf (device-local) back to a transient
// host-visible staging buffer sized to activeSatCount (not MAX_SATELLITES, so cost scales
// with what's actually simulated — a few MB at the current constellation roster), then scans
// it on the CPU for the nearest currently-visible satellite within its own hit radius. The
// synchronous stall from ctx.beginOneTimeCommands()/endOneTimeCommands() is fine here — this
// only runs once per user click, never per frame (contrast the tiny per-frame tracking copy
// in recordCompute above, which deliberately avoids any such stall).
int SatelliteSim::pickSatelliteAt(float clickX, float clickY, float screenW, float screenH)
{
    if (activeSatCount == 0 || !ctx_)
        return -1;

    VulkanContext &ctx = *ctx_;
    VkDeviceSize copySize = (VkDeviceSize)activeSatCount * sizeof(GpuSatVisible);

    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    ctx.createBuffer(copySize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuf, stagingMem);

    VkCommandBuffer cmd = ctx.beginOneTimeCommands();
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = copySize;
    vkCmdCopyBuffer(cmd, satVisibleBuf, stagingBuf, 1, &region);
    ctx.endOneTimeCommands(cmd);

    void *mapped = nullptr;
    vkMapMemory(ctx.device, stagingMem, 0, copySize, 0, &mapped);
    const GpuSatVisible *entries = static_cast<const GpuSatVisible *>(mapped);

    constexpr float kMinHitRadiusPx = 8.0f; // dim/tiny points stay clickable

    int best = -1;
    float bestDist = 0.0f;
    for (uint32_t i = 0; i < activeSatCount; ++i)
    {
        const GpuSatVisible &v = entries[i];
        if (v.flareIntensity <= 0.0f)
            continue;
        float sx, sy;
        if (!projectSkyDirToScreen(v.skyDir, screenW, screenH, sx, sy))
            continue;
        float dx = sx - clickX, dy = sy - clickY;
        float dist = sqrtf(dx * dx + dy * dy);
        float hitRadius = std::max(v.angularSize * 0.5f, kMinHitRadiusPx);
        if (dist <= hitRadius && (best < 0 || dist < bestDist))
        {
            best = (int)i;
            bestDist = dist;
        }
    }

    vkUnmapMemory(ctx.device, stagingMem);
    vkDestroyBuffer(ctx.device, stagingBuf, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    return best;
}

// ─── pickPlanetAt ───────────────────────────────────────────────────────────────
// Cheaper than pickSatelliteAt(): planetBuf is HOST_VISIBLE/COHERENT and already holds this
// frame's data (updatePlanets() writes it directly from the CPU, no device-local buffer, no
// staging copy needed at all) — just a small loop over kPlanetCount entries. Same hit-test
// convention (nearest within its own angularSize-derived radius, kMinHitRadiusPx floor) as
// pickSatelliteAt for a consistent feel.
int SatelliteSim::pickPlanetAt(float clickX, float clickY, float screenW, float screenH)
{
    if (!planetMapped || !showPlanets)
        return -1;

    const GpuSatVisible *entries = static_cast<const GpuSatVisible *>(planetMapped);
    constexpr float kMinHitRadiusPx = 8.0f;

    int best = -1;
    float bestDist = 0.0f;
    for (int i = 0; i < kPlanetCount; ++i)
    {
        if (!planetEnabled[i])
            continue;
        const GpuSatVisible &v = entries[i];
        if (v.flareIntensity <= 0.0f)
            continue;
        float sx, sy;
        if (!projectSkyDirToScreen(v.skyDir, screenW, screenH, sx, sy))
            continue;
        float dx = sx - clickX, dy = sy - clickY;
        float dist = sqrtf(dx * dx + dy * dy);
        float hitRadius = std::max(v.angularSize * 0.5f, kMinHitRadiusPx);
        if (dist <= hitRadius && (best < 0 || dist < bestDist))
        {
            best = i;
            bestDist = dist;
        }
    }
    return best;
}

// ─── formatSelectedPlanetInfo ────────────────────────────────────────────────────
// Mirrors formatSelectedSatInfo below, but the planet's astronomy (distance/phase/magnitude) is
// dynamic — recomputed fresh from this frame's planetStates[] rather than static orbital elements
// — so unlike the satellite version, re-call this every frame the selection is active, not only
// when the selection changes (see buildSelectedSatPanel).
void SatelliteSim::formatSelectedPlanetInfo()
{
    if (selectedPlanetIndex < 0 || selectedPlanetIndex >= kPlanetCount)
    {
        for (auto &line : planetInfoLine)
            line[0] = '\0';
        return;
    }

    const PlanetState &ps = planetStates[selectedPlanetIndex];
    float vmag = planetApparentMagnitude((PlanetId)selectedPlanetIndex, ps.sunDistAU, ps.distanceAU, ps.phaseAngleDeg);
    float illumFrac = (1.0f + cosf(glm::radians(ps.phaseAngleDeg))) * 0.5f; // 1=full, 0=new

    snprintf(planetInfoLine[0], sizeof(planetInfoLine[0]), "%s", kPlanetNames[selectedPlanetIndex]);
    snprintf(planetInfoLine[1], sizeof(planetInfoLine[1]), "Planet");
    snprintf(planetInfoLine[2], sizeof(planetInfoLine[2]), "Mag: %.1f", vmag);
    snprintf(planetInfoLine[3], sizeof(planetInfoLine[3]), "Distance: %.2f AU", ps.distanceAU);
    snprintf(planetInfoLine[4], sizeof(planetInfoLine[4]), "Phase: %.0f%%", illumFrac * 100.0f);
    snprintf(planetInfoLine[5], sizeof(planetInfoLine[5]), "Sun dist: %.2f AU", ps.sunDistAU);
    planetInfoLine[6][0] = '\0'; // 7th slot is satellite-only (flare-mitigation power readout)
}

// ─── formatSelectedSatInfo ─────────────────────────────────────────────────────
// Fills selInfoLine[] from static, CPU-resident orbital-element data (satOrbits/constellations/
// satTypes) — call once when selectedSatIndex changes, not every frame; nothing here is
// per-frame dynamic (only screen position is, handled separately via lastPickedSkyDir).
void SatelliteSim::formatSelectedSatInfo()
{
    if (selectedSatIndex < 0 || selectedSatIndex >= (int)satOrbits.size())
    {
        for (auto &line : selInfoLine)
            line[0] = '\0';
        return;
    }

    const SatOrbit &orb = satOrbits[selectedSatIndex];
    const char *constName = "?";
    const char *typeName = "?";
    const SatelliteType *type = nullptr;
    int localId = selectedSatIndex;
    if (orb.constIdx < constellations.size())
    {
        const ConstellationConfig &c = constellations[orb.constIdx];
        constName = c.name.c_str();
        localId = selectedSatIndex - (int)c.orbitStart;
        if (c.typeIdx < satTypes.size())
        {
            type = &satTypes[c.typeIdx];
            typeName = type->name.c_str();
        }
    }

    float altKm = orb.altM / 1000.0f;
    float inclDeg = glm::degrees(orb.incl);
    float periodMin = (orb.meanMot > 0.0f) ? (2.0f * glm::pi<float>() / orb.meanMot) / 60.0f : 0.0f;

    snprintf(selInfoLine[0], sizeof(selInfoLine[0]), "%s #%d", constName, localId);
    snprintf(selInfoLine[1], sizeof(selInfoLine[1]), "%s", typeName);
    snprintf(selInfoLine[2], sizeof(selInfoLine[2]), "Alt: %.0f km", altKm);
    snprintf(selInfoLine[3], sizeof(selInfoLine[3]), "Incl: %.1f deg", inclDeg);
    if (orb.alignTerminator)
        snprintf(selInfoLine[4], sizeof(selInfoLine[4]), "RAAN: sun-sync (precessing)");
    else
        snprintf(selInfoLine[4], sizeof(selInfoLine[4]), "RAAN: %.1f deg", glm::degrees(orb.raan));
    snprintf(selInfoLine[5], sizeof(selInfoLine[5]), "Period: %.1f min", periodMin);
    // Flare-mitigation power readout — only for satellites whose primary surface actually uses
    // the tilt (datacenter types), so an unrelated constellation's panel doesn't show a
    // meaningless "100%" row. Power loss is exactly cos(tiltDeg): computeNormal()'s
    // AM_SUN_TRACKING_TILTED case rotates sunDirECI by exactly tiltRad, so
    // dot(tiltedNormal, sunDirECI) == cos(tiltRad) — see that shader comment.
    if (type && type->primary.attitude == AttitudeMode::SunTrackingTilted)
    {
        float powerPct = cosf(glm::radians(flareMitigationTiltDeg)) * 100.0f;
        snprintf(selInfoLine[6], sizeof(selInfoLine[6]), "Power: %.0f%% (tilt %.0f deg)",
                 powerPct, flareMitigationTiltDeg);
    }
    else
    {
        selInfoLine[6][0] = '\0';
    }
}

// ─── recordDraw ───────────────────────────────────────────────────────────────
// Every field the sky/satellite/star pipelines' push constant needs — shared by recordPrePass
// (low-res sky background, when scaled) and recordDraw (satellites/stars always; sky background
// too when renderScale==1.0), so the two never drift out of sync on what they push. targetExtent
// is THIS draw's own actual framebuffer size (skyLowResExtent when rendering the scaled
// background, ctx.swapExtent for everything else) — aspect always uses the true swap extent
// (the camera's real aspect ratio never changes just because the sky pass rendered smaller), but
// screenSizePx must reflect the actual target so gl_FragCoord-based UV math in the shader stays
// correct (see that field's comment in sat_sky.frag for why).
SatDrawPC SatelliteSim::buildSatDrawPC(VulkanContext &ctx, VkExtent2D targetExtent)
{
    SatDrawPC pc{};
    pc.skyView = camera.viewMatrix();
    pc.fovYRad = glm::radians(camera.fovYDeg);
    pc.aspect = (float)ctx.swapExtent.width / (float)ctx.swapExtent.height;
    pc.screenSizePx = glm::vec2((float)targetExtent.width, (float)targetExtent.height);
    pc.gmst = (float)fmod(kOmegaEarth * (simDayJ2000 * 86400.0 + simSecInDay), glm::two_pi<double>());
    // Wave time relative to sim epoch: pauses when paused, scales with time warp.
    // Sim sec works great as it resets before any crazy floating point issues happen. Great for any animations that need a time variable.
    // There is probably a looping artifact when it rolls over but who cares it's a tiny blip that most won't notice
    pc.waveTime = simSecInDay * 1.0;
    pc.sunDirENU = sunDirENU;
    pc.moonDirENU = moonDirENU; // xyz = moon dir in ENU, w = illuminated fraction
    // obsTerrainH and the CloudParams UBO fill both moved to recordCompute() — the new
    // cloud_march.comp dispatch there needs them before this function even runs. See the
    // comment at that relocation site for why.
    pc.obsECEFDir = glm::vec4(obsDir, obsHeightOffset); // w = user altitude offset above terrain (m); GPU computes ground height
    pc.debugDisableMask = debugDisableMask;             // perf knockout toggles — see SatelliteSim.h member comment
    pc.skyGlareVisibility = skyGlareEased;              // sun-glare gate for the Milky Way — see skyGlareEased member comment
    pc.beamMaxRangeM = beamMaxRangeM;                   // C12 follow-up #6
    pc.beamSkyGlowGain = beamSkyGlowGain;               // C12 follow-up #18 — shared with cloud_march.comp's copy
    pc.beamGlowBleedGain = beamGlowBleedGain;           // C12 follow-up #39 — moved here from CloudMarchPC;
                                                        // now drives this shader's own beam sky-glow wash
    pc.beamProximityGlow = beamProximityGlow;           // C12 follow-up #41
    pc.mwSuppressEased = mwSuppressEased;               // Milky Way's own pollution hysteresis — see member comment
    return pc;
}

void SatelliteSim::recordPrePass(VkCommandBuffer cmd, VulkanContext &ctx, float /*dt*/, uint32_t imgIdx)
{
    if (renderScale >= 0.999f)
        return; // full-res: Pass 1 draws inline in recordDraw as before, nothing to pre-render here

    SatDrawPC pc = buildSatDrawPC(ctx, skyLowResExtent);

    // ── Low-res sky/ground background, into its own offscreen target ─────────────────────────
    VkClearValue clear = clearColor();
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = skyLowResRenderPass;
    rbi.framebuffer = skyLowResFramebuffer;
    rbi.renderArea = {{0, 0}, skyLowResExtent};
    rbi.clearValueCount = 1;
    rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyLowResPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            skyBgPipeLayout, 0, 1, &skyDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, skyBgPipeLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd); // finalLayout=TRANSFER_SRC_OPTIMAL — ready for the blit below, no extra barrier
    // Moved here from recordDraw's Pass 1 (same meaning: sky/terrain/ocean/cloud-composite
    // shader's own cost) — this is now the only place that timestamp gets written when scaled.
    ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 6);

    // ── Blit (linear-filtered upscale) into the swapchain image ──────────────────────────────
    // UNDEFINED oldLayout: we're about to overwrite the whole image via blit, so previous
    // contents (whatever they were — PRESENT_SRC_KHR from a prior present, or truly undefined on
    // this image's very first use) don't need to be preserved. The wait on ctx.semImageAvailable
    // (App.cpp's submit, now gated on TRANSFER too — see that comment) already guarantees the
    // presentation engine is done reading this image before this write can start.
    ctx.imageBarrier(cmd, ctx.swapImages[imgIdx],
                     0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {(int32_t)skyLowResExtent.width, (int32_t)skyLowResExtent.height, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)ctx.swapExtent.width, (int32_t)ctx.swapExtent.height, 1};
    vkCmdBlitImage(cmd, skyLowResColorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   ctx.swapImages[imgIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_LINEAR);
    // No further barrier here — activeRenderPass() returns ctx.renderPassLoad when scaled, whose
    // color attachment initialLayout is TRANSFER_DST_OPTIMAL (exactly what the blit just left it
    // in); the render pass's own automatic transition takes it to COLOR_ATTACHMENT_OPTIMAL.
}

void SatelliteSim::recordDraw(VkCommandBuffer cmd, VulkanContext &ctx, float /*dt*/)
{
    SatDrawPC pc = buildSatDrawPC(ctx, ctx.swapExtent);

    // ── Pass 1: sky/ground background (fullscreen triangle, opaque) ──────────
    // Skipped when renderScale < 1.0 — already rendered (at low res) and blitted into this
    // frame's swapchain image by recordPrePass, before this render pass even began. See
    // SatelliteSim.h's resolution-scaling member comment for the full design and the accepted
    // depth-occlusion tradeoff.
    if (renderScale >= 0.999f)
    {
        if (!dbgEnv("SATLIGHTSIM_SKIP_SKYBG")) // diagnostic: drop the fullscreen sky/ground draw
        {
            // Potato preset (bit 262144): swap in the minimal fragment shader. sat_sky.frag is
            // ~490 ms/frame on a 2015 AMD GPU via MoltenVK — the whole frame — and no quality
            // slider or knockout reduces it. Same layout / descriptor set / push constant.
            VkPipeline skyPipe = (debugDisableMask & 262144u) ? skyBgMinimalPipeline : skyBgPipeline;
            {
                // Always-on breadcrumb (into satlight_log.txt) for the intermittent Potato
                // slow-start bug — see [[potato-mode-intermittent-slow-start]]. Logs only on a
                // change, so it costs nothing per frame but is always present when it recurs.
                static VkPipeline lastSkyPipe = VK_NULL_HANDLE;
                static uint32_t lastSkyMask = 0xFFFFFFFFu;
                if (skyPipe != lastSkyPipe || debugDisableMask != lastSkyMask)
                {
                    Log::line("sky pipeline: " +
                              std::string((skyPipe == skyBgMinimalPipeline) ? "MINIMAL" : "FULL sat_sky.frag") +
                              " (mask " + std::to_string(debugDisableMask) +
                              ", renderScale " + std::to_string(renderScale) + ")");
                    lastSkyPipe = skyPipe;
                    lastSkyMask = debugDisableMask;
                }
                static uint64_t dbgSkyFrame = 0;
                if (dbgEnv("SATLIGHTSIM_FRAME_TRACE") && (dbgSkyFrame++ % 60 == 0))
                    fprintf(stderr, "[sky] mask=%u rs=%.3f minimalBit=%d pipe=%s(%p) full=%p min=%p\n",
                            debugDisableMask, renderScale, (debugDisableMask & 262144u) ? 1 : 0,
                            (skyPipe == skyBgMinimalPipeline) ? "MIN" : "FULL", (void *)skyPipe,
                            (void *)skyBgPipeline, (void *)skyBgMinimalPipeline);
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    skyBgPipeLayout, 0, 1, &skyDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, skyBgPipeLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        // Isolates the sky/terrain/ocean/cloud-composite fragment shader's own cost from the
        // satellite + star point draws that follow (previously all three were lumped into one
        // timestamp bucket in App.cpp — see VulkanContext::kTimestampCount). Written even when the
        // draw above is diagnostically skipped, so the query pool never has an unwritten slot.
        ctx.writeTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 6);
    }

    // ── Pass 2: satellite points (additive blending) ──────────────────────────
    if (activeSatCount > 0)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                drawPipeLayout, 0, 1, &descSet, 0, nullptr);
        // C12 follow-up #33: FRAGMENT added (sat_point.frag now reads screenSizePx for cloud
        // occlusion) — must match drawPipeLayout's push constant range exactly.
        vkCmdPushConstants(cmd, drawPipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDraw(cmd, activeSatCount, 1, 0, 0);
    }

    // ── Pass 3: background stars (additive blending) ──────────────────────────
    if (starCount > 0 && starPipeline != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, starPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                starPipeLayout, 0, 1, &starDescSet, 0, nullptr);
        // FRAGMENT added (session 30 bug fix): star_point.frag now reads screenSizePx for cloud
        // occlusion — must match starPipeLayout's push constant range exactly.
        vkCmdPushConstants(cmd, starPipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDraw(cmd, starCount, 1, 0, 0);
    }

    // ── Pass 3.5: planets (additive blending, same pipeline/shaders as stars) ──────────────
    // Reuses starPipeline/starPipeLayout unchanged — only the descriptor set (planetDescSet,
    // bound to planetBuf instead of starBuf) differs. noTwinkle=1 for this draw only: real
    // planets are small resolved discs and don't atmospheric-scintillate the way point-source
    // stars do (see SatDrawPC's noTwinkle comment and star_point.vert's gating of it). pc is
    // otherwise identical to the star draw above; mutating it here doesn't affect that already-
    // recorded vkCmdDraw.
    if (showPlanets && planetDescSet != VK_NULL_HANDLE && starPipeline != VK_NULL_HANDLE)
    {
        pc.noTwinkle = 1.0f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, starPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                starPipeLayout, 0, 1, &planetDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, starPipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDraw(cmd, kPlanetCount, 1, 0, 0);
    }

    // ── Pass 4: flare/corona composite (flare architecture overhaul) ──────────
    // Additive fullscreen triangle sampling the blurred/streaked buffer recordCompute() built
    // this frame — replaces the old per-pixel flareEntries loop in sat_sky.frag. Deliberately
    // last: lands over terrain/ocean/clouds/satellites/stars, under the UI (drawn afterward by
    // App), matching where the deleted flareAccum add used to land (post-tonemap).
    {
        // S3 (RELEASE_v1_1_PLAN.md): same darkness-scaled gain as the blur pass in recordCompute()
        // (kept as a local duplicate — recordDraw() and recordCompute() are separate functions, and
        // sunDirENU is a member updated earlier this same frame in both).
        const float kFlareDayFloor = 0.35f;
        float flareDarkness = glm::clamp(-sunDirENU.w * 5.0f, 0.0f, 1.0f);
        float flareEyeAdaptGain = glm::mix(kFlareDayFloor, 1.0f, flareDarkness);

        FlareCompositePC cpc{};
        cpc.gain = flareGlowGain * flareEyeAdaptGain;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, flareCompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                flareCompositePipeLayout, 0, 1, &flareCompositeDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, flareCompositePipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(cpc), &cpc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // ── Pass 5: long-exposure trail composite ──────────────────────────────────
    // Additive fullscreen triangle sampling trailAccumImg (decayed + splatted this frame in
    // recordCompute()). Order relative to the flare composite above doesn't matter — both are
    // ONE/ONE additive blending, which is commutative.
    if (trailEnabled)
    {
        TrailCompositePC tcpc{};
        tcpc.gain = trailCompositeGain;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trailCompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                trailCompositePipeLayout, 0, 1, &trailCompositeDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, trailCompositePipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(tcpc), &tcpc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
}

// ─── setAudio ─────────────────────────────────────────────────────────────────
// Called by App after both sim and audio are initialised.
// Configures the music playlist and stores the pointer for buildUI UI sounds.
void SatelliteSim::setAudio(AudioSystem *audio)
{
    audio_ = audio;
    if (!audio_)
        return;

    audio_->addTrack("assets/sound/music/gravity_wave.mp3");
    audio_->addTrack("assets/sound/music/fuse.mp3");
    audio_->startMusic();
    // Apply any volumes loaded from settings.json before the audio system was ready.
    audio_->setMasterVolume(masterVol_);
    audio_->setMusicVolume(musicVol_);
    audio_->setSfxVolume(sfxVol_);
}

// ─── cleanup ──────────────────────────────────────────────────────────────────
void SatelliteSim::cleanup(VkDevice device)
{
    // UC6: join any in-flight screenshot encode thread before anything below (including this
    // object's own eventual destruction) can happen — it captures `this` for the result-handoff
    // members, so letting it outlive the object would be a use-after-free.
    if (screenshotThread.joinable())
        screenshotThread.join();

    saveSettings();

    // NEW-3: reaching this point IS the clean-exit signal — remove the sentinel so the NEXT
    // launch doesn't think this run crashed. Best-effort; a failed delete just means the next
    // launch conservatively (and harmlessly) treats this run as unclean too.
    {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::path(userDataDir_) / "session.lock", ec);
    }

    // ── Orbit pipeline ─────────────────────────────────────────────────────────
    vkDestroyPipeline(device, orbitPipeline, nullptr);
    vkDestroyPipelineLayout(device, orbitPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, orbitDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, orbitDescLayout, nullptr);
    // ── Cloud march pipeline (C15-perf) ────────────────────────────────────────
    vkDestroyPipeline(device, cloudMarchPipeline, nullptr);
    vkDestroyPipelineLayout(device, cloudMarchPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, cloudMarchDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, cloudMarchDescLayout, nullptr);
    // ── Scene depth pipeline (pipeline unification) ────────────────────────────
    vkDestroyPipeline(device, sceneDepthPipeline, nullptr);
    vkDestroyPipelineLayout(device, sceneDepthPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, sceneDepthDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, sceneDepthDescLayout, nullptr);
    vkDestroySampler(device, sceneDepthSampler, nullptr);
    vkDestroyImageView(device, sceneDepthView, nullptr);
    vkDestroyImage(device, sceneDepthImg, nullptr);
    vkFreeMemory(device, sceneDepthMem, nullptr);
    // ── Beam self-march pipeline (2026-08-09, replaces beam_cloud_block.comp) ──
    vkDestroyPipeline(device, beamSelfMarchPipeline, nullptr);
    vkDestroyPipelineLayout(device, beamSelfMarchPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, beamSelfMarchDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, beamSelfMarchDescLayout, nullptr);
    // ── Flare + draw + sky pipelines ───────────────────────────────────────────
    vkDestroyPipeline(device, compPipeline, nullptr);
    vkDestroyPipeline(device, skyBgPipeline, nullptr);
    vkDestroyPipeline(device, skyBgMinimalPipeline, nullptr);
    destroySkyLowResResources(device);
    vkDestroyPipeline(device, drawPipeline, nullptr);
    vkDestroyPipelineLayout(device, compPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, skyBgPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, drawPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyDescriptorPool(device, skyDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, skyDescLayout, nullptr);
    // ── Flare/corona render-to-texture pipeline (flare architecture overhaul) ──
    vkDestroyPipeline(device, flareSourcePipeline, nullptr);
    vkDestroyPipelineLayout(device, flareSourcePipeLayout, nullptr);
    vkDestroyPipeline(device, flareBlurPipeline, nullptr);
    vkDestroyPipelineLayout(device, flareBlurPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, flareBlurDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, flareBlurDescLayout, nullptr);
    vkDestroyPipeline(device, flareCompositePipeline, nullptr);
    vkDestroyPipelineLayout(device, flareCompositePipeLayout, nullptr);
    vkDestroyDescriptorPool(device, flareCompositeDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, flareCompositeDescLayout, nullptr);
    destroyFlareResources(device);
    if (flareSampler)
    {
        vkDestroySampler(device, flareSampler, nullptr);
        flareSampler = VK_NULL_HANDLE;
    }
    // ── Long-exposure trail pipeline ──
    vkDestroyPipeline(device, trailSatPipeline, nullptr);
    vkDestroyPipeline(device, trailStarPipeline, nullptr);
    vkDestroyPipeline(device, trailFadePipeline, nullptr);
    vkDestroyPipelineLayout(device, trailFadePipeLayout, nullptr);
    vkDestroyDescriptorPool(device, trailFadeDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, trailFadeDescLayout, nullptr);
    vkDestroyPipeline(device, trailCompositePipeline, nullptr);
    vkDestroyPipelineLayout(device, trailCompositePipeLayout, nullptr);
    vkDestroyDescriptorPool(device, trailCompositeDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, trailCompositeDescLayout, nullptr);
    destroyTrailResources(device);
    vkUnmapMemory(device, glowMem);
    if (noiseSampler)
    {
        vkDestroySampler(device, noiseSampler, nullptr);
        noiseSampler = VK_NULL_HANDLE;
    }
    if (noiseTexView)
    {
        vkDestroyImageView(device, noiseTexView, nullptr);
        noiseTexView = VK_NULL_HANDLE;
    }
    if (noiseTex)
    {
        vkDestroyImage(device, noiseTex, nullptr);
        noiseTex = VK_NULL_HANDLE;
    }
    if (noiseTexMem)
    {
        vkFreeMemory(device, noiseTexMem, nullptr);
        noiseTexMem = VK_NULL_HANDLE;
    }
    if (moonSampler)
    {
        vkDestroySampler(device, moonSampler, nullptr);
        moonSampler = VK_NULL_HANDLE;
    }
    if (moonTexView)
    {
        vkDestroyImageView(device, moonTexView, nullptr);
        moonTexView = VK_NULL_HANDLE;
    }
    if (moonTex)
    {
        vkDestroyImage(device, moonTex, nullptr);
        moonTex = VK_NULL_HANDLE;
    }
    if (moonTexMem)
    {
        vkFreeMemory(device, moonTexMem, nullptr);
        moonTexMem = VK_NULL_HANDLE;
    }
    if (earthDaySampler)
    {
        vkDestroySampler(device, earthDaySampler, nullptr);
        earthDaySampler = VK_NULL_HANDLE;
    }
    if (earthDayView)
    {
        vkDestroyImageView(device, earthDayView, nullptr);
        earthDayView = VK_NULL_HANDLE;
    }
    if (earthDayImg)
    {
        vkDestroyImage(device, earthDayImg, nullptr);
        earthDayImg = VK_NULL_HANDLE;
    }
    if (earthDayMem)
    {
        vkFreeMemory(device, earthDayMem, nullptr);
        earthDayMem = VK_NULL_HANDLE;
    }
    if (milkyWaySampler)
    {
        vkDestroySampler(device, milkyWaySampler, nullptr);
        milkyWaySampler = VK_NULL_HANDLE;
    }
    if (milkyWayView)
    {
        vkDestroyImageView(device, milkyWayView, nullptr);
        milkyWayView = VK_NULL_HANDLE;
    }
    if (milkyWayImg)
    {
        vkDestroyImage(device, milkyWayImg, nullptr);
        milkyWayImg = VK_NULL_HANDLE;
    }
    if (milkyWayMem)
    {
        vkFreeMemory(device, milkyWayMem, nullptr);
        milkyWayMem = VK_NULL_HANDLE;
    }
    if (earthNightSampler)
    {
        vkDestroySampler(device, earthNightSampler, nullptr);
        earthNightSampler = VK_NULL_HANDLE;
    }
    if (earthNightView)
    {
        vkDestroyImageView(device, earthNightView, nullptr);
        earthNightView = VK_NULL_HANDLE;
    }
    if (earthNightImg)
    {
        vkDestroyImage(device, earthNightImg, nullptr);
        earthNightImg = VK_NULL_HANDLE;
    }
    if (earthNightMem)
    {
        vkFreeMemory(device, earthNightMem, nullptr);
        earthNightMem = VK_NULL_HANDLE;
    }
    if (cityDayDetailSampler)
    {
        vkDestroySampler(device, cityDayDetailSampler, nullptr);
        cityDayDetailSampler = VK_NULL_HANDLE;
    }
    if (cityDayDetailView)
    {
        vkDestroyImageView(device, cityDayDetailView, nullptr);
        cityDayDetailView = VK_NULL_HANDLE;
    }
    if (cityDayDetailImg)
    {
        vkDestroyImage(device, cityDayDetailImg, nullptr);
        cityDayDetailImg = VK_NULL_HANDLE;
    }
    if (cityDayDetailMem)
    {
        vkFreeMemory(device, cityDayDetailMem, nullptr);
        cityDayDetailMem = VK_NULL_HANDLE;
    }
    if (cityNightDetailSampler)
    {
        vkDestroySampler(device, cityNightDetailSampler, nullptr);
        cityNightDetailSampler = VK_NULL_HANDLE;
    }
    if (cityNightDetailView)
    {
        vkDestroyImageView(device, cityNightDetailView, nullptr);
        cityNightDetailView = VK_NULL_HANDLE;
    }
    if (cityNightDetailImg)
    {
        vkDestroyImage(device, cityNightDetailImg, nullptr);
        cityNightDetailImg = VK_NULL_HANDLE;
    }
    if (cityNightDetailMem)
    {
        vkFreeMemory(device, cityNightDetailMem, nullptr);
        cityNightDetailMem = VK_NULL_HANDLE;
    }
    if (earthElevSampler)
    {
        vkDestroySampler(device, earthElevSampler, nullptr);
        earthElevSampler = VK_NULL_HANDLE;
    }
    if (earthElevView)
    {
        vkDestroyImageView(device, earthElevView, nullptr);
        earthElevView = VK_NULL_HANDLE;
    }
    if (earthElevImg)
    {
        vkDestroyImage(device, earthElevImg, nullptr);
        earthElevImg = VK_NULL_HANDLE;
    }
    if (earthElevMem)
    {
        vkFreeMemory(device, earthElevMem, nullptr);
        earthElevMem = VK_NULL_HANDLE;
    }
    if (earthSpecSampler)
    {
        vkDestroySampler(device, earthSpecSampler, nullptr);
        earthSpecSampler = VK_NULL_HANDLE;
    }
    if (earthSpecView)
    {
        vkDestroyImageView(device, earthSpecView, nullptr);
        earthSpecView = VK_NULL_HANDLE;
    }
    if (earthSpecImg)
    {
        vkDestroyImage(device, earthSpecImg, nullptr);
        earthSpecImg = VK_NULL_HANDLE;
    }
    if (earthSpecMem)
    {
        vkFreeMemory(device, earthSpecMem, nullptr);
        earthSpecMem = VK_NULL_HANDLE;
    }
    if (earthCloudsSampler)
    {
        vkDestroySampler(device, earthCloudsSampler, nullptr);
        earthCloudsSampler = VK_NULL_HANDLE;
    }
    if (earthCloudsView)
    {
        vkDestroyImageView(device, earthCloudsView, nullptr);
        earthCloudsView = VK_NULL_HANDLE;
    }
    if (earthCloudsImg)
    {
        vkDestroyImage(device, earthCloudsImg, nullptr);
        earthCloudsImg = VK_NULL_HANDLE;
    }
    if (earthCloudsMem)
    {
        vkFreeMemory(device, earthCloudsMem, nullptr);
        earthCloudsMem = VK_NULL_HANDLE;
    }
    if (cloudNoiseSampler)
    {
        vkDestroySampler(device, cloudNoiseSampler, nullptr);
        cloudNoiseSampler = VK_NULL_HANDLE;
    }
    if (cloudNoiseView)
    {
        vkDestroyImageView(device, cloudNoiseView, nullptr);
        cloudNoiseView = VK_NULL_HANDLE;
    }
    if (cloudNoiseImg)
    {
        vkDestroyImage(device, cloudNoiseImg, nullptr);
        cloudNoiseImg = VK_NULL_HANDLE;
    }
    if (cloudNoiseMem)
    {
        vkFreeMemory(device, cloudNoiseMem, nullptr);
        cloudNoiseMem = VK_NULL_HANDLE;
    }
    if (cloudWarpNoiseSampler)
    {
        vkDestroySampler(device, cloudWarpNoiseSampler, nullptr);
        cloudWarpNoiseSampler = VK_NULL_HANDLE;
    }
    if (cloudWarpNoiseView)
    {
        vkDestroyImageView(device, cloudWarpNoiseView, nullptr);
        cloudWarpNoiseView = VK_NULL_HANDLE;
    }
    if (cloudWarpNoiseImg)
    {
        vkDestroyImage(device, cloudWarpNoiseImg, nullptr);
        cloudWarpNoiseImg = VK_NULL_HANDLE;
    }
    if (cloudWarpNoiseMem)
    {
        vkFreeMemory(device, cloudWarpNoiseMem, nullptr);
        cloudWarpNoiseMem = VK_NULL_HANDLE;
    }
    if (auroraNoiseSampler)
    {
        vkDestroySampler(device, auroraNoiseSampler, nullptr);
        auroraNoiseSampler = VK_NULL_HANDLE;
    }
    if (auroraNoiseView)
    {
        vkDestroyImageView(device, auroraNoiseView, nullptr);
        auroraNoiseView = VK_NULL_HANDLE;
    }
    if (auroraNoiseImg)
    {
        vkDestroyImage(device, auroraNoiseImg, nullptr);
        auroraNoiseImg = VK_NULL_HANDLE;
    }
    if (auroraNoiseMem)
    {
        vkFreeMemory(device, auroraNoiseMem, nullptr);
        auroraNoiseMem = VK_NULL_HANDLE;
    }
    if (cloudParamsBuf)
    {
        vkDestroyBuffer(device, cloudParamsBuf, nullptr);
        cloudParamsBuf = VK_NULL_HANDLE;
    }
    if (cloudParamsMem)
    {
        vkFreeMemory(device, cloudParamsMem, nullptr);
        cloudParamsMem = VK_NULL_HANDLE;
    }
    if (cloudMarchSampler)
    {
        vkDestroySampler(device, cloudMarchSampler, nullptr);
        cloudMarchSampler = VK_NULL_HANDLE;
    }
    if (cloudMarchTargetAView)
    {
        vkDestroyImageView(device, cloudMarchTargetAView, nullptr);
        cloudMarchTargetAView = VK_NULL_HANDLE;
    }
    if (cloudMarchTargetAImg)
    {
        vkDestroyImage(device, cloudMarchTargetAImg, nullptr);
        cloudMarchTargetAImg = VK_NULL_HANDLE;
    }
    if (cloudMarchTargetAMem)
    {
        vkFreeMemory(device, cloudMarchTargetAMem, nullptr);
        cloudMarchTargetAMem = VK_NULL_HANDLE;
    }
    if (cloudMarchTargetBView)
    {
        vkDestroyImageView(device, cloudMarchTargetBView, nullptr);
        cloudMarchTargetBView = VK_NULL_HANDLE;
    }
    if (cloudMarchTargetBImg)
    {
        vkDestroyImage(device, cloudMarchTargetBImg, nullptr);
        cloudMarchTargetBImg = VK_NULL_HANDLE;
    }
    if (cloudMarchTargetBMem)
    {
        vkFreeMemory(device, cloudMarchTargetBMem, nullptr);
        cloudMarchTargetBMem = VK_NULL_HANDLE;
    }
    vkDestroyBuffer(device, glowBuf, nullptr);
    vkFreeMemory(device, glowMem, nullptr);
    vkDestroyBuffer(device, oceanGlintBuf, nullptr);
    vkFreeMemory(device, oceanGlintMem, nullptr);
    if (pickedVisibleMapped)
        vkUnmapMemory(device, pickedVisibleMem);
    vkDestroyBuffer(device, pickedVisibleBuf, nullptr);
    vkFreeMemory(device, pickedVisibleMem, nullptr);
    // satInputBuf is now device-local (no host mapping to release).
    vkDestroyBuffer(device, satInputBuf, nullptr);
    vkFreeMemory(device, satInputMem, nullptr);
    vkDestroyBuffer(device, satVisibleBuf, nullptr);
    vkFreeMemory(device, satVisibleMem, nullptr);
    if (lightDomeMapped)
        vkUnmapMemory(device, lightDomeMem);
    vkDestroyBuffer(device, lightDomeBuf, nullptr);
    vkFreeMemory(device, lightDomeMem, nullptr);
    vkDestroyBuffer(device, satOrbitBuf, nullptr);
    vkFreeMemory(device, satOrbitMem, nullptr);
    if (reflectorTargetsECEFMapped)
        vkUnmapMemory(device, reflectorTargetsECEFMem);
    vkDestroyBuffer(device, reflectorTargetsECEFBuf, nullptr);
    vkFreeMemory(device, reflectorTargetsECEFMem, nullptr);
    // beamCloudBlockBuf destroy removed 2026-08-09 — that buffer no longer exists (see its
    // creation-site comment, createBuffers()).
    if (reflectBeamsMapped)
        vkUnmapMemory(device, reflectBeamsMem);
    vkDestroyBuffer(device, reflectBeamsBuf, nullptr);
    vkFreeMemory(device, reflectBeamsMem, nullptr);
    if (beamCloudLightMapped)
        vkUnmapMemory(device, beamCloudLightMem);
    vkDestroyBuffer(device, beamCloudLightBuf, nullptr);
    vkFreeMemory(device, beamCloudLightMem, nullptr);
    if (groundBeamsMapped)
        vkUnmapMemory(device, groundBeamsMem);
    vkDestroyBuffer(device, groundBeamsBuf, nullptr);
    vkFreeMemory(device, groundBeamsMem, nullptr);
    if (beamGlowDomeMapped)
        vkUnmapMemory(device, beamGlowDomeMem);
    vkDestroyBuffer(device, beamGlowDomeBuf, nullptr);
    vkFreeMemory(device, beamGlowDomeMem, nullptr);

    vkDestroyPipeline(device, starPipeline, nullptr);
    vkDestroyPipelineLayout(device, starPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, starDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, starDescLayout, nullptr);
    if (starMapped)
        vkUnmapMemory(device, starMem);
    vkDestroyBuffer(device, starBuf, nullptr);
    vkFreeMemory(device, starMem, nullptr);

    // Planets: own descriptor pool only (planetDescSet freed with it); pipeline/pipeline-layout/
    // desc-layout are starPipeline/starPipeLayout/starDescLayout, already destroyed above.
    vkDestroyDescriptorPool(device, planetDescPool, nullptr);
    if (planetMapped)
        vkUnmapMemory(device, planetMem);
    vkDestroyBuffer(device, planetBuf, nullptr);
    vkFreeMemory(device, planetMem, nullptr);

    // UC6: screenshot staging buffer, if a capture happened this run (recreated per-capture, so
    // this is the only place it's torn down for real).
    if (screenshotStagingBuf != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, screenshotStagingBuf, nullptr);
        vkFreeMemory(device, screenshotStagingMem, nullptr);
    }

    if (win)
        glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

// ─── updateIntroCinematic ───────────────────────────────────────────────────
// UC3: drives the intro's fixed beat sheet (kIntroKeyframes, SatelliteSim.h). obsDir/obsFacing's
// tangent basis is captured once (introEastEF/introNorthEF) since the intro never changes lat/lon
// after its one-time init below — only altitude, look elevation/FOV, and a facing-azimuth
// rotation change, so azDeg(t) can be turned into obsFacing directly every frame instead of
// accumulating incremental rotations.
void SatelliteSim::updateIntroCinematic(float dt)
{
    if (!introBasisValid)
    {
        // Lock the intro's starting location/orientation to the fixed vantage
        // (kIntroObserverLatDeg/LonDeg, SatelliteSim.h) rather than wherever obsDir/camera
        // happen to already be — otherwise a replay from a different in-game location would
        // play the whole cinematic somewhere else, and the beat sheet's altitude/az/el values
        // (tuned against this exact spot) would no longer line up with what's actually below.
        float lat = glm::radians(kIntroObserverLatDeg);
        float lon = glm::radians(kIntroObserverLonDeg);
        obsDir = {cosf(lat) * cosf(lon), cosf(lat) * sinf(lon), sinf(lat)};
        obsFacing = {-sinf(lat) * cosf(lon), -sinf(lat) * sinf(lon), cosf(lat)};
        obsLatDeg = kIntroObserverLatDeg;
        obsLonDeg = kIntroObserverLonDeg;
        obsHeightOffset = 0.0f;
        camera.azDeg = kIntroStartAzDeg;
        camera.elDeg = kIntroStartElDeg;
        camera.fovYDeg = kIntroStartFovDeg;

        // "Ensure we are at 1x speed on start": the intro is a fixed, tuned-at-1x cinematic —
        // force this regardless of whatever timeScaleIdx/pause/direction the player last saved
        // or left the sim in (a replay in particular runs on live state, not a fresh boot).
        timeScaleIdx = 0;
        timePaused = false;
        timeDir = 1.0f;

        // Cloud drift is not just a speed — cloudPhase is fmod(cloudDriftRate * simTime, 2pi),
        // and the intro always starts at the same fixed epoch, so this rate alone decides WHERE
        // the cloud noise pattern sits over the California vantage on frame 1. Later cloud-noise
        // changes moved that pattern so the intro opened socked in under overcast, hiding the
        // satellites the whole cinematic is framed around. Pinned to the rate that puts a clear
        // patch overhead. Forced here (not just changed as the compiled default) for the same
        // reason as timeScaleIdx above: settings.json would otherwise restore the player's own
        // saved rate and put the overcast back.
        cloudDriftRate = kIntroCloudDriftRate;

        float sL = obsDir.z;
        float cLH = sqrtf(obsDir.x * obsDir.x + obsDir.y * obsDir.y);
        float inv = (cLH > 1e-7f) ? 1.0f / cLH : 0.0f;
        float cLn = obsDir.x * inv, sLn = obsDir.y * inv;
        introEastEF = {-sLn, cLn, 0.0f};
        introNorthEF = {-sL * cLn, -sL * sLn, cLH};
        introBasisValid = true;
    }

    introElapsed += dt;
    float tEnd = kIntroKeyframes[kIntroKeyframeCount - 1].t;
    float tClamped = std::min(introElapsed, tEnd);

    int i = 0;
    while (i < kIntroKeyframeCount - 2 && kIntroKeyframes[i + 1].t <= tClamped)
        ++i;

    // Follow-up: per-segment smoothstep easing (old code) has zero velocity at EVERY waypoint,
    // not just the first and last — so the camera visibly decelerates to a stop and re-accelerates
    // at every single beat boundary, which read as a stutter/stop-start rather than one continuous
    // move. Replaced with a Catmull-Rom/cubic-Hermite spline through all the keyframes: the
    // tangent at each interior key is estimated from its two neighbors (time-weighted, since beats
    // aren't evenly spaced), so velocity carries through a waypoint instead of resetting there.
    // Endpoints fall back to the one-sided neighbor difference, which happens to already be ~0 for
    // this beat sheet (beats 0-1 and the final hold beats share identical values), so the start and
    // end still ease naturally without a special case.
    auto hermite = [&](float IntroKeyframe::*field) -> float
    {
        float p0 = kIntroKeyframes[i].*field;
        float p1 = kIntroKeyframes[i + 1].*field;
        float t0 = kIntroKeyframes[i].t;
        float t1 = kIntroKeyframes[i + 1].t;
        float segDt = t1 - t0;
        float m0 = (i == 0)
                       ? (p1 - p0) / segDt
                       : (kIntroKeyframes[i + 1].*field - kIntroKeyframes[i - 1].*field) /
                             (kIntroKeyframes[i + 1].t - kIntroKeyframes[i - 1].t);
        float m1 = (i + 1 == kIntroKeyframeCount - 1)
                       ? (p1 - p0) / segDt
                       : (kIntroKeyframes[i + 2].*field - kIntroKeyframes[i].*field) /
                             (kIntroKeyframes[i + 2].t - kIntroKeyframes[i].t);
        float u = (segDt > 0.0f) ? glm::clamp((tClamped - t0) / segDt, 0.0f, 1.0f) : 1.0f;
        float u2 = u * u, u3 = u2 * u;
        float h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
        float h10 = u3 - 2.0f * u2 + u;
        float h01 = -2.0f * u3 + 3.0f * u2;
        float h11 = u3 - u2;
        return h00 * p0 + h10 * (m0 * segDt) + h01 * p1 + h11 * (m1 * segDt);
    };

    // UC3 follow-up: once the controls-hint beat is reached, WASD/Q-E become live (see
    // recordCompute) — the camera has already arrived at its final framing there, so simply
    // stopping the forced overwrite is enough; nothing further needs blending in.
    bool controlsLive = introCaptionIndex >= kIntroControlsIndex;
    if (!controlsLive)
    {
        obsHeightOffset = std::max(0.0f, hermite(&IntroKeyframe::altM));
        camera.elDeg = hermite(&IntroKeyframe::elDeg);
        camera.fovYDeg = hermite(&IntroKeyframe::fovDeg);
        float azDegMixed = hermite(&IntroKeyframe::azDeg);
        // Both must be set: camera.azDeg is what viewMatrix() actually renders with, while
        // obsFacing is the ground-movement tangent buildUI's post-intro block re-derives
        // camera.azDeg FROM — leaving either one stale caused a one-frame camera snap the moment
        // the intro handed off (camera.azDeg was previously never touched here, so the rendered
        // view didn't pan at all during playback and then jumped to match obsFacing on the very
        // first post-intro frame).
        camera.azDeg = azDegMixed;
        float az = glm::radians(azDegMixed);
        obsFacing = glm::normalize(cosf(az) * introNorthEF + sinf(az) * introEastEF);
    }

    // Track the most recently reached non-null caption (see IntroKeyframe's text field doc).
    for (int k = 0; k <= i; ++k)
        if (kIntroKeyframes[k].text)
            introCaptionIndex = k;

    // UC1 mechanism 2: accumulate GPU frame time across the camera-motion beats for the
    // end-of-intro preset promote/demote in finishIntro().
    if (introElapsed <= kIntroBenchEndT)
    {
        introBenchMsSum += gpuMsTotalSmoothed;
        ++introBenchFrames;
    }

    if (introElapsed >= tEnd)
        finishIntro(false);
}

// ─── finishIntro ─────────────────────────────────────────────────────────────
void SatelliteSim::finishIntro(bool wasSkipped)
{
    showIntro = false;
    introSkipped = wasSkipped;

    // UC1 mechanisms 2+3: only decide anything when the intro played to completion (a skip
    // means no representative frame-time average was collected) and never during crash recovery
    // (that launch already forced Planetarium for an unrelated reason — see init()).
    if (!wasSkipped && !crashRecoveryMode && !introIsReplay && introBenchFrames > 0)
    {
        float avgMs = introBenchMsSum / (float)introBenchFrames;
        constexpr float kTargetMs = 12.0f; // ~3 tiers of headroom, per RELEASE_v1_1_PLAN.md UC1
        GraphicsPreset newPreset = graphicsPreset;
        if (avgMs > kTargetMs * 1.5f)
        {
            if (graphicsPreset == GraphicsPreset::Medium)
                newPreset = GraphicsPreset::Low;
            else if (graphicsPreset == GraphicsPreset::Low)
                newPreset = GraphicsPreset::Planetarium;
            else if (graphicsPreset == GraphicsPreset::Planetarium)
                newPreset = GraphicsPreset::Potato;
        }
        else if (avgMs < kTargetMs * 0.4f)
        {
            if (graphicsPreset == GraphicsPreset::Potato)
                newPreset = GraphicsPreset::Planetarium;
            else if (graphicsPreset == GraphicsPreset::Low)
                newPreset = GraphicsPreset::Medium;
            else if (graphicsPreset == GraphicsPreset::Medium)
                newPreset = GraphicsPreset::High;
        }

        if (newPreset != graphicsPreset)
            applyGraphicsPreset(newPreset);

        // UC1 mechanism 3: always tell the user, never silently re-decide — this timer/text pair
        // is separate from crashRecoveryNoticeTimer since the two can in principle both be live.
        snprintf(graphicsAutoNoticeText, sizeof(graphicsAutoNoticeText),
                 "Graphics set to %s based on your hardware -- change in Settings > Display.",
                 kGraphicsPresetNames[(int)graphicsPreset]);
        graphicsAutoNoticeTimer = 8.0f;
    }
    introBenchMsSum = 0.0f;
    introBenchFrames = 0;
    introIsReplay = false;
}

// ─── recordScreenshotCopy ────────────────────────────────────────────────────
// UC6. See Simulation.h for the calling convention: App calls this once per frame, right after
// the render pass ends (image is back in PRESENT_SRC_KHR, holding the fully composited frame —
// or just the scene if wantsCleanScreenshot() suppressed ui.record() this frame), before the
// command buffer is ended.
void SatelliteSim::recordScreenshotCopy(VkCommandBuffer cmd, VulkanContext &ctx, VkImage image)
{
    if (!screenshotRequested)
        return;
    screenshotRequested = false; // consume — this is the one frame it's true for

    if (!ctx.screenshotSupported)
    {
        snprintf(screenshotToastText, sizeof(screenshotToastText),
                 "Screenshot not supported (GPU/driver lacks swapchain TRANSFER_SRC).");
        screenshotToastTimer = 4.0f;
        return;
    }

    screenshotW = ctx.swapExtent.width;
    screenshotH = ctx.swapExtent.height;
    screenshotFormat = ctx.swapFormat;
    VkDeviceSize bufSize = (VkDeviceSize)screenshotW * (VkDeviceSize)screenshotH * 4;

    // Fresh buffer per capture rather than tracking size across resizes between requests —
    // screenshots are rare (a user keypress), not a hot path worth caching for.
    if (screenshotStagingBuf != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(ctx.device, screenshotStagingBuf, nullptr);
        vkFreeMemory(ctx.device, screenshotStagingMem, nullptr);
    }
    ctx.createBuffer(bufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     screenshotStagingBuf, screenshotStagingMem);

    // PRESENT_SRC_KHR -> TRANSFER_SRC_OPTIMAL, copy, then back — the presentation engine still
    // needs to consume this same image right after this command buffer's submission.
    ctx.imageBarrier(cmd, image, VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {screenshotW, screenshotH, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, screenshotStagingBuf, 1, &region);

    ctx.imageBarrier(cmd, image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    screenshotCopyPending = true;
}

// ─── finalizeScreenshot ──────────────────────────────────────────────────────
// UC6. Called once per frame right after App waits on the frame fence — the same point
// ctx.resolveTimestamps() already reads back the previous frame's GPU-written data, so the copy
// recorded last frame's recordScreenshotCopy is guaranteed complete by the time this runs.
void SatelliteSim::finalizeScreenshot()
{
    if (!screenshotCopyPending)
        return;
    screenshotCopyPending = false;
    if (!ctx_)
        return;

    void *mapped = nullptr;
    if (vkMapMemory(ctx_->device, screenshotStagingMem, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped)
        return;

    // Copy out of the GPU-mapped memory into a plain heap buffer we own outright, then unmap
    // immediately — the mapped memory is only needed long enough for this memcpy, and holding it
    // open for the (much slower) encode below would block the NEXT capture from reusing/recreating
    // screenshotStagingBuf for no reason.
    size_t count = (size_t)screenshotW * (size_t)screenshotH;
    std::vector<uint8_t> pixels(count * 4);
    memcpy(pixels.data(), mapped, count * 4);
    vkUnmapMemory(ctx_->device, screenshotStagingMem);

    // Swizzle BGRA->RGBA (the swapchain format is almost certainly B8G8R8A8 — see
    // VulkanContext::createSwapchain's format preference — but read the real format rather than
    // assuming) and force alpha opaque: a color-attachment copy's alpha channel isn't meaningful
    // for a screenshot (this app's render pass never blends against destination alpha).
    bool isBgra = (screenshotFormat == VK_FORMAT_B8G8R8A8_SRGB || screenshotFormat == VK_FORMAT_B8G8R8A8_UNORM);
    for (size_t i = 0; i < count; ++i)
    {
        if (isBgra)
            std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
        pixels[i * 4 + 3] = 255;
    }

    // PNG encoding (stbi_write_png) is genuinely slow in an unoptimized Debug build — tens of
    // seconds at 1080p+ is normal for its unoptimized DEFLATE-style compressor — which reads as
    // "the game froze" when run synchronously on the main thread. Encode on a background thread
    // instead; it only touches the copied `pixels` buffer (not GPU/Vulkan state) and the
    // result-handoff members below, so nothing here needs to synchronize with the render loop.
    if (screenshotThread.joinable())
        screenshotThread.join(); // previous capture's thread — screenshotEncoding guarantees it's
                                 // already finished (or about to), so this never blocks noticeably
    screenshotEncoding = true;
    uint32_t w = screenshotW, h = screenshotH;
    std::string path = screenshotPath;
    screenshotThread = std::thread([this, pixels = std::move(pixels), w, h, path]() mutable
                                   {
        // Default compression level (8) is tuned for size over speed; screenshots don't need
        // that, and it's the dominant cost of the encode (see the comment above this thread is
        // spawned from). Only one of these threads ever runs at a time (join-before-start above),
        // so mutating this global is safe. Global, not thread-local — stb_image_write's own API.
        stbi_write_png_compression_level = 4;
        bool ok = stbi_write_png(path.c_str(), (int)w, (int)h, 4, pixels.data(), (int)w * 4) != 0;
        char buf[192];
        if (ok)
            snprintf(buf, sizeof(buf), "Saved %s", std::filesystem::path(path).filename().string().c_str());
        else
            snprintf(buf, sizeof(buf), "Screenshot failed to write.");
        {
            std::lock_guard<std::mutex> lock(screenshotResultMutex);
            screenshotResultText = buf;
        }
        screenshotResultReady.store(true);
        screenshotEncoding.store(false); });
}

// ─── requestScreenshot ────────────────────────────────────────────────────────
// UC6: shared by KB_SCREENSHOT and the left HUD panel's camera button (buildLeftHudPanel).
void SatelliteSim::requestScreenshot()
{
    if (screenshotEncoding.load() || screenshotCopyPending || screenshotRequested)
        return; // a previous capture is still in flight (copy pending, or already encoding) — drop this one
    // Build the output path now, at request time — not later at copy/encode time — so the
    // timestamp reflects the moment the shot was actually taken, not whenever the async copy
    // happens to be read back a frame later. Saved next to the exe (screenshots/, user-requested
    // — same directory userDataDir_ now resolves to by default anyway, but this path is
    // independent of that so screenshots always land next to the exe even on a read-only
    // install where userDataDir_ has fallen back to AppData), falling back to userDataDir_ only
    // if the exe dir genuinely can't be created/written.
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::path(exeDir_) / "screenshots";
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        fprintf(stderr, "[SatelliteSim] Couldn't create '%s' (%s); saving screenshots to the "
                        "user data directory instead.\n",
                dir.string().c_str(), ec.message().c_str());
        dir = std::filesystem::path(userDataDir_) / "screenshots";
        std::filesystem::create_directories(dir, ec);
    }
    std::time_t now = std::time(nullptr);
    struct tm *lt = localtime(&now);
    char nameBuf[64];
    if (lt)
        strftime(nameBuf, sizeof(nameBuf), "satlight_%Y-%m-%d_%H-%M-%S.png", lt);
    else
        snprintf(nameBuf, sizeof(nameBuf), "satlight_screenshot.png");
    screenshotPath = (dir / nameBuf).string();
    screenshotRequested = true;
}

// ─── onKey ────────────────────────────────────────────────────────────────────
// Shared event-action dispatch — see declaration in SatelliteSim.h for why this is split
// out of onKey (used by both the keyboard callback and pollGamepad's button edge-detect).
void SatelliteSim::dispatchKeyAction(int bindIdx)
{
    switch (bindIdx)
    {
    case KB_TOGGLE_UI:
        uiVisible = !uiVisible;
        break;
    case KB_PAUSE:
        timePaused = !timePaused;
        break;
    case KB_SLOWER:
        timeScaleIdx = std::max(0, timeScaleIdx - 1);
        break;
    case KB_FASTER:
        timeScaleIdx = std::min(kNumTimeScales - 1, timeScaleIdx + 1);
        break;
    case KB_REVERSE:
        toggleTimeDirection();
        break;
    case KB_CINEMATIC:
        // Mouse-drag-flavored feature (RMB capture is required to mean anything); gated
        // the same way regardless of whether the press came from a key or a gamepad button.
        if (camera.captured)
            cinematicMode = !cinematicMode;
        break;
    case KB_RESET_ELEV:
        obsHeightOffset = 0.0f;
        break;
    case KB_ZOOM_RESET:
        camera.fovYDeg = SkyCamera{}.fovYDeg; // reads the struct's own default rather than duplicating the literal
        break;
    case KB_SELECT_SAT:
    {
        // Center-of-screen equivalent of the mouse left-click pick in buildUI — same
        // pickPlanetAt/pickSatelliteAt priority and selectedSatIndex/selectedPlanetIndex path,
        // just at (screenW/2, screenH/2) instead of the cursor. ctx_ (set in init()) gives
        // swapExtent without needing a UIInput here.
        if (!ctx_)
            break;
        float w = (float)ctx_->swapExtent.width;
        float h = (float)ctx_->swapExtent.height;
        int planetHit = pickPlanetAt(w * 0.5f, h * 0.5f, w, h);
        if (planetHit >= 0)
        {
            if (planetHit != selectedPlanetIndex || selectedSatIndex >= 0)
            {
                selectedPlanetIndex = planetHit;
                selectedSatIndex = -1;
                formatSelectedPlanetInfo();
                if (audio_)
                    audio_->playSfx("assets/sound/ui/buttonclick.wav");
            }
        }
        else
        {
            int hit = pickSatelliteAt(w * 0.5f, h * 0.5f, w, h);
            if (hit != selectedSatIndex || selectedPlanetIndex >= 0)
            {
                selectedSatIndex = hit;
                selectedPlanetIndex = -1;
                formatSelectedSatInfo();
                if (hit >= 0 && audio_)
                    audio_->playSfx("assets/sound/ui/buttonclick.wav");
            }
        }
        break;
    }
    case KB_SCREENSHOT:
        requestScreenshot();
        break;
    case KB_TOGGLE_CURSOR:
        // UC5: effective activation also requires uiVisible && !showIntro (see pollGamepad's
        // cursorActive computation) — this just flips the player-facing on/off state.
        vCursorToggled = !vCursorToggled;
        break;
    case KB_TOGGLE_TRAILS:
        // Single consolidated control (session follow-up) — same action the HUD "TrailsBtn" icon
        // button performs. OFF hides the trail immediately (recordDraw()'s composite draw is
        // itself gated on trailEnabled); ON always starts from a blank buffer.
        trailEnabled = !trailEnabled;
        if (trailEnabled)
            trailClearPending = true;
        break;
    default:
        break; // KB_MOVE_BOOST/FINE, KB_RAISE_ELEV/LOWER_ELEV, KB_ZOOM_IN/OUT are held keys — polled directly.
    }
}

void SatelliteSim::onKey(GLFWwindow *w, int key, int action)
{
    win = w;
    if (action != GLFW_PRESS)
        return;

    lastInputWasGamepad = false; // UC4: any keypress means the player is at the keyboard

    // F11: toggle fullscreen. Checked before the showIntro early-return below (and unlike every
    // other key, not folded into the keybindings dispatch table) — a player stuck in a badly
    // scaled/positioned window needs to be able to fix that without first having to sit through
    // or skip the cinematic to reach it.
    if (key == GLFW_KEY_F11)
    {
        bool isFs = glfwGetWindowMonitor(win) != nullptr;
        if (!isFs)
        {
            glfwGetWindowPos(win, &windowedX, &windowedY);
            glfwGetWindowSize(win, &windowedW, &windowedH);
            GLFWmonitor *mon = glfwGetPrimaryMonitor();
            const GLFWvidmode *mode = glfwGetVideoMode(mon);
            glfwSetWindowMonitor(win, mon, 0, 0, mode->width, mode->height, mode->refreshRate);
        }
        else
        {
            glfwSetWindowMonitor(win, nullptr, windowedX, windowedY, windowedW, windowedH, 0);
        }
        return;
    }

    if (showIntro)
    {
        // A defined single skip key, not "any key": the intro used to be dismissed by literally
        // any keypress (or a click — see buildIntroOverlay), which meant an incidental tap or a
        // click into the window skipped it before anyone had a chance to see it. Space is the
        // one deliberate escape — chosen as a literal key rather than whatever KB_PAUSE is
        // currently bound to, since the intro isn't "pausing" anything and shouldn't move if the
        // player has rebound Pause/Resume elsewhere.
        if (key == GLFW_KEY_SPACE)
            finishIntro(true);
        return;
    }

    // If any binding is listening for a rebind (keyboard or gamepad), capture/cancel here.
    // Esc cancels either kind; a non-Esc key only assigns kb.key (a keyboard press can't
    // satisfy a listeningPad row — the user must press a controller button for that, handled
    // in pollGamepad — but it still consumes the event so it doesn't leak through to dispatch).
    for (auto &kb : keybindings)
    {
        if (!kb.listening && !kb.listeningPad)
            continue;
        if (key == GLFW_KEY_ESCAPE)
        {
            kb.listening = false;
            kb.listeningPad = false;
        }
        else if (kb.listening)
        {
            kb.key = key;
            kb.listening = false;
        }
        return; // consume the key event
    }

    // Dispatch via keybindings array
    for (size_t i = 0; i < keybindings.size(); ++i)
        if (key == keybindings[i].key)
            dispatchKeyAction((int)i);
}

// ─── onCursorPos ──────────────────────────────────────────────────────────────
void SatelliteSim::onCursorPos(GLFWwindow *w, double x, double y)
{
    win = w;
    if (firstMouse)
    {
        prevX = x;
        prevY = y;
        firstMouse = false;
    }
    dmx += (float)(x - prevX);
    dmy += (float)(y - prevY);
    prevX = x;
    prevY = y;
}

// ─── createBuffers ────────────────────────────────────────────────────────────
void SatelliteSim::createBuffers(VulkanContext &ctx)
{
    // satInputBuf: device-local. sat_orbit.comp writes each frame; sat_flare.comp reads.
    ctx.createBuffer(sizeof(GpuSatInput) * MAX_SATELLITES,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     satInputBuf, satInputMem);

    // satVisibleBuf: device-local. sat_flare.comp writes, vertex reads.
    ctx.createBuffer(sizeof(GpuSatVisible) * MAX_SATELLITES,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     satVisibleBuf, satVisibleMem);

    // lightDomeBuf: host-visible, updated each frame by updateLightPollutionDome().
    ctx.createBuffer(sizeof(float) * kNumLightSectors,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     lightDomeBuf, lightDomeMem);
    vkMapMemory(ctx.device, lightDomeMem, 0, sizeof(float) * kNumLightSectors, 0, &lightDomeMapped);
    memset(lightDomeMapped, 0, sizeof(float) * kNumLightSectors);

    // satOrbitBuf: device-local, uploaded once. sat_orbit.comp reads every frame.
    ctx.createBuffer(sizeof(GpuSatOrbit) * MAX_SATELLITES,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     satOrbitBuf, satOrbitMem);

    // mirrorNormalsBuf (persistent lock/slew state) and reflectorTargetsBuf (per-frame CPU-
    // compacted night-side buffer) were removed 2026-08-06 — sat_orbit.comp now derives
    // everything from reflectorTargetsECEFBuf below, a pure function of sim time each frame, with
    // no persisted GPU state. See CLAUDE.md's TargetedReflector section.

    // reflectorTargetsECEFBuf: host-visible + coherent, but written ONCE (right after
    // initConstellation() in init(), not every frame — see the member comment). Sized as
    // vec4 per target (xyz=unit ECEF dir, w=radius); consumed by sat_orbit.comp (TargetedReflector
    // target search — the 2026-08-06 reversibility rework). beam_cloud_block.comp, the buffer's
    // other former reader, was retired 2026-08-09 — beam_self_march.comp doesn't need this buffer
    // at all, since it reconstructs each beam's ECEF endpoints from ReflectBeamsBuf's own ENU
    // offsets instead (see that shader's header).
    VkDeviceSize reflECEFSize = sizeof(glm::vec4) * kNumReflectorTargets;
    ctx.createBuffer(reflECEFSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     reflectorTargetsECEFBuf, reflectorTargetsECEFMem);
    vkMapMemory(ctx.device, reflectorTargetsECEFMem, 0, reflECEFSize, 0, &reflectorTargetsECEFMapped);
    memset(reflectorTargetsECEFMapped, 0, reflECEFSize);

    // beamCloudBlockBuf (beam_cloud_block.comp's own output buffer) retired 2026-08-09 —
    // beam_self_march.comp writes directly into reflectBeamsBuf below, no intermediate buffer.

    // reflectBeamsBuf: HOST_VISIBLE|HOST_COHERENT (same reasoning as glowBuf: single frame in
    // flight, so the previous frame's atomicMax writes from sat_orbit.comp are safely readable
    // by the CPU at the top of recordCompute, used for the active-beam-count/nearest-distance
    // diagnostic). Zeroed every frame via vkCmdFillBuffer, same as before this change.
    ctx.createBuffer(sizeof(GpuReflectBeams),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     reflectBeamsBuf, reflectBeamsMem);
    vkMapMemory(ctx.device, reflectBeamsMem, 0, sizeof(GpuReflectBeams), 0, &reflectBeamsMapped);
    memset(reflectBeamsMapped, 0, sizeof(GpuReflectBeams));

    // beamCloudLightBuf: HOST_VISIBLE|HOST_COHERENT, written wholesale by the CPU every frame in
    // recordCompute() (a full memcpy of a freshly-built GpuBeamCloudLights — see its own comment
    // for the 2026-08-09 design) — no vkCmdFillBuffer zero-fill needed, since each frame's write
    // already fully overwrites the struct including its own count.
    ctx.createBuffer(sizeof(GpuBeamCloudLights),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     beamCloudLightBuf, beamCloudLightMem);
    vkMapMemory(ctx.device, beamCloudLightMem, 0, sizeof(GpuBeamCloudLights), 0, &beamCloudLightMapped);
    memset(beamCloudLightMapped, 0, sizeof(GpuBeamCloudLights));

    // groundBeamsBuf (perf follow-up): HOST_VISIBLE|HOST_COHERENT, written wholesale by the CPU
    // every frame in recordCompute() (full memcpy of a freshly-built GpuGroundBeams) — no
    // vkCmdFillBuffer zero-fill needed.
    ctx.createBuffer(sizeof(GpuGroundBeams),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     groundBeamsBuf, groundBeamsMem);
    vkMapMemory(ctx.device, groundBeamsMem, 0, sizeof(GpuGroundBeams), 0, &groundBeamsMapped);
    memset(groundBeamsMapped, 0, sizeof(GpuGroundBeams));

    // beamGlowDomeBuf (C12 follow-up #31): HOST_VISIBLE|HOST_COHERENT, same reasoning as
    // reflectBeamsBuf — written by sat_orbit.comp (atomicMax per sector), zeroed every frame via
    // vkCmdFillBuffer, and safely readable by the CPU (one-frame-stale) for updateStars().
    ctx.createBuffer(sizeof(float) * kNumBeamGlowSectors,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     beamGlowDomeBuf, beamGlowDomeMem);
    vkMapMemory(ctx.device, beamGlowDomeMem, 0, sizeof(float) * kNumBeamGlowSectors, 0, &beamGlowDomeMapped);
    memset(beamGlowDomeMapped, 0, sizeof(float) * kNumBeamGlowSectors);
}

// ─── createDescriptors ────────────────────────────────────────────────────────
void SatelliteSim::createDescriptors(VulkanContext &ctx)
{
    VkDescriptorSetLayoutBinding bindings[9] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // glowBuf: atomic writes from flare shader
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // lightDomeBuf: host-visible, CPU-written
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // beamGlowDomeBuf: C12 follow-up #31
    // C12 follow-up #33: cloud occlusion for satellite/flare points — sat_point.frag reads these,
    // same underlying views/samplers already bound into skyDescSet (bindings 10/11 there). Also
    // read by flare_source.frag (flare architecture overhaul) — same layout, VERTEX not needed.
    bindings[5] = {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // cloudTargetA
    bindings[6] = {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // cloudTargetB
    // Flare architecture overhaul: flare_source.frag needs terrain occlusion too (this render pass
    // has no shared hardware depth buffer of its own to test against, unlike sat_point.frag).
    bindings[7] = {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // sceneDepthTex
    // Ocean-glint list (GpuOceanGlintBuf) — written here by sat_flare.comp, read by sat_sky.frag
    // via its OWN descriptor set (skyDescSet binding 20) pointed at the same underlying buffer,
    // same split as glowBuf's binding 2 here vs skyDescSet binding 0.
    bindings[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // oceanGlintBuf

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 9;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &descLayout);

    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 2;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &descSet);

    VkDescriptorBufferInfo inpInfo{satInputBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo visInfo{satVisibleBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo glowInfo{glowBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo domeInfo{lightDomeBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo beamDomeInfo{beamGlowDomeBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo cloudAInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cloudBInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo sceneDepthInfo{sceneDepthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo oceanGlintInfo{oceanGlintBuf, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[9] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inpInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visInfo, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &glowInfo, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &domeInfo, nullptr};
    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &beamDomeInfo, nullptr};
    writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudAInfo, nullptr, nullptr};
    writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudBInfo, nullptr, nullptr};
    writes[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr};
    writes[8] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &oceanGlintInfo, nullptr};
    vkUpdateDescriptorSets(ctx.device, 9, writes, 0, nullptr);
}

// ─── createComputePipeline ────────────────────────────────────────────────────
void SatelliteSim::createComputePipeline(VulkanContext &ctx)
{
    VkShaderModule mod = ctx.loadShader("shaders/sat_flare.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SatFlarePC)};

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &descLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &compPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = compPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &compPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create compute pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createOrbitDescriptors ───────────────────────────────────────────────────
// Descriptor set for sat_orbit.comp:
//   binding 0  satOrbitBuf       (readonly  SSBO)
//   binding 1  satInputBuf       (write     SSBO — same buffer that sat_flare.comp reads)
//   binding 2  reflectorTargetsECEFBuf (readonly SSBO — static; replaces the old
//                                       mirrorNormalsBuf/reflectorTargetsBuf pair as of the
//                                       2026-08-06 reversibility rework)
//   binding 3  reflectBeamsBuf   (readwrite SSBO — capped atomic-append beam list, C12)
//   binding 4  beamGlowDomeBuf  (readwrite SSBO — 16-sector beam sky-glow dome, C12 follow-up #31)
// Binding 5 (beamCloudBlockBuf, per-target cloud occlusion, C12 follow-up #33) removed 2026-08-09
// — beam_self_march.comp now writes blockAltM/blockOpacity directly, per beam, in its own later
// dispatch; this shader no longer reads or writes those two fields at all.
void SatelliteSim::createOrbitDescriptors(VulkanContext &ctx)
{
    VkDescriptorSetLayoutBinding bindings[5] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 5;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &orbitDescLayout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &orbitDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = orbitDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &orbitDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &orbitDescSet);

    VkDescriptorBufferInfo orbitInfo{satOrbitBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo inputInfo{satInputBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo targetEcefInfo{reflectorTargetsECEFBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo beamInfo{reflectBeamsBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo beamDomeInfo{beamGlowDomeBuf, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[5] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, orbitDescSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &orbitInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, orbitDescSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inputInfo, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, orbitDescSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &targetEcefInfo, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, orbitDescSet, 3, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &beamInfo, nullptr};
    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, orbitDescSet, 4, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &beamDomeInfo, nullptr};
    vkUpdateDescriptorSets(ctx.device, 5, writes, 0, nullptr);
}

// ─── createOrbitPipeline ──────────────────────────────────────────────────────
void SatelliteSim::createOrbitPipeline(VulkanContext &ctx)
{
    VkShaderModule mod = ctx.loadShader("shaders/sat_orbit.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SatOrbitPC)};

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &orbitDescLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &orbitPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = orbitPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &orbitPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create orbit compute pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── uploadSatOrbits ─────────────────────────────────────────────────────────
// Bakes GpuSatOrbit data from satOrbits+satTypes and uploads to satOrbitBuf.
// Stores orbitEpochDay/Sec = current simTime so deltaT resets to 0.
// Auto-called from recordCompute when |simDayJ2000-orbitEpochDay| >= kOrbitRebakeDays.
void SatelliteSim::uploadSatOrbits(VulkanContext &ctx)
{
    if (satOrbits.empty())
        return;

    // A rebake re-writes what every dispatch index means, but TargetedReflector selection and
    // orientation no longer persist any per-index GPU state (2026-08-06 reversibility rework) —
    // nothing here needs invalidating; the next frame's sat_orbit.comp dispatch derives everything
    // fresh from the rebaked orbits and the current sim time regardless.

    orbitEpochDay = simDayJ2000;
    orbitEpochSec = simSecInDay;
    const double orbitEpochT0 = (double)orbitEpochDay * 86400.0 + orbitEpochSec;
    // SSO RAAN is anchored at sim-start, so bake only the precession since then.
    const double t_start = (double)simInitDayJ2000 * 86400.0 + simInitSecInDay;

    std::vector<GpuSatOrbit> gpuOrbits(activeSatCount);
    for (uint32_t ci = 0; ci < (uint32_t)constellations.size(); ++ci)
    {
        const ConstellationConfig &c = constellations[ci];
        for (uint32_t i = c.orbitStart; i < c.orbitStart + c.orbitCount; ++i)
        {
            if (i >= activeSatCount)
                break;
            const SatOrbit &src = satOrbits[i];
            const SatelliteType &type = satTypes[src.typeIdx];
            GpuSatOrbit &dst = gpuOrbits[i];
            dst.raan = src.alignTerminator
                           ? (float)fmod((double)src.raan + kSSOPrecRate * (orbitEpochT0 - t_start),
                                         glm::two_pi<double>())
                           : src.raan;
            dst.u0 = (float)fmod((double)src.u0 + (double)src.meanMot * orbitEpochT0,
                                 glm::two_pi<double>());
            dst.R_sat = src.R_sat;
            dst.meanMot = src.meanMot;
            dst.cosI = src.cosI;
            dst.sinI = src.sinI;
            dst.cosRaan = src.cosRaan;
            dst.sinRaan = src.sinRaan;

            dst.tumbleRate = src.tumbleRate;
            dst.tumblePhase = (float)fmod((double)src.tumblePhase +
                                              (double)src.tumbleRate * orbitEpochT0,
                                          glm::two_pi<double>());
            dst.alignTerminator = src.alignTerminator ? 1.0f : 0.0f;
            dst.tumbleAxisX = src.tumbleAxis.x;
            dst.tumbleAxisY = src.tumbleAxis.y;
            dst.tumbleAxisZ = src.tumbleAxis.z;

            dst.primaryAttitude = (uint32_t)type.primary.attitude;
            dst.secondaryAttitude = (uint32_t)type.secondary.attitude;

            dst.baseColorR = type.baseColor.r;
            dst.baseColorG = type.baseColor.g;
            dst.baseColorB = type.baseColor.b;
            dst.crossSection = sqrtf(type.crossSectionM2 / 10.0f);
            dst.specExp0 = type.primary.specExp;
            dst.specExp1 = type.secondary.specExp;
            dst.w1 = type.secondary.weight;
            dst.diffuse = type.diffuse;
            dst.mirrorFrac = type.mirrorFrac;
            dst.constIdx = src.constIdx;
            dst.pad0 = dst.pad1 = 0;
        }
    }

    VkDeviceSize bufSize = activeSatCount * sizeof(GpuSatOrbit);
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
    void *mapped;
    vkMapMemory(ctx.device, stagingMem, 0, bufSize, 0, &mapped);
    memcpy(mapped, gpuOrbits.data(), bufSize);
    vkUnmapMemory(ctx.device, stagingMem);

    VkCommandBuffer cmd = ctx.beginOneTimeCommands();
    VkBufferCopy region{0, 0, bufSize};
    vkCmdCopyBuffer(cmd, staging, satOrbitBuf, 1, &region);
    ctx.endOneTimeCommands(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
}

// ─── createCloudNoisePipeline ────────────────────────────────────────────────
// Allocates the 128³ RGBA cloud noise volume, dispatches cloud_noise.comp to bake
// Perlin-Worley + Worley channels into it in one shot, transitions to
// SHADER_READ_ONLY_OPTIMAL, then destroys the bake pipeline/descriptor set.
// Must be called before createGlowResources() so cloudNoiseView+Sampler exist
// when the sky descriptor writes are assembled.
void SatelliteSim::createCloudNoisePipeline(VulkanContext &ctx)
{
    static constexpr uint32_t kSz = 192;

    // ── Create 192³ RGBA8 3D image (storage + sampled) ───────────────────────
    ctx.createImage(kSz, kSz, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    cloudNoiseImg, cloudNoiseMem,
                    1,    // mipLevels
                    kSz); // depth > 1 → createImage produces VK_IMAGE_TYPE_3D

    // 3D image view (layerCount=1; depth lives in extent, not array layers)
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = cloudNoiseImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &cloudNoiseView);
    }

    // Trilinear REPEAT sampler — noise must tile seamlessly across UVW [0,1)
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod = 1.0f;
        vkCreateSampler(ctx.device, &sci, nullptr, &cloudNoiseSampler);
    }

    // ── Bake descriptor set layout: single STORAGE_IMAGE binding 0 ───────────
    VkDescriptorSetLayout bakeDescLayout = VK_NULL_HANDLE;
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 1;
        li.pBindings = &b;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &bakeDescLayout);
    }

    VkDescriptorPool bakePool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        pi.maxSets = 1;
        vkCreateDescriptorPool(ctx.device, &pi, nullptr, &bakePool);
    }

    VkDescriptorSet bakeSet = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = bakePool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &bakeDescLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &bakeSet);
    }

    // ── Pipeline layout + compute pipeline ────────────────────────────────────
    VkPipelineLayout bakePipeLayout = VK_NULL_HANDLE;
    {
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &bakeDescLayout;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &bakePipeLayout);
    }

    VkPipeline bakePipeline = VK_NULL_HANDLE;
    {
        VkShaderModule mod = ctx.loadShader("shaders/cloud_noise.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = bakePipeLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &bakePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create cloud_noise bake pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    // ── One-shot bake: barrier → bind → dispatch → barrier ───────────────────
    {
        auto cmd = ctx.beginOneTimeCommands();

        // Transition UNDEFINED → GENERAL so the compute shader can write
        ctx.imageBarrier(cmd, cloudNoiseImg,
                         0, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Descriptor write: STORAGE_IMAGE pointing at cloudNoiseView in GENERAL layout
        VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, cloudNoiseView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = bakeSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bakePipeLayout, 0, 1, &bakeSet, 0, nullptr);
        vkCmdDispatch(cmd, 24, 24, 24); // 24×8=192 threads per axis

        // Transition GENERAL → SHADER_READ_ONLY_OPTIMAL for use by sat_sky.frag
        ctx.imageBarrier(cmd, cloudNoiseImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        ctx.endOneTimeCommands(cmd);
    }

    // ── Destroy bake-only Vulkan objects (view+sampler are kept as members) ───
    vkDestroyPipeline(ctx.device, bakePipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, bakePipeLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, bakePool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, bakeDescLayout, nullptr);
}

// ─── createCloudWarpNoisePipeline ───────────────────────────────────────────────
// Allocates the 192³ RGB cloud/cirrus domain-warp noise volume, dispatches cloud_warp_noise.comp
// to bake it in one shot, transitions to SHADER_READ_ONLY_OPTIMAL, then destroys the bake
// pipeline/descriptor set. Structurally identical to createCloudNoisePipeline above (same
// one-shot-bake pattern) — see cloud_warp_noise.comp for what's baked, why, and the tiling/
// repetition trade-off it deliberately accepts. Resolution matches createCloudNoisePipeline's
// own 192³ exactly — see that file's header comment for why (fixing visible interpolation
// faceting at an earlier, smaller 128³/single-octave attempt). Must run before
// createCloudMarchDescriptors() so cloudWarpNoiseView+Sampler exist when that descriptor set's
// writes are assembled.
void SatelliteSim::createCloudWarpNoisePipeline(VulkanContext &ctx)
{
    static constexpr uint32_t kSz = 192;

    // ── Create 192³ RGBA8 3D image (storage + sampled) ───────────────────────
    ctx.createImage(kSz, kSz, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    cloudWarpNoiseImg, cloudWarpNoiseMem,
                    1,    // mipLevels
                    kSz); // depth > 1 → createImage produces VK_IMAGE_TYPE_3D

    // 3D image view (layerCount=1; depth lives in extent, not array layers)
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = cloudWarpNoiseImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &cloudWarpNoiseView);
    }

    // Trilinear REPEAT sampler — the bake tiles seamlessly across UVW [0,1), and the continuous
    // wind-drift term in cloudWarpOffset relies on hardware wrap to scroll through it smoothly.
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod = 1.0f;
        vkCreateSampler(ctx.device, &sci, nullptr, &cloudWarpNoiseSampler);
    }

    // ── Bake descriptor set layout: single STORAGE_IMAGE binding 0 ───────────
    VkDescriptorSetLayout bakeDescLayout = VK_NULL_HANDLE;
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 1;
        li.pBindings = &b;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &bakeDescLayout);
    }

    VkDescriptorPool bakePool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        pi.maxSets = 1;
        vkCreateDescriptorPool(ctx.device, &pi, nullptr, &bakePool);
    }

    VkDescriptorSet bakeSet = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = bakePool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &bakeDescLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &bakeSet);
    }

    // ── Pipeline layout + compute pipeline ────────────────────────────────────
    VkPipelineLayout bakePipeLayout = VK_NULL_HANDLE;
    {
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &bakeDescLayout;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &bakePipeLayout);
    }

    VkPipeline bakePipeline = VK_NULL_HANDLE;
    {
        VkShaderModule mod = ctx.loadShader("shaders/cloud_warp_noise.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = bakePipeLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &bakePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create cloud_warp_noise bake pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    // ── One-shot bake: barrier → bind → dispatch → barrier ───────────────────
    {
        auto cmd = ctx.beginOneTimeCommands();

        // Transition UNDEFINED → GENERAL so the compute shader can write
        ctx.imageBarrier(cmd, cloudWarpNoiseImg,
                         0, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Descriptor write: STORAGE_IMAGE pointing at cloudWarpNoiseView in GENERAL layout
        VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, cloudWarpNoiseView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = bakeSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bakePipeLayout, 0, 1, &bakeSet, 0, nullptr);
        vkCmdDispatch(cmd, kSz / 8, kSz / 8, kSz / 8); // local_size (8,8,8)

        // Transition GENERAL → SHADER_READ_ONLY_OPTIMAL for use by cloud_march.comp
        ctx.imageBarrier(cmd, cloudWarpNoiseImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        ctx.endOneTimeCommands(cmd);
    }

    // ── Destroy bake-only Vulkan objects (view+sampler are kept as members) ───
    vkDestroyPipeline(ctx.device, bakePipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, bakePipeLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, bakePool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, bakeDescLayout, nullptr);
}

// ─── createAuroraNoisePipeline ─────────────────────────────────────────────────
// Allocates the 1024x16x256 RGBA aurora noise volume, dispatches aurora_noise.comp to bake the
// curtain fold base (R) + column-window colA/colB (G/B) into it in one shot, transitions to
// SHADER_READ_ONLY_OPTIMAL, then destroys the bake pipeline/descriptor set. Structurally identical
// to createCloudNoisePipeline above (same one-shot-bake pattern) — see aurora_noise.comp for what's
// baked and why. Must run before createGlowResources() so auroraNoiseView+Sampler exist when the
// sky descriptor writes are assembled.
void SatelliteSim::createAuroraNoisePipeline(VulkanContext &ctx)
{
    static constexpr uint32_t kSzU = 1024, kSzV = 16, kSzW = 256;

    // ── Create 1024x16x256 RGBA8 3D image (storage + sampled) ────────────────
    ctx.createImage(kSzU, kSzV, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    auroraNoiseImg, auroraNoiseMem,
                    1,     // mipLevels
                    kSzW); // depth > 1 → createImage produces VK_IMAGE_TYPE_3D

    // 3D image view (layerCount=1; depth lives in extent, not array layers)
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = auroraNoiseImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &auroraNoiseView);
    }

    // U (azimuth) wraps — REPEAT. V (altitude) and W (colatitude) never wrap at runtime — sat_sky.
    // frag always clamps both to their baked ranges — so CLAMP_TO_EDGE there matches how they're
    // actually sampled and avoids any bake-edge value leaking in from the opposite side.
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 1.0f;
        vkCreateSampler(ctx.device, &sci, nullptr, &auroraNoiseSampler);
    }

    // ── Bake descriptor set layout: single STORAGE_IMAGE binding 0 ───────────
    VkDescriptorSetLayout bakeDescLayout = VK_NULL_HANDLE;
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 1;
        li.pBindings = &b;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &bakeDescLayout);
    }

    VkDescriptorPool bakePool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        pi.maxSets = 1;
        vkCreateDescriptorPool(ctx.device, &pi, nullptr, &bakePool);
    }

    VkDescriptorSet bakeSet = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = bakePool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &bakeDescLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &bakeSet);
    }

    // ── Pipeline layout + compute pipeline ────────────────────────────────────
    VkPipelineLayout bakePipeLayout = VK_NULL_HANDLE;
    {
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &bakeDescLayout;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &bakePipeLayout);
    }

    VkPipeline bakePipeline = VK_NULL_HANDLE;
    {
        VkShaderModule mod = ctx.loadShader("shaders/aurora_noise.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = bakePipeLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &bakePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create aurora_noise bake pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    // ── One-shot bake: barrier → bind → dispatch → barrier ───────────────────
    {
        auto cmd = ctx.beginOneTimeCommands();

        // Transition UNDEFINED → GENERAL so the compute shader can write
        ctx.imageBarrier(cmd, auroraNoiseImg,
                         0, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Descriptor write: STORAGE_IMAGE pointing at auroraNoiseView in GENERAL layout
        VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, auroraNoiseView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = bakeSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bakePipeLayout, 0, 1, &bakeSet, 0, nullptr);
        vkCmdDispatch(cmd, kSzU / 8, kSzV / 8, kSzW / 8); // local_size (8,8,8)

        // Transition GENERAL → SHADER_READ_ONLY_OPTIMAL for use by sat_sky.frag
        ctx.imageBarrier(cmd, auroraNoiseImg,
                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        ctx.endOneTimeCommands(cmd);
    }

    // ── Destroy bake-only Vulkan objects (view+sampler are kept as members) ───
    vkDestroyPipeline(ctx.device, bakePipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, bakePipeLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, bakePool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, bakeDescLayout, nullptr);
}

// ─── createCloudMarchResources ────────────────────────────────────────────────
// Two half-resolution RGBA16F storage+sampled images written by cloud_march.comp each frame.
// Unlike cloudNoiseImg's bake-once volume, these are swapchain-size-dependent — recreated in
// onResize (see there for the matching descriptor-set patch).
void SatelliteSim::createCloudMarchResources(VulkanContext &ctx)
{
    uint32_t w = (ctx.swapExtent.width + 1) / 2;
    uint32_t h = (ctx.swapExtent.height + 1) / 2;

    auto createTarget = [&](VkImage &img, VkDeviceMemory &mem, VkImageView &view)
    {
        ctx.createImage(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        img, mem);
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &view);
    };
    createTarget(cloudMarchTargetAImg, cloudMarchTargetAMem, cloudMarchTargetAView);
    createTarget(cloudMarchTargetBImg, cloudMarchTargetBMem, cloudMarchTargetBView);

    // Bilinear, clamp-to-edge — resolution-independent, created once and reused across resizes.
    if (!cloudMarchSampler)
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        vkCreateSampler(ctx.device, &sci, nullptr, &cloudMarchSampler);
    }

    // Leave both images in SHADER_READ_ONLY_OPTIMAL — the layout createGlowResources' descriptor
    // write (bindings 10/11) declares and the layout recordCompute's per-frame pre-dispatch
    // barrier expects to transition FROM (see recordCompute: SHADER_READ_ONLY_OPTIMAL → GENERAL
    // before each dispatch, back to SHADER_READ_ONLY_OPTIMAL after — this call only establishes
    // that starting state once, for both init and after an onResize recreation).
    auto cmd = ctx.beginOneTimeCommands();
    ctx.imageBarrier(cmd, cloudMarchTargetAImg, 0, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    ctx.imageBarrier(cmd, cloudMarchTargetBImg, 0, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    ctx.endOneTimeCommands(cmd);
}

// ─── createCloudMarchDescriptors ──────────────────────────────────────────────
// Descriptor set for cloud_march.comp:
//   binding 0  earthCloudsTex  (sampler2D)
//   binding 1  cloudNoiseTex   (sampler3D)
//   binding 2  earthNightTex   (sampler2D)
//   binding 3  noiseTex        (sampler2D)
//   binding 4  CloudParams UBO (same underlying buffer as skyDescSet binding 9)
//   binding 5  targetA (storage image, rgba16f)
//   binding 6  targetB (storage image, rgba16f)
//   binding 9  cloudWarpNoiseTex (sampler3D) — baked domain-warp field, see cloud_warp_noise.comp
//   binding 10 reflectBeamsBuf  (readonly SSBO) — Reflect-Orbital beams; the visible pointing ray
//              reads this directly (main()'s per-pixel debug-ray loop)
//   binding 11 earthElevTex    (sampler2D) — needed by observerEffHeight() in main()
//   binding 12 earthSpecTex    (sampler2D) — land/ocean mask, same pair sat_sky.frag already binds
//   binding 14 beamCloudLightBuf (readonly SSBO) — 2026-08-09 (third design, see
//              GpuBeamCloudLights' own comment): small capped list of individual real-beam light
//              sources, fed forward into cloudMarchCS's per-sample loop the same way sun/moon
//              light the cloud — NOT the per-pixel screen-space glow that shipped briefly the same
//              day and produced visible ring artifacts (removed).
// Requires createGlowResources() to already have run (needs cloudParamsBuf, earthClouds/Night
// textures) — see init() ordering.
void SatelliteSim::createCloudMarchDescriptors(VulkanContext &ctx)
{
    VkDescriptorSetLayoutBinding bindings[15] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    // lightDomeBuf: same buffer as sat_flare.comp/sat_sky.frag's own read — needed now that the
    // aurora sky curtain march moved here (perf: folded into this half-res pass alongside clouds).
    bindings[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};   // aurora noise sampler3D
    bindings[9] = {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};   // cloud warp noise sampler3D
    bindings[10] = {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};         // reflectBeamsBuf
    bindings[11] = {11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // earthElevTex
    bindings[12] = {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // earthSpecTex
    // sceneDepthTex: written by scene_depth.comp earlier in the same recordCompute. Read 1:1 by
    // texelFetch — this set's dispatch grid and that image are the same half-swapExtent size.
    bindings[13] = {13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // sceneDepthTex
    bindings[14] = {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};         // beamCloudLightBuf

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 15;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &cloudMarchDescLayout);

    VkDescriptorPoolSize ps[4] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 4;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &cloudMarchDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = cloudMarchDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &cloudMarchDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &cloudMarchDescSet);

    VkDescriptorImageInfo cloudsInfo{earthCloudsSampler, earthCloudsView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo noise3DInfo{cloudNoiseSampler, cloudNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo nightInfo{earthNightSampler, earthNightView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo noiseInfo{noiseSampler, noiseTexView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo cloudParamsInfo{cloudParamsBuf, 0, sizeof(GpuCloudParams)};
    VkDescriptorImageInfo targetAInfo{VK_NULL_HANDLE, cloudMarchTargetAView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo targetBInfo{VK_NULL_HANDLE, cloudMarchTargetBView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo lightDomeInfo{lightDomeBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo auroraNoiseInfo{auroraNoiseSampler, auroraNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo warpNoiseInfo{cloudWarpNoiseSampler, cloudWarpNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo beamInfo{reflectBeamsBuf, 0, VK_WHOLE_SIZE};
    // Same fallback pattern as the sky descriptor set (SatelliteSim.cpp:3432-3435) — elevation
    // texture may have failed to load, fall back to the always-valid noise sampler so this
    // descriptor set is never left pointing at a null image.
    VkSampler elevSamplerFinal2 = earthElevSampler ? earthElevSampler : noiseSampler;
    VkImageView elevViewFinal2 = earthElevView ? earthElevView : noiseTexView;
    VkSampler specSamplerFinal2 = earthSpecSampler ? earthSpecSampler : noiseSampler;
    VkImageView specViewFinal2 = earthSpecView ? earthSpecView : noiseTexView;
    VkDescriptorImageInfo elevInfo{elevSamplerFinal2, elevViewFinal2, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo specInfo{specSamplerFinal2, specViewFinal2, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo sceneDepthInfo{sceneDepthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo beamCloudLightInfo{beamCloudLightBuf, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[15] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudsInfo, nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &noise3DInfo, nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &nightInfo, nullptr, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 3, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &noiseInfo, nullptr, nullptr};
    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 4, 0, 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &cloudParamsInfo, nullptr};
    writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 5, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &targetAInfo, nullptr, nullptr};
    writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 6, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &targetBInfo, nullptr, nullptr};
    writes[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 7, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDomeInfo, nullptr};
    writes[8] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 8, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &auroraNoiseInfo, nullptr, nullptr};
    writes[9] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 9, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &warpNoiseInfo, nullptr, nullptr};
    writes[10] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 10, 0, 1,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &beamInfo, nullptr};
    writes[11] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 11, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &elevInfo, nullptr, nullptr};
    writes[12] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 12, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &specInfo, nullptr, nullptr};
    writes[13] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 13, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr};
    writes[14] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cloudMarchDescSet, 14, 0, 1,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &beamCloudLightInfo, nullptr};
    vkUpdateDescriptorSets(ctx.device, 15, writes, 0, nullptr);
}

// ─── createCloudMarchPipeline ──────────────────────────────────────────────────
void SatelliteSim::createCloudMarchPipeline(VulkanContext &ctx)
{
    VkShaderModule mod = ctx.loadShader("shaders/cloud_march.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CloudMarchPC)};

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &cloudMarchDescLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &cloudMarchPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = cloudMarchPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &cloudMarchPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create cloud_march compute pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createSceneDepthResources ────────────────────────────────────────────────
// One half-resolution R32_SFLOAT storage+sampled image written by scene_depth.comp each frame,
// holding the linear distance to the first terrain/ocean surface along each view ray.
//
// Sized identically to cloudMarchTargetA/B — half of the SWAP extent, NOT of the render-scaled
// extent — so cloud_march.comp (which dispatches on exactly this grid) can read it with a plain
// texelFetch at its own gl_GlobalInvocationID, with no UV math to get wrong. Swapchain-size
// dependent, so recreated in onResize alongside those targets.
//
// A colour-aspect image rather than a real depth attachment, deliberately: that keeps
// ctx.imageBarrier (which hardcodes COLOR aspect / 1 mip / 1 layer) usable unchanged, and avoids
// the depth-format blit portability problem that made the render-scale path skip depth entirely.
void SatelliteSim::createSceneDepthResources(VulkanContext &ctx)
{
    uint32_t w = (ctx.swapExtent.width + 1) / 2;
    uint32_t h = (ctx.swapExtent.height + 1) / 2;

    ctx.createImage(w, h, VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    sceneDepthImg, sceneDepthMem);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = sceneDepthImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(ctx.device, &vci, nullptr, &sceneDepthView);

    // NEAREST, not LINEAR. Filtering a distance field across a terrain silhouette interpolates
    // between "ridge at 8 km" and "sky at 1e30", producing meaningless intermediate distances
    // along every skyline. Point sampling keeps every fetched value one the depth pass actually
    // wrote. Created once and reused across resizes (resolution-independent).
    if (!sceneDepthSampler)
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        vkCreateSampler(ctx.device, &sci, nullptr, &sceneDepthSampler);
    }

    // Clear to kNoSurfaceT (1e30 = "no terrain/ocean anywhere on any ray") ONCE here, then
    // establish SHADER_READ_ONLY_OPTIMAL — the layout the descriptor writes declare and that
    // recordCompute's per-frame pre-dispatch barrier transitions FROM. The clear matters because
    // knockout bit 1024 now skips the scene_depth dispatch (and its two per-frame barriers)
    // entirely on the CPU side rather than early-returning in the shader — a skipped dispatch
    // leaves whatever is here, so it must be a valid "nothing occludes" buffer. Same one-time-
    // setup role createCloudMarchResources' matching barriers play, for first init and onResize.
    auto cmd = ctx.beginOneTimeCommands();
    ctx.imageBarrier(cmd, sceneDepthImg, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkClearColorValue noSurface{};
    noSurface.float32[0] = noSurface.float32[1] = noSurface.float32[2] = noSurface.float32[3] = 1e30f;
    VkImageSubresourceRange fullColor{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, sceneDepthImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &noSurface, 1, &fullColor);
    ctx.imageBarrier(cmd, sceneDepthImg, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    ctx.endOneTimeCommands(cmd);
}

// ─── createSceneDepthDescriptors ──────────────────────────────────────────────
// Descriptor set for scene_depth.comp:
//   binding 0  earthElevTex (sampler2D)
//   binding 1  earthSpecTex (sampler2D)
//   binding 2  sceneDepth   (storage image, r32f)
void SatelliteSim::createSceneDepthDescriptors(VulkanContext &ctx)
{
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 3;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &sceneDepthDescLayout);

    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 2;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &sceneDepthDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = sceneDepthDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &sceneDepthDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &sceneDepthDescSet);

    // Same fallback pattern the sky and cloud-march sets use — the elevation/spec textures may
    // have failed to load, so fall back to the always-valid noise sampler rather than leaving a
    // descriptor pointing at a null image. With that fallback the depth pass reads noise as
    // "terrain", which is wrong but bounded; a null view is a device loss.
    VkSampler elevSamplerFinal = earthElevSampler ? earthElevSampler : noiseSampler;
    VkImageView elevViewFinal = earthElevView ? earthElevView : noiseTexView;
    VkSampler specSamplerFinal = earthSpecSampler ? earthSpecSampler : noiseSampler;
    VkImageView specViewFinal = earthSpecView ? earthSpecView : noiseTexView;
    VkDescriptorImageInfo elevInfo{elevSamplerFinal, elevViewFinal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo specInfo{specSamplerFinal, specViewFinal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depthInfo{VK_NULL_HANDLE, sceneDepthView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[3] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDepthDescSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &elevInfo, nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDepthDescSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &specInfo, nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDepthDescSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
}

// ─── createSceneDepthPipeline ─────────────────────────────────────────────────
void SatelliteSim::createSceneDepthPipeline(VulkanContext &ctx)
{
    VkShaderModule mod = ctx.loadShader("shaders/scene_depth.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SceneDepthPC)};

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &sceneDepthDescLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &sceneDepthPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = sceneDepthPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &sceneDepthPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create scene_depth compute pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// (createCloudShadowResources / Descriptors / Pipeline lived here — the 128x128 R16_SFLOAT grid,
//  its own 5-binding descriptor set, sampler, and compute pipeline. All deleted in the
//  pipeline-unification pass; cloud_march.comp::cloudGroundShadow now produces the same value
//  per pixel from the terrain hit point the scene-depth pass supplies, using bindings that pass
//  already had.)

// ─── createBeamSelfMarchDescriptors ────────────────────────────────────────────
// Descriptor set for beam_self_march.comp (2026-08-09, replaces beam_cloud_block.comp):
//   binding 0  reflectBeamsBuf (readwrite SSBO) — same buffer sat_orbit.comp/cloud_march.comp/
//                               sat_sky.frag all reference; this pass reads satENU/targetENU and
//                               overwrites blockAltM/blockOpacity in place, no separate output
//                               buffer needed (unlike beam_cloud_block.comp's own beamCloudBlockBuf)
//   binding 1  earthCloudsTex  (sampler2D)
//   binding 2  cloudNoiseTex   (sampler3D)
//   binding 3  cloudWarpNoiseTex (sampler3D)
//   binding 4  CloudParams UBO (same underlying buffer as skyDescSet binding 9)
// Same shape as beam_cloud_block.comp's own set otherwise (deliberately its own small descriptor
// set — see that shader's header, still applicable, for why this isn't built on any shared grid).
void SatelliteSim::createBeamSelfMarchDescriptors(VulkanContext &ctx)
{
    VkDescriptorSetLayoutBinding bindings[5] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 5;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &beamSelfMarchDescLayout);

    VkDescriptorPoolSize ps[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 3;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &beamSelfMarchDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = beamSelfMarchDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &beamSelfMarchDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &beamSelfMarchDescSet);

    VkDescriptorBufferInfo beamInfo{reflectBeamsBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo cloudsInfo{earthCloudsSampler, earthCloudsView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo noise3DInfo{cloudNoiseSampler, cloudNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo warpNoiseInfo{cloudWarpNoiseSampler, cloudWarpNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo cloudParamsInfo{cloudParamsBuf, 0, sizeof(GpuCloudParams)};

    VkWriteDescriptorSet writes[5] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, beamSelfMarchDescSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &beamInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, beamSelfMarchDescSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudsInfo, nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, beamSelfMarchDescSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &noise3DInfo, nullptr, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, beamSelfMarchDescSet, 3, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &warpNoiseInfo, nullptr, nullptr};
    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, beamSelfMarchDescSet, 4, 0, 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &cloudParamsInfo, nullptr};
    vkUpdateDescriptorSets(ctx.device, 5, writes, 0, nullptr);
}

// ─── createBeamSelfMarchPipeline ───────────────────────────────────────────────
void SatelliteSim::createBeamSelfMarchPipeline(VulkanContext &ctx)
{
    VkShaderModule mod = ctx.loadShader("shaders/beam_self_march.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BeamSelfMarchPC)};

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &beamSelfMarchDescLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &beamSelfMarchPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = beamSelfMarchPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &beamSelfMarchPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create beam_self_march compute pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createSkyBgPipeline ──────────────────────────────────────────────────────
// ─── createGlowResources ──────────────────────────────────────────────────────
// Allocates the host-visible SSBO that holds up to kMaxGlows bright-flare entries,
// and creates the descriptor set used by the sky background pipeline to read it.
void SatelliteSim::createGlowResources(VulkanContext &ctx)
{
    // ── SSBO: top-N glow entries written every frame ──────────────────────────
    VkDeviceSize bufSize = sizeof(GpuGlowBuf);
    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     glowBuf, glowMem);
    vkMapMemory(ctx.device, glowMem, 0, bufSize, 0, &glowMapped);
    memset(glowMapped, 0, bufSize);

    // ── Ocean-glint list (GpuOceanGlintBuf, flare architecture overhaul) ───────
    // Device-local (unlike glowBuf — nothing on the CPU ever reads this back), zeroed every frame
    // via vkCmdFillBuffer in recordCompute() (same idiom as glowBuf).
    ctx.createBuffer(sizeof(GpuOceanGlintBuf),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     oceanGlintBuf, oceanGlintMem);

    // ── Picked-satellite tracking buffer ───────────────────────────────────────
    // 32-byte host-visible mirror of the selected satellite's GpuSatVisible entry, written by a
    // tiny vkCmdCopyBuffer in recordCompute (only while a selection is active) and read back
    // one-frame-stale at the top of recordCompute — same idiom as glowBuf/peakMagnitude above.
    // Never bound as an SSBO, so TRANSFER_DST is the only usage it needs.
    ctx.createBuffer(sizeof(GpuSatVisible),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     pickedVisibleBuf, pickedVisibleMem);
    vkMapMemory(ctx.device, pickedVisibleMem, 0, sizeof(GpuSatVisible), 0, &pickedVisibleMapped);
    memset(pickedVisibleMapped, 0, sizeof(GpuSatVisible));

    // ── Noise texture: RGBA PNG for lens-flare angular corona variation ────────
    // Loaded from assets/noise/rgba_noise.png (tiled REPEAT sampler).
    // The sky shader samples it at angular coordinates around each flare source
    // to produce the irregular spiky corona shape (see lensFlare() in sat_sky.frag).
    {
        int w = 0, h = 0, ch = 0;
        stbi_uc *pixels = stbi_load("assets/noise/rgba_noise.png", &w, &h, &ch, 4);
        if (!pixels)
            throw std::runtime_error("SatelliteSim: failed to load assets/noise/rgba_noise.png");

        VkDeviceSize imgBytes = (VkDeviceSize)w * h * 4;

        // Staging buffer
        VkBuffer stageBuf;
        VkDeviceMemory stageMem;
        ctx.createBuffer(imgBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stageBuf, stageMem);
        void *mapped;
        vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
        memcpy(mapped, pixels, (size_t)imgBytes);
        vkUnmapMemory(ctx.device, stageMem);
        stbi_image_free(pixels);

        // Device image
        ctx.createImage((uint32_t)w, (uint32_t)h,
                        VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        noiseTex, noiseTexMem);

        // Upload via one-time command
        {
            auto cmd = ctx.beginOneTimeCommands();
            ctx.imageBarrier(cmd, noiseTex,
                             0, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, noiseTex,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.imageBarrier(cmd, noiseTex,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            ctx.endOneTimeCommands(cmd);
        }
        vkDestroyBuffer(ctx.device, stageBuf, nullptr);
        vkFreeMemory(ctx.device, stageMem, nullptr);

        // Image view
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = noiseTex;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &noiseTexView);

        // Sampler: REPEAT so the noise tiles seamlessly around the full angular range
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        vkCreateSampler(ctx.device, &sci, nullptr, &noiseSampler);
    }

    // ── Moon texture: near-side face disc image (binding 2) ──────────────────
    {
        int w = 0, h = 0, ch = 0;
        stbi_uc *pixels = stbi_load("assets/textures/full_moon.png", &w, &h, &ch, 4);
        if (!pixels)
            throw std::runtime_error("SatelliteSim: failed to load assets/textures/full_moon.png");

        VkDeviceSize imgBytes = (VkDeviceSize)w * h * 4;

        VkBuffer stageBuf;
        VkDeviceMemory stageMem;
        ctx.createBuffer(imgBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stageBuf, stageMem);
        void *mapped;
        vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
        memcpy(mapped, pixels, (size_t)imgBytes);
        vkUnmapMemory(ctx.device, stageMem);
        stbi_image_free(pixels);

        ctx.createImage((uint32_t)w, (uint32_t)h,
                        VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        moonTex, moonTexMem);

        {
            auto cmd = ctx.beginOneTimeCommands();
            ctx.imageBarrier(cmd, moonTex,
                             0, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, moonTex,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.imageBarrier(cmd, moonTex,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            ctx.endOneTimeCommands(cmd);
        }
        vkDestroyBuffer(ctx.device, stageBuf, nullptr);
        vkFreeMemory(ctx.device, stageMem, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = moonTex;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &moonTexView);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(ctx.device, &sci, nullptr, &moonSampler);
    }

    // ── Earth day texture (binding 3): 8K equirectangular colour map ─────────
    {
        int w = 0, h = 0, ch = 0;
        stbi_uc *pixels = stbi_load("assets/textures/8k_earth_daymap.jpg", &w, &h, &ch, 4);
        if (!pixels)
            throw std::runtime_error("SatelliteSim: failed to load assets/textures/8k_earth_daymap.jpg");

        earthDayMips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
        VkDeviceSize imgBytes = (VkDeviceSize)w * h * 4;

        VkBuffer stageBuf;
        VkDeviceMemory stageMem;
        ctx.createBuffer(imgBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stageBuf, stageMem);
        void *mapped;
        vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
        memcpy(mapped, pixels, (size_t)imgBytes);
        vkUnmapMemory(ctx.device, stageMem);
        stbi_image_free(pixels);

        ctx.createImage((uint32_t)w, (uint32_t)h, VK_FORMAT_R8G8B8A8_SRGB,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        earthDayImg, earthDayMem, earthDayMips);

        {
            auto cmd = ctx.beginOneTimeCommands();
            // Transition ALL mips to TRANSFER_DST_OPTIMAL for upload + blit
            VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            allMips.srcAccessMask = 0;
            allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.image = earthDayImg;
            allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthDayMips, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, earthDayImg,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.generateMipmaps(cmd, earthDayImg, VK_FORMAT_R8G8B8A8_SRGB,
                                (uint32_t)w, (uint32_t)h, earthDayMips);
            ctx.endOneTimeCommands(cmd);
        }
        vkDestroyBuffer(ctx.device, stageBuf, nullptr);
        vkFreeMemory(ctx.device, stageMem, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = earthDayImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthDayMips, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &earthDayView);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = (float)earthDayMips;
        vkCreateSampler(ctx.device, &sci, nullptr, &earthDaySampler);
    }

    // ── Milky Way skybox texture (binding 13): 8K equirectangular galactic panorama ──
    // Same load pattern as earthDay (SRGB, mipmapped). Sampled in sat_sky.frag against a
    // CPU-computed ENU->galactic direction; see the "Milky Way skybox basis" block in
    // updatePositions().
    {
        int w = 0, h = 0, ch = 0;
        stbi_uc *pixels = stbi_load("assets/textures/8k_stars_milky_way.jpg", &w, &h, &ch, 4);
        if (!pixels)
            throw std::runtime_error("SatelliteSim: failed to load assets/textures/8k_stars_milky_way.jpg");

        milkyWayMips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
        VkDeviceSize imgBytes = (VkDeviceSize)w * h * 4;

        VkBuffer stageBuf;
        VkDeviceMemory stageMem;
        ctx.createBuffer(imgBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stageBuf, stageMem);
        void *mapped;
        vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
        memcpy(mapped, pixels, (size_t)imgBytes);
        vkUnmapMemory(ctx.device, stageMem);
        stbi_image_free(pixels);

        ctx.createImage((uint32_t)w, (uint32_t)h, VK_FORMAT_R8G8B8A8_SRGB,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        milkyWayImg, milkyWayMem, milkyWayMips);

        {
            auto cmd = ctx.beginOneTimeCommands();
            VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            allMips.srcAccessMask = 0;
            allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.image = milkyWayImg;
            allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, milkyWayMips, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, milkyWayImg,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.generateMipmaps(cmd, milkyWayImg, VK_FORMAT_R8G8B8A8_SRGB,
                                (uint32_t)w, (uint32_t)h, milkyWayMips);
            ctx.endOneTimeCommands(cmd);
        }
        vkDestroyBuffer(ctx.device, stageBuf, nullptr);
        vkFreeMemory(ctx.device, stageMem, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = milkyWayImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, milkyWayMips, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &milkyWayView);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = (float)milkyWayMips;
        vkCreateSampler(ctx.device, &sci, nullptr, &milkyWaySampler);
    }

    // ── Earth night texture (binding 4): 8K equirectangular night-lights map ─
    {
        int w = 0, h = 0, ch = 0;
        stbi_uc *pixels = stbi_load("assets/textures/8k_earth_nightmap.jpg", &w, &h, &ch, 4);
        if (!pixels)
            throw std::runtime_error("SatelliteSim: failed to load assets/textures/8k_earth_nightmap.jpg");

        earthNightMips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
        VkDeviceSize imgBytes = (VkDeviceSize)w * h * 4;

        VkBuffer stageBuf;
        VkDeviceMemory stageMem;
        ctx.createBuffer(imgBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stageBuf, stageMem);
        void *mapped;
        vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
        memcpy(mapped, pixels, (size_t)imgBytes);
        vkUnmapMemory(ctx.device, stageMem);

        // CPU-side downsample to 2160×1080 (~18km/px, matches earthElevCpu) for the observer
        // light-pollution lookup — stores precomputed Rec.709 luminance, one byte per texel.
        // Box-filtered (average every source pixel in each cell), not nearest-neighbor picking
        // one corner pixel — the latter throws away ~93% of the source data per cell and bakes
        // real aliasing/moiré into the array before updateLightPollutionDome() ever samples it.
        earthNightCpuW = 2160;
        earthNightCpuH = 1080;
        earthNightCpu.resize((size_t)earthNightCpuW * earthNightCpuH);
        for (int cy = 0; cy < earthNightCpuH; ++cy)
        {
            int sy0 = cy * h / earthNightCpuH;
            int sy1 = std::max(sy0 + 1, (cy + 1) * h / earthNightCpuH);
            for (int cx = 0; cx < earthNightCpuW; ++cx)
            {
                int sx0 = cx * w / earthNightCpuW;
                int sx1 = std::max(sx0 + 1, (cx + 1) * w / earthNightCpuW);
                float sum = 0.0f;
                int count = 0;
                for (int sy = sy0; sy < sy1; ++sy)
                {
                    for (int sx = sx0; sx < sx1; ++sx)
                    {
                        const stbi_uc *px = &pixels[((size_t)sy * w + sx) * 4];
                        sum += 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
                        ++count;
                    }
                }
                earthNightCpu[cy * earthNightCpuW + cx] = (uint8_t)std::clamp(sum / (float)count, 0.0f, 255.0f);
            }
        }

        // Half-resolution box-blur (~37km/px) — see the member comment in SatelliteSim.h. One 2×2
        // averaging pass over the already-box-filtered earthNightCpu above.
        earthNightCpuBlurW = earthNightCpuW / 2;
        earthNightCpuBlurH = earthNightCpuH / 2;
        earthNightCpuBlur.resize((size_t)earthNightCpuBlurW * earthNightCpuBlurH);
        for (int by = 0; by < earthNightCpuBlurH; ++by)
        {
            for (int bx = 0; bx < earthNightCpuBlurW; ++bx)
            {
                int x0 = bx * 2, y0 = by * 2;
                int sum = earthNightCpu[y0 * earthNightCpuW + x0] + earthNightCpu[y0 * earthNightCpuW + x0 + 1] + earthNightCpu[(y0 + 1) * earthNightCpuW + x0] + earthNightCpu[(y0 + 1) * earthNightCpuW + x0 + 1];
                earthNightCpuBlur[by * earthNightCpuBlurW + bx] = (uint8_t)(sum / 4);
            }
        }
        stbi_image_free(pixels);

        ctx.createImage((uint32_t)w, (uint32_t)h, VK_FORMAT_R8G8B8A8_SRGB,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        earthNightImg, earthNightMem, earthNightMips);

        {
            auto cmd = ctx.beginOneTimeCommands();
            VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            allMips.srcAccessMask = 0;
            allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.image = earthNightImg;
            allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthNightMips, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, earthNightImg,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.generateMipmaps(cmd, earthNightImg, VK_FORMAT_R8G8B8A8_SRGB,
                                (uint32_t)w, (uint32_t)h, earthNightMips);
            ctx.endOneTimeCommands(cmd);
        }
        vkDestroyBuffer(ctx.device, stageBuf, nullptr);
        vkFreeMemory(ctx.device, stageMem, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = earthNightImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthNightMips, 0, 1};
        vkCreateImageView(ctx.device, &vci, nullptr, &earthNightView);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = (float)earthNightMips;
        vkCreateSampler(ctx.device, &sci, nullptr, &earthNightSampler);
    }

    // ── City day/night detail textures (bindings 14/15): small tileable maps blended onto
    // dayColor/nightColor near cities (see terrain block in sat_sky.frag).
    {
        struct
        {
            const char *path;
            VkImage *img;
            VkDeviceMemory *mem;
            VkImageView *view;
            VkSampler *sampler;
            uint32_t *mips;
        } detailTexes[2] = {
            {"assets/textures/city_day_detail.png", &cityDayDetailImg, &cityDayDetailMem,
             &cityDayDetailView, &cityDayDetailSampler, &cityDayDetailMips},
            {"assets/textures/city_night_detail.png", &cityNightDetailImg, &cityNightDetailMem,
             &cityNightDetailView, &cityNightDetailSampler, &cityNightDetailMips},
        };
        for (auto &t : detailTexes)
        {
            int w = 0, h = 0, ch = 0;
            stbi_uc *pixels = stbi_load(t.path, &w, &h, &ch, 4);
            if (!pixels)
                throw std::runtime_error(std::string("SatelliteSim: failed to load ") + t.path);

            *t.mips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
            VkDeviceSize imgBytes = (VkDeviceSize)w * h * 4;

            VkBuffer stageBuf;
            VkDeviceMemory stageMem;
            ctx.createBuffer(imgBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             stageBuf, stageMem);
            void *mapped;
            vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
            memcpy(mapped, pixels, (size_t)imgBytes);
            vkUnmapMemory(ctx.device, stageMem);
            stbi_image_free(pixels);

            ctx.createImage((uint32_t)w, (uint32_t)h, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            *t.img, *t.mem, *t.mips);

            {
                auto cmd = ctx.beginOneTimeCommands();
                VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                allMips.srcAccessMask = 0;
                allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                allMips.image = *t.img;
                allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, *t.mips, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
                vkCmdCopyBufferToImage(cmd, stageBuf, *t.img,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                ctx.generateMipmaps(cmd, *t.img, VK_FORMAT_R8G8B8A8_SRGB,
                                    (uint32_t)w, (uint32_t)h, *t.mips);
                ctx.endOneTimeCommands(cmd);
            }
            vkDestroyBuffer(ctx.device, stageBuf, nullptr);
            vkFreeMemory(ctx.device, stageMem, nullptr);

            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = *t.img;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R8G8B8A8_SRGB;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, *t.mips, 0, 1};
            vkCreateImageView(ctx.device, &vci, nullptr, t.view);

            // Tileable in both U and V (unlike the equirect Earth maps, which only repeat U).
            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.maxLod = (float)*t.mips;
            vkCreateSampler(ctx.device, &sci, nullptr, t.sampler);
        }
    }

    // ── Load earth elevation map (binding 5): 21600×10800 R8_UNORM , 0 = sea level
    {
        int w, h, ch;
        unsigned char *pixels = stbi_load("assets/textures/earth_elevation.png", &w, &h, &ch, 1);
        if (pixels)
        {
            earthElevMips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
            VkDeviceSize imgBytes = (VkDeviceSize)w * h * 1;

            VkBuffer stageBuf;
            VkDeviceMemory stageMem;
            ctx.createBuffer(imgBytes,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             stageBuf, stageMem);
            void *mapped;
            vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
            memcpy(mapped, pixels, imgBytes);
            vkUnmapMemory(ctx.device, stageMem);

            // Downsample to 2160×1080 (10:1 each axis, ~18 km/pixel) for CPU observer height
            earthElevCpuW = 2160;
            earthElevCpuH = 1080;
            earthElevCpu.resize((size_t)earthElevCpuW * earthElevCpuH);
            for (int cy = 0; cy < earthElevCpuH; ++cy)
            {
                for (int cx = 0; cx < earthElevCpuW; ++cx)
                {
                    int sx = std::min(cx * w / earthElevCpuW, w - 1);
                    int sy = std::min(cy * h / earthElevCpuH, h - 1);
                    earthElevCpu[cy * earthElevCpuW + cx] = pixels[sy * w + sx];
                }
            }
            stbi_image_free(pixels);

            ctx.createImage((uint32_t)w, (uint32_t)h,
                            VK_FORMAT_R8_UNORM,
                            VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            earthElevImg, earthElevMem, earthElevMips);

            VkCommandBuffer cmd = ctx.beginOneTimeCommands();
            VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            allMips.srcAccessMask = 0;
            allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.image = earthElevImg;
            allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthElevMips, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, earthElevImg,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.generateMipmaps(cmd, earthElevImg, VK_FORMAT_R8_UNORM,
                                (uint32_t)w, (uint32_t)h, earthElevMips);
            ctx.endOneTimeCommands(cmd);
            vkDestroyBuffer(ctx.device, stageBuf, nullptr);
            vkFreeMemory(ctx.device, stageMem, nullptr);

            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = earthElevImg;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R8_UNORM;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthElevMips, 0, 1};
            vkCreateImageView(ctx.device, &vci, nullptr, &earthElevView);

            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod = (float)earthElevMips;
            vkCreateSampler(ctx.device, &sci, nullptr, &earthElevSampler);
        }
        else
        {
            fprintf(stderr, "Warning: could not load earth_elevation terrain march disabled\n");
        }
    }

    // ── Load earth specular map (binding 6): 8K R8_UNORM ocean mask ──────────────
    {
        int w, h, ch;
        unsigned char *pixels = stbi_load("assets/textures/8k_earth_specular_map.png", &w, &h, &ch, 1);
        if (pixels)
        {
            earthSpecMips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
            VkDeviceSize imgBytes = (VkDeviceSize)w * h;

            VkBuffer stageBuf;
            VkDeviceMemory stageMem;
            ctx.createBuffer(imgBytes,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             stageBuf, stageMem);
            void *mapped;
            vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
            memcpy(mapped, pixels, (size_t)imgBytes);
            vkUnmapMemory(ctx.device, stageMem);
            stbi_image_free(pixels);

            ctx.createImage((uint32_t)w, (uint32_t)h,
                            VK_FORMAT_R8_UNORM,
                            VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            earthSpecImg, earthSpecMem, earthSpecMips);

            VkCommandBuffer cmd = ctx.beginOneTimeCommands();
            VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            allMips.srcAccessMask = 0;
            allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.image = earthSpecImg;
            allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthSpecMips, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, earthSpecImg,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.generateMipmaps(cmd, earthSpecImg, VK_FORMAT_R8_UNORM,
                                (uint32_t)w, (uint32_t)h, earthSpecMips);
            ctx.endOneTimeCommands(cmd);
            vkDestroyBuffer(ctx.device, stageBuf, nullptr);
            vkFreeMemory(ctx.device, stageMem, nullptr);

            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = earthSpecImg;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R8_UNORM;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthSpecMips, 0, 1};
            vkCreateImageView(ctx.device, &vci, nullptr, &earthSpecView);

            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod = (float)earthSpecMips;
            vkCreateSampler(ctx.device, &sci, nullptr, &earthSpecSampler);
        }
        else
        {
            fprintf(stderr, "Warning: could not load 8k_earth_specular_map.png; ocean shader disabled\n");
        }
    }

    // ── Load earth cloud map (binding 7): 8K R8_UNORM grayscale coverage ─────────
    {
        int w, h, ch;
        unsigned char *pixels = stbi_load("assets/textures/8k_earth_clouds.jpg", &w, &h, &ch, 1);
        if (pixels)
        {
            earthCloudsMips = (uint32_t)std::floor(std::log2((float)std::max(w, h))) + 1;
            VkDeviceSize imgBytes = (VkDeviceSize)w * h;

            VkBuffer stageBuf;
            VkDeviceMemory stageMem;
            ctx.createBuffer(imgBytes,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             stageBuf, stageMem);
            void *mapped;
            vkMapMemory(ctx.device, stageMem, 0, imgBytes, 0, &mapped);
            memcpy(mapped, pixels, (size_t)imgBytes);
            vkUnmapMemory(ctx.device, stageMem);
            stbi_image_free(pixels);

            ctx.createImage((uint32_t)w, (uint32_t)h,
                            VK_FORMAT_R8_UNORM,
                            VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            earthCloudsImg, earthCloudsMem, earthCloudsMips);

            VkCommandBuffer cmd = ctx.beginOneTimeCommands();
            VkImageMemoryBarrier allMips{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            allMips.srcAccessMask = 0;
            allMips.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            allMips.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allMips.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            allMips.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            allMips.image = earthCloudsImg;
            allMips.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthCloudsMips, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &allMips);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cmd, stageBuf, earthCloudsImg,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ctx.generateMipmaps(cmd, earthCloudsImg, VK_FORMAT_R8_UNORM,
                                (uint32_t)w, (uint32_t)h, earthCloudsMips);
            ctx.endOneTimeCommands(cmd);
            vkDestroyBuffer(ctx.device, stageBuf, nullptr);
            vkFreeMemory(ctx.device, stageMem, nullptr);

            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = earthCloudsImg;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R8_UNORM;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, earthCloudsMips, 0, 1};
            vkCreateImageView(ctx.device, &vci, nullptr, &earthCloudsView);

            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod = (float)earthCloudsMips;
            vkCreateSampler(ctx.device, &sci, nullptr, &earthCloudsSampler);
        }
        else
        {
            fprintf(stderr, "Warning: could not load 8k_earth_clouds.jpg; cloud map disabled\n");
        }
    }

    // ── Cloud params UBO (binding 9): host-visible, persistently mapped ──────────
    {
        VkDeviceSize sz = sizeof(GpuCloudParams);
        ctx.createBuffer(sz,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         cloudParamsBuf, cloudParamsMem);
        vkMapMemory(ctx.device, cloudParamsMem, 0, sz, 0, &cloudParamsMapped);
        memset(cloudParamsMapped, 0, sz);
    }

    // ── Descriptor set layout: 0=GlowBuf, 1=noise, 2=moon, 3=earthDay, 4=earthNight, 5=earthElev, 6=earthSpec, 7=earthClouds, 8=cloudNoise3D, 9=CloudParams UBO, 10/11=half-res cloud march targets A/B, 12=lightDomeBuf, 13=milkyWayTex, 14=cityDayDetail, 15=cityNightDetail, 16=auroraNoise3D, 17=reflectBeamsBuf, 18=beamGlowDomeBuf, 19=sceneDepthTex, 20=oceanGlintBuf, 21=groundBeamsBuf
    VkDescriptorSetLayoutBinding bindings[22] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[5] = {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[6] = {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[7] = {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // cloudNoise sampler3D
    bindings[9] = {9, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[10] = {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // half-res cloud march target A
    bindings[11] = {11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // half-res cloud march target B
    // lightDomeBuf: same buffer as sat_flare.comp's binding 3 — sat_sky.frag needs its own read of
    // it to dim the Milky Way directionally, matching how satellites/stars are already dimmed.
    bindings[12] = {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[13] = {13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // milky way skybox
    bindings[14] = {14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // city day detail
    bindings[15] = {15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // city night detail
    bindings[16] = {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // aurora noise sampler3D
    // reflectBeamsBuf: same buffer as sat_orbit.comp's binding 4 — ground-spot direct lighting (C12).
    bindings[17] = {17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    // beamGlowDomeBuf: same buffer as sat_orbit.comp's binding 5 / sat_flare.comp's binding 4 —
    // dims the Milky Way near an active beam the same way the light pollution dome already does
    // (C12 follow-up #31). Was binding 19; compacted down when cloudShadowTex (18) was deleted.
    bindings[18] = {18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    // sceneDepthTex: same image scene_depth.comp writes at the top of recordCompute — the shared
    // terrain/ocean distance every occlusion test now reads instead of re-deriving.
    bindings[19] = {19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    // oceanGlintBuf: same buffer as sat_flare.comp's binding 8 (descSet) — sat_sky.frag's
    // ocean-glint block reads it here (flare architecture overhaul), same split as GlowBuf's
    // binding 0 here vs descSet's binding 2.
    bindings[20] = {20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    // groundBeamsBuf (perf follow-up): CPU-compacted, observer-range-culled beam list — see
    // GpuGroundBeams comment in SatelliteSim.h and the GroundBeamsBuf declaration in sat_sky.frag.
    bindings[21] = {21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 22;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &skyDescLayout);

    VkDescriptorPoolSize ps[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 15},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 3;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &skyDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = skyDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &skyDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &skyDescSet);

    // Use noise texture as 1×1 placeholder if optional textures failed to load
    VkSampler elevSamplerFinal = earthElevSampler ? earthElevSampler : noiseSampler;
    VkImageView elevViewFinal = earthElevView ? earthElevView : noiseTexView;
    VkSampler specSamplerFinal = earthSpecSampler ? earthSpecSampler : noiseSampler;
    VkImageView specViewFinal = earthSpecView ? earthSpecView : noiseTexView;
    VkSampler cloudsSamplerFinal = earthCloudsSampler ? earthCloudsSampler : noiseSampler;
    VkImageView cloudsViewFinal = earthCloudsView ? earthCloudsView : noiseTexView;

    VkDescriptorBufferInfo bufInfo{glowBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo noiseImgInfo{noiseSampler, noiseTexView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo moonImgInfo{moonSampler, moonTexView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dayImgInfo{earthDaySampler, earthDayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo nightImgInfo{earthNightSampler, earthNightView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo elevImgInfo{elevSamplerFinal, elevViewFinal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo specImgInfo{specSamplerFinal, specViewFinal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cloudsImgInfo{cloudsSamplerFinal, cloudsViewFinal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cloudNoiseImgInfo{cloudNoiseSampler, cloudNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo cloudParamsInfo{cloudParamsBuf, 0, sizeof(GpuCloudParams)};
    VkDescriptorImageInfo cloudMarchAImgInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cloudMarchBImgInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo lightDomeInfo{lightDomeBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo milkyWayImgInfo{milkyWaySampler, milkyWayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cityDayDetailImgInfo{cityDayDetailSampler, cityDayDetailView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cityNightDetailImgInfo{cityNightDetailSampler, cityNightDetailView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo auroraNoiseImgInfo{auroraNoiseSampler, auroraNoiseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo reflectBeamsInfo{reflectBeamsBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo beamGlowDomeInfo{beamGlowDomeBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo oceanGlintInfo{oceanGlintBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo groundBeamsInfo{groundBeamsBuf, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[22] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = skyDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bufInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = skyDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &noiseImgInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = skyDescSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &moonImgInfo;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = skyDescSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &dayImgInfo;
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = skyDescSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &nightImgInfo;
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = skyDescSet;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &elevImgInfo;
    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = skyDescSet;
    writes[6].dstBinding = 6;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = &specImgInfo;
    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = skyDescSet;
    writes[7].dstBinding = 7;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &cloudsImgInfo;
    writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[8].dstSet = skyDescSet;
    writes[8].dstBinding = 8;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &cloudNoiseImgInfo;
    writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[9].dstSet = skyDescSet;
    writes[9].dstBinding = 9;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[9].pBufferInfo = &cloudParamsInfo;
    writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[10].dstSet = skyDescSet;
    writes[10].dstBinding = 10;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[10].pImageInfo = &cloudMarchAImgInfo;
    writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[11].dstSet = skyDescSet;
    writes[11].dstBinding = 11;
    writes[11].descriptorCount = 1;
    writes[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[11].pImageInfo = &cloudMarchBImgInfo;
    writes[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[12].dstSet = skyDescSet;
    writes[12].dstBinding = 12;
    writes[12].descriptorCount = 1;
    writes[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[12].pBufferInfo = &lightDomeInfo;
    writes[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[13].dstSet = skyDescSet;
    writes[13].dstBinding = 13;
    writes[13].descriptorCount = 1;
    writes[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[13].pImageInfo = &milkyWayImgInfo;
    writes[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[14].dstSet = skyDescSet;
    writes[14].dstBinding = 14;
    writes[14].descriptorCount = 1;
    writes[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[14].pImageInfo = &cityDayDetailImgInfo;
    writes[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[15].dstSet = skyDescSet;
    writes[15].dstBinding = 15;
    writes[15].descriptorCount = 1;
    writes[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[15].pImageInfo = &cityNightDetailImgInfo;
    writes[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[16].dstSet = skyDescSet;
    writes[16].dstBinding = 16;
    writes[16].descriptorCount = 1;
    writes[16].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[16].pImageInfo = &auroraNoiseImgInfo;
    writes[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[17].dstSet = skyDescSet;
    writes[17].dstBinding = 17;
    writes[17].descriptorCount = 1;
    writes[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[17].pBufferInfo = &reflectBeamsInfo;
    writes[18].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[18].dstSet = skyDescSet;
    writes[18].dstBinding = 18;
    writes[18].descriptorCount = 1;
    writes[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[18].pBufferInfo = &beamGlowDomeInfo;
    VkDescriptorImageInfo sceneDepthImgInfo{sceneDepthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[19].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[19].dstSet = skyDescSet;
    writes[19].dstBinding = 19;
    writes[19].descriptorCount = 1;
    writes[19].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[19].pImageInfo = &sceneDepthImgInfo;
    writes[20].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[20].dstSet = skyDescSet;
    writes[20].dstBinding = 20;
    writes[20].descriptorCount = 1;
    writes[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[20].pBufferInfo = &oceanGlintInfo;
    writes[21].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[21].dstSet = skyDescSet;
    writes[21].dstBinding = 21;
    writes[21].descriptorCount = 1;
    writes[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[21].pBufferInfo = &groundBeamsInfo;
    vkUpdateDescriptorSets(ctx.device, 22, writes, 0, nullptr);
}

// Fullscreen triangle that colors pixels sky or ground based on camera elevation.
// Uses same push constant layout as the satellite draw pass (SatDrawPC).
void SatelliteSim::createSkyBgPipeline(VulkanContext &ctx)
{
    VkShaderModule vert = ctx.loadShader("shaders/sat_sky.vert.spv");
    VkShaderModule frag = ctx.loadShader("shaders/sat_sky.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D sc{{0, 0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Write terrain depth so satellite/star passes can test against it with LESS.
    // ALWAYS compare op so the sky background always wins (it's the first pass).
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    // Opaque: simply overwrite what the clear left.
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_FALSE;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    if (skyBgPipeLayout == VK_NULL_HANDLE)
    {
        // Fragment stage needs push constants too (sun disc reads sunDirENU).
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                0, sizeof(SatDrawPC)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &skyDescLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &skyBgPipeLayout);
    }

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.layout = skyBgPipeLayout;
    ci.renderPass = ctx.renderPass;
    ci.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &skyBgPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create sky background pipeline");

    // Minimal variant — identical state, same layout/render pass, cheap fragment module. Used by
    // the Potato preset (debugDisableMask bit 262144) where the full sat_sky.frag is too slow.
    {
        if (std::getenv("SATLIGHTSIM_FRAME_TRACE"))
            fprintf(stderr, "[sky] createSkyBgPipeline: recreating both sky pipelines\n");
        VkShaderModule minFrag = ctx.loadShader("shaders/sat_sky_minimal.frag.spv");
        VkPipelineShaderStageCreateInfo minStages[2] = {stages[0], stages[1]};
        minStages[1].module = minFrag;
        VkGraphicsPipelineCreateInfo minCi = ci;
        minCi.pStages = minStages;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &minCi, nullptr, &skyBgMinimalPipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create minimal sky background pipeline");
        vkDestroyShaderModule(ctx.device, minFrag, nullptr);
    }

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── createSkyLowResResources (resolution scaling, session 29) ────────────────────────────────
// Swapchain-size-AND-renderScale-dependent, so this recreates on both resize and any renderScale
// change (see buildSettingsDisplayTab). A single color-only render pass (no depth attachment —
// nothing else draws in this pass to depth-test against, and depth is deliberately not blitted
// downstream — see the member comments in SatelliteSim.h), CLEARed fresh each frame, with
// finalLayout TRANSFER_SRC_OPTIMAL so recordPrePass's blit needs no extra barrier on this side.
void SatelliteSim::createSkyLowResResources(VulkanContext &ctx)
{
    skyLowResExtent.width = std::max(1u, (uint32_t)(ctx.swapExtent.width * renderScale));
    skyLowResExtent.height = std::max(1u, (uint32_t)(ctx.swapExtent.height * renderScale));

    // ── Render pass: single color attachment, CLEAR -> TRANSFER_SRC_OPTIMAL ──────────────────
    VkAttachmentDescription color{};
    color.format = ctx.swapFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &color;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    if (vkCreateRenderPass(ctx.device, &rpci, nullptr, &skyLowResRenderPass) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create skyLowRes render pass");

    // ── Color image + view ────────────────────────────────────────────────────────────────────
    ctx.createImage(skyLowResExtent.width, skyLowResExtent.height, ctx.swapFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    skyLowResColorImg, skyLowResColorMem);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = skyLowResColorImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ctx.swapFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(ctx.device, &vci, nullptr, &skyLowResColorView);

    // ── Framebuffer ────────────────────────────────────────────────────────────────────────────
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = skyLowResRenderPass;
    fci.attachmentCount = 1;
    fci.pAttachments = &skyLowResColorView;
    fci.width = skyLowResExtent.width;
    fci.height = skyLowResExtent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(ctx.device, &fci, nullptr, &skyLowResFramebuffer) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create skyLowRes framebuffer");

    // ── Pipeline: same shaders/layout as skyBgPipeline, low-res viewport, no depth ────────────
    VkShaderModule vert = ctx.loadShader("shaders/sat_sky.vert.spv");
    VkShaderModule frag = ctx.loadShader("shaders/sat_sky.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)skyLowResExtent.width, (float)skyLowResExtent.height, 0, 1};
    VkRect2D sc{{0, 0}, skyLowResExtent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // No depth attachment in this render pass at all — test/write both off.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_FALSE;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    // skyBgPipeLayout already exists by the time this is first called (createSkyBgPipeline runs
    // first in init() — see there) and is reused as-is: identical push constants/descriptor set.
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.layout = skyBgPipeLayout;
    ci.renderPass = skyLowResRenderPass;
    ci.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &skyLowResPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create skyLowRes pipeline");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

void SatelliteSim::destroySkyLowResResources(VkDevice device)
{
    if (skyLowResPipeline)
        vkDestroyPipeline(device, skyLowResPipeline, nullptr);
    if (skyLowResFramebuffer)
        vkDestroyFramebuffer(device, skyLowResFramebuffer, nullptr);
    if (skyLowResColorView)
        vkDestroyImageView(device, skyLowResColorView, nullptr);
    if (skyLowResColorImg)
        vkDestroyImage(device, skyLowResColorImg, nullptr);
    if (skyLowResColorMem)
        vkFreeMemory(device, skyLowResColorMem, nullptr);
    if (skyLowResRenderPass)
        vkDestroyRenderPass(device, skyLowResRenderPass, nullptr);
    skyLowResPipeline = VK_NULL_HANDLE;
    skyLowResFramebuffer = VK_NULL_HANDLE;
    skyLowResColorView = VK_NULL_HANDLE;
    skyLowResColorImg = VK_NULL_HANDLE;
    skyLowResColorMem = VK_NULL_HANDLE;
    skyLowResRenderPass = VK_NULL_HANDLE;
}

// ─── createDrawPipeline ───────────────────────────────────────────────────────
void SatelliteSim::createDrawPipeline(VulkanContext &ctx)
{
    VkShaderModule vert = ctx.loadShader("shaders/sat_point.vert.spv");
    VkShaderModule frag = ctx.loadShader("shaders/sat_point.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D sc{{0, 0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth test against terrain written by the sky background pass (gl_FragDepth).
    // Satellites at fixed depth 0.5 fail LESS where terrain depth < 0.5 (close terrain hits).
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    // Additive blending.
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    if (drawPipeLayout == VK_NULL_HANDLE)
    {
        // C12 follow-up #33: FRAGMENT added so sat_point.frag can read screenSizePx for its new
        // cloud-occlusion sampling (previously vertex-only, since the fragment shader used no
        // push constants at all before this).
        VkPushConstantRange drawPcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SatDrawPC)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &descLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &drawPcr;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &drawPipeLayout);
    }

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.layout = drawPipeLayout;
    ci.renderPass = ctx.renderPass;
    ci.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &drawPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create draw pipeline");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── createFlareResources ─────────────────────────────────────────────────────
// Flare architecture overhaul — see FlareSourcePC's comment in SatelliteSim.h for the design.
// Images + render pass + framebuffer for stage 1 (the bright-source render). Swapchain-size
// dependent (flareExtent derives from ctx.swapExtent) — destroyed/recreated in onResize alongside
// cloudMarchTargetA/B and sceneDepthImg.
void SatelliteSim::createFlareResources(VulkanContext &ctx)
{
    flareExtent.width = std::max(1u, (ctx.swapExtent.width + 3) / 4);
    flareExtent.height = std::max(1u, (ctx.swapExtent.height + 3) / 4);

    // ── Render pass: single color attachment, CLEAR -> GENERAL ───────────────────────────────
    // finalLayout=GENERAL (not COLOR_ATTACHMENT_OPTIMAL) so the immediately-following compute blur
    // dispatch can imageLoad/imageStore it with no extra transition barrier — the same "finalLayout
    // matches what the next command needs" convention skyLowResRenderPass already established
    // (there: TRANSFER_SRC_OPTIMAL, ready for its own following blit with no extra barrier).
    VkAttachmentDescription color{};
    color.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &color;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    if (vkCreateRenderPass(ctx.device, &rpci, nullptr, &flareSourceRenderPass) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create flare source render pass");

    // ── Images + views ────────────────────────────────────────────────────────────────────────
    // flareSourceImg: rendered into by stage 1, then read+written by stage 2's compute ping-pong.
    ctx.createImage(flareExtent.width, flareExtent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    flareSourceImg, flareSourceMem);
    // flareScratchImg: compute-only ping-pong target; also what stage 3 samples for the final composite.
    ctx.createImage(flareExtent.width, flareExtent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    flareScratchImg, flareScratchMem);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vci.image = flareSourceImg;
    vkCreateImageView(ctx.device, &vci, nullptr, &flareSourceView);
    vci.image = flareScratchImg;
    vkCreateImageView(ctx.device, &vci, nullptr, &flareScratchView);

    // flareScratchImg is never touched by a render pass — its very first use each frame is a
    // compute WRITE (dispatch 1 of flare_blur.comp) — so it needs a one-time UNDEFINED->GENERAL
    // transition here, same idiom cloudMarchTargetA/B's initial setup uses.
    {
        auto cmd = ctx.beginOneTimeCommands();
        ctx.imageBarrier(cmd, flareScratchImg, 0, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        ctx.endOneTimeCommands(cmd);
    }

    // Shared sampler, resolution-independent — created once, kept across resizes (matches
    // sceneDepthSampler/cloudMarchSampler convention).
    if (!flareSampler)
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(ctx.device, &sci, nullptr, &flareSampler);
    }

    // ── Framebuffer ───────────────────────────────────────────────────────────────────────────
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = flareSourceRenderPass;
    fci.attachmentCount = 1;
    fci.pAttachments = &flareSourceView;
    fci.width = flareExtent.width;
    fci.height = flareExtent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(ctx.device, &fci, nullptr, &flareSourceFramebuffer) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create flare source framebuffer");
}

void SatelliteSim::destroyFlareResources(VkDevice device)
{
    if (flareSourceFramebuffer)
        vkDestroyFramebuffer(device, flareSourceFramebuffer, nullptr);
    if (flareSourceRenderPass)
        vkDestroyRenderPass(device, flareSourceRenderPass, nullptr);
    if (flareSourceView)
        vkDestroyImageView(device, flareSourceView, nullptr);
    if (flareSourceImg)
        vkDestroyImage(device, flareSourceImg, nullptr);
    if (flareSourceMem)
        vkFreeMemory(device, flareSourceMem, nullptr);
    if (flareScratchView)
        vkDestroyImageView(device, flareScratchView, nullptr);
    if (flareScratchImg)
        vkDestroyImage(device, flareScratchImg, nullptr);
    if (flareScratchMem)
        vkFreeMemory(device, flareScratchMem, nullptr);
    flareSourceFramebuffer = VK_NULL_HANDLE;
    flareSourceRenderPass = VK_NULL_HANDLE;
    flareSourceView = VK_NULL_HANDLE;
    flareSourceImg = VK_NULL_HANDLE;
    flareSourceMem = VK_NULL_HANDLE;
    flareScratchView = VK_NULL_HANDLE;
    flareScratchImg = VK_NULL_HANDLE;
    flareScratchMem = VK_NULL_HANDLE;
    // flareSampler NOT destroyed here — resolution-independent, persists across resize (destroyed
    // only in cleanup()).
}

// ─── createFlareDescriptors ────────────────────────────────────────────────────
// Two small, NEW descriptor sets this overhaul needs (flareSourcePipeline reuses the existing
// descLayout/descSet directly — see createDescriptors()'s bindings 1/5/6/7 — since satVisibleBuf/
// cloudTargetA/cloudTargetB/sceneDepthTex are exactly what it needs and already live there).
void SatelliteSim::createFlareDescriptors(VulkanContext &ctx)
{
    // ── Stage 2 (blur/streak compute): 2 STORAGE_IMAGE bindings ──────────────────────────────
    VkDescriptorSetLayoutBinding blurBindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo blurLi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    blurLi.bindingCount = 2;
    blurLi.pBindings = blurBindings;
    vkCreateDescriptorSetLayout(ctx.device, &blurLi, nullptr, &flareBlurDescLayout);

    VkDescriptorPoolSize blurPs{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
    VkDescriptorPoolCreateInfo blurPi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    blurPi.poolSizeCount = 1;
    blurPi.pPoolSizes = &blurPs;
    blurPi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &blurPi, nullptr, &flareBlurDescPool);

    VkDescriptorSetAllocateInfo blurAi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    blurAi.descriptorPool = flareBlurDescPool;
    blurAi.descriptorSetCount = 1;
    blurAi.pSetLayouts = &flareBlurDescLayout;
    vkAllocateDescriptorSets(ctx.device, &blurAi, &flareBlurDescSet);

    VkDescriptorImageInfo flareAInfo{VK_NULL_HANDLE, flareSourceView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo flareBInfo{VK_NULL_HANDLE, flareScratchView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet blurWrites[2] = {};
    blurWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                     flareBlurDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &flareAInfo, nullptr, nullptr};
    blurWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                     flareBlurDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &flareBInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, blurWrites, 0, nullptr);

    // ── Stage 3 (composite draw): 1 COMBINED_IMAGE_SAMPLER binding ────────────────────────────
    // Sampled directly in VK_IMAGE_LAYOUT_GENERAL — legal, and this image is small/short-lived
    // enough per frame that skipping a dedicated layout-transition barrier costs nothing measurable.
    VkDescriptorSetLayoutBinding compBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo compLi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    compLi.bindingCount = 1;
    compLi.pBindings = &compBinding;
    vkCreateDescriptorSetLayout(ctx.device, &compLi, nullptr, &flareCompositeDescLayout);

    VkDescriptorPoolSize compPs{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo compPi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    compPi.poolSizeCount = 1;
    compPi.pPoolSizes = &compPs;
    compPi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &compPi, nullptr, &flareCompositeDescPool);

    VkDescriptorSetAllocateInfo compAi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    compAi.descriptorPool = flareCompositeDescPool;
    compAi.descriptorSetCount = 1;
    compAi.pSetLayouts = &flareCompositeDescLayout;
    vkAllocateDescriptorSets(ctx.device, &compAi, &flareCompositeDescSet);

    // Final result lands in flareScratchImg (see the dispatch-order comment at the flare_blur.comp
    // call site in recordCompute()).
    VkDescriptorImageInfo flareFinalInfo{flareSampler, flareScratchView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet compWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                   flareCompositeDescSet, 0, 0, 1,
                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &flareFinalInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &compWrite, 0, nullptr);
}

// ─── createFlarePipelines ──────────────────────────────────────────────────────
void SatelliteSim::createFlarePipelines(VulkanContext &ctx)
{
    // ── Stage 1: flareSourcePipeline (graphics, point list, additive blend, no depth) ─────────
    {
        VkShaderModule vert = ctx.loadShader("shaders/flare_source.vert.spv");
        VkShaderModule frag = ctx.loadShader("shaders/flare_source.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

        VkViewport vp{0, 0, (float)flareExtent.width, (float)flareExtent.height, 0, 1};
        VkRect2D sc{{0, 0}, flareExtent};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.pViewports = &vp;
        vps.scissorCount = 1;
        vps.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        if (flareSourcePipeLayout == VK_NULL_HANDLE)
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FlareSourcePC)};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.setLayoutCount = 1;
            li.pSetLayouts = &descLayout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &pcr;
            vkCreatePipelineLayout(ctx.device, &li, nullptr, &flareSourcePipeLayout);
        }

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rast;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds;
        ci.pColorBlendState = &cb;
        ci.layout = flareSourcePipeLayout;
        ci.renderPass = flareSourceRenderPass;
        ci.subpass = 0;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &flareSourcePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create flare source pipeline");

        vkDestroyShaderModule(ctx.device, vert, nullptr);
        vkDestroyShaderModule(ctx.device, frag, nullptr);
    }

    // ── Stage 2: flareBlurPipeline (compute) — swapchain-size independent, NOT recreated on
    // resize, so guard both the layout and the pipeline itself against being called again.
    if (flareBlurPipeline == VK_NULL_HANDLE)
    {
        VkShaderModule mod = ctx.loadShader("shaders/flare_blur.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FlareBlurPC)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &flareBlurDescLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &flareBlurPipeLayout);

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = flareBlurPipeLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &flareBlurPipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create flare blur pipeline");

        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    // ── Stage 3: flareCompositePipeline (graphics, fullscreen tri, additive blend) ────────────
    {
        VkShaderModule vert = ctx.loadShader("shaders/flare_composite.vert.spv");
        VkShaderModule frag = ctx.loadShader("shaders/flare_composite.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
        VkRect2D sc{{0, 0}, ctx.swapExtent};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.pViewports = &vp;
        vps.scissorCount = 1;
        vps.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        if (flareCompositePipeLayout == VK_NULL_HANDLE)
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FlareCompositePC)};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.setLayoutCount = 1;
            li.pSetLayouts = &flareCompositeDescLayout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &pcr;
            vkCreatePipelineLayout(ctx.device, &li, nullptr, &flareCompositePipeLayout);
        }

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rast;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds;
        ci.pColorBlendState = &cb;
        ci.layout = flareCompositePipeLayout;
        ci.renderPass = ctx.renderPass;
        ci.subpass = 0;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &flareCompositePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create flare composite pipeline");

        vkDestroyShaderModule(ctx.device, vert, nullptr);
        vkDestroyShaderModule(ctx.device, frag, nullptr);
    }
}

// ─── createTrailResources ──────────────────────────────────────────────────────
// Long-exposure trail pipeline (see the trailAccumImg member block comment in SatelliteSim.h).
// trailAccumExtent is the FULL ctx.swapExtent (unlike flareExtent's quarter-res) — satellites and
// stars always render at native resolution in this app, and a downscaled trail buffer would blur
// thin streaks.
void SatelliteSim::createTrailResources(VulkanContext &ctx)
{
    trailAccumExtent = ctx.swapExtent;

    // ── Render pass: single color attachment, LOAD -> GENERAL throughout ─────────────────────
    // LOAD_OP_LOAD (not CLEAR) because persistence across frames is the whole point. initialLayout/
    // finalLayout are both GENERAL — the image never leaves GENERAL between frames (fade compute
    // read/write, splat render pass, composite sampled read all use it), same "stay in GENERAL"
    // convention flareScratchImg already uses for the same reason (skip pointless transitions on a
    // small/short-lived-per-frame image).
    VkAttachmentDescription color{};
    color.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    color.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &color;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    if (vkCreateRenderPass(ctx.device, &rpci, nullptr, &trailAccumRenderPass) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create trail accum render pass");

    // ── Image + view ──────────────────────────────────────────────────────────────────────────
    ctx.createImage(trailAccumExtent.width, trailAccumExtent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    trailAccumImg, trailAccumMem);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vci.image = trailAccumImg;
    vkCreateImageView(ctx.device, &vci, nullptr, &trailAccumView);

    // First use each frame is either vkCmdClearColorImage (trailClearPending, recordCompute()) or
    // the fade compute's imageLoad/imageStore — both need GENERAL, so transition it here (same
    // one-time UNDEFINED->GENERAL idiom flareScratchImg's own setup uses) and zero it so a
    // fresh/resized buffer starts blank regardless of trailClearPending's own state.
    {
        auto cmd = ctx.beginOneTimeCommands();
        ctx.imageBarrier(cmd, trailAccumImg, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, trailAccumImg, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        ctx.imageBarrier(cmd, trailAccumImg, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        ctx.endOneTimeCommands(cmd);
    }

    // ── Framebuffer ───────────────────────────────────────────────────────────────────────────
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = trailAccumRenderPass;
    fci.attachmentCount = 1;
    fci.pAttachments = &trailAccumView;
    fci.width = trailAccumExtent.width;
    fci.height = trailAccumExtent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(ctx.device, &fci, nullptr, &trailAccumFramebuffer) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create trail accum framebuffer");
}

void SatelliteSim::destroyTrailResources(VkDevice device)
{
    if (trailAccumFramebuffer)
        vkDestroyFramebuffer(device, trailAccumFramebuffer, nullptr);
    if (trailAccumRenderPass)
        vkDestroyRenderPass(device, trailAccumRenderPass, nullptr);
    if (trailAccumView)
        vkDestroyImageView(device, trailAccumView, nullptr);
    if (trailAccumImg)
        vkDestroyImage(device, trailAccumImg, nullptr);
    if (trailAccumMem)
        vkFreeMemory(device, trailAccumMem, nullptr);
    trailAccumFramebuffer = VK_NULL_HANDLE;
    trailAccumRenderPass = VK_NULL_HANDLE;
    trailAccumView = VK_NULL_HANDLE;
    trailAccumImg = VK_NULL_HANDLE;
    trailAccumMem = VK_NULL_HANDLE;
}

// ─── createTrailDescriptors ─────────────────────────────────────────────────────
// Two small, NEW descriptor sets (trailSatPipeline/trailStarPipeline reuse the existing
// descSet/starDescSet/planetDescSet directly — see createTrailPipelines()).
void SatelliteSim::createTrailDescriptors(VulkanContext &ctx)
{
    // ── Stage A (fade compute): 1 STORAGE_IMAGE binding ───────────────────────────────────────
    VkDescriptorSetLayoutBinding fadeBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo fadeLi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    fadeLi.bindingCount = 1;
    fadeLi.pBindings = &fadeBinding;
    vkCreateDescriptorSetLayout(ctx.device, &fadeLi, nullptr, &trailFadeDescLayout);

    VkDescriptorPoolSize fadePs{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo fadePi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    fadePi.poolSizeCount = 1;
    fadePi.pPoolSizes = &fadePs;
    fadePi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &fadePi, nullptr, &trailFadeDescPool);

    VkDescriptorSetAllocateInfo fadeAi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    fadeAi.descriptorPool = trailFadeDescPool;
    fadeAi.descriptorSetCount = 1;
    fadeAi.pSetLayouts = &trailFadeDescLayout;
    vkAllocateDescriptorSets(ctx.device, &fadeAi, &trailFadeDescSet);

    VkDescriptorImageInfo fadeImgInfo{VK_NULL_HANDLE, trailAccumView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet fadeWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                   trailFadeDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fadeImgInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &fadeWrite, 0, nullptr);

    // ── Stage C (composite draw): 1 COMBINED_IMAGE_SAMPLER binding ────────────────────────────
    // Sampled directly in VK_IMAGE_LAYOUT_GENERAL — same convention flareCompositeDescSet already
    // uses for flareScratchImg. Reuses flareSampler (LINEAR/CLAMP_TO_EDGE, resolution-independent).
    VkDescriptorSetLayoutBinding compBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo compLi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    compLi.bindingCount = 1;
    compLi.pBindings = &compBinding;
    vkCreateDescriptorSetLayout(ctx.device, &compLi, nullptr, &trailCompositeDescLayout);

    VkDescriptorPoolSize compPs{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo compPi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    compPi.poolSizeCount = 1;
    compPi.pPoolSizes = &compPs;
    compPi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &compPi, nullptr, &trailCompositeDescPool);

    VkDescriptorSetAllocateInfo compAi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    compAi.descriptorPool = trailCompositeDescPool;
    compAi.descriptorSetCount = 1;
    compAi.pSetLayouts = &trailCompositeDescLayout;
    vkAllocateDescriptorSets(ctx.device, &compAi, &trailCompositeDescSet);

    VkDescriptorImageInfo compImgInfo{flareSampler, trailAccumView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet compWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                   trailCompositeDescSet, 0, 0, 1,
                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &compImgInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 1, &compWrite, 0, nullptr);
}

// ─── createTrailPipelines ───────────────────────────────────────────────────────
void SatelliteSim::createTrailPipelines(VulkanContext &ctx)
{
    // ── Stage A: trailFadePipeline (compute) — swapchain-size independent, NOT recreated on
    // resize, so guard both the layout and the pipeline itself (same convention as flareBlurPipeline).
    if (trailFadePipeline == VK_NULL_HANDLE)
    {
        VkShaderModule mod = ctx.loadShader("shaders/trail_fade.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TrailFadePC)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &trailFadeDescLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &trailFadePipeLayout);

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = trailFadePipeLayout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &trailFadePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create trail fade pipeline");

        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    // ── Stage B: trailSatPipeline / trailStarPipeline (graphics, point list, additive blend, no
    // depth — this offscreen target has no depth attachment). Reuse the EXISTING drawPipeLayout/
    // starPipeLayout + sat_point/star_point shaders UNCHANGED — only new VkPipeline objects,
    // targeting trailAccumRenderPass instead of ctx.renderPass.
    {
        VkShaderModule vert = ctx.loadShader("shaders/sat_point.vert.spv");
        VkShaderModule frag = ctx.loadShader("shaders/sat_point.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

        VkViewport vp{0, 0, (float)trailAccumExtent.width, (float)trailAccumExtent.height, 0, 1};
        VkRect2D sc{{0, 0}, trailAccumExtent};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.pViewports = &vp;
        vps.scissorCount = 1;
        vps.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rast;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds;
        ci.pColorBlendState = &cb;
        ci.layout = drawPipeLayout; // reused unchanged — same SatDrawPC push-constant range
        ci.renderPass = trailAccumRenderPass;
        ci.subpass = 0;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &trailSatPipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create trail satellite pipeline");

        vkDestroyShaderModule(ctx.device, vert, nullptr);
        vkDestroyShaderModule(ctx.device, frag, nullptr);
    }
    {
        VkShaderModule vert = ctx.loadShader("shaders/star_point.vert.spv");
        VkShaderModule frag = ctx.loadShader("shaders/star_point.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

        VkViewport vp{0, 0, (float)trailAccumExtent.width, (float)trailAccumExtent.height, 0, 1};
        VkRect2D sc{{0, 0}, trailAccumExtent};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.pViewports = &vp;
        vps.scissorCount = 1;
        vps.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rast;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds;
        ci.pColorBlendState = &cb;
        ci.layout = starPipeLayout; // reused unchanged — also used for the planet trail draw
        ci.renderPass = trailAccumRenderPass;
        ci.subpass = 0;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &trailStarPipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create trail star pipeline");

        vkDestroyShaderModule(ctx.device, vert, nullptr);
        vkDestroyShaderModule(ctx.device, frag, nullptr);
    }

    // ── Stage C: trailCompositePipeline (graphics, fullscreen tri, additive blend into ctx.renderPass)
    {
        VkShaderModule vert = ctx.loadShader("shaders/trail_composite.vert.spv");
        VkShaderModule frag = ctx.loadShader("shaders/trail_composite.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
        VkRect2D sc{{0, 0}, ctx.swapExtent};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.pViewports = &vp;
        vps.scissorCount = 1;
        vps.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        if (trailCompositePipeLayout == VK_NULL_HANDLE)
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TrailCompositePC)};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.setLayoutCount = 1;
            li.pSetLayouts = &trailCompositeDescLayout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &pcr;
            vkCreatePipelineLayout(ctx.device, &li, nullptr, &trailCompositePipeLayout);
        }

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rast;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds;
        ci.pColorBlendState = &cb;
        ci.layout = trailCompositePipeLayout;
        ci.renderPass = ctx.renderPass;
        ci.subpass = 0;
        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &trailCompositePipeline) != VK_SUCCESS)
            throw std::runtime_error("SatelliteSim: failed to create trail composite pipeline");

        vkDestroyShaderModule(ctx.device, vert, nullptr);
        vkDestroyShaderModule(ctx.device, frag, nullptr);
    }
}

// ─── initStars ────────────────────────────────────────────────────────────────
// Parses the embedded Yale BSC catalog, builds star records with ECI vectors,
// creates a host-visible GPU buffer, and sets up the star descriptor set + pipeline.
void SatelliteSim::initStars(VulkanContext &ctx)
{
    starRecords.clear();
    const int kCatSize = sizeof(kStarCatalog) / sizeof(kStarCatalog[0]);
    starRecords.reserve(kCatSize);

    for (int i = 0; i < kCatSize; ++i)
    {
        const auto &s = kStarCatalog[i];

        // RA/Dec (degrees, J2000) → ECI unit vector.
        // ECI is the J2000 equatorial frame: X toward vernal equinox, Z toward north pole.
        float ra = glm::radians(s.ra_deg);
        float dec = glm::radians(s.dec_deg);
        glm::vec3 eciDir{cosf(dec) * cosf(ra),
                         cosf(dec) * sinf(ra),
                         sinf(dec)};

        // Visual magnitude → intensity: mag 0 → 1.0; Sirius (−1.46) → ~3.84.
        float rawInt = glm::clamp(powf(10.0f, -s.vmag / 2.5f), 0.0f, 8.0f);

        // B-V colour index → approximate RGB (hot blue at low B-V, red at high B-V).
        float bv = s.bv;
        glm::vec3 col{glm::clamp(0.90f + 0.10f * bv, 0.60f, 1.0f),  // R
                      glm::clamp(1.00f - 0.15f * bv, 0.50f, 1.0f),  // G
                      glm::clamp(1.00f - 0.90f * bv, 0.10f, 1.0f)}; // B

        // Point sprite size: magnitude-driven, floored near 2 px (S2a, RELEASE_v1_1_PLAN.md;
        // floor raised 1px->2px in a later pass to fight movement flicker — a 1px sprite covers
        // so few pixels that the rasterizer's covered-pixel SET changes in discrete jumps as the
        // sprite's sub-pixel center drifts frame-to-frame under camera motion, with no MSAA
        // anywhere in this project to smooth that transition; 2px gives more pixels to spread
        // that quantization error across).
        // The old formula `1.5 + min(rawInt, 4.0)` let its additive 1.5 floor dominate for every
        // faint star (rawInt is tiny once vmag > ~2), so a mag 6 and a mag 3 star both landed at
        // ~6 px post-scale — invisible at 287 stars, but with the catalog expanded to 8404 (down
        // to mag 6.5) that turned the sky to porridge. sqrt(rawInt) keeps faint stars close to the
        // floor (fine dust) while still giving Sirius roughly its old ~21 px size.
        float starScale = 4.0f; // tweak this to make stars bigger/smaller overall
        float angSize = 0.5f + 2.5f * sqrtf(rawInt);
        angSize *= starScale;

        starRecords.push_back({eciDir, rawInt, col, angSize});
    }
    starCount = (uint32_t)starRecords.size();

    // Host-visible buffer (tiny: ~8404 × 32 bytes = ~263 KB — negligible against 78,000 satellites).
    VkDeviceSize bufSize = starCount * sizeof(GpuSatVisible);
    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     starBuf, starMem);
    vkMapMemory(ctx.device, starMem, 0, bufSize, 0, &starMapped);

    // Descriptor layout: binding=1 (vertex shader reads GpuSatVisible) + bindings=2/3 (cloud
    // occlusion, session 30 bug fix — see star_point.frag's own comment) + binding=4 (shared
    // terrain/ocean depth, added for the long-exposure trail's manual terrain-occlusion test — see
    // star_point.frag's own comment and SatDrawPC::manualTerrainTest). cloudMarchSampler/
    // cloudMarchTargetAView/BView/sceneDepthSampler/sceneDepthView already exist by this point in
    // init() (createCloudMarchResources/createSceneDepthResources run well before initStars — see
    // init()'s call order), so it's safe to bind them here.
    VkDescriptorSetLayoutBinding bindings[4] = {};
    bindings[0].binding = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 2;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[2].binding = 3;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[3].binding = 4;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 4;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &starDescLayout);

    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 2;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &starDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = starDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &starDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &starDescSet);

    VkDescriptorBufferInfo bufInfo{starBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo cloudAInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cloudBInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo sceneDepthInfo{sceneDepthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr[4] = {};
    wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             starDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufInfo, nullptr};
    wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             starDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudAInfo, nullptr, nullptr};
    wr[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             starDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudBInfo, nullptr, nullptr};
    wr[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             starDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 4, wr, 0, nullptr);

    createStarPipeline(ctx);

    // Do an initial upload so stars are visible from frame 1.
    updateStars();
}

// ─── initPlanets ──────────────────────────────────────────────────────────────
// Must run after initStars() (reuses starDescLayout, and starPipeline must already exist —
// see recordDraw()'s planet draw call, which binds starPipeline with a different desc set).
void SatelliteSim::initPlanets(VulkanContext &ctx)
{
    VkDeviceSize bufSize = kPlanetCount * sizeof(GpuSatVisible);
    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     planetBuf, planetMem);
    vkMapMemory(ctx.device, planetMem, 0, bufSize, 0, &planetMapped);

    // Reuses starDescLayout (binding=1 STORAGE_BUFFER vertex-stage + bindings=2/3 cloud occlusion +
    // binding=4 shared terrain/ocean depth) unchanged — same shape, different buffer. starDescPool
    // is sized maxSets=1 (already holds starDescSet), so this gets its own tiny pool rather than
    // resizing that one.
    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 2;
    pi.pPoolSizes = ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &planetDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = planetDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &starDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &planetDescSet);

    VkDescriptorBufferInfo bufInfo{planetBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo cloudAInfo{cloudMarchSampler, cloudMarchTargetAView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo cloudBInfo{cloudMarchSampler, cloudMarchTargetBView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo sceneDepthInfo{sceneDepthSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr[4] = {};
    wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             planetDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufInfo, nullptr};
    wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             planetDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudAInfo, nullptr, nullptr};
    wr[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             planetDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudBInfo, nullptr, nullptr};
    wr[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
             planetDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(ctx.device, 4, wr, 0, nullptr);

    // Do an initial upload so planets are visible from frame 1 (mirrors initStars() above).
    // Requires updatePositions() to have already run at least once — see init()'s call order.
    updatePlanets();
}

// ─── createStarPipeline ───────────────────────────────────────────────────────
// Uses star_point.vert (twinkling + shared layout) + star_point.frag (tight core only,
// no satellite-style outer glow — prevents bright stars from becoming blobs).
void SatelliteSim::createStarPipeline(VulkanContext &ctx)
{
    VkShaderModule vert = ctx.loadShader("shaders/star_point.vert.spv");
    VkShaderModule frag = ctx.loadShader("shaders/star_point.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D sc{{0, 0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Same depth test as satellites: stars at fixed depth 0.5 are culled by close terrain.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    if (starPipeLayout == VK_NULL_HANDLE)
    {
        // FRAGMENT added (session 30 bug fix): star_point.frag now reads screenSizePx for cloud
        // occlusion, same reason sat_point.frag's drawPipeLayout adds it (C12 follow-up #33).
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SatDrawPC)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &starDescLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &starPipeLayout);
    }

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.layout = starPipeLayout;
    ci.renderPass = ctx.renderPass;
    ci.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &starPipeline) != VK_SUCCESS)
        throw std::runtime_error("SatelliteSim: failed to create star pipeline");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── updateLightPollutionDome ────────────────────────────────────────────────
// Builds an 8-azimuth-sector "how much city glow is in this compass direction" dome around
// the observer, replacing the old single-scalar "brightness at the observer's own position"
// approximation that dimmed stars/satellites uniformly regardless of which way they appeared
// in the sky (session 25 follow-up per user feedback). Each sector samples earthNightCpuBlur
// (bilinearly) at a few radii along that bearing — a flat-Earth tangent-plane lat/lon offset,
// adequate at the tens-of-km scale light pollution actually reaches — and combines them with a
// weighted max
// (one nearby bright city should dominate that direction's glow, not get averaged away by
// darker samples at other radii in the same sector). Sector convention matches GlowBuf's
// existing 8-sector azBin in sat_flare.comp exactly (bearing clockwise from North, 45° each)
// so both consumers read consistent geometry. Uploaded to lightDomeBuf for sat_flare.comp;
// updateStars() (called right after this) reads lightDomeAz[] directly, no upload needed there.
void SatelliteSim::updateLightPollutionDome()
{
    float obsR = glm::length(obsECI);
    float obsHeight = obsR - kEarthRadius;
    // Altitude falloff: light pollution's visible skyglow washes out within a few km — a much
    // tighter scale than the atmosphere's own 80km Rayleigh height. Effectively zero by aircraft
    // cruise altitude, let alone orbit. Same for every sector (observer's own altitude).
    float altFalloff = glm::clamp(glm::exp(-obsHeight / 3000.0f), 0.0f, 1.0f);

    if (earthNightCpuBlur.empty() || altFalloff <= 0.0f)
    {
        for (int i = 0; i < kNumLightSectors; ++i)
            lightDomeAz[i] = 0.0f;
        memcpy(lightDomeMapped, lightDomeAz, sizeof(lightDomeAz));
        mwPollutionRaw = 0.0f; // no data (or above the skyglow altitude falloff) — reads as "dark"
        return;
    }

    // Bilinear sample of earthNightCpuBlur (the coarser, box-blurred level — see the member
    // comment in SatelliteSim.h) — previously a nearest-pixel lookup against the sharp array,
    // which made each of the 4 per-sector radius samples snap between ~18km cells as the
    // observer moved or as neighboring sectors sampled nearby bearings, reading as sharp,
    // blocky transitions. Longitude wraps; latitude clamps at the poles.
    auto sampleDomeLum = [this](float u, float v) -> float
    {
        float fx = u * (float)earthNightCpuBlurW - 0.5f;
        float fy = v * (float)earthNightCpuBlurH - 0.5f;
        int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
        float tx = fx - (float)x0, ty = fy - (float)y0;
        auto wrapX = [this](int x)
        { return ((x % earthNightCpuBlurW) + earthNightCpuBlurW) % earthNightCpuBlurW; };
        int x0w = wrapX(x0), x1w = wrapX(x0 + 1);
        int y0c = std::clamp(y0, 0, earthNightCpuBlurH - 1);
        int y1c = std::clamp(y0 + 1, 0, earthNightCpuBlurH - 1);
        float v00 = earthNightCpuBlur[y0c * earthNightCpuBlurW + x0w] / 255.0f;
        float v10 = earthNightCpuBlur[y0c * earthNightCpuBlurW + x1w] / 255.0f;
        float v01 = earthNightCpuBlur[y1c * earthNightCpuBlurW + x0w] / 255.0f;
        float v11 = earthNightCpuBlur[y1c * earthNightCpuBlurW + x1w] / 255.0f;
        return glm::mix(glm::mix(v00, v10, tx), glm::mix(v01, v11, tx), ty);
    };

    // Same brightness-response curve (kNightFloor/kCityCompressK) as the sky-glow/cloud
    // city-light effects in sat_sky.frag, so all of these read one consistent "how bright is
    // this city" signal instead of drifting out of tune with each other.
    const float kNightFloor = 0.06f, kCityCompressK = 0.08f;
    // 2 km near sample added (session 26 follow-up): the observer's *own* position can sit inside
    // a bright pixel while every 8+ km ring around it is already dark countryside (small/isolated
    // towns) — without a near sample the dome can miss the pollution source entirely and read as
    // "no effect" even directly under city lights. This is the direct analog of the old scalar's
    // distance-0 sample, which this replaced.
    //
    // Radii/falloff widened (session follow-up, per user feedback): the old 45km outer sample with
    // a 20km falloff scale meant a city's contribution was already down to ~10% by 45km and
    // essentially gone by 60-80km, so any gap between two cities much wider than that read as flat
    // pitch black — reads as far more localized than real skyglow, which stays visible over tens
    // to a hundred+ km for a sizeable city (Falchi et al., World Atlas of Artificial Night Sky
    // Brightness). Two more (90/150km) far radii plus a longer 35km falloff scale spread the same
    // weighted-max combine further out: close-in behaviour barely changes (all four original radii
    // still land at broadly the same relative weights), 45km roughly triples in contribution, and
    // the two new far radii add a gentle regional tail that only a genuinely large/bright city can
    // still reach by 150km. A spot with no city within that whole radius still correctly falls to
    // ~0 and stays dark enough for the Milky Way — this is reach, not a raised floor.
    const float kSampleRadiiM[6] = {2000.0f, 8000.0f, 20000.0f, 45000.0f, 90000.0f, 150000.0f};
    const float kRadiusFalloffM = 35000.0f;
    float obsLatRad = glm::radians(obsLatDeg);
    float obsLonRad = glm::radians(obsLonDeg);
    float cosObsLat = std::max(0.05f, cosf(obsLatRad)); // guard near the poles

    float mwRawMax = 0.0f; // max cityBrightness*altFalloff across sectors, BEFORE lightPollutionGain
                            // — feeds the Milky Way's own threshold (mwPollutionThresholdLo/Hi),
                            // deliberately independent of the gain slider that scales lightDomeAz[]
                            // for stars/satellites below.
    for (int sec = 0; sec < kNumLightSectors; ++sec)
    {
        float bearing = (float(sec) + 0.5f) * (2.0f * glm::pi<float>() / float(kNumLightSectors));
        float domeRaw = 0.0f;
        for (float D : kSampleRadiiM)
        {
            float dLat = (D / kEarthRadius) * cosf(bearing);
            float dLon = (D / kEarthRadius) * sinf(bearing) / cosObsLat;
            float sampleLatRad = glm::clamp(obsLatRad + dLat, -glm::half_pi<float>(), glm::half_pi<float>());
            float sampleLonRad = obsLonRad + dLon;
            while (sampleLonRad > glm::pi<float>())
                sampleLonRad -= 2.0f * glm::pi<float>();
            while (sampleLonRad < -glm::pi<float>())
                sampleLonRad += 2.0f * glm::pi<float>();

            float u = (sampleLonRad + glm::pi<float>()) / (2.0f * glm::pi<float>());
            float v = (0.5f * glm::pi<float>() - sampleLatRad) / glm::pi<float>();
            float lum = sampleDomeLum(u, v);
            float raw = std::max(0.0f, lum - kNightFloor);
            float weight = expf(-D / kRadiusFalloffM);
            domeRaw = std::max(domeRaw, raw * weight);
        }
        float cityBrightness = domeRaw / (domeRaw + kCityCompressK);
        // lightPollutionGain applied once at the source so satellites (via lightDomeBuf) and
        // stars (reading lightDomeAz[] directly) stay coherently scaled by construction — same
        // array, not two separately-tuned gains that could drift apart like daySuppression vs.
        // the stars' fixed kStarPollutionMaxDim did.
        // NOT clamped to 1.0 here (session 26 follow-up — this was a real bug): elevFalloff
        // (applied downstream by each consumer, ≤1 for anything off the horizon) was multiplying
        // an already-saturated value, so no gain above the point where cityBrightness*altFalloff*
        // gain first hit 1.0 (around gain≈5) could push non-horizon directions any brighter —
        // gain=500 read identically to gain=5. Leaving this unclamped lets high gain compensate
        // for elevFalloff's reduction; the final domeVal clamp downstream still bounds the result.
        lightDomeAz[sec] = cityBrightness * altFalloff * lightPollutionGain;
        mwRawMax = std::max(mwRawMax, cityBrightness * altFalloff);
    }
    mwPollutionRaw = mwRawMax;

    // Circular smoothing pass (session 26 follow-up): each sector is a single bearing ray, so a
    // real city's edge — which doesn't line up with 22.5° sector boundaries — can put a bright
    // sector directly next to a dark one. Center-to-center interpolation (in the consumers) only
    // smooths *within* a sector's span; it doesn't reduce how different two neighboring raw
    // values are, so sharp swings still read as fast, unsubtle pops when panning across a sector
    // boundary — worst exactly at the horizon, where elevFalloff is largest and fully exposes any
    // sampling noise. 5-tap blur (~±45°) trades a little directional sharpness for removing that
    // noise while keeping the broad "city here, dark ocean there" structure intact.
    float smoothed[kNumLightSectors];
    const float kBlurWeights[5] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};
    for (int i = 0; i < kNumLightSectors; ++i)
    {
        float acc = 0.0f;
        for (int k = -2; k <= 2; ++k)
        {
            int idx = ((i + k) % kNumLightSectors + kNumLightSectors) % kNumLightSectors;
            acc += lightDomeAz[idx] * kBlurWeights[k + 2];
        }
        smoothed[i] = acc;
    }
    memcpy(lightDomeAz, smoothed, sizeof(smoothed));

    memcpy(lightDomeMapped, lightDomeAz, sizeof(lightDomeAz));
}

// ─── updateStars ──────────────────────────────────────────────────────────────
// Transforms star ECI unit vectors into ENU each frame (Earth rotates under stars).
// Altitude of a ray's closest approach to Earth's center, restricted to the forward ray (t >= 0) —
// CPU mirror of the GLSL rayTangentAltM in common.glsl (see that copy's comment for the full
// derivation/rationale). Used by updateStars()/updatePlanets() so a star/planet's atmospheric
// extinction is gated on how deep ITS OWN line of sight dips toward the ground, not on the
// observer's own altitude — an observer in orbit looking near the horizon still sends that ray
// through a long real atmospheric column even though the observer itself is above all air.
// maxT (meters, default effectively-infinite) bounds the closest-approach parameter to a finite
// target range along the ray, instead of the unconstrained infinite forward ray — needed for any
// target that actually SITS somewhere finite (a planet) rather than being effectively background/
// infinite (a star's direction-only eciDir). Mirrors the GLSL 3-arg overload added to
// common.glsl/sat_flare.comp for the same reason: a target appearing in front of Earth from a
// space-based observer has a ray direction pointing roughly toward the planet, so the unbounded
// tangent point can sit far beyond the target's own much closer actual position, wrongly applying
// extinction to a sightline that never gets anywhere near the atmosphere. The default (1e15, far
// beyond anything this sim renders — Saturn, the farthest body, sits at ~1.5e12 m) reproduces the
// old always-unbounded behavior exactly for every call site that doesn't pass its own maxT.
static inline float rayTangentAltM(const glm::vec3 &ro, const glm::vec3 &rd, float maxT = 1e15f)
{
    float t = glm::clamp(-glm::dot(ro, rd), 0.0f, maxT);
    glm::vec3 p = ro + rd * t;
    return glm::length(p) - kEarthRadius;
}

// Stars fade out during civil/nautical twilight — invisible in full daylight.
void SatelliteSim::updateStars()
{
    if (!starMapped || starCount == 0)
        return;

    // Stars become visible as the sun sinks below the horizon.
    // sin(elevation) = sunDirENU.w: 0 at horizon, -0.2 at ~11.5° below.
    float nightFactor = glm::clamp(-sunDirENU.w * 5.0f, 0.0f, 1.0f);

    // In space the sky is dark regardless of sun angle — no atmosphere to scatter.
    // atmFrac used to decay with the same 80 km scale height used for satellites' orbital day
    // suppression (correct for them — satellites fly hundreds of km up) — but reused here that
    // decayed too fast for a still-in-atmosphere observer: even a modest few-km cloud-deck
    // altitude already left atmFrac ~0.9, leaking ~10% of full night brightness into a clear
    // daytime sky, visible on the brightest stars. Real daytime sky glow doesn't meaningfully
    // thin until far above any cloud deck, so hold atmFrac at 1.0 (fully day/night gated, no
    // leak) through the whole flyable atmosphere and only fade toward "space, nothing to hide
    // behind" over the last stretch of the simulated atmosphere shell (R_ATMOS - R_EARTH = 100 km).
    const float kStarSpaceFadeStartM = 40000.0f;
    const float kStarSpaceFadeEndM = 100000.0f;
    float obsR = glm::length(obsECI);
    float obsHeight = obsR - kEarthRadius;
    float atmFrac = 1.0f - glm::clamp((obsHeight - kStarSpaceFadeStartM) / (kStarSpaceFadeEndM - kStarSpaceFadeStartM), 0.0f, 1.0f);
    // skyGlareEased (computed once per frame in recordCompute(), right before this call) replaces
    // the old flat 1.0 space target — sun-on-screen or unshielded sunlight still gates visibility
    // even with no atmosphere left to explain it away. At atmFrac==1 (fully in-atmosphere) this
    // has no effect, since it's weighted out by (1-atmFrac)==0 and nightFactor alone governs.
    float nightFactorEff = glm::mix(skyGlareEased, nightFactor, atmFrac);

    // Earth-limb elevation cutoff: from altitude, stars are visible below the 0° horizon.
    float r = kEarthRadius / obsR;
    float limbSin = (obsHeight > 1.0f) ? -sqrtf(glm::max(0.0f, 1.0f - r * r)) : 0.0f;

    // Light pollution dome caps: kStarPollutionMaxDim caps how dark the directional dome
    // (lightDomeAz[], filled by updateLightPollutionDome() just above) can push a star, so the
    // brightest stars/planets still peek through even in a maximally lit city, like reality.
    const float kStarPollutionMaxDim = 0.99f;

    // Moonlight sky-brightness dimming: same physical ramp (elevation × phase illumination) as
    // sat_flare.comp's moonBright term, computed here CPU-side since stars are drawn from this
    // same per-frame pass. Reuses moonDirENU.w (illuminated fraction, already computed this frame
    // in updatePositions()) instead of re-deriving it from sunDirECI/moonDirECI. Unlike satellites'
    // user-tunable moonSuppression gain, this uses a fixed response cap — mirrors how nightFactor
    // above and kStarPollutionMaxDim are both fixed formulas with no settings-window knob; stars
    // have never exposed per-suppression-source sliders, only the geometry-driven inputs.
    float moonElevStar = glm::dot(moonDirECI, glm::normalize(obsECI));
    float tmStar = glm::clamp(moonElevStar / 0.5f, 0.0f, 1.0f);
    float moonBrightStar = tmStar * tmStar * moonDirENU.w;
    const float kStarMoonMaxDim = 0.9f; // full moon dims naked-eye stars but never Venus/Jupiter/Sirius

    auto *dst = static_cast<GpuSatVisible *>(starMapped);
    for (uint32_t i = 0; i < starCount; ++i)
    {
        const auto &rec = starRecords[i];

        // Rotate from inertial ECI into the observer's local ENU frame.
        glm::vec3 enu{glm::dot(rec.eciDir, glm::vec3(eci2enuX)),
                      glm::dot(rec.eciDir, glm::vec3(eci2enuY)),
                      glm::dot(rec.eciDir, glm::vec3(eci2enuZ))};

        // Directional light pollution: same interpolated-dome/elevFalloff formula as
        // sat_flare.comp, so stars and satellites read the same dome consistently in a given
        // direction. Interpolated between the two nearest sector CENTERS (not hard-binned) —
        // 16 discrete wedges still showed visible blocky transitions over wide, fairly uniform
        // bright regions (e.g. flying over Europe).
        float bearing = atan2f(enu.x, enu.y); // matches GPU's atan(skyDir.x, skyDir.y) convention
        if (bearing < 0.0f)
            bearing += 2.0f * glm::pi<float>();
        float secF = bearing * (float(kNumLightSectors) / (2.0f * glm::pi<float>())) - 0.5f;
        int sec0 = (int)floorf(secF);
        float secFrac = secF - float(sec0);
        int sec0w = ((sec0 % kNumLightSectors) + kNumLightSectors) % kNumLightSectors;
        int sec1w = (sec0w + 1) % kNumLightSectors;
        float domeAz = glm::mix(lightDomeAz[sec0w], lightDomeAz[sec1w], secFrac);
        // 0.35 (not 0.15): matches the sat_flare.comp softening — the steeper curve crushed the
        // effect to near-zero above ~20° elevation, where most visible stars actually sit.
        float elevFalloff = 0.35f / (std::max(enu.z, 0.0f) + 0.35f); // 1.0 at horizon, ~0.26 at zenith
        // domeAz is intentionally unclamped upstream — the clamp only happens here, after
        // elevFalloff, so high lightPollutionGain can compensate for elevFalloff's reduction at
        // non-horizon angles instead of saturating uselessly before elevFalloff is even applied.
        // S2c (RELEASE_v1_1_PLAN.md): elevFalloff alone bottoms out at ~0.26 at zenith, wrong for
        // real (isotropically-scattered) urban skyglow, which raises zenith brightness far more
        // than that. kIsotropicFrac gives the dome a real floor at zenith; horizon behaviour is
        // unchanged. Same constant/formula in sat_flare.comp, sat_sky.frag (Milky Way), and
        // cloud_march.comp (aurora) — keep them coherent.
        const float kIsotropicFrac = 0.4f;
        float domeVal = glm::clamp(domeAz * (kIsotropicFrac + (1.0f - kIsotropicFrac) * elevFalloff), 0.0f, 1.0f);

        // C12 follow-up #31: same suppression shape, second independent source — a nearby
        // Reflect-Orbital beam should wash out this star the same way real light pollution does.
        // beamGlowDomeAz[] is the one-frame-stale CPU readback of sat_orbit.comp's atomicMax'd
        // dome (see recordCompute()'s top-of-frame read), same interpolation convention as
        // lightDomeAz above.
        float beamDomeAz = glm::mix(beamGlowDomeAz[sec0w], beamGlowDomeAz[sec1w], secFrac);
        float beamDomeVal = glm::clamp(beamDomeAz * elevFalloff, 0.0f, 1.0f);
        const float kStarBeamPollutionMaxDim = 0.99f;

        // Atmospheric extinction (airmass) — same Kasten & Young 1989 approximation as
        // sat_flare.comp, applied identically so a star and a satellite at the same elevation
        // dim by the same amount. Independent of light pollution/moon; this is what gives the
        // pollution dome's directional variation a smooth baseline to sit on top of instead of
        // being the only source of horizon-vs-zenith brightness difference.
        // Gated on THIS star's own tangent altitude (rayTangentAltM above), not the observer-height
        // atmFrac used for nightFactorEff above — a star near the observer's local horizon can
        // still have a line of sight that grazes deep into the atmosphere even from orbit.
        float sinElClamped = glm::clamp(enu.z, 0.0f, 1.0f);
        float elDeg = glm::degrees(asinf(sinElClamped));
        float airmass = 1.0f / (sinElClamped + 0.50572f * powf(elDeg + 6.07995f, -1.6364f));
        float atmFracExtinct = expf(-std::max(rayTangentAltM(obsECI, rec.eciDir), 0.0f) / 80000.0f);
        float extinctMag = extinctionCoeff * (airmass - 1.0f) * atmFracExtinct;
        float extinction = powf(10.0f, -0.4f * extinctMag);

        // Above the Earth limb: visible. Below: culled.
        float intensity = (enu.z >= limbSin)
                              ? rec.rawIntensity * nightFactorEff * extinction * (1.0f - domeVal * kStarPollutionMaxDim) * (1.0f - beamDomeVal * kStarBeamPollutionMaxDim) * (1.0f - moonBrightStar * kStarMoonMaxDim)
                              : 0.0f;

        dst[i].skyDir = enu;
        dst[i].flareIntensity = intensity;
        dst[i].baseColor = rec.color;
        dst[i].angularSize = rec.angSize;
    }
}

// ─── updatePlanets ────────────────────────────────────────────────────────────
// Mirrors updateStars() above — same suppression chain (day/moon/pollution-dome/extinction),
// same point-sprite size convention — but unlike stars, planetStates[].eciDir moves every frame
// (updatePositions() recomputed it this frame), so this recomputes ENU direction AND brightness
// from scratch each call rather than only re-touching brightness on a fixed direction. Duplicated
// suppression math rather than shared with updateStars() — matches this codebase's established
// per-consumer-duplication convention for this exact formula (see CLAUDE.md's "Subsystem: Light
// Pollution Dome").
void SatelliteSim::updatePlanets()
{
    if (!planetMapped)
        return;

    if (!showPlanets)
    {
        auto *dst = static_cast<GpuSatVisible *>(planetMapped);
        for (int i = 0; i < kPlanetCount; ++i)
            dst[i].flareIntensity = 0.0f;
        return;
    }

    float nightFactor = glm::clamp(-sunDirENU.w * 5.0f, 0.0f, 1.0f);
    const float kPlanetSpaceFadeStartM = 40000.0f;
    const float kPlanetSpaceFadeEndM = 100000.0f;
    float obsR = glm::length(obsECI);
    float obsHeight = obsR - kEarthRadius;
    float atmFrac = 1.0f - glm::clamp((obsHeight - kPlanetSpaceFadeStartM) / (kPlanetSpaceFadeEndM - kPlanetSpaceFadeStartM), 0.0f, 1.0f);
    float nightFactorEff = glm::mix(skyGlareEased, nightFactor, atmFrac);

    float r = kEarthRadius / obsR;
    float limbSin = (obsHeight > 1.0f) ? -sqrtf(glm::max(0.0f, 1.0f - r * r)) : 0.0f;

    const float kPlanetPollutionMaxDim = 0.99f;
    const float kPlanetBeamPollutionMaxDim = 0.99f;

    float moonElevP = glm::dot(moonDirECI, glm::normalize(obsECI));
    float tmP = glm::clamp(moonElevP / 0.5f, 0.0f, 1.0f);
    float moonBrightP = tmP * tmP * moonDirENU.w;
    const float kPlanetMoonMaxDim = 0.9f; // matches updateStars()'s kStarMoonMaxDim

    const float starScale = 4.0f; // must match initStars()'s starScale — same size convention so
                                  // a planet reads at the same visual weight as an equally-bright star

    auto *dst = static_cast<GpuSatVisible *>(planetMapped);
    for (int i = 0; i < kPlanetCount; ++i)
    {
        const PlanetState &ps = planetStates[i];

        glm::vec3 enu{glm::dot(ps.eciDir, glm::vec3(eci2enuX)),
                      glm::dot(ps.eciDir, glm::vec3(eci2enuY)),
                      glm::dot(ps.eciDir, glm::vec3(eci2enuZ))};

        if (!planetEnabled[i])
        {
            dst[i].skyDir = enu;
            dst[i].flareIntensity = 0.0f;
            continue;
        }

        float vmag = planetApparentMagnitude((PlanetId)i, ps.sunDistAU, ps.distanceAU, ps.phaseAngleDeg);
        float rawIntensity = powf(10.0f, -vmag / 2.5f);

        float bearing = atan2f(enu.x, enu.y);
        if (bearing < 0.0f)
            bearing += 2.0f * glm::pi<float>();
        float secF = bearing * (float(kNumLightSectors) / (2.0f * glm::pi<float>())) - 0.5f;
        int sec0 = (int)floorf(secF);
        float secFrac = secF - float(sec0);
        int sec0w = ((sec0 % kNumLightSectors) + kNumLightSectors) % kNumLightSectors;
        int sec1w = (sec0w + 1) % kNumLightSectors;
        float domeAz = glm::mix(lightDomeAz[sec0w], lightDomeAz[sec1w], secFrac);
        float elevFalloff = 0.35f / (std::max(enu.z, 0.0f) + 0.35f);
        const float kIsotropicFrac = 0.4f; // S2c — same constant as updateStars()/sat_flare.comp
        float domeVal = glm::clamp(domeAz * (kIsotropicFrac + (1.0f - kIsotropicFrac) * elevFalloff), 0.0f, 1.0f);

        float beamDomeAz = glm::mix(beamGlowDomeAz[sec0w], beamGlowDomeAz[sec1w], secFrac);
        float beamDomeVal = glm::clamp(beamDomeAz * elevFalloff, 0.0f, 1.0f);

        // Gated on this planet's own tangent altitude, not the observer-height atmFrac used for
        // nightFactorEff above — same fix, same reason, as updateStars()'s copy of this block.
        // Bounded by the planet's own real distance (unlike stars, a planet has one) — same
        // in-front-of-Earth fix as the satellite call site in sat_flare.comp: without this, a
        // planet whose ray direction happens to point roughly toward Earth (a rare but real transit/
        // occultation-adjacent geometry) would wrongly pick up extinction from the unbounded ray's
        // tangent point sitting near the surface, far beyond where the planet actually is.
        const float kAuM = 1.495978707e11f; // IAU-defined astronomical unit, meters
        float sinElClamped = glm::clamp(enu.z, 0.0f, 1.0f);
        float elDeg = glm::degrees(asinf(sinElClamped));
        float airmass = 1.0f / (sinElClamped + 0.50572f * powf(elDeg + 6.07995f, -1.6364f));
        float atmFracExtinct = expf(-std::max(rayTangentAltM(obsECI, ps.eciDir, ps.distanceAU * kAuM), 0.0f) / 80000.0f);
        float extinctMag = extinctionCoeff * (airmass - 1.0f) * atmFracExtinct;
        float extinction = powf(10.0f, -0.4f * extinctMag);

        float intensity = (enu.z >= limbSin)
                              ? rawIntensity * nightFactorEff * extinction * (1.0f - domeVal * kPlanetPollutionMaxDim) * (1.0f - beamDomeVal * kPlanetBeamPollutionMaxDim) * (1.0f - moonBrightP * kPlanetMoonMaxDim)
                              : 0.0f;

        // Same size curve as initStars() (S2a, RELEASE_v1_1_PLAN.md) — floors near 1px for a
        // faint/distant planet, grows for a bright one like Venus.
        float angSize = (1.0f + 1.0f * sqrtf(std::max(rawIntensity, 0.0f))) * starScale;

        dst[i].skyDir = enu;
        dst[i].flareIntensity = intensity;
        dst[i].baseColor = kPlanetColor[i]; // hand-picked approximate true color — see its own comment
        dst[i].angularSize = angSize;
    }
}

// ─── initConstellation ────────────────────────────────────────────────────────
// Entry point called once from init().  Loads satellite type and constellation
// definitions (from constellations.json or hardcoded fallback), then builds the
// flat satOrbits array that drives per-frame position updates.
void SatelliteSim::initConstellation()
{
    loadDefinitions();
    buildOrbits();
}

// ─── JSON helpers (file-local) ────────────────────────────────────────────────
static AttitudeMode parseAttitudeMode(const std::string &s)
{
    if (s == "NadirPointing")
        return AttitudeMode::NadirPointing;
    if (s == "SunTracking")
        return AttitudeMode::SunTracking;
    if (s == "Tumbling")
        return AttitudeMode::Tumbling;
    if (s == "Perpendicular")
        return AttitudeMode::Perpendicular;
    if (s == "AntiNadir")
        return AttitudeMode::AntiNadir;
    if (s == "FlatMirror45")
        return AttitudeMode::FlatMirror45;
    if (s == "TargetedReflector")
        return AttitudeMode::TargetedReflector;
    if (s == "KnifeEdge")
        return AttitudeMode::KnifeEdge;
    if (s == "SunPerp")
        return AttitudeMode::SunPerp;
    if (s == "SunTrackingTilted")
        return AttitudeMode::SunTrackingTilted;
    fprintf(stderr, "[SatelliteSim] Unknown AttitudeMode '%s'; using NadirPointing.\n", s.c_str());
    return AttitudeMode::NadirPointing;
}

static SurfaceSpec parseSurfaceSpec(const nlohmann::json &j)
{
    return {
        parseAttitudeMode(j.value("attitude", std::string("NadirPointing"))),
        j.value("spec_exp", 0.0f),
        j.value("weight", 0.0f),
    };
}

// ─── loadDefinitions ─────────────────────────────────────────────────────────
// Reads constellations.json from the exe directory.  If the file is missing or
// malformed, falls back to loadHardcoded() and logs the reason to stderr.
void SatelliteSim::loadDefinitions()
{
    auto jsonPath = (std::filesystem::path(exeDir_) / "constellations.json").string();
    std::ifstream f(jsonPath);
    if (!f.is_open())
    {
        fprintf(stderr, "[SatelliteSim] constellations.json not found at '%s';"
                        " using hardcoded defaults.\n",
                jsonPath.c_str());
        loadHardcoded();
        return;
    }

    nlohmann::json j;
    try
    {
        f >> j;
    }
    catch (const nlohmann::json::exception &e)
    {
        fprintf(stderr, "[SatelliteSim] Failed to parse constellations.json: %s\n"
                        "              Using hardcoded defaults.\n",
                e.what());
        loadHardcoded();
        return;
    }

    // ── Satellite types ───────────────────────────────────────────────────────
    satTypes.clear();
    for (const auto &jt : j.value("satellite_types", nlohmann::json::array()))
    {
        SatelliteType t;
        t.name = jt["name"].get<std::string>();
        auto col = jt["base_color"];
        t.baseColor = {col[0].get<float>(), col[1].get<float>(), col[2].get<float>()};
        t.crossSectionM2 = jt["cross_section_m2"].get<float>();
        t.primary = parseSurfaceSpec(jt["primary"]);
        t.secondary = jt.contains("secondary")
                          ? parseSurfaceSpec(jt["secondary"])
                          : SurfaceSpec{AttitudeMode::Perpendicular, 0.0f, 0.0f};
        t.diffuse = jt.value("diffuse", 0.02f);
        t.mirrorFrac = jt.value("mirror_frac", 0.0f);
        satTypes.push_back(std::move(t));
    }

    // Build name → index map so constellations can reference types by name.
    std::unordered_map<std::string, uint32_t> typeMap;
    for (uint32_t i = 0; i < (uint32_t)satTypes.size(); ++i)
        typeMap[satTypes[i].name] = i;

    // ── Constellations ────────────────────────────────────────────────────────
    constellations.clear();
    for (const auto &jc : j.value("constellations", nlohmann::json::array()))
    {
        ConstellationConfig c;
        c.name = jc["name"].get<std::string>();
        c.numPlanes = jc["num_planes"].get<int>();
        c.perPlane = jc["per_plane"].get<int>();
        c.altM = jc["alt_km"].get<float>() * 1000.0f;
        c.incl = glm::radians(jc.value("incl_deg", 0.0f));
        c.enabled = jc.value("enabled", true);

        std::string typeName = jc["type"].get<std::string>();
        auto it = typeMap.find(typeName);
        if (it == typeMap.end())
        {
            fprintf(stderr, "[SatelliteSim] Constellation '%s' references unknown type '%s';"
                            " skipping.\n",
                    c.name.c_str(), typeName.c_str());
            continue;
        }
        c.typeIdx = it->second;

        std::string dist = jc.value("distribution", std::string("Walker"));
        if (dist == "RandomShell")
            c.distribution = OrbitDistribution::RandomShell;
        else if (dist == "Disk")
            c.distribution = OrbitDistribution::Disk;
        else
            c.distribution = OrbitDistribution::Walker;

        c.altJitterM = jc.value("alt_jitter_km", 0.0f) * 1000.0f;
        c.raan = glm::radians(jc.value("raan_deg", 0.0f));
        c.alignTerminator = jc.value("align_terminator", false);
        c.numRings = jc.value("num_rings", 1);
        c.ringSpacingM = jc.value("ring_spacing_km", 0.0f) * 1000.0f;

        constellations.push_back(std::move(c));
    }

    hovConst.assign(constellations.size(), false);
    hovHighlightConst.assign(constellations.size(), false);
    fprintf(stderr, "[SatelliteSim] Loaded %zu satellite type(s) and %zu constellation(s)"
                    " from %s\n",
            satTypes.size(), constellations.size(), jsonPath.c_str());
}

// ─── loadHardcoded ────────────────────────────────────────────────────────────
// Hardcoded satellite type catalogue and constellation shells — used as fallback
// when constellations.json is absent or malformed.
void SatelliteSim::loadHardcoded()
{
    // AI SAT
    // 175 m estimate, 1820 x 72
    // scale 1820px / 175 m
    // 870 * 72 px panels
    // 200 x 55 px radiator panels

    float px_scale = 1820.0f / 175.0f;                                                      // ~10.4 px/meter for the bus/antenna face
    float ai_sat_panel_area = (870.0f * 72.0f * 2.0f) / (px_scale * px_scale);              // ~1158 m² total panel area
    [[maybe_unused]] float ai_sat_radiator_area = (200.0f * 55.0f) / (px_scale * px_scale); // ~102 m² total radiator area

    satTypes = {
        {// 0 — Starlink: flat phased-array face toward Earth, brief intense flares
         "Starlink",
         {0.80f, 0.87f, 1.00f},                      // cool blue-white
         10.0f,                                      // ~10 m² bus + visor
         {AttitudeMode::NadirPointing, 18.0f, 1.0f}, // very sharp specular (flat mirror-like face)
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},  // no significant secondary surface
         0.01f,                                      // no diffuse floor (visor-darkened)
         0.05f},                                     // mirrorFrac: polished phased-array glass → mag ~-2.7 at perfect alignment
        {                                            // 1 — LEO broadband (OneWeb/Kuiper/Xingwang/Telesat): sun-tracking panels
         "LEO Broadband",
         {1.00f, 0.92f, 0.75f}, // warm white
         5.0f,                  // ~12 m² typical LEO broadband bus + panels
         {AttitudeMode::SunTracking, 18.0f, 1.0f},
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},
         0.01f,  // no diffuse floor
         0.02f}, // mirrorFrac: moderate — sun-tracking panels occasionally flash
        {        // 2 — GEO Comsat: large sun-tracking panels + body radiators facing away from Earth
         "GEO Comsat",
         {0.95f, 0.95f, 1.00f},                    // near-white
         50.0f,                                    // ~50 m² (large GEO body + wings)
         {AttitudeMode::SunTracking, 3.0f, 1.00f}, // broad lobe solar wings
         {AttitudeMode::AntiNadir, 2.0f, 0.10f},   // body radiators face deep space
         0.05f,                                    // slight structural glow
         0.10f},                                   // mirrorFrac: large polished antenna dishes, well-aligned
        {                                          // 3 — ISS: enormous truss-mounted solar arrays AND large radiator panels.
         // The PVTCS and EATCS radiators (~900 m² NH3 panels on the ITS) face away from
         // Earth for maximum view factor to cold space. From the ground: ISS at zenith shows
         // the back of the radiators (dim); ISS near the horizon shows the radiator face.
         "ISS",
         {1.00f, 0.85f, 0.70f}, // warm golden (solar array color)
         250.0f,
         {AttitudeMode::SunTracking, 12.0f, 1.00f}, // truss-mounted solar arrays
         {AttitudeMode::AntiNadir, 4.0f, 0.35f},    // large radiator panels (ITS) face deep space
         0.04f,                                     // complex truss/module body
         0.05f},                                    // mirrorFrac: highly polished solar panel glass → mag ~-7.5 at peak
        {                                           // 4 — SpaceX AI1 datacenter satellite (revealed 2026).
         // Bus: nadir-pointing (phased-array antenna faces Earth). Modeled via diffuse.
         // Solar arrays: ~600 m², two-axis tracked (bus yaw + panel gimbal) → always face sun.
         // Radiators: 110 m² deployable liquid panels, hard-mounted perpendicular to bus.
         //   The bus yaws to let solar wings track the sun, which constrains the radiators
         //   to normal = cross(sunDir, satNadir) — always edge-on to the sun by design.
         //   irr = 0 always (correct: radiator must never see the sun to reject heat).
         //   Visual contribution is through the diffuse parameter (large structure scatter).
         // crossSection = sqrt(600/10) ≈ 7.75 for the dominant solar wing area.

         "SpaceX AI Sats",
         {1.00f, 1.00f, 0.92f},                          // cyan-teal (distinct from Starlink blue-white)
         600.0f,                                         // 600 m² solar array area (150 kW / 250 W/m²)
         {AttitudeMode::SunTrackingTilted, 25.0f, 1.0f}, // solar wings — sun-tracking with the global
                                                          // flareMitigationTiltDeg pitch applied (see
                                                          // that enum value's comment); tiltDeg=0
                                                          // is bit-identical to plain SunTracking
         {AttitudeMode::SunPerp, 3.0f, 0.18f},           // radiators 110 m² — edge-on to sun, irr=0
         0.06f,                                          // bus body + radiator structure bulk scatter
         0.01f},                                         // mirrorFrac: polished solar panel glass
        {                                          // 5 — Reflect Orbital mirror (speculative, 55 m diameter flat mirror).
         // FlatMirror45: normal = normalize(sunDir + satNadir).
         // By construction reflect(-sunDir, n) = satNadir — reflected sunlight
         // hits the ground directly below the satellite.  In SSO the mirror spends
         // most of its time near the terminator, so ground observers see a brilliant
         // slow-moving point of light just before dawn or just after dusk.
         //
         // To switch to targeted multi-beam focusing, change AttitudeMode to
         // TargetedReflector and call the orbit post-processing loop below.
         //
         // Area: 55 m diameter circular mirror → π × 27.5² ≈ 2,376 m²
         // crossSection = sqrt(2376/10) ≈ 15.4  (vs. Starlink ~1.0)
         // At 500 km, perfect alignment: peak effectFlare ≈ 10,000+ → mag ≈ −11
         // (comparable to a quarter moon; visible in daylight sky)
         "Reflect Mirror",
         {1.00f, 0.97f, 0.94f},                           // warm silver-white
         2376.0f,                                         // 55 m diameter mirror area (m²)
         {AttitudeMode::TargetedReflector, 200.0f, 1.0f}, // near-perfect flat mirror; tight but not laser-narrow lobe
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},       // no secondary surface
         0.02f,                                           // no diffuse scatter (mirror absorbs nothing)
         0.97f},                                          // mirrorFrac: near-perfect specular mirror
        {                                                 // 6 — Space debris: defunct satellites, rocket bodies, fragments.
         // Tumbling attitude — chaotic rotation around a random body axis.
         // Rate varies per object from near-stationary to ~1 Hz flicker.
         // Small area, rough surfaces, no attitude control.
         "Debris",
         {0.78f, 0.74f, 0.68f},                     // dull grey-tan (aged thermal blanket / oxidised metal)
         1.0f,                                      // ~3 m² effective cross-section (fragment to small bus)
         {AttitudeMode::Tumbling, 6.0f, 1.0f},      // rough diffuse tumble; occasional glint
         {AttitudeMode::Perpendicular, 6.0f, 0.0f}, // no secondary surface
         0.001f,                                    // high diffuse floor — structural clutter scatters everywhere
         0.03f},                                    // rare metallic glints from exposed foil or polished surfaces
        {                                           // 7 — Starlink (knife-edge): SpaceX roll-angle policy adopted 2020.
         // Body rolls around the along-track axis so the phased-array face is
         // edge-on to the sun; solar panel gimbals counter-rotate to compensate.
         // Roll is clamped to ±kKnifeMaxRollDeg (80°) — solar power constraint.
         // Measured effect: ~90% brightness reduction vs. NadirPointing at standard
         // distance (Mallama & Respler 2023, 2303.01431).  Residual brightness at
         // the clamp limit: specular ∝ cos(80°) ≈ 0.17 of fully-lit nadir face.
         "Starlink KE",
         {0.80f, 0.87f, 1.00f},                     // same cool blue-white as Starlink
         10.0f,                                     // same bus area
         {AttitudeMode::KnifeEdge, 18.0f, 1.0f},    // sharp specular; normal set by roll solver
         {AttitudeMode::Perpendicular, 0.0f, 0.0f}, // no secondary surface
         0.01f,                                     // visor-darkened diffuse floor
         0.05f},                                    // same polished array face as baseline Starlink
    };

    // ── Constellation shells ───────────────────────────────────────────────────
    // Walker:  numPlanes × perPlane satellites, regular spacing.
    // Random:  numPlanes satellites total, random orbital params.
    // Disk:    numPlanes satellites in one or more concentric rings.
    //   .altJitterM   = per-satellite altitude scatter (random)
    //   .raan         = orbital plane RAAN (unless alignTerminator=true)
    //   .alignTerminator = derive incl+raan from sunDirECI at init
    //   .numRings     = concentric rings (1 = single ring)
    //   .ringSpacingM = altitude step between rings
    // ── Real mega-constellation data (source: planet4589.org/space/con/conlist.html) ──
    // Walker field order: name, altM, incl, numPlanes, perPlane, typeIdx, enabled, distribution
    //   total sats = numPlanes × perPlane
    // Disk field order (extra trailing args): ..., altJitterM, raan, alignTerminator, numRings, ringSpacingM
    //   total sats = numPlanes × perPlane, spread evenly across numRings concentric rings
    //   alignTerminator=true: overrides incl+raan to track sunDirECI (orbital plane = terminator plane)
    // All totals fit within MAX_SATELLITES=100,000 when all enabled simultaneously (~98,907).

    constellations = {
        // SpaceX Starlink Gen1 — FCC filing: 4,408 sats; 72 planes × 61 = 4,392
        {"Starlink Gen1",
         550'000.0f,          // altM:      550 km
         glm::radians(53.0f), // incl:      53° — primary mid-inclination shell
         72,                  // numPlanes: orbital planes
         61,                  // perPlane:  sats per plane (72×61 = 4,392)
         0u,                  // typeIdx:   Starlink (NadirPointing)
         true,                // enabled
         OrbitDistribution::Walker},

        // SpaceX Starlink Gen2 — FCC filing: 30,456 sats; 120 planes × 254 = 30,480
        // Uses knife-edge roll (type 7) to model SpaceX's 2020 roll-angle policy.
        {"Starlink Gen2",
         525'000.0f,          // altM:      525 km (slightly lower than Gen1)
         glm::radians(53.2f), // incl:      53.2°
         120,                 // numPlanes: orbital planes
         254,                 // perPlane:  sats per plane (120×254 = 30,480)
         7u,                  // typeIdx:   Starlink KE (KnifeEdge roll)
         true,                // enabled
         OrbitDistribution::Walker},

        // OneWeb (UK/Eutelsat) — planned: 648 sats; 18 planes × 36 = 648
        {"OneWeb",
         1'200'000.0f,        // altM:      1,200 km
         glm::radians(87.9f), // incl:      87.9° — near-polar
         18,                  // numPlanes: orbital planes
         36,                  // perPlane:  sats per plane (18×36 = 648)
         1u,                  // typeIdx:   LEO Broadband (SunTracking)
         true,                // enabled
         OrbitDistribution::Walker},

        // Amazon Kuiper — FCC filing: 7,774 sats; 98 planes × 79 = 7,742
        {"Amazon LEO",
         630'000.0f,          // altM:      630 km
         glm::radians(51.9f), // incl:      51.9°
         98,                  // numPlanes: orbital planes
         79,                  // perPlane:  sats per plane (98×79 = 7,742)
         1u,                  // typeIdx:   LEO Broadband (SunTracking)
         true,                // enabled
         OrbitDistribution::Walker},

        // China Xingwang/GW (CASC/CASIC) — planned: ~13,952 sats; 80 planes × 174 = 13,920
        {"Guowang",
         508'000.0f,          // altM:      508 km
         glm::radians(85.0f), // incl:      85° — near-polar
         80,                  // numPlanes: orbital planes
         174,                 // perPlane:  sats per plane (80×174 = 13,920)
         1u,                  // typeIdx:   LEO Broadband (SunTracking)
         true,                // enabled
         OrbitDistribution::Walker},

        // International Space Station — single object for visual reference
        {"ISS",
         408'000.0f,          // altM:      408 km
         glm::radians(51.6f), // incl:      51.6°
         1,                   // numPlanes: 1 plane
         1,                   // perPlane:  1 satellite
         3u,                  // typeIdx:   ISS (SunTracking + large radiators)
         true,                // enabled
         OrbitDistribution::Walker},

        // SpaceX Orbital Data Center — sun-synchronous Disk shell (FCC filing Jan 2026)
        //   Disk+alignTerminator places the ring in the Earth-Sun terminator plane, visually
        //   representing where SSO satellites dwell relative to the day/night boundary.
        //   200 × 100 = 20,000 sats spread across 10 rings from ~575 km to ~1,925 km.
        {"SpaceX AI Sat",
         1'250'000.0f, // altM:      1,250 km — ring centre altitude
         0.0f,         // incl:      ignored (alignTerminator=true overrides)
         200,          // numPlanes: × perPlane = total sats (200×100 = 20,000)
         100,          // perPlane:  × numPlanes = total sats
         4u,           // typeIdx:   SpaceX ODC (NadirPointing + AntiNadir radiators)
         true,         // enabled
         OrbitDistribution::Disk,
         5000.0f,         // altJitterM:      no per-satellite altitude scatter
         0.0f,            // raan:            ignored (alignTerminator=true)
         true,            // alignTerminator: orbital plane = terminator plane (tracks Sun)
         10 * 2,          // numRings:        10 concentric rings
         150'000.0f / 2}, // ringSpacingM:    150 km between rings (575–1,925 km range)

        // Reflect Orbital — speculative 55 m flat mirror constellation (SSO Walker).
        // FlatMirror45 attitude keeps mirror normal = normalize(sunDir+satNadir) each frame,
        // reflecting sunlight straight down to the ground below.
        // Disabled by default: enabling while all other constellations are on exceeds
        // MAX_SATELLITES=100,000.  Toggle others off first, or raise the cap.
        //
        // To enable focused multi-beam targeting (10 ground spots, ~500 mirrors each):
        //   1. Change typeIdx to 5 — switch sat type primary to TargetedReflector.
        //      (Add a type-6 TargetedReflector variant and reference it here, or edit type-5.)
        //   2. Ensure the post-processing loop below runs (it already does for typeIdx==5).
        {"Reflect Orbital",
         500'000.0f, // altM:   500 km — low LEO for maximum ground flux
         0,          // incl:   SSO retrograde (~97.4°) from J2 formula
         10,         // numPlanes: orbital planes
         100,        // perPlane:  50×100 = 5,000 satellites
         5u,         // typeIdx:   Reflect Mirror (FlatMirror45)
         true,       // enabled:   OFF — enabling pushes total > MAX_SATELLITES
         OrbitDistribution::Disk,
         1000.0f, // altJitterM:      no per-satellite altitude scatter
         0.0f,    // raan:            ignored (alignTerminator=true)
         true,    // alignTerminator: orbital plane = terminator plane (tracks Sun)
         10,      // numRings:        10 concentric rings
         10000},  // ringSpacingM:    150 km between rings (575–1,925 km range)

        // Space Junk — LEO debris shell modelling defunct satellites, rocket bodies,
        // and large fragments.  Random inclinations (0–180°) give isotropic coverage.
        // Each object gets an independent tumble axis + rate (0–1 Hz) so flickers
        // are desynchronised across the shell.
        {"Space Junk",
         1'000'000.0f,     // altM:       600 km — centre of dense LEO debris band
         glm::pi<float>(), // incl: random 0..180° → full spherical coverage
         100,              // numPlanes:  }
         30,               // perPlane:   } 1 × 3,000 = 3,000 debris objects
         6u,               // typeIdx:    Debris (typeIdx 6)
         true,             // enabled
         OrbitDistribution::RandomShell,
         500'000.0f}, // altJitterM: ±200 km → 400–800 km altitude band
    };

    hovConst.assign(constellations.size(), false);
    hovHighlightConst.assign(constellations.size(), false);
}

// ─── buildOrbits ──────────────────────────────────────────────────────────────
// Generates the flat satOrbits array from whatever is currently in satTypes and
// constellations (loaded by loadDefinitions or loadHardcoded).
// Also generates reflector ground targets and applies the MAX_SATELLITES cap.
void SatelliteSim::buildOrbits()
{
    // ── Populate satOrbits ────────────────────────────────────────────────────
    satOrbits.clear();
    for (ConstellationConfig &c : constellations)
    {
        c.orbitStart = (uint32_t)satOrbits.size();

        if (c.distribution == OrbitDistribution::Walker)
        {
            for (int p = 0; p < c.numPlanes; ++p)
            {
                float raan = (float)p / c.numPlanes * glm::two_pi<float>();
                for (int s = 0; s < c.perPlane; ++s)
                {
                    float u0 = (float)rand() / (float)RAND_MAX * glm::two_pi<float>();
                    satOrbits.push_back({raan, c.incl, u0, c.typeIdx, c.altM, 0.0f, 0.0f, {0.0f, 0.0f, 1.0f}, false});
                }
            }
        }
        else if (c.distribution == OrbitDistribution::RandomShell)
        {
            int total = c.numPlanes * c.perPlane;
            for (int i = 0; i < total; ++i)
            {
                float raan = (float)rand() / RAND_MAX * glm::two_pi<float>();
                float incl = (float)rand() / RAND_MAX * c.incl;
                float u0 = (float)rand() / RAND_MAX * glm::two_pi<float>();
                float jitter = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * c.altJitterM;
                float altM = c.altM + jitter;

                float phi = (float)rand() / RAND_MAX * glm::two_pi<float>();
                float cosTheta = (float)rand() / RAND_MAX * 2.0f - 1.0f;
                float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
                glm::vec3 axis{sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta};

                // Randomise rotation rate 0..2π rad/s (0..1 Hz) so debris
                // objects tumble independently rather than blinking in unison.
                float tumbleRate = (float)rand() / (float)RAND_MAX * glm::two_pi<float>() * 0.001;
                float tumblePhase = (float)rand() / RAND_MAX * glm::two_pi<float>();

                satOrbits.push_back({raan, incl, u0, c.typeIdx, altM,
                                     tumbleRate, tumblePhase, axis, false});
            }
        }
        else if (c.distribution == OrbitDistribution::Disk)
        {
            // Determine orbital plane.  For alignTerminator, RAAN is shared across all rings
            // but inclination is computed per-ring from each ring's actual altitude.
            float incl_d = c.incl;
            float raan_d = c.raan;
            if (c.alignTerminator)
            {
                // Anchor RAAN at the simulation start time using the sun direction already
                // computed by updatePositions().  uploadSatOrbits() then precesses only
                // (orbitEpochT0 - t_start) seconds forward, so liveRaan = raan_start +
                // kSSOPrecRate*(simTime - t_start).  Anchoring here rather than at J2000
                // eliminates the ~3° obliquity-driven phase error that accumulates when
                // extrapolating 36+ years with a constant precession rate.
                raan_d = atan2f(sunDirECI.x, -sunDirECI.y);
            }

            // Distribute satellites across numRings concentric rings.
            // The rings are centred on c.altM and spaced by c.ringSpacingM.
            int totalSats = c.numPlanes * c.perPlane;
            int nr = glm::max(1, c.numRings);
            int perRing = (totalSats + nr - 1) / nr; // ceiling division

            for (int r = 0; r < nr; ++r)
            {
                // Altitude: centre-offset each ring around c.altM.
                float ringAlt = c.altM + (r - (nr - 1) * 0.5f) * c.ringSpacingM;
                // For SSO constellations compute the exact J2 inclination for each ring's
                // altitude rather than using the centre altitude for all rings.  A 1500 km
                // span (e.g. 500–2000 km) otherwise biases every ring by up to ±3.7°.
                float ringIncl = c.alignTerminator ? computeSSOInclination(ringAlt) : incl_d;

                // Model incomplete constellation, vary number of sats per ring to fill totalSats without exceeding it.
                int satsInThisRing = glm::min(perRing, totalSats - r * perRing);

                for (int s = 0; s < satsInThisRing; ++s)
                {
                    // Evenly spaced around the ring + optional small jitter.
                    float u0 = (float)s / satsInThisRing * glm::two_pi<float>();
                    float jitter = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * c.altJitterM;
                    satOrbits.push_back({raan_d, ringIncl, u0, c.typeIdx, ringAlt + jitter, 0.0f, 0.0f, {0.0f, 0.0f, 1.0f}, c.alignTerminator});
                }
            }
        }

        c.orbitCount = (uint32_t)satOrbits.size() - c.orbitStart;

        // Stamp constIdx on every orbit that belongs to this constellation so
        // updatePositions() can look up highlight/enabled state by orbit index.
        uint32_t ci = (uint32_t)(&c - constellations.data());
        for (uint32_t oi = c.orbitStart; oi < c.orbitStart + c.orbitCount; ++oi)
            satOrbits[oi].constIdx = ci;
    }
    // ── TargetedReflector ground targets (S1, RELEASE_v1_1_PLAN.md) ─────────
    // Real solar-farm sites loaded from reflector_targets.json, falling back to random points —
    // see loadReflectorTargets(). Must run after createGlowResources() has populated earthElevCpu
    // (same ordering constraint the old inline code here had).
    loadReflectorTargets();

    // ── Safety cap ────────────────────────────────────────────────────────────
    // satInputBuf and satVisibleBuf are allocated for exactly MAX_SATELLITES
    // entries.  Exceeding this causes a buffer overflow in recordCompute()'s
    // memcpy, corrupting heap memory or triggering a GPU fault.  Satellites
    // beyond the cap are silently dropped.
    //
    // Common overflow source: Starlink G1 at 7200 planes × 22 sats = 158,400 —
    // already 58% over the 100,000 limit.  Raise MAX_SATELLITES and resize the
    // GPU buffers (createBuffers) if more capacity is needed.  Alternatively,
    // move orbit computation to a second compute shader so the CPU loop and
    // the host-visible upload buffer are no longer the bottleneck.
    if ((uint32_t)satOrbits.size() > MAX_SATELLITES)
    {
        fprintf(stderr, "[SatelliteSim] Warning: %zu total satellites exceeds "
                        "MAX_SATELLITES=%u; truncating.\n",
                satOrbits.size(), MAX_SATELLITES);
        satOrbits.resize(MAX_SATELLITES);
    }
    // Precompute frame-invariant constants into each SatOrbit so updatePositions()
    // doesn't recompute them every frame (saves sqrt + 4 trig calls per satellite).
    for (SatOrbit &orb : satOrbits)
    {
        orb.R_sat = kEarthRadius + orb.altM;
        orb.meanMot = (float)sqrt(kGM / ((double)orb.R_sat * orb.R_sat * orb.R_sat));
        orb.cosI = cosf(orb.incl);
        orb.sinI = sinf(orb.incl);
        if (!orb.alignTerminator)
        {
            orb.cosRaan = cosf(orb.raan);
            orb.sinRaan = sinf(orb.raan);
        }
    }

    activeSatCount = (uint32_t)satOrbits.size();
}

// ─── computeReflectorTargetElevationRadius ───────────────────────────────────
// Shared by loadReflectorTargets() and generateReflectorTargetsRandomFallback(). Same CPU-side
// earthElevCpu lookup/formula as the observer's own terrain height (see "Elevation texture
// encoding" in CLAUDE.md): MAX over a 3x3 texel neighborhood, not a single point sample (C12
// follow-up #23) — earthElevCpu is itself a 10x downsample, so a single lookup can land a full
// ~9km from the target's true lat/lon and miss a nearby peak. Taking the max can only raise the
// estimate toward a real nearby peak, never lower it, so at worst a target ends up slightly ABOVE
// ground (reads as "beam floats a little," far less objectionable than "beam sinks into the
// hillside"). Combined with a small fixed margin below for the same reason.
void SatelliteSim::computeReflectorTargetElevationRadius(int ti)
{
    // 2026-08-12: also build this site's own local ENU frame here. Both callers set
    // reflectorTargetsECEF[ti] immediately before calling this, and neither the direction nor the
    // frame ever changes afterwards, so this is the one place that can't be forgotten by a future
    // third loading path. Done BEFORE the early-return below — the frame doesn't depend on
    // earthElevCpu, and a target whose elevation lookup was skipped still needs a valid frame.
    {
        glm::vec3 z = reflectorTargetsECEF[ti];
        float zLen = glm::length(z);
        if (zLen > 1e-6f)
        {
            z /= zLen;
            // Reference axis chosen away from z so the cross product can't degenerate. terrain.glsl's
            // enuBasis() always uses +Z and simply breaks at the poles; that's tolerable there
            // (the observer is never exactly polar) but not here, where the result feeds an integer
            // bucket index and a NaN would propagate silently.
            glm::vec3 ref = (std::fabs(z.z) < 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                                     : glm::vec3(1.0f, 0.0f, 0.0f);
            glm::vec3 x = glm::normalize(glm::cross(ref, z));
            reflectorSiteEnuX[ti] = x;
            reflectorSiteEnuY[ti] = glm::cross(z, x);
            reflectorSiteEnuZ[ti] = z;
        }
        else
        {
            // Degenerate/unpopulated slot (the fallback path leaves the last one zeroed). Give it an
            // identity frame rather than NaNs — it is never a real beam's targetIdx anyway.
            reflectorSiteEnuX[ti] = glm::vec3(1.0f, 0.0f, 0.0f);
            reflectorSiteEnuY[ti] = glm::vec3(0.0f, 1.0f, 0.0f);
            reflectorSiteEnuZ[ti] = glm::vec3(0.0f, 0.0f, 1.0f);
        }
    }

    reflectorTargetsRadiusM[ti] = kEarthRadius; // default: sea level
    if (earthElevCpu.empty())
        return;

    const glm::vec3 &ef = reflectorTargetsECEF[ti];
    float lonRad = atan2f(ef.y, ef.x);
    float latRad = asinf(glm::clamp(ef.z, -1.0f, 1.0f));
    float u = (lonRad + glm::pi<float>()) / (2.0f * glm::pi<float>());
    float v = (0.5f * glm::pi<float>() - latRad) / glm::pi<float>();
    int px = std::clamp((int)(u * (float)earthElevCpuW), 0, earthElevCpuW - 1);
    int py = std::clamp((int)(v * (float)earthElevCpuH), 0, earthElevCpuH - 1);

    const float kSeaLevel = 15.0f / 255.0f;
    uint8_t maxPix = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int py2 = std::clamp(py + dy, 0, earthElevCpuH - 1);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int px2 = ((px + dx) % earthElevCpuW + earthElevCpuW) % earthElevCpuW; // wrap longitude
            maxPix = std::max(maxPix, earthElevCpu[py2 * earthElevCpuW + px2]);
        }
    }
    float pixVal = maxPix / 255.0f;
    float terrainH = (pixVal <= kSeaLevel) ? 0.0f : std::max(0.0f, (pixVal - kSeaLevel) * 8848.0f);
    const float kElevSafetyMarginM = 75.0f; // small fixed bias toward "above ground, not below"
    reflectorTargetsRadiusM[ti] = kEarthRadius + terrainH + kElevSafetyMarginM;
}

// ─── generateReflectorTargetsRandomFallback ──────────────────────────────────
void SatelliteSim::generateReflectorTargetsRandomFallback()
{
    // Index 0: real fixed target at the observer spawn point (67°S, 67°W) — see loadReflectorTargets()'s
    // doc comment for the bug this replaces (index 0 used to be silently left as a degenerate
    // zero-vector in this exact fallback). Matches CLAUDE.md's "Fixed Simulation State" spawn.
    {
        const float kSpawnLatDeg = -67.0f, kSpawnLonDeg = -67.0f;
        float latRad = glm::radians(kSpawnLatDeg), lonRad = glm::radians(kSpawnLonDeg);
        float cosLat = cosf(latRad);
        reflectorTargetsECEF[0] = glm::vec3(cosLat * cosf(lonRad), cosLat * sinf(lonRad), sinf(latRad));
        computeReflectorTargetElevationRadius(0);
        reflectorObserverSpawnIdx = 0;
    }
    // Remaining slots: uniformly-random lat/lon points stored as unit ECEF vectors. Leaves the
    // last slot (kNumReflectorTargets-1) untouched/degenerate, same as the old behavior — not
    // worth using every last slot of a 201-capacity buffer for a fallback path.
    for (int ti = 1; ti < kNumReflectorTargets - 1; ++ti)
    {
        // Uniform sampling on sphere: latitude from arcsin of uniform[-1,1], longitude uniform [0, 2π).
        float sinLat = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        float cosLat = sqrtf(std::max(0.0f, 1.0f - sinLat * sinLat));
        float lon = (float)rand() / RAND_MAX * glm::two_pi<float>();
        reflectorTargetsECEF[ti] = glm::vec3(cosLat * cosf(lon), cosLat * sinf(lon), sinLat);
        computeReflectorTargetElevationRadius(ti);
    }
    reflectorTargetCount = kNumReflectorTargets - 1;
    fprintf(stderr, "[SatelliteSim] Using %d procedurally-random reflector targets.\n", reflectorTargetCount);
}

// ─── loadReflectorTargets ─────────────────────────────────────────────────────
// S1 (RELEASE_v1_1_PLAN.md): reads reflector_targets.json (real, hand-curated solar-farm sites —
// see that file's own header comment for provenance/license notes), moddable exactly like
// constellations.json. Falls back to generateReflectorTargetsRandomFallback() on any failure.
//
// Fixes a real bug along the way: CLAUDE.md's "TargetedReflector" subsystem doc claimed index
// (kNumReflectorTargets-1) was a fixed pin at the observer spawn point "guaranteeing a mirror
// always aims here when in darkness" — but the code that wrote it had already been removed (see
// the historical comment this replaced), leaving that slot a degenerate zero-vector that
// normalize() turns into NaN, silently marked invalid, and never selected. There was no working
// observer-spawn-pinned target at all. reflector_targets.json's first entry (observer_spawn:
// true) is a REAL, correctly-populated site at the exact same coordinates, so the guarantee is
// real again — see generateReflectorTargetsRandomFallback() for the same fix in the fallback path.
void SatelliteSim::loadReflectorTargets()
{
    reflectorTargetCount = 0;
    reflectorObserverSpawnIdx = -1;

    auto jsonPath = (std::filesystem::path(exeDir_) / "reflector_targets.json").string();
    std::ifstream f(jsonPath);
    if (!f.is_open())
    {
        fprintf(stderr, "[SatelliteSim] reflector_targets.json not found at '%s';"
                        " using procedurally-random targets.\n",
                jsonPath.c_str());
        generateReflectorTargetsRandomFallback();
        return;
    }

    nlohmann::json j;
    try
    {
        f >> j;
    }
    catch (const nlohmann::json::exception &e)
    {
        fprintf(stderr, "[SatelliteSim] Failed to parse reflector_targets.json: %s\n"
                        "              Using procedurally-random targets.\n",
                e.what());
        generateReflectorTargetsRandomFallback();
        return;
    }

    int count = 0;
    for (const auto &jt : j.value("targets", nlohmann::json::array()))
    {
        if (count >= kNumReflectorTargets)
        {
            fprintf(stderr, "[SatelliteSim] reflector_targets.json has more than %d entries; "
                            "truncating.\n",
                    kNumReflectorTargets);
            break;
        }
        float latDeg = jt.value("lat", 0.0f);
        float lonDeg = jt.value("lon", 0.0f);
        float latRad = glm::radians(latDeg), lonRad = glm::radians(lonDeg);
        float cosLat = cosf(latRad);
        reflectorTargetsECEF[count] = glm::vec3(cosLat * cosf(lonRad), cosLat * sinf(lonRad), sinf(latRad));
        computeReflectorTargetElevationRadius(count);
        if (jt.value("observer_spawn", false) && reflectorObserverSpawnIdx < 0)
            reflectorObserverSpawnIdx = count;
        ++count;
    }

    if (count == 0)
    {
        fprintf(stderr, "[SatelliteSim] reflector_targets.json has no usable \"targets\" entries; "
                        "using procedurally-random targets.\n");
        generateReflectorTargetsRandomFallback();
        return;
    }

    reflectorTargetCount = count;
    fprintf(stderr, "[SatelliteSim] Loaded %d reflector targets from reflector_targets.json%s.\n",
            reflectorTargetCount,
            reflectorObserverSpawnIdx >= 0 ? " (observer-spawn pin found)" : " (no observer-spawn pin!)");
}

// ─── updatePositions ──────────────────────────────────────────────────────────
// Recomputes: observer ECI position + ECI→ENU matrix, sun direction,
// and per-satellite geometry + panel attitude (nested per-constellation).
//
// Performance characteristics:
//   This function runs on the CPU main thread every frame, O(N) in satellite
//   count.  Per satellite it executes ~15–20 floating-point operations including
//   double-precision fmod, cosf/sinf, asinf, length, and conditionally cross
//   products for tumbling/sun-tracking attitude modes.
//
//   Approximate wall time on a modern desktop CPU:
//     1,000  sats  →  ~0.1 ms
//    10,000  sats  →  ~1   ms
//   100,000  sats  →  ~10  ms (hits frame budget at 60 Hz)
//
//   For larger constellations the orbit computation should be moved to a
//   dedicated GPU compute pass.  The CPU would then only upload simTime + the
//   ECI→ENU matrix (~120 bytes) rather than the full GpuSatInput array
//   (~6.4 MB at 100k sats).
void SatelliteSim::updatePositions(double t, float dt)
{
    // ── Observer ECI position (rotates with Earth) ────────────────────────────
    // fmod keeps the angle small so float trig precision is maintained at large t.
    // Add the Earth-fixed longitude offset to the GMST angle.
    // kOmegaEarth * t  = Greenwich Meridian Sidereal Time (Earth's rotation since epoch).
    // obsLonRad        = observer's geodetic longitude in the Earth-fixed frame.
    // Together: the observer sits at geodetic (obsLatDeg, obsLonDeg) rotating with Earth.
    // Derive lat/lon from obsDir (canonical state) — stable at all latitudes.
    float sinLat = obsDir.z;
    float cosLat = sqrtf(obsDir.x * obsDir.x + obsDir.y * obsDir.y);
    float obsLonRad = atan2f(obsDir.y, obsDir.x); // safe: cosLat >= 0 always
    float theta = (float)fmod(kOmegaEarth * t + (double)obsLonRad, glm::two_pi<double>());
    // Refresh display caches each frame so UI stays in sync regardless of who moved obsDir.
    obsLatDeg = glm::degrees(asinf(glm::clamp(sinLat, -1.0f, 1.0f)));
    obsLonDeg = glm::degrees(obsLonRad);
    float cosLon = cosf(theta), sinLon = sinf(theta);

    float obsRadius = kEarthRadius + obsTerrainH + obsHeightOffset;
    obsECI = glm::vec3{obsRadius * cosLat * cosLon,
                       obsRadius * cosLat * sinLon,
                       obsRadius * sinLat};

    // ── ECI → ENU basis vectors ───────────────────────────────────────────────
    glm::vec3 east{-sinLon, cosLon, 0.0f};
    glm::vec3 north{-sinLat * cosLon, -sinLat * sinLon, cosLat};
    glm::vec3 up{cosLat * cosLon, cosLat * sinLon, sinLat};

    eci2enuX = glm::vec4(east, 0.0f);
    eci2enuY = glm::vec4(north, 0.0f);
    eci2enuZ = glm::vec4(up, 0.0f);

    // ── Milky Way skybox basis: ENU -> galactic, recomputed each frame ───────────────────────
    // The base ECI->Galactic rotation only depends on fixed IAU constants (galX/Y/Z below are
    // static in ECI, which is itself inertial), so this could be cached — recomputing is cheap
    // (a handful of dot/cross products) and keeps it self-contained alongside eci2enu above,
    // which is rebuilt every frame for the same reason despite most of its own inputs (lat/lon)
    // changing slowly too.
    // Alignment was confirmed by eye against assets/textures/8k_stars_milky_way.jpg: the only
    // correction needed versus the raw IAU rotation was a longitude mirror (galY negated) — no
    // yaw/pitch/roll tweak or V-flip. That was previously exposed as runtime sliders/toggles;
    // removed once confirmed correct, this is just the fixed result.
    {
        auto raDecToVec = [](float raDeg, float decDeg)
        {
            float ra = glm::radians(raDeg);
            float dec = glm::radians(decDeg);
            return glm::vec3{cosf(dec) * cosf(ra), cosf(dec) * sinf(ra), sinf(dec)};
        };
        // IAU 1958 galactic coordinate system constants (J2000 equatorial).
        glm::vec3 galZ = raDecToVec(192.859508f, 27.128336f);   // North Galactic Pole
        glm::vec3 gcDir = raDecToVec(266.405100f, -28.936175f); // Galactic center direction
        glm::vec3 galX = glm::normalize(gcDir - glm::dot(gcDir, galZ) * galZ);
        glm::vec3 galY = -glm::cross(galZ, galX); // negated: confirmed longitude mirror (was "Flip U")

        // dirGal = M * dirECI where M's rows are (galX,galY,galZ) expressed in ECI coords, and
        // dirECI = east*enu.x + north*enu.y + up*enu.z, so each row dotted against enuDir in the
        // shader needs to already be expressed in the ENU basis — project each gal axis onto
        // east/north/up here so the shader can do a single dot(enuDir, mwRowN).
        mwRow0 = glm::vec3(glm::dot(galX, east), glm::dot(galX, north), glm::dot(galX, up));
        mwRow1 = glm::vec3(glm::dot(galY, east), glm::dot(galY, north), glm::dot(galY, up));
        mwRow2 = glm::vec3(glm::dot(galZ, east), glm::dot(galZ, north), glm::dot(galZ, up));
    }

    // ── Sun direction in ECI (low-accuracy Astronomical Almanac) ─────────────
    double dJ2000 = t / 86400.0;
    double L = fmod(280.46 + 0.9856474 * dJ2000, 360.0);
    double g = fmod(357.528 + 0.9856003 * dJ2000, 360.0);
    double gR = g * (glm::pi<double>() / 180.0);
    double lambdaR = (L + 1.915 * sin(gR) + 0.020 * sin(2.0 * gR)) * (glm::pi<double>() / 180.0);
    double epsR = (23.439 - 0.0000004 * dJ2000) * (glm::pi<double>() / 180.0);

    sunDirECI = glm::normalize(glm::vec3{
        (float)cos(lambdaR),
        (float)(sin(lambdaR) * cos(epsR)),
        (float)(sin(lambdaR) * sin(epsR))});

    glm::vec3 sunENU{
        glm::dot(sunDirECI, east),
        glm::dot(sunDirECI, north),
        glm::dot(sunDirECI, up)};
    sunDirENU = glm::vec4(glm::normalize(sunENU), sunENU.z); // w = sin(elevation)

    // ── Moon direction in ECI (Keplerian two-body ellipse — see kMoonElements) ───────────────
    // Was a circular equatorial orbit with a phase constant hand-calibrated for one epoch
    // (2026-03-30) that drifted for any other date (see kMoonElements' comment) — replaced as
    // part of the planets feature (RELEASE_v1_1_PLAN follow-up, session 30), same T this frame's
    // sun calc already computed.
    double Tcent = dJ2000 / 36525.0; // Julian centuries since J2000 — shared by Moon + all planets
    glm::dvec3 moonGeoEcl = keplerEclipticPos(kMoonElements, Tcent);
    glm::dvec3 moonDirEcl = glm::normalize(moonGeoEcl);
    glm::dvec3 moonDirEciD{
        moonDirEcl.x,
        moonDirEcl.y * cos(epsR) - moonDirEcl.z * sin(epsR),
        moonDirEcl.y * sin(epsR) + moonDirEcl.z * cos(epsR)};
    moonDirECI = glm::normalize(glm::vec3(moonDirEciD));

    // Moon in ENU
    glm::vec3 moonENU_local{
        glm::dot(moonDirECI, east),
        glm::dot(moonDirECI, north),
        glm::dot(moonDirECI, up)};
    // Illuminated fraction = (1 − dot(sunDir, moonDir)) / 2
    // Full moon when moon is opposite the sun; new moon when aligned.
    float moonIllum = (1.0f - glm::dot(sunDirECI, moonDirECI)) * 0.5f;
    moonDirENU = glm::vec4(moonENU_local, moonIllum);

    // ── Planets (Mercury..Uranus): heliocentric Keplerian ephemeris ──────────────────────────
    // Same Tcent, same epsR obliquity rotation the Moon/Sun above use. See kPlanetElements'
    // comment (top of file) for the source and keplerEclipticPos() for the shared math.
    {
        glm::dvec3 earthHelio = keplerEclipticPos(kEarthElements, Tcent);
        for (int p = 0; p < kPlanetCount; ++p)
        {
            glm::dvec3 planetHelio = keplerEclipticPos(kPlanetElements[p], Tcent);
            glm::dvec3 geo = planetHelio - earthHelio;
            double distAU = glm::length(geo);
            double sunDistAU = glm::length(planetHelio);
            glm::dvec3 geoDirEcl = geo / distAU;
            glm::dvec3 geoDirEciD{
                geoDirEcl.x,
                geoDirEcl.y * cos(epsR) - geoDirEcl.z * sin(epsR),
                geoDirEcl.y * sin(epsR) + geoDirEcl.z * cos(epsR)};
            // Phase angle = angle at the PLANET between (planet->sun) and (planet->earth).
            // planet->sun = -planetHelio; planet->earth = -geo (geo is earth->planet, i.e.
            // planetHelio - earthHelio). So cos(phase) = dot(-planetHelio, -geo) / (r*delta) =
            // dot(planetHelio, geo) / (r*delta) — NOT dot(-planetHelio, geo), which computes the
            // supplementary angle (verified against Jupiter at this sim's epoch: the wrong sign
            // gives ~176° where the true phase angle is ~4°, i.e. reads as new when it's near-full).
            double cosPhase = glm::dot(planetHelio, geo) / (sunDistAU * distAU);
            planetStates[p].eciDir = glm::normalize(glm::vec3(geoDirEciD));
            planetStates[p].distanceAU = (float)distAU;
            planetStates[p].sunDistAU = (float)sunDistAU;
            planetStates[p].phaseAngleDeg = (float)glm::degrees(acos(glm::clamp(cosPhase, -1.0, 1.0)));
        }
    }

    // TargetedReflector per-frame ECEF→ECI rotation + night-side compaction was removed
    // 2026-08-06 — sat_orbit.comp now does this rotation itself (reflectorTargetsECEFBuf +
    // SatOrbitPC::gmstNow/windowFrac), scanning the full static target set every frame instead of
    // reading a CPU-precompacted subset. See CLAUDE.md's TargetedReflector section.

    // ── Satellite loop runs on GPU (sat_orbit.comp + sat_flare.comp) ─────────────
    // peakMagnitude is computed in recordCompute() from the previous frame's glowBuf.
    visibleCount = activeSatCount;
    gpuSatCount = activeSatCount;
    loopMs = 0.0f;
}
