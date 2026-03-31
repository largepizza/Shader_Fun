#include "SatelliteSim.h"
#include "../UIRenderer.h"
#include "../AudioSystem.h"
#include "clay.h"
#include "star_catalog.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <stdexcept>

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
static constexpr float kBrightnessScale = 6.0f;  // mirror BRIGHTNESS_SCALE in sat_flare.comp
static constexpr float kDaySuppression = 150.0f; // mirror DAY_SUPPRESSION in sat_flare.comp
static constexpr float kRefRange = 500'000.0f;   // 500 km normalisation range (m)
static constexpr float kMagRef = 6.0f;           // apparent magnitude at kMagRefFlare
static constexpr float kMagRefFlare = 0.008f;    // effectFlare corresponding to kMagRef
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

// Sun direction in ECI at J2000.0 (2000-01-01 12:00 TT), using the same
// low-accuracy Astronomical Almanac formula as updatePositions() at t=0.
// This is a fixed reference used to anchor the SSO epoch RAAN.
static glm::vec3 sunDirECIAtJ2000()
{
    constexpr double L = 280.46;   // mean longitude at J2000 (degrees)
    constexpr double g = 357.528;  // mean anomaly at J2000 (degrees)
    constexpr double eps = 23.439; // obliquity at J2000 (degrees)
    const double pi180 = glm::pi<double>() / 180.0;
    const double gR = g * pi180;
    const double lamR = (L + 1.915 * sin(gR) + 0.020 * sin(2.0 * gR)) * pi180;
    const double epsR = eps * pi180;
    return glm::normalize(glm::vec3{float(cos(lamR)),
                                    float(sin(lamR) * cos(epsR)),
                                    float(sin(lamR) * sin(epsR))});
}

// ─── init ─────────────────────────────────────────────────────────────────────
void SatelliteSim::init(VulkanContext &ctx)
{
    // Fixed start time: 2026-03-30 05:53:58 UTC
    // J2000.0 = 2000-01-01 12:00:00 UTC = Unix 946728000
    // 2026-03-30 05:53:58 UTC = Unix 1774849038
    simTime = simTimeAtInit = 1774849038.0 - 946728000.0 + 11 * 3600; // 828121038 s from J2000

    ctx_ = &ctx;

    keybindings = {
        {"Toggle UI", GLFW_KEY_TAB, false},
        {"Pause/Resume", GLFW_KEY_SPACE, false},
        {"Slow Down", GLFW_KEY_COMMA, false},
        {"Speed Up", GLFW_KEY_PERIOD, false},
        {"Reverse Time", GLFW_KEY_R, false},
    };

    createBuffers(ctx);
    createDescriptors(ctx);
    createGlowResources(ctx);
    createComputePipeline(ctx);
    createSkyBgPipeline(ctx);
    createDrawPipeline(ctx);
    updatePositions(simTime); // must run first — initConstellation reads sunDirECI
    initConstellation();
    initStars(ctx);
}

// ─── onResize ─────────────────────────────────────────────────────────────────
void SatelliteSim::onResize(VulkanContext &ctx)
{
    vkDestroyPipeline(ctx.device, skyBgPipeline, nullptr);
    skyBgPipeline = VK_NULL_HANDLE;
    createSkyBgPipeline(ctx);

    vkDestroyPipeline(ctx.device, drawPipeline, nullptr);
    drawPipeline = VK_NULL_HANDLE;
    createDrawPipeline(ctx);

    vkDestroyPipeline(ctx.device, starPipeline, nullptr);
    starPipeline = VK_NULL_HANDLE;
    createStarPipeline(ctx);
}

// ─── recordCompute ────────────────────────────────────────────────────────────
void SatelliteSim::recordCompute(VkCommandBuffer cmd, VulkanContext &ctx, float dt)
{
    // ── WASD surface navigation ───────────────────────────────────────────────
    // Pure 3D ECEF — no lat/lon arithmetic, no gimbal lock, works at any latitude.
    //
    // obsDir    : unit position vector on the Earth-fixed sphere.
    // obsFacing : unit tangent vector (forward), always ⊥ obsDir.
    //
    // W/S move along obsFacing; A/D move along cross(obsFacing, obsDir) (right).
    // After each step obsFacing is parallel-transported to stay tangent at newPos.
    if (win)
    {
        bool shift = glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        float speed = shift ? 0.5f : 0.08f; // radians of arc per real second

        float fwd = (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS ? 1.0f : 0.0f) - (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS ? 1.0f : 0.0f);
        float right = (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS ? 1.0f : 0.0f) - (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS ? 1.0f : 0.0f);

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
    }

    if (!timePaused)
        simTime += (double)dt * kTimeScales[timeScaleIdx] * timeDir;
    updatePositions(simTime);
    updateStars();

    // Upload top-N glow entries to the sky shader's SSBO.
    {
        GpuGlowBuf *gb = static_cast<GpuGlowBuf *>(glowMapped);
        gb->count = glowEntryCount;
        for (int gi = 0; gi < glowEntryCount; ++gi)
            gb->entries[gi] = glowEntries[gi];
    }

    if (activeSatCount == 0)
        return;

    // Upload input positions to the GPU-visible buffer.
    memcpy(satInputMapped, satInputData.data(), activeSatCount * sizeof(GpuSatInput));

    // ECI→ENU matrix, sun direction, and observer position — all from updatePositions().
    SatFlarePC pc{};
    pc.enuX = eci2enuX;
    pc.enuY = eci2enuY;
    pc.enuZ = eci2enuZ;
    pc.sunDirECI = sunDirECI;
    pc.satCount = activeSatCount;
    pc.obsECI = obsECI;
    pc.pad = 0.0f;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compPipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t groups = (activeSatCount + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute SSBO write → vertex shader SSBO read.
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = satVisibleBuf;
    bmb.offset = 0;
    bmb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         0, 0, nullptr, 1, &bmb, 0, nullptr);
}

// ─── recordDraw ───────────────────────────────────────────────────────────────
void SatelliteSim::recordDraw(VkCommandBuffer cmd, VulkanContext &ctx, float /*dt*/)
{
    // Camera push constants shared by all sky passes.
    SatDrawPC pc{};
    pc.skyView = camera.viewMatrix();
    pc.fovYRad = glm::radians(camera.fovYDeg);
    pc.aspect = (float)ctx.swapExtent.width / (float)ctx.swapExtent.height;
    pc.pad[0] = pc.pad[1] = 0.0f;
    pc.sunDirENU = sunDirENU;
    pc.moonDirENU = moonDirENU; // xyz = moon dir in ENU, w = illuminated fraction

    // ── Pass 1: sky/ground background (fullscreen triangle, opaque) ──────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyBgPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            skyBgPipeLayout, 0, 1, &skyDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, skyBgPipeLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // ── Pass 2: satellite points (additive blending) ──────────────────────────
    if (activeSatCount > 0)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                drawPipeLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, drawPipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDraw(cmd, activeSatCount, 1, 0, 0);
    }

    // ── Pass 3: background stars (additive blending) ──────────────────────────
    if (starCount > 0 && starPipeline != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, starPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                starPipeLayout, 0, 1, &starDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, starPipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDraw(cmd, starCount, 1, 0, 0);
    }
}

// Icon index constants (match order passed to ui.loadIcons)
static constexpr int kIconAngleLeft = 0;  // pixel--angle-left.png  → slow down
static constexpr int kIconAngleRight = 1; // pixel--angle-right.png → speed up
// 2 = controller (unused in time controls)
static constexpr int kIconPause = 3;    // pixel--pause.png
static constexpr int kIconPlay = 4;     // pixel--play.png
static constexpr int kIconSettings = 5; // pixel--settings.png

// Helper: short display name for a GLFW key code (used in settings window).
static const char *keyDisplayName(int key)
{
    switch (key)
    {
    case GLFW_KEY_SPACE:
        return "Space";
    case GLFW_KEY_TAB:
        return "Tab";
    case GLFW_KEY_COMMA:
        return ",";
    case GLFW_KEY_PERIOD:
        return ".";
    case GLFW_KEY_ESCAPE:
        return "Esc";
    case GLFW_KEY_ENTER:
        return "Enter";
    default:
        if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z)
        {
            static char buf[2] = {};
            buf[0] = (char)('A' + (key - GLFW_KEY_A));
            return buf;
        }
        if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
        {
            static char buf[2] = {};
            buf[0] = (char)('0' + (key - GLFW_KEY_0));
            return buf;
        }
        return "?";
    }
}

// ── UI color palette ──────────────────────────────────────────────────────────
// Edit here to restyle the entire UI. All buildUI colors reference these names.
namespace Pal
{
    // Backgrounds
    constexpr Clay_Color panelBg = {8, 8, 9, 210};        // floating panel
    constexpr Clay_Color panelBgFade = {8, 8, 9, 180};    // panel, slightly transparent
    constexpr Clay_Color panelSolid = {12, 12, 13, 245};  // settings window
    constexpr Clay_Color titleBar = {18, 18, 19, 255};    // title / header strip
    constexpr Clay_Color sectionHdr = {22, 22, 23, 130};  // section divider strip
    constexpr Clay_Color rowEnabled = {45, 10, 10, 180};  // enabled constellation row
    constexpr Clay_Color rowDisabled = {16, 16, 17, 160}; // disabled constellation row
    constexpr Clay_Color listenRow = {50, 10, 10, 185};   // keybind capture row
    // Buttons
    constexpr Clay_Color btnIdle = {30, 30, 31, 210};      // default button
    constexpr Clay_Color btnHover = {52, 52, 54, 230};     // hovered button
    constexpr Clay_Color btnAccent = {150, 20, 20, 240};   // ON / active (red)
    constexpr Clay_Color btnAccentHv = {100, 15, 15, 230}; // accent hovered
    constexpr Clay_Color closeBgIdle = {50, 16, 16, 180};  // [X] idle
    constexpr Clay_Color closeBgHov = {170, 30, 30, 220};  // [X] hovered
    constexpr Clay_Color pauseActive = {140, 25, 25, 230}; // pause btn when paused
    constexpr Clay_Color listenBtn = {120, 18, 18, 220};   // rebind btn while listening
    // Chrome
    constexpr Clay_Color divider = {48, 48, 50, 120}; // separator line
    // Text
    constexpr Clay_Color textPrimary = {205, 205, 210, 255}; // main readable text
    constexpr Clay_Color textDim = {130, 130, 135, 200};     // secondary / dim
    constexpr Clay_Color textHint = {72, 72, 76, 160};       // hint / footer
    constexpr Clay_Color textSection = {155, 155, 165, 200}; // section header labels
    constexpr Clay_Color textCamera = {110, 110, 115, 180};  // dim descriptive text
    constexpr Clay_Color volLabel = {185, 185, 195, 220};    // vol/scale label
    constexpr Clay_Color volValue = {210, 210, 215, 255};    // vol/scale value readout
    constexpr Clay_Color btnLabel = {210, 210, 215, 255};    // text inside +/- buttons
    constexpr Clay_Color listenKey = {255, 85, 85, 255};     // key label while listening
    constexpr Clay_Color keyText = {140, 140, 145, 200};     // normal key label
    // Speed indicator
    constexpr Clay_Color speedFwd = {200, 55, 55, 220};    // forward (red)
    constexpr Clay_Color speedRev = {155, 155, 165, 220};  // reverse (grey)
    constexpr Clay_Color speedPaused = {95, 95, 100, 220}; // paused (dark grey)
}

// ─── buildUI ──────────────────────────────────────────────────────────────────
void SatelliteSim::buildUI(float dt, UIRenderer &ui)
{
    // Apply camera mouse look.
    // Yaw  (dmx): rotate obsFacing around obsDir via Rodrigues — no ENU frame, no pole issue.
    // Pitch (dmy): handled by camera.update → camera.elDeg as usual.
    if (win)
    {
        camera.update(win, 0.0f, dmy); // pitch only; we handle yaw below

        if (camera.captured && dmx != 0.0f)
        {
            // cross(obsDir, obsFacing) is the LEFT tangent, so negate angle for look-right.
            float angle = glm::radians(-dmx * camera.sens);
            glm::vec3 leftDir = glm::cross(obsDir, obsFacing);
            obsFacing = glm::normalize(cosf(angle) * obsFacing + sinf(angle) * leftDir);
        }

        // Derive camera.azDeg from obsFacing projected into the local Earth-fixed ENU.
        // Only used for the view matrix — never fed back into movement math.
        {
            float sL = obsDir.z;
            float cLH = sqrtf(obsDir.x * obsDir.x + obsDir.y * obsDir.y);
            float inv = (cLH > 1e-7f) ? 1.0f / cLH : 0.0f;
            float cLn = obsDir.x * inv, sLn = obsDir.y * inv;
            glm::vec3 eastEF = {-sLn, cLn, 0.0f};
            glm::vec3 northEF = {-sL * cLn, -sL * sLn, cLH};
            camera.azDeg = glm::degrees(atan2f(
                glm::dot(obsFacing, eastEF),
                glm::dot(obsFacing, northEF)));
        }
    }
    dmx = dmy = 0.0f;

    const UIInput &inp = ui.input();

    // ── Font-size helper: scales base pixel size by uiScale ───────────────────
    auto fs = [&](int base) -> uint16_t
    {
        return (uint16_t)std::max(8, (int)(base * uiScale + 0.5f));
    };

    // ── Audio helpers: rollover fires on hover transition, click on LMB press ─
    // Both are no-ops when audio_ is null (audio init failed or not yet set).
    auto sndRollover = [&](bool nowHov, bool prevHov)
    {
        if (audio_ && nowHov && !prevHov)
            audio_->playSfx("assets/sound/ui/buttonrollover.wav");
    };
    auto sndClick = [&](bool nowHov)
    {
        if (audio_ && nowHov && inp.lmbPressed)
            audio_->playSfx("assets/sound/ui/buttonclick.wav");
    };

    // ── Lazy icon loading (first buildUI call after init) ─────────────────────
    if (!iconsLoaded && ctx_)
    {
        const char *iconPaths[] = {
            "assets/icons/ui/pixel--angle-left.png",
            "assets/icons/ui/pixel--angle-right.png",
            "assets/icons/ui/pixel--controller.png",
            "assets/icons/ui/pixel--pause.png",
            "assets/icons/ui/pixel--play.png",
            "assets/icons/ui/pixel--settings.png",
        };
        ui.loadIcons(*ctx_, iconPaths, 6);
        iconsLoaded = true;
    }

    // ── Scroll wheel → FOV zoom (when not hovering over UI panels) ───────────
    if (inp.scrollY != 0.0f && !ui.mouseOverUI())
    {
        camera.fovYDeg = glm::clamp(camera.fovYDeg - inp.scrollY * 3.0f, 10.0f, 120.0f);
    }

    // ── Tab: skip all UI when hidden ─────────────────────────────────────────
    if (!uiVisible)
        return;

    // ── Simulated UTC time string ─────────────────────────────────────────────
    static char timeBuf[32];
    {
        time_t unixSim = (time_t)(simTime) + 946738000;
        struct tm *utc = gmtime(&unixSim);
        if (utc)
            snprintf(timeBuf, sizeof(timeBuf), "UTC %04d-%02d-%02d %02d:%02d:%02d",
                     utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
                     utc->tm_hour, utc->tm_min, utc->tm_sec);
        else
            snprintf(timeBuf, sizeof(timeBuf), "UTC --");
    }

    // ── Status bar data (bottom-right toolbar) ────────────────────────────────
    // setLat: moves the observer to a new latitude while preserving longitude direction
    // and parallel-transporting obsFacing so it stays tangent after the position jump.
    auto setLat = [this](float newLatDeg)
    {
        newLatDeg = glm::clamp(newLatDeg, -90.0f, 90.0f);
        float sinL = sinf(glm::radians(newLatDeg));
        float cosL = cosf(glm::radians(newLatDeg));
        glm::vec2 xy = glm::vec2(obsDir.x, obsDir.y);
        float xyMag = glm::length(xy);
        if (xyMag > 1e-6f)
            xy /= xyMag;
        else
            xy = {1.0f, 0.0f};
        obsDir = {xy.x * cosL, xy.y * cosL, sinL};
        obsFacing = glm::normalize(obsFacing - glm::dot(obsFacing, obsDir) * obsDir);
        obsLatDeg = newLatDeg;
        obsLonDeg = glm::degrees(atan2f(obsDir.y, obsDir.x));
    };

    static char latBuf[20], lonBuf[20], statFpsBuf[24], statVisBuf[32];
    {
        float absLat = fabsf(obsLatDeg);
        float absLon = fabsf(obsLonDeg);
        snprintf(latBuf, sizeof(latBuf), "%.1f\xc2\xb0 %c", absLat, obsLatDeg >= 0.0f ? 'N' : 'S');
        snprintf(lonBuf, sizeof(lonBuf), "%.1f\xc2\xb0 %c", absLon, obsLonDeg >= 0.0f ? 'E' : 'W');
        snprintf(statFpsBuf, sizeof(statFpsBuf), "%.0f fps", dt > 0.0f ? 1.0f / dt : 0.0f);
        snprintf(statVisBuf, sizeof(statVisBuf), "%u vis", visibleCount);
    }
    Clay_String latStr{false, (int32_t)strlen(latBuf), latBuf};
    Clay_String lonStr{false, (int32_t)strlen(lonBuf), lonBuf};
    Clay_String statFpsStr{false, (int32_t)strlen(statFpsBuf), statFpsBuf};
    Clay_String statVisStr{false, (int32_t)strlen(statVisBuf), statVisBuf};

    // ── Time controls (bottom-left) ───────────────────────────────────────────
    // Shows UTC time, a slow/pause|play/fast icon button row, speed label, and reverse indicator.
    static char speedBuf[24];
    snprintf(speedBuf, sizeof(speedBuf), "%s%s",
             timeDir < 0.0f ? "REV " : "", kTimeLabels[timeScaleIdx]);

    CLAY(CLAY_ID("TimePanel"), {.layout = {
                                    .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                    .padding = {10, 10, 8, 8},
                                    .childGap = 6,
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                .backgroundColor = Pal::panelBg,
                                .cornerRadius = CLAY_CORNER_RADIUS(6),
                                .floating = {.offset = {12, inp.screenH - 110.0f}, .zIndex = 5, .attachTo = CLAY_ATTACH_TO_ROOT}})
    {
        // UTC time + speed indicator in one row
        CLAY(CLAY_ID("TimeHeaderRow"), {.layout = {
                                            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)},
                                            .childGap = 8,
                                            .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                            .layoutDirection = CLAY_LEFT_TO_RIGHT}})
        {
            Clay_String timeStr{false, (int32_t)strlen(timeBuf), timeBuf};
            CLAY_TEXT(timeStr,
                      CLAY_TEXT_CONFIG({.textColor = Pal::textDim, .fontSize = fs(12)}));

            // Speed / direction label
            Clay_Color speedCol = timePaused       ? Pal::speedPaused
                                  : timeDir < 0.0f ? Pal::speedRev
                                                   : Pal::speedFwd;
            Clay_String speedStr{false, (int32_t)strlen(speedBuf), speedBuf};
            CLAY_TEXT(speedStr, CLAY_TEXT_CONFIG({.textColor = speedCol, .fontSize = fs(12)}));
        }

        // Icon button row: [◀] [⏸/▶] [▶]
        const int kBtnSize = 28;
        const int kIconSize = 18;
        CLAY(CLAY_ID("TimeBtnRow"), {.layout = {
                                         .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                         .childGap = 5,
                                         .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                         .layoutDirection = CLAY_LEFT_TO_RIGHT}})
        {
            // ── Slow down ─────────────────────────────────────────────────────
            Clay_Color slowBg = hovTimeSlower ? Pal::btnHover : Pal::btnIdle;
            CLAY(CLAY_ID("TimeSlowerBtn"), {.layout = {
                                                .sizing = {CLAY_SIZING_FIXED(kBtnSize), CLAY_SIZING_FIXED(kBtnSize)},
                                                .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                            .backgroundColor = slowBg,
                                            .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                {
                    bool n = Clay_Hovered();
                    sndRollover(n, hovTimeSlower);
                    sndClick(n);
                    hovTimeSlower = n;
                }
                if (hovTimeSlower && inp.lmbPressed)
                    timeScaleIdx = std::max(0, timeScaleIdx - 1);
                CLAY(CLAY_ID("TimeSlowerIcon"), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(kIconSize), CLAY_SIZING_FIXED(kIconSize)}},
                                                 .image = {.imageData = (void *)(intptr_t)(kIconAngleLeft + 1)}}) {}
            }

            // ── Pause / Play ──────────────────────────────────────────────────
            Clay_Color pauseBg = timePaused
                                     ? Pal::pauseActive
                                     : (hovTimePause ? Pal::btnHover : Pal::btnIdle);
            CLAY(CLAY_ID("TimePauseBtn"), {.layout = {
                                               .sizing = {CLAY_SIZING_FIXED(kBtnSize), CLAY_SIZING_FIXED(kBtnSize)},
                                               .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                           .backgroundColor = pauseBg,
                                           .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                {
                    bool n = Clay_Hovered();
                    sndRollover(n, hovTimePause);
                    sndClick(n);
                    hovTimePause = n;
                }
                if (hovTimePause && inp.lmbPressed)
                    timePaused = !timePaused;
                int pauseIcon = timePaused ? kIconPlay : kIconPause;
                CLAY(CLAY_ID("TimePauseIcon"), {.layout = {
                                                    .sizing = {CLAY_SIZING_FIXED(kIconSize), CLAY_SIZING_FIXED(kIconSize)}},
                                                .image = {.imageData = (void *)(intptr_t)(pauseIcon + 1)}}) {}
            }

            // ── Speed up ──────────────────────────────────────────────────────
            Clay_Color fastBg = hovTimeFaster ? Pal::btnHover : Pal::btnIdle;
            CLAY(CLAY_ID("TimeFasterBtn"), {.layout = {
                                                .sizing = {CLAY_SIZING_FIXED(kBtnSize), CLAY_SIZING_FIXED(kBtnSize)},
                                                .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                            .backgroundColor = fastBg,
                                            .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                {
                    bool n = Clay_Hovered();
                    sndRollover(n, hovTimeFaster);
                    sndClick(n);
                    hovTimeFaster = n;
                }
                if (hovTimeFaster && inp.lmbPressed)
                    timeScaleIdx = std::min(kNumTimeScales - 1, timeScaleIdx + 1);
                CLAY(CLAY_ID("TimeFasterIcon"), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(kIconSize), CLAY_SIZING_FIXED(kIconSize)}},
                                                 .image = {.imageData = (void *)(intptr_t)(kIconAngleRight + 1)}}) {}
            }
        }

        // Hint text
        CLAY_TEXT(CLAY_STRING(",/. = speed  Space = pause  R = reverse  Tab = hide UI"),
                  CLAY_TEXT_CONFIG({.textColor = Pal::textHint, .fontSize = fs(10)}));
    }

    // ── Status bar (bottom-right): lat/lon, fps, vis count, settings gear ────
    {
        const int kSBBtnSz = 22;
        const int kSBIconSz = 14;
        const int kGearSz = 28;
        Clay_Color settingsBg = hovSettings ? Pal::btnHover : Pal::panelBgFade;

        CLAY(CLAY_ID("StatusBar"), {.layout = {
                                        .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIXED(38)},
                                        .padding = {10, 10, 6, 6},
                                        .childGap = 7,
                                        .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                        .layoutDirection = CLAY_LEFT_TO_RIGHT},
                                    .backgroundColor = Pal::panelBg,
                                    .cornerRadius = CLAY_CORNER_RADIUS(6),
                                    .floating = {.offset = {-12.0f, -12.0f}, .zIndex = 5, .attachPoints = {.element = CLAY_ATTACH_POINT_RIGHT_BOTTOM, .parent = CLAY_ATTACH_POINT_RIGHT_BOTTOM}, .attachTo = CLAY_ATTACH_TO_ROOT}})
        {
            // ── Lat south button ──────────────────────────────────────────────
            Clay_Color sbSBg = hovLatSouth ? Pal::btnHover : Pal::btnIdle;
            CLAY(CLAY_ID("SBLatSBtn"), {.layout = {
                                            .sizing = {CLAY_SIZING_FIXED(kSBBtnSz), CLAY_SIZING_FIXED(kSBBtnSz)},
                                            .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                        .backgroundColor = sbSBg,
                                        .cornerRadius = CLAY_CORNER_RADIUS(3)})
            {
                {
                    bool n = Clay_Hovered();
                    sndRollover(n, hovLatSouth);
                    sndClick(n);
                    hovLatSouth = n;
                }
                if (hovLatSouth && inp.lmbPressed)
                    setLat(obsLatDeg - 5.0f);
                CLAY(CLAY_ID("SBLatSIcon"), {.layout = {.sizing = {CLAY_SIZING_FIXED(kSBIconSz), CLAY_SIZING_FIXED(kSBIconSz)}},
                                             .image = {.imageData = (void *)(intptr_t)(kIconAngleLeft + 1)}}) {}
            }

            // ── Lat display (scroll to adjust) ────────────────────────────────
            CLAY(CLAY_ID("SBLatDisplay"), {.layout = {
                                               .sizing = {CLAY_SIZING_FIXED(62), CLAY_SIZING_FIT(0)},
                                               .childAlignment = {.x = CLAY_ALIGN_X_CENTER}}})
            {
                if (Clay_Hovered() && inp.scrollY != 0.0f)
                    setLat(obsLatDeg + inp.scrollY * 5.0f);
                CLAY_TEXT(latStr, CLAY_TEXT_CONFIG({.textColor = Pal::volValue, .fontSize = fs(12)}));
            }

            // ── Lat north button ──────────────────────────────────────────────
            Clay_Color sbNBg = hovLatNorth ? Pal::btnHover : Pal::btnIdle;
            CLAY(CLAY_ID("SBLatNBtn"), {.layout = {
                                            .sizing = {CLAY_SIZING_FIXED(kSBBtnSz), CLAY_SIZING_FIXED(kSBBtnSz)},
                                            .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                        .backgroundColor = sbNBg,
                                        .cornerRadius = CLAY_CORNER_RADIUS(3)})
            {
                {
                    bool n = Clay_Hovered();
                    sndRollover(n, hovLatNorth);
                    sndClick(n);
                    hovLatNorth = n;
                }
                if (hovLatNorth && inp.lmbPressed)
                    setLat(obsLatDeg + 5.0f);
                CLAY(CLAY_ID("SBLatNIcon"), {.layout = {.sizing = {CLAY_SIZING_FIXED(kSBIconSz), CLAY_SIZING_FIXED(kSBIconSz)}},
                                             .image = {.imageData = (void *)(intptr_t)(kIconAngleRight + 1)}}) {}
            }

            CLAY(CLAY_ID("SBDiv1"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1), CLAY_SIZING_FIXED(20)}},
                                     .backgroundColor = Pal::divider}) {}

            // ── Lon display ───────────────────────────────────────────────────
            CLAY(CLAY_ID("SBLonDisplay"), {.layout = {
                                               .sizing = {CLAY_SIZING_FIXED(62), CLAY_SIZING_FIT(0)},
                                               .childAlignment = {.x = CLAY_ALIGN_X_CENTER}}})
            {
                CLAY_TEXT(lonStr, CLAY_TEXT_CONFIG({.textColor = Pal::volValue, .fontSize = fs(12)}));
            }

            CLAY(CLAY_ID("SBDiv2"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1), CLAY_SIZING_FIXED(20)}},
                                     .backgroundColor = Pal::divider}) {}

            // ── FPS ───────────────────────────────────────────────────────────
            CLAY(CLAY_ID("SBFps"), {.layout = {
                                        .sizing = {CLAY_SIZING_FIXED(50), CLAY_SIZING_FIT(0)},
                                        .childAlignment = {.x = CLAY_ALIGN_X_CENTER}}})
            {
                CLAY_TEXT(statFpsStr, CLAY_TEXT_CONFIG({.textColor = Pal::textDim, .fontSize = fs(12)}));
            }

            CLAY(CLAY_ID("SBDiv3"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1), CLAY_SIZING_FIXED(20)}},
                                     .backgroundColor = Pal::divider}) {}

            // ── Visible sat count ─────────────────────────────────────────────
            CLAY(CLAY_ID("SBVis"), {.layout = {
                                        .sizing = {CLAY_SIZING_FIXED(60), CLAY_SIZING_FIT(0)},
                                        .childAlignment = {.x = CLAY_ALIGN_X_CENTER}}})
            {
                CLAY_TEXT(statVisStr, CLAY_TEXT_CONFIG({.textColor = Pal::textSection, .fontSize = fs(12)}));
            }

            CLAY(CLAY_ID("SBDiv4"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1), CLAY_SIZING_FIXED(20)}},
                                     .backgroundColor = Pal::divider}) {}

            // ── Settings gear button ──────────────────────────────────────────
            CLAY(CLAY_ID("SettingsBtn"), {.layout = {
                                              .sizing = {CLAY_SIZING_FIXED(kGearSz), CLAY_SIZING_FIXED(kGearSz)},
                                              .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                          .backgroundColor = settingsBg,
                                          .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                {
                    bool n = Clay_Hovered();
                    sndRollover(n, hovSettings);
                    sndClick(n);
                    if (n && inp.lmbPressed)
                        settingsOpen = !settingsOpen;
                    hovSettings = n;
                }
                CLAY(CLAY_ID("SettingsIcon"), {.layout = {.sizing = {CLAY_SIZING_FIXED(18), CLAY_SIZING_FIXED(18)}},
                                               .image = {.imageData = (void *)(intptr_t)(kIconSettings + 1)}}) {}
            }
        }
    }

    // ── Settings window ───────────────────────────────────────────────────────
    if (settingsOpen)
    {
        const float kWinW = 500.0f;
        const float kWinH = 500.0f;
        const float kWinX = (inp.screenW - kWinW) * 0.5f;
        const float kWinY = (inp.screenH - kWinH) * 0.5f;

        CLAY(CLAY_ID("SettingsWin"), {.layout = {
                                          .sizing = {CLAY_SIZING_FIXED(kWinW), CLAY_SIZING_FIXED(kWinH)},
                                          .padding = {0, 0, 0, 0},
                                          .childGap = 0,
                                          .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                      .backgroundColor = Pal::panelSolid,
                                      .cornerRadius = CLAY_CORNER_RADIUS(8),
                                      .floating = {.offset = {kWinX, kWinY}, .zIndex = 10, .attachTo = CLAY_ATTACH_TO_ROOT}})
        {
            // Title bar
            CLAY(CLAY_ID("SettingsTitleBar"), {.layout = {
                                                   .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(36)},
                                                   .padding = {14, 14, 0, 0},
                                                   .childGap = 0,
                                                   .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                                   .layoutDirection = CLAY_LEFT_TO_RIGHT},
                                               .backgroundColor = Pal::titleBar,
                                               .cornerRadius = {8, 8, 0, 0}})
            {
                CLAY_TEXT(CLAY_STRING("Settings"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(16)}));

                // Spacer
                CLAY(CLAY_ID("SettingsTitleSpacer"), {.layout = {
                                                          .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)}}}) {}

                // Close button [X]
                Clay_Color closeBg = hovSettingsClose ? Pal::closeBgHov : Pal::closeBgIdle;
                CLAY(CLAY_ID("SettingsCloseBtn"), {.layout = {
                                                       .sizing = {CLAY_SIZING_FIXED(24), CLAY_SIZING_FIXED(24)},
                                                       .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                   .backgroundColor = closeBg,
                                                   .cornerRadius = CLAY_CORNER_RADIUS(4)})
                {
                    {
                        bool n = Clay_Hovered();
                        sndRollover(n, hovSettingsClose);
                        sndClick(n);
                        hovSettingsClose = n;
                    }
                    if (hovSettingsClose && inp.lmbPressed)
                        settingsOpen = false;
                    CLAY_TEXT(CLAY_STRING("X"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(12)}));
                }
            }

            // Scrollable controls list
            CLAY(CLAY_ID("SettingsScroll"), {.layout = {
                                                 .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                                                 .padding = {14, 14, 10, 10},
                                                 .childGap = 4,
                                                 .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                             .clip = {.vertical = true, .childOffset = Clay_GetScrollOffset()}})
            {
                // ── Constellations ────────────────────────────────────────────
                CLAY_TEXT(CLAY_STRING("Constellations"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textSection, .fontSize = fs(14)}));
                CLAY(CLAY_ID("ConstSep"), {.layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                      .padding = {0, 0, 4, 4}},
                                           .backgroundColor = Pal::sectionHdr}) {}

                static char constCntBuf[10][16];
                for (int ci = 0; ci < (int)constellations.size() && ci < 10; ++ci)
                {
                    ConstellationConfig &c = constellations[ci];
                    snprintf(constCntBuf[ci], sizeof(constCntBuf[ci]), "%u", c.orbitCount);

                    Clay_Color rowBg = c.enabled
                                           ? Pal::rowEnabled
                                           : Pal::rowDisabled;
                    CLAY(CLAY_IDI("ConstRow", ci), {.layout = {
                                                        .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(24)},
                                                        .padding = {4, 4, 3, 3},
                                                        .childGap = 6,
                                                        .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                                        .layoutDirection = CLAY_LEFT_TO_RIGHT},
                                                    .backgroundColor = rowBg,
                                                    .cornerRadius = CLAY_CORNER_RADIUS(3)})
                    {
                        Clay_Color btnBg = c.enabled
                                               ? Pal::btnAccent
                                               : (hovConst[ci] ? Pal::btnAccentHv : Pal::btnIdle);
                        CLAY(CLAY_IDI("ConstBtn", ci), {.layout = {
                                                            .sizing = {CLAY_SIZING_FIXED(30), CLAY_SIZING_FIXED(18)},
                                                            .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                        .backgroundColor = btnBg,
                                                        .cornerRadius = CLAY_CORNER_RADIUS(3)})
                        {
                            {
                                bool n = Clay_Hovered();
                                sndRollover(n, hovConst[ci]);
                                sndClick(n);
                                hovConst[ci] = n;
                            }
                            if (hovConst[ci] && inp.lmbPressed)
                                c.enabled = !c.enabled;
                            CLAY_TEXT(c.enabled ? CLAY_STRING("ON") : CLAY_STRING("OFF"),
                                      CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(10)}));
                        }
                        CLAY(CLAY_IDI("ConstName", ci), {.layout = {.sizing = {CLAY_SIZING_FIXED(150), CLAY_SIZING_FIT(0)}}})
                        {
                            Clay_String nameStr{false, (int32_t)strlen(c.name), c.name};
                            CLAY_TEXT(nameStr, CLAY_TEXT_CONFIG({.textColor = Pal::volLabel, .fontSize = fs(12)}));
                        }
                        Clay_String cntStr{false, (int32_t)strlen(constCntBuf[ci]), constCntBuf[ci]};
                        CLAY_TEXT(cntStr, CLAY_TEXT_CONFIG({.textColor = Pal::textCamera, .fontSize = fs(11)}));
                    }
                }

                // ── Sound ────────────────────────────────────────────────────
                CLAY(CLAY_ID("SndTopSep"), {.layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                       .padding = {0, 0, 6, 4}},
                                            .backgroundColor = Pal::sectionHdr}) {}
                CLAY_TEXT(CLAY_STRING("Sound"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textSection, .fontSize = fs(14)}));
                CLAY(CLAY_ID("SndSep"), {.layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                    .padding = {0, 0, 4, 4}},
                                         .backgroundColor = Pal::sectionHdr}) {}

                // Helper: one volume row (label, spacer, −, value, +)
                // We use a macro-like lambda to avoid copy-pasting Clay layout 3 times.
                static char volBufs[3][8];
                struct VolRow
                {
                    const char *label;
                    float vol;
                    bool &hMinus;
                    bool &hPlus;
                    const char *idMinus;
                    const char *idPlus;
                    const char *idVal;
                    int bufIdx;
                };
                VolRow volRows[] = {
                    {"Master vol", audio_ ? audio_->getMasterVolume() : masterVol_, hovMasterVolMinus, hovMasterVolPlus, "MasterVolMinus", "MasterVolPlus", "MasterVolVal", 0},
                    {"Music vol", audio_ ? audio_->getMusicVolume() : musicVol_, hovMusicVolMinus, hovMusicVolPlus, "MusicVolMinus", "MusicVolPlus", "MusicVolVal", 1},
                    {"SFX vol", audio_ ? audio_->getSfxVolume() : sfxVol_, hovSfxVolMinus, hovSfxVolPlus, "SfxVolMinus", "SfxVolPlus", "SfxVolVal", 2},
                };
                for (auto &vr : volRows)
                {
                    snprintf(volBufs[vr.bufIdx], sizeof(volBufs[0]), "%3.0f%%", vr.vol * 100.0f);
                    Clay_String volStr{false, (int32_t)strlen(volBufs[vr.bufIdx]), volBufs[vr.bufIdx]};
                    CLAY(CLAY_IDI("VolRow", vr.bufIdx), {.layout = {
                                                             .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(26)},
                                                             .padding = {4, 4, 2, 2},
                                                             .childGap = 6,
                                                             .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                                             .layoutDirection = CLAY_LEFT_TO_RIGHT}})
                    {
                        CLAY(CLAY_IDI("VolLabel", vr.bufIdx), {.layout = {.sizing = {CLAY_SIZING_FIXED(76), CLAY_SIZING_FIT(0)}}})
                        {
                            Clay_String lblStr{false, (int32_t)strlen(vr.label), vr.label};
                            CLAY_TEXT(lblStr, CLAY_TEXT_CONFIG({.textColor = Pal::volLabel, .fontSize = fs(12)}));
                        }
                        CLAY(CLAY_IDI("VolSpc", vr.bufIdx), {.layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)}}}) {}
                        // − button
                        Clay_Color cMinus = vr.hMinus ? Pal::btnHover : Pal::btnIdle;
                        CLAY(CLAY_IDI("VolMinus", vr.bufIdx), {.layout = {
                                                                   .sizing = {CLAY_SIZING_FIXED(20), CLAY_SIZING_FIXED(20)},
                                                                   .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                               .backgroundColor = cMinus,
                                                               .cornerRadius = CLAY_CORNER_RADIUS(3)})
                        {
                            bool n = Clay_Hovered();
                            sndRollover(n, vr.hMinus);
                            sndClick(n);
                            vr.hMinus = n;
                            if (vr.hMinus && inp.lmbPressed)
                            {
                                if (vr.bufIdx == 0 && audio_)
                                    audio_->setMasterVolume(audio_->getMasterVolume() - 0.05f);
                                else if (vr.bufIdx == 1 && audio_)
                                    audio_->setMusicVolume(audio_->getMusicVolume() - 0.05f);
                                else if (vr.bufIdx == 2 && audio_)
                                    audio_->setSfxVolume(audio_->getSfxVolume() - 0.05f);
                            }
                            CLAY_TEXT(CLAY_STRING("-"), CLAY_TEXT_CONFIG({.textColor = Pal::btnLabel, .fontSize = fs(12)}));
                        }
                        // value
                        CLAY(CLAY_IDI("VolVal", vr.bufIdx), {.layout = {
                                                                 .sizing = {CLAY_SIZING_FIXED(38), CLAY_SIZING_FIT(0)},
                                                                 .childAlignment = {.x = CLAY_ALIGN_X_CENTER}}})
                        {
                            CLAY_TEXT(volStr, CLAY_TEXT_CONFIG({.textColor = Pal::volValue, .fontSize = fs(12)}));
                        }
                        // + button
                        Clay_Color cPlus = vr.hPlus ? Pal::btnHover : Pal::btnIdle;
                        CLAY(CLAY_IDI("VolPlus", vr.bufIdx), {.layout = {
                                                                  .sizing = {CLAY_SIZING_FIXED(20), CLAY_SIZING_FIXED(20)},
                                                                  .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                              .backgroundColor = cPlus,
                                                              .cornerRadius = CLAY_CORNER_RADIUS(3)})
                        {
                            bool n = Clay_Hovered();
                            sndRollover(n, vr.hPlus);
                            sndClick(n);
                            vr.hPlus = n;
                            if (vr.hPlus && inp.lmbPressed)
                            {
                                if (vr.bufIdx == 0 && audio_)
                                    audio_->setMasterVolume(audio_->getMasterVolume() + 0.05f);
                                else if (vr.bufIdx == 1 && audio_)
                                    audio_->setMusicVolume(audio_->getMusicVolume() + 0.05f);
                                else if (vr.bufIdx == 2 && audio_)
                                    audio_->setSfxVolume(audio_->getSfxVolume() + 0.05f);
                            }
                            CLAY_TEXT(CLAY_STRING("+"), CLAY_TEXT_CONFIG({.textColor = Pal::btnLabel, .fontSize = fs(12)}));
                        }
                    }
                }

                // ── Controls ──────────────────────────────────────────────────
                CLAY(CLAY_ID("SndCtrlSep"), {.layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                        .padding = {0, 0, 6, 4}},
                                             .backgroundColor = Pal::sectionHdr}) {}
                CLAY_TEXT(CLAY_STRING("Controls"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textSection, .fontSize = fs(14)}));

                // Thin separator
                CLAY(CLAY_ID("CtrlSep"), {.layout = {
                                              .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                              .padding = {0, 0, 4, 4}},
                                          .backgroundColor = Pal::sectionHdr}) {}

                // One row per keybinding
                static char kbKeyBuf[8][16];
                for (int ki = 0; ki < (int)keybindings.size() && ki < 8; ++ki)
                {
                    KeyBinding &kb = keybindings[ki];
                    snprintf(kbKeyBuf[ki], sizeof(kbKeyBuf[ki]), "[%s]", keyDisplayName(kb.key));

                    Clay_Color rowBg = kb.listening
                                           ? Pal::listenRow
                                           : Clay_Color{0, 0, 0, 0};
                    CLAY(CLAY_IDI("KbRow", ki), {.layout = {
                                                     .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(28)},
                                                     .padding = {4, 4, 4, 4},
                                                     .childGap = 6,
                                                     .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                                     .layoutDirection = CLAY_LEFT_TO_RIGHT},
                                                 .backgroundColor = rowBg,
                                                 .cornerRadius = CLAY_CORNER_RADIUS(3)})
                    {
                        // Action name (fixed width)
                        CLAY(CLAY_IDI("KbAction", ki), {.layout = {
                                                            .sizing = {CLAY_SIZING_FIXED(130), CLAY_SIZING_FIT(0)}}})
                        {
                            Clay_String actStr{false, (int32_t)strlen(kb.action), kb.action};
                            CLAY_TEXT(actStr,
                                      CLAY_TEXT_CONFIG({.textColor = Pal::volLabel, .fontSize = fs(13)}));
                        }

                        // Current key label (fixed width)
                        CLAY(CLAY_IDI("KbKey", ki), {.layout = {
                                                         .sizing = {CLAY_SIZING_FIXED(60), CLAY_SIZING_FIT(0)}}})
                        {
                            Clay_String keyStr{false, (int32_t)strlen(kbKeyBuf[ki]), kbKeyBuf[ki]};
                            Clay_Color keyCol = kb.listening
                                                    ? Pal::listenKey
                                                    : Pal::keyText;
                            CLAY_TEXT(keyStr,
                                      CLAY_TEXT_CONFIG({.textColor = keyCol, .fontSize = fs(13)}));
                        }

                        // Rebind button
                        Clay_Color rebindBg = kb.listening
                                                  ? Pal::listenBtn
                                                  : (hovRebind[ki] ? Pal::btnHover : Pal::btnIdle);
                        CLAY(CLAY_IDI("KbRebind", ki), {.layout = {
                                                            .sizing = {CLAY_SIZING_FIXED(80), CLAY_SIZING_FIXED(20)},
                                                            .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                        .backgroundColor = rebindBg,
                                                        .cornerRadius = CLAY_CORNER_RADIUS(3)})
                        {
                            {
                                bool n = Clay_Hovered();
                                sndRollover(n, hovRebind[ki]);
                                sndClick(n);
                                hovRebind[ki] = n;
                            }
                            if (hovRebind[ki] && inp.lmbPressed)
                            {
                                // Cancel any other listening binding
                                for (auto &other : keybindings)
                                    other.listening = false;
                                kb.listening = true;
                            }
                            CLAY_TEXT(kb.listening ? CLAY_STRING("PRESS KEY") : CLAY_STRING("Rebind"),
                                      CLAY_TEXT_CONFIG({.textColor = Pal::btnLabel, .fontSize = fs(10)}));
                        }
                    }
                }

                // Camera controls section
                CLAY(CLAY_ID("CamCtrlSep"), {.layout = {
                                                 .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                 .padding = {0, 0, 6, 4}},
                                             .backgroundColor = Pal::sectionHdr}) {}
                CLAY_TEXT(CLAY_STRING("Camera"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textSection, .fontSize = fs(14)}));
                CLAY_TEXT(CLAY_STRING("Right-click drag   Look around"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textCamera, .fontSize = fs(12)}));
                CLAY_TEXT(CLAY_STRING("Scroll wheel        Zoom (FOV)"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textCamera, .fontSize = fs(12)}));

                // ── UI Scale ──────────────────────────────────────────────────
                CLAY(CLAY_ID("UiScaleSep"), {.layout = {
                                                 .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                 .padding = {0, 0, 6, 4}},
                                             .backgroundColor = Pal::sectionHdr}) {}
                CLAY_TEXT(CLAY_STRING("Display"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textSection, .fontSize = fs(14)}));

                CLAY(CLAY_ID("UiScaleRow"), {.layout = {
                                                 .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(28)},
                                                 .padding = {4, 4, 4, 4},
                                                 .childGap = 8,
                                                 .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                                 .layoutDirection = CLAY_LEFT_TO_RIGHT}})
                {
                    CLAY_TEXT(CLAY_STRING("Text scale"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::volLabel, .fontSize = fs(13)}));

                    // Spacer
                    CLAY(CLAY_ID("UiScaleSpacer"), {.layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)}}}) {}

                    // − button
                    Clay_Color scaleMinusBg = hovScaleMinus ? Pal::btnHover : Pal::btnIdle;
                    CLAY(CLAY_ID("UiScaleMinus"), {.layout = {
                                                       .sizing = {CLAY_SIZING_FIXED(22), CLAY_SIZING_FIXED(22)},
                                                       .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                   .backgroundColor = scaleMinusBg,
                                                   .cornerRadius = CLAY_CORNER_RADIUS(3)})
                    {
                        {
                            bool n = Clay_Hovered();
                            sndRollover(n, hovScaleMinus);
                            sndClick(n);
                            hovScaleMinus = n;
                        }
                        if (hovScaleMinus && inp.lmbPressed)
                            uiScale = std::max(0.75f, uiScale - 0.125f);
                        CLAY_TEXT(CLAY_STRING("-"), CLAY_TEXT_CONFIG({.textColor = Pal::btnLabel, .fontSize = fs(13)}));
                    }

                    // Scale readout
                    static char scaleBuf[8];
                    snprintf(scaleBuf, sizeof(scaleBuf), "%.2fx", uiScale);
                    Clay_String scaleStr{false, (int32_t)strlen(scaleBuf), scaleBuf};
                    CLAY(CLAY_ID("UiScaleVal"), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(44), CLAY_SIZING_FIT(0)},
                                                     .childAlignment = {.x = CLAY_ALIGN_X_CENTER}}})
                    {
                        CLAY_TEXT(scaleStr, CLAY_TEXT_CONFIG({.textColor = Pal::volValue, .fontSize = fs(13)}));
                    }

                    // + button
                    Clay_Color scalePlusBg = hovScalePlus ? Pal::btnHover : Pal::btnIdle;
                    CLAY(CLAY_ID("UiScalePlus"), {.layout = {
                                                      .sizing = {CLAY_SIZING_FIXED(22), CLAY_SIZING_FIXED(22)},
                                                      .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                  .backgroundColor = scalePlusBg,
                                                  .cornerRadius = CLAY_CORNER_RADIUS(3)})
                    {
                        {
                            bool n = Clay_Hovered();
                            sndRollover(n, hovScalePlus);
                            sndClick(n);
                            hovScalePlus = n;
                        }
                        if (hovScalePlus && inp.lmbPressed)
                            uiScale = std::min(2.0f, uiScale + 0.125f);
                        CLAY_TEXT(CLAY_STRING("+"), CLAY_TEXT_CONFIG({.textColor = Pal::btnLabel, .fontSize = fs(13)}));
                    }
                }
            }
        }
    }

    // ── Mouse capture rects ───────────────────────────────────────────────────
    ui.addMouseCaptureRect(12, inp.screenH - 110.0f, 340, 100); // time panel (bottom-left)
    ui.addMouseCaptureRect(inp.screenW - 460.0f, inp.screenH - 52.0f,
                           450.0f, 44.0f); // status bar (bottom-right)
    if (settingsOpen)
        ui.addMouseCaptureRect((inp.screenW - 500.0f) * 0.5f,
                               (inp.screenH - 500.0f) * 0.5f,
                               500.0f, 500.0f); // settings window

    // ── Cinematic intro overlay ───────────────────────────────────────────────
    if (showIntro)
    {
        if (inp.lmbPressed || inp.rmbPressed)
            showIntro = false;
        ui.addMouseCaptureRect(0, 0, inp.screenW, inp.screenH);

        CLAY(CLAY_ID("IntroOverlay"), {.layout = {
                                           .sizing = {CLAY_SIZING_FIXED((float)inp.screenW),
                                                      CLAY_SIZING_FIXED((float)inp.screenH)},
                                           .childAlignment = {.x = CLAY_ALIGN_X_CENTER,
                                                              .y = CLAY_ALIGN_Y_CENTER},
                                           .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                       .backgroundColor = {0, 0, 0, 185},
                                       .floating = {.zIndex = 30, .attachTo = CLAY_ATTACH_TO_ROOT}})
        {
            CLAY(CLAY_ID("IntroPanel"), {.layout = {
                                             .sizing = {CLAY_SIZING_FIXED(660),
                                                        CLAY_SIZING_FIT(0)},
                                             .childGap = 0,
                                             .layoutDirection = CLAY_TOP_TO_BOTTOM}})
            {
                // ── Title ─────────────────────────────────────────────────────
                CLAY_TEXT(CLAY_STRING("Welcome to the future!"),
                          CLAY_TEXT_CONFIG({.textColor = {255, 255, 255, 255},
                                            .fontSize = fs(34)}));

                CLAY(CLAY_ID("IP1"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1),
                                                            CLAY_SIZING_FIXED(22)}}}) {}

                // ── Body ──────────────────────────────────────────────────────
                CLAY(CLAY_ID("IntroBody"), {.layout = {
                                                .sizing = {CLAY_SIZING_FIXED(660), CLAY_SIZING_FIT(0)},
                                                .childGap = 14,
                                                .layoutDirection = CLAY_TOP_TO_BOTTOM}})
                {
                    CLAY_TEXT(CLAY_STRING("Every planned major space constellation has been constructed."),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(18)}));
                    CLAY_TEXT(CLAY_STRING("Perpetual solar power lies in sun synchronous orbit, which has become competitive real estate for football field-sized space datacenters"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(18)}));
                    CLAY_TEXT(CLAY_STRING(
                                  "Whether or not they are profitable, useful, or even still functional, "
                                  "they are going to be up there for a very long time."),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(18)}));
                    CLAY_TEXT(CLAY_STRING("We will come to miss the quiet sky."),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textPrimary, .fontSize = fs(18)}));
                }

                CLAY(CLAY_ID("IP2"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1),
                                                            CLAY_SIZING_FIXED(48)}}}) {}

                // ── Controls ──────────────────────────────────────────────────
                CLAY(CLAY_ID("IntroControls"), {.layout = {
                                                    .sizing = {CLAY_SIZING_FIXED(660),
                                                               CLAY_SIZING_FIT(0)},
                                                    .childGap = 7,
                                                    .layoutDirection = CLAY_TOP_TO_BOTTOM}})
                {
                    CLAY_TEXT(CLAY_STRING("WASD  =  Move"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textSection,
                                                .fontSize = fs(13)}));
                    CLAY_TEXT(CLAY_STRING("Right click  =  Look"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textSection,
                                                .fontSize = fs(13)}));
                    CLAY_TEXT(CLAY_STRING("Shift  =  Boost"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textSection,
                                                .fontSize = fs(13)}));
                    CLAY_TEXT(CLAY_STRING("Space  =  Play / Pause"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textSection,
                                                .fontSize = fs(13)}));
                    CLAY_TEXT(CLAY_STRING("Comma  =  Slow down time"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textSection,
                                                .fontSize = fs(13)}));
                    CLAY_TEXT(CLAY_STRING("Period  =  Speed up time"),
                              CLAY_TEXT_CONFIG({.textColor = Pal::textSection,
                                                .fontSize = fs(13)}));
                }

                CLAY(CLAY_ID("IP3"), {.layout = {.sizing = {CLAY_SIZING_FIXED(1),
                                                            CLAY_SIZING_FIXED(32)}}}) {}

                // ── Dismiss hint ──────────────────────────────────────────────
                CLAY_TEXT(CLAY_STRING("Click or press any key to continue"),
                          CLAY_TEXT_CONFIG({.textColor = Pal::textHint,
                                            .fontSize = fs(11)}));
            }
        }
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
}

// ─── cleanup ──────────────────────────────────────────────────────────────────
void SatelliteSim::cleanup(VkDevice device)
{
    vkDestroyPipeline(device, compPipeline, nullptr);
    vkDestroyPipeline(device, skyBgPipeline, nullptr);
    vkDestroyPipeline(device, drawPipeline, nullptr);
    vkDestroyPipelineLayout(device, compPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, skyBgPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, drawPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyDescriptorPool(device, skyDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, skyDescLayout, nullptr);
    vkUnmapMemory(device, glowMem);
    vkDestroyBuffer(device, glowBuf, nullptr);
    vkFreeMemory(device, glowMem, nullptr);
    vkUnmapMemory(device, satInputMem);
    vkDestroyBuffer(device, satInputBuf, nullptr);
    vkFreeMemory(device, satInputMem, nullptr);
    vkDestroyBuffer(device, satVisibleBuf, nullptr);
    vkFreeMemory(device, satVisibleMem, nullptr);

    vkDestroyPipeline(device, starPipeline, nullptr);
    vkDestroyPipelineLayout(device, starPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, starDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, starDescLayout, nullptr);
    if (starMapped)
        vkUnmapMemory(device, starMem);
    vkDestroyBuffer(device, starBuf, nullptr);
    vkFreeMemory(device, starMem, nullptr);

    if (win)
        glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

// ─── onKey ────────────────────────────────────────────────────────────────────
void SatelliteSim::onKey(GLFWwindow *w, int key, int action)
{
    win = w;
    if (action != GLFW_PRESS)
        return;

    if (showIntro)
    {
        showIntro = false;
        return;
    }

    // If any binding is listening, capture this key press and assign it.
    for (auto &kb : keybindings)
    {
        if (!kb.listening)
            continue;
        if (key == GLFW_KEY_ESCAPE)
        {
            kb.listening = false; // cancel rebind
        }
        else
        {
            kb.key = key;
            kb.listening = false;
        }
        return; // consume the key event
    }

    // Dispatch via keybindings array
    auto pressed = [&](int bindIdx)
    {
        return bindIdx < (int)keybindings.size() && key == keybindings[bindIdx].key;
    };

    if (pressed(0))
        uiVisible = !uiVisible; // Toggle UI
    if (pressed(1))
        timePaused = !timePaused; // Pause/Resume
    if (pressed(2))
        timeScaleIdx = std::max(0, timeScaleIdx - 1); // Slow Down
    if (pressed(3))
        timeScaleIdx = std::min(kNumTimeScales - 1, timeScaleIdx + 1); // Speed Up
    if (pressed(4))
        timeDir = -timeDir; // Reverse Time
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
    // satInputBuf: host-visible + coherent, persistently mapped.
    ctx.createBuffer(sizeof(GpuSatInput) * MAX_SATELLITES,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     satInputBuf, satInputMem);
    vkMapMemory(ctx.device, satInputMem, 0,
                sizeof(GpuSatInput) * MAX_SATELLITES, 0, &satInputMapped);

    // satVisibleBuf: device-local. Compute writes, vertex reads.
    ctx.createBuffer(sizeof(GpuSatVisible) * MAX_SATELLITES,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     satVisibleBuf, satVisibleMem);
}

// ─── createDescriptors ────────────────────────────────────────────────────────
void SatelliteSim::createDescriptors(VulkanContext &ctx)
{
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 2;
    li.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &descLayout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &descSet);

    VkDescriptorBufferInfo inpInfo{satInputBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo visInfo{satVisibleBuf, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[2] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inpInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visInfo, nullptr};
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
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

// ─── createSkyBgPipeline ──────────────────────────────────────────────────────
// ─── createGlowResources ──────────────────────────────────────────────────────
// Allocates the host-visible SSBO that holds up to kMaxGlows bright-flare entries,
// and creates the descriptor set used by the sky background pipeline to read it.
void SatelliteSim::createGlowResources(VulkanContext &ctx)
{
    VkDeviceSize bufSize = sizeof(GpuGlowBuf);
    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     glowBuf, glowMem);
    vkMapMemory(ctx.device, glowMem, 0, bufSize, 0, &glowMapped);
    memset(glowMapped, 0, bufSize);

    VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 1;
    li.pBindings = &b;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &skyDescLayout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &skyDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = skyDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &skyDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &skyDescSet);

    VkDescriptorBufferInfo bi{glowBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = skyDescSet;
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr.pBufferInfo = &bi;
    vkUpdateDescriptorSets(ctx.device, 1, &wr, 0, nullptr);
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

    // No depth test — background overwrites the clear color.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

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

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
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

    // No depth test — satellite points are a sky overlay.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

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
        VkPushConstantRange drawPcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SatDrawPC)};
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

        // Point sprite size: 1.5 px for faint stars, up to ~5.5 px for Sirius.
        float starScale = 4.0f; // tweak this to make stars bigger/smaller overall
        float angSize = 1.5f + glm::min(rawInt, 4.0f) * 1.0f;
        angSize *= starScale;

        starRecords.push_back({eciDir, rawInt, col, angSize});
    }
    starCount = (uint32_t)starRecords.size();

    // Host-visible buffer (tiny: ~287 × 32 bytes = ~9 KB).
    VkDeviceSize bufSize = starCount * sizeof(GpuSatVisible);
    ctx.createBuffer(bufSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     starBuf, starMem);
    vkMapMemory(ctx.device, starMem, 0, bufSize, 0, &starMapped);

    // Descriptor layout: only binding=1 (vertex shader reads GpuSatVisible).
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 1;
    li.pBindings = &binding;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &starDescLayout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &starDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = starDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &starDescLayout;
    vkAllocateDescriptorSets(ctx.device, &ai, &starDescSet);

    VkDescriptorBufferInfo bufInfo{starBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = starDescSet;
    wr.dstBinding = 1;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr.pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(ctx.device, 1, &wr, 0, nullptr);

    createStarPipeline(ctx);

    // Do an initial upload so stars are visible from frame 1.
    updateStars();
}

// ─── createStarPipeline ───────────────────────────────────────────────────────
// Reuses sat_point.vert/frag with a separate descriptor layout (binding=1 only).
void SatelliteSim::createStarPipeline(VulkanContext &ctx)
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
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    if (starPipeLayout == VK_NULL_HANDLE)
    {
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SatDrawPC)};
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

// ─── updateStars ──────────────────────────────────────────────────────────────
// Transforms star ECI unit vectors into ENU each frame (Earth rotates under stars).
// Stars fade out during civil/nautical twilight — invisible in full daylight.
void SatelliteSim::updateStars()
{
    if (!starMapped || starCount == 0)
        return;

    // Stars become visible as the sun sinks below the horizon.
    // sin(elevation) = sunDirENU.w: 0 at horizon, -0.2 at ~11.5° below.
    float nightFactor = glm::clamp(-sunDirENU.w * 5.0f, 0.0f, 1.0f);

    auto *dst = static_cast<GpuSatVisible *>(starMapped);
    for (uint32_t i = 0; i < starCount; ++i)
    {
        const auto &rec = starRecords[i];

        // Rotate from inertial ECI into the observer's local ENU frame.
        glm::vec3 enu{glm::dot(rec.eciDir, glm::vec3(eci2enuX)),
                      glm::dot(rec.eciDir, glm::vec3(eci2enuY)),
                      glm::dot(rec.eciDir, glm::vec3(eci2enuZ))};

        // Stars below the local horizon are discarded by the vertex shader
        // (flareIntensity ≤ 0 → position pushed out of clip space).
        float intensity = (enu.z >= 0.0f) ? rec.rawIntensity * nightFactor : 0.0f;

        dst[i].skyDir = enu;
        dst[i].flareIntensity = intensity;
        dst[i].baseColor = rec.color;
        dst[i].angularSize = rec.angSize;
    }
}

// ─── initConstellation ────────────────────────────────────────────────────────
// Define satellite types + constellation shells; populate flat satOrbits array.
// Orbit distribution types:
//   Walker      — regular planes × sats grid (classic constellation)
//   RandomShell — random RAAN, random incl in [0, c.incl], jittered altitude
//   Disk        — one or more concentric rings in a fixed orbital plane (incl + raan).
//                 Set alignTerminator=true to auto-derive the plane from sunDirECI.
void SatelliteSim::initConstellation()
{
    // ── Satellite type catalogue ───────────────────────────────────────────────
    // crossSectionM2: effective reflective area in m².  Brightness ∝ sqrt(area/10 m²)
    // so doubling the area → ~1.4× brighter (avoids runaway scale for giant structures).
    //
    // Each type composes up to two reflective surfaces plus an isotropic diffuse floor:
    //   primary   — dominant surface (solar panels, antenna face)
    //   secondary — optional second surface (radiators perpendicular to primay)
    //               weight=0 disables it; Perpendicular attitude = cross(surfN0, satNadir)
    //   diffuse   — constant Lambertian floor for structural body scatter (always visible)

    // AI SAT
    // 175 m estimate, 1820 x 72
    // scale 1820px / 175 m
    // 870 * 72 px panels
    // 200 x 55 px radiator panels

    float px_scale = 1820.0f / 175.0f;                                         // ~10.4 px/meter for the bus/antenna face
    float ai_sat_panel_area = (870.0f * 72.0f * 2.0f) / (px_scale * px_scale); // ~600 m² total panel area
    float ai_sat_radiator_area = (200.0f * 55.0f) / (px_scale * px_scale);     // ~100 m² total radiator area

    satTypes = {
        {// 0 — Starlink: flat phased-array face toward Earth, brief intense flares
         "Starlink",
         {0.80f, 0.87f, 1.00f},                      // cool blue-white
         10.0f,                                      // ~10 m² bus + visor
         {AttitudeMode::NadirPointing, 18.0f, 1.0f}, // very sharp specular (flat mirror-like face)
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},  // no significant secondary surface
         0.0f,                                       // no diffuse floor (visor-darkened)
         0.05f},                                     // mirrorFrac: polished phased-array glass → mag ~-2.7 at perfect alignment
        {                                            // 1 — LEO broadband (OneWeb/Kuiper/Xingwang/Telesat): sun-tracking panels
         "LEO Broadband",
         {1.00f, 0.92f, 0.75f}, // warm white
         5.0f,                  // ~12 m² typical LEO broadband bus + panels
         {AttitudeMode::SunTracking, 18.0f, 1.0f},
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},
         0.0f,   // no diffuse floor
         0.02f}, // mirrorFrac: moderate — sun-tracking panels occasionally flash
        {        // 2 — GEO Comsat: large sun-tracking panels + body radiators facing away from Earth
         "GEO Comsat",
         {0.95f, 0.95f, 1.00f},                    // near-white
         50.0f,                                    // ~50 m² (large GEO body + wings)
         {AttitudeMode::SunTracking, 3.0f, 1.00f}, // broad lobe solar wings
         {AttitudeMode::AntiNadir, 2.0f, 0.10f},   // body radiators face deep space
         0.01f,                                    // slight structural glow
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
        {                                           // 4 — SpaceX ODC (FCC filing Jan 2026): Starlink-class bus with large radiator panels
         // for compute heat rejection. Nadir-pointing phased array + deep-space-facing radiators.
         // Flat compute/antenna face produces brief nadir flares; radiator face brightens near
         // the horizon (AntiNadir face tilts toward the observer as satellite descends).

         "SpaceX AI Sats",
         {1.00f, 1.00f, 0.92f},                    // cyan-teal (distinct from Starlink blue-white)
         ai_sat_panel_area,                        // ~15 m² — Starlink-class bus + extra radiator area
         {AttitudeMode::SunTracking, 18.0f, 1.0f}, // phased-array/compute face toward Earth
         {AttitudeMode::AntiNadir, 2.0f, 0.25f},   // large radiator panels face deep space
         0.005f,                                   // minimal structural body scatter
         0.01f},                                   // mirrorFrac: similar to Starlink — polished compute face
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
        {"Starlink Gen2",
         525'000.0f,          // altM:      525 km (slightly lower than Gen1)
         glm::radians(53.2f), // incl:      53.2°
         120,                 // numPlanes: orbital planes
         254,                 // perPlane:  sats per plane (120×254 = 30,480)
         0u,                  // typeIdx:   Starlink (NadirPointing)
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
        {"Xingwang",
         508'000.0f,          // altM:      508 km
         glm::radians(85.0f), // incl:      85° — near-polar
         80,                  // numPlanes: orbital planes
         174,                 // perPlane:  sats per plane (80×174 = 13,920)
         1u,                  // typeIdx:   LEO Broadband (SunTracking)
         true,                // enabled
         OrbitDistribution::Walker},

        // GEO commercial belt — representative sample of operational comsats
        {"GEO Belt",
         35'786'000.0f,      // altM:      35,786 km (geostationary)
         glm::radians(0.0f), // incl:      0° — equatorial
         1,                  // numPlanes: single equatorial ring
         50,                 // perPlane:  50 representative comsats
         2u,                 // typeIdx:   GEO Comsat (SunTracking + AntiNadir radiators)
         true,               // enabled
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
         5000.0f,     // altJitterM:      no per-satellite altitude scatter
         0.0f,        // raan:            ignored (alignTerminator=true)
         true,        // alignTerminator: orbital plane = terminator plane (tracks Sun)
         10,          // numRings:        10 concentric rings
         150'000.0f}, // ringSpacingM:    150 km between rings (575–1,925 km range)
    };

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

                float tumbleRate = 0.008f * (float)rand() / (float)RAND_MAX;
                float tumblePhase = (float)rand() / RAND_MAX * glm::two_pi<float>();

                satOrbits.push_back({raan, incl, u0, c.typeIdx, altM,
                                     tumbleRate, tumblePhase, axis, false});
            }
        }
        else if (c.distribution == OrbitDistribution::Disk)
        {
            // Determine orbital plane.
            float incl_d = c.incl;
            float raan_d = c.raan;
            if (c.alignTerminator)
            {
                // Physically correct SSO inclination from the J2 nodal precession formula.
                // Depends only on orbit altitude (~100.6° at 1250 km); does NOT vary with season.
                incl_d = computeSSOInclination(c.altM);

                // Reference RAAN anchored at J2000 epoch (2000-01-01 12:00 TT).
                // At J2000 the orbit is at the dawn-dusk terminator.  updatePositions()
                // then applies liveRaan = raan_j2000 + kSSOPrecRate * t (absolute J2000 s),
                // giving a fully deterministic sky position at any simulation date.
                glm::vec3 sunJ2000 = sunDirECIAtJ2000();
                float sinI = sinf(incl_d);
                raan_d = (sinI > 1e-5f) ? atan2f(sunJ2000.x, -sunJ2000.y) : 0.0f;
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
                float ringIncl = incl_d; // same inclination for all rings (flat disk)

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
    }

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
    activeSatCount = (uint32_t)satOrbits.size();
    satInputData.resize(activeSatCount);
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
void SatelliteSim::updatePositions(double t)
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

    obsECI = glm::vec3{kEarthRadius * cosLat * cosLon,
                       kEarthRadius * cosLat * sinLon,
                       kEarthRadius * sinLat};

    // ── ECI → ENU basis vectors ───────────────────────────────────────────────
    glm::vec3 east{-sinLon, cosLon, 0.0f};
    glm::vec3 north{-sinLat * cosLon, -sinLat * sinLon, cosLat};
    glm::vec3 up{cosLat * cosLon, cosLat * sinLon, sinLat};

    eci2enuX = glm::vec4(east, 0.0f);
    eci2enuY = glm::vec4(north, 0.0f);
    eci2enuZ = glm::vec4(up, 0.0f);

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

    // ── Moon direction in ECI (simple circular equatorial orbit) ─────────────
    // Period: 27.3217 days. The moon orbits in the ecliptic plane (~5° tilt);
    // for rendering purposes an equatorial approximation is sufficient.
    static constexpr double kMoonPeriodSec = 27.3217 * 86400.0;
    // Phase offset calibrated so the moon is 91% illuminated (waxing gibbous) at the
    // fixed sim start epoch 2026-03-30 05:53:58 UTC (t=828121038 J2000 s).
    // Derived: sun at ecliptic lon ~14°, moon must be at ~158° for dot=-0.82.
    // At t_start the bare formula gives ~294°; offset = 158°-294° = +224° = 3.916 rad.
    static constexpr double kMoonPhaseOffsetRad = 3.916;
    double moonAngle = fmod(2.0 * glm::pi<double>() * (t) / kMoonPeriodSec + kMoonPhaseOffsetRad,
                            glm::two_pi<double>());
    moonDirECI = glm::vec3{(float)cos(moonAngle), (float)sin(moonAngle), 0.0f};

    // Moon in ENU
    glm::vec3 moonENU_local{
        glm::dot(moonDirECI, east),
        glm::dot(moonDirECI, north),
        glm::dot(moonDirECI, up)};
    // Illuminated fraction = (1 − dot(sunDir, moonDir)) / 2
    // Full moon when moon is opposite the sun; new moon when aligned.
    float moonIllum = (1.0f - glm::dot(sunDirECI, moonDirECI)) * 0.5f;
    moonDirENU = glm::vec4(moonENU_local, moonIllum);

    // ── Per-constellation satellite geometry ──────────────────────────────────
    visibleCount = 0;
    peakMagnitude = 99.0f; // reset each frame; updated below for each visible sat
    glowEntryCount = 0;
    glowMinIntensity = 0.0f;
    for (const ConstellationConfig &c : constellations)
    {
        for (uint32_t i = c.orbitStart; i < c.orbitStart + c.orbitCount; ++i)
        {
            if (!c.enabled)
            {
                satInputData[i].elevation = -glm::half_pi<float>(); // force-cull
                continue;
            }

            const SatOrbit &orb = satOrbits[i];
            const SatelliteType &type = satTypes[orb.typeIdx];

            // Per-satellite altitude and mean motion (allows jitter for debris shells).
            float R_sat = kEarthRadius + orb.altM;
            float meanMot = (float)sqrt(kGM / ((double)R_sat * R_sat * R_sat));

            float cosI = cosf(orb.incl), sinI = sinf(orb.incl);

            // Double-precision phase then fmod to [0,2π) keeps float trig accurate.
            float u = (float)fmod((double)orb.u0 + (double)meanMot * t,
                                  glm::two_pi<double>());
            float cosU = cosf(u), sinU = sinf(u);

            // alignTerminator: precess RAAN at the SSO rate from the J2000-epoch reference RAAN.
            // Using absolute t (J2000 seconds) makes the orbit position deterministic at any
            // simulation date — changing the init time now produces a different sky position.
            float liveRaan = orb.alignTerminator
                                 ? (float)fmod((double)orb.raan + kSSOPrecRate * t, glm::two_pi<double>())
                                 : orb.raan;
            float cosR = cosf(liveRaan), sinR = sinf(liveRaan);

            // Satellite ECI position (Rz(RAAN) · Rx(incl) · perifocal).
            float ex = cosR * cosU - sinR * sinU * cosI;
            float ey = sinR * cosU + cosR * sinU * cosI;
            float ez = sinU * sinI;

            glm::vec3 satECI{ex * R_sat, ey * R_sat, ez * R_sat};
            glm::vec3 relPos = satECI - obsECI;
            float range = glm::length(relPos);

            float sinEl = glm::dot(relPos / range, up);
            float el = asinf(glm::clamp(sinEl, -1.0f, 1.0f));

            glm::vec3 satECI_abs = obsECI + relPos;
            glm::vec3 satNadir = glm::normalize(-satECI_abs); // unit vector toward Earth centre

            // ── Compute surface normals ───────────────────────────────────────
            // surfN0 = primary surface (solar panels, antenna face, tumbling body).
            // surfN1 = secondary surface (radiators, body panels).
            //   AntiNadir  — normal = -satNadir (faces deep space, away from Earth)
            //   Perpendicular — normal = cross(surfN0, satNadir) (along orbital track)
            // The irradiance term in sat_flare.comp gates each surface by solar flux received.
            auto computeNormal = [&](AttitudeMode mode,
                                     const glm::vec3 &primary) -> glm::vec3
            {
                switch (mode)
                {
                case AttitudeMode::NadirPointing:
                    return satNadir;
                case AttitudeMode::Tumbling:
                {
                    // Spin around a body-fixed random axis; angle advances with simTime.
                    // fmod before float cast: simTime ≈ 8e8 s gives only ~96 s float precision,
                    // causing (float)t * rate to jump by ~96 rad between frames → staggering.
                    float angle = orb.tumblePhase +
                                  (float)fmod((double)orb.tumbleRate * t, glm::two_pi<double>());
                    glm::vec3 ax = orb.tumbleAxis;
                    glm::vec3 ref = (fabsf(ax.z) < 0.9f) ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
                    glm::vec3 axA = glm::normalize(glm::cross(ax, ref));
                    glm::vec3 axB = glm::cross(ax, axA); // already unit length
                    return cosf(angle) * axA + sinf(angle) * axB;
                }
                case AttitudeMode::Perpendicular:
                {
                    // 90° to primary panel normal in the nadir plane, along the orbital track.
                    glm::vec3 perp = glm::cross(primary, satNadir);
                    float len = glm::length(perp);
                    return (len > 1e-5f) ? perp / len : satNadir;
                }
                case AttitudeMode::AntiNadir:
                {
                    // Thermal radiators facing deep space (away from Earth center).
                    // From the ground: edge-on to observers directly beneath the satellite
                    // (zenith pass) — they see the back of the panel; horizon observers
                    // see the face tilted toward them, making flares more likely at low elevation.
                    return -satNadir;
                }
                default: // SunTracking
                {
                    // Two-axis sun-tracking: panel normal points directly at the sun.
                    // reflect(-sunDir, sunDir) = sunDir, so specular peaks when
                    // satToObs ≈ sunDir — i.e. when the satellite is at the antisolar
                    // point of the sky.  irr0 = 1.0 (panel always fully lit).
                    return sunDirECI;
                }
                }
            };

            glm::vec3 surfN0 = computeNormal(type.primary.attitude, glm::vec3(0.0f));
            glm::vec3 surfN1 = computeNormal(type.secondary.attitude, surfN0);

            satInputData[i].eciRelPos = relPos;
            satInputData[i].range = range;
            satInputData[i].surfN0 = surfN0;
            satInputData[i].elevation = el;
            satInputData[i].surfN1 = surfN1;
            satInputData[i].specExp0 = type.primary.specExp;
            satInputData[i].baseColor = type.baseColor;
            satInputData[i].specExp1 = type.secondary.specExp;
            satInputData[i].crossSection = sqrtf(type.crossSectionM2 / 10.0f);
            satInputData[i].w1 = type.secondary.weight;
            satInputData[i].diffuse = type.diffuse;
            satInputData[i].mirrorFrac = type.mirrorFrac;

            if (el > -0.01f)
            {
                ++visibleCount;

                // ── Steady-state apparent magnitude ───────────────────────────
                // Mirrors the GPU formula (sat_flare.comp) using only the diffuse
                // floor — no specular — giving the baseline brightness of the sat
                // when not actively flaring.  Cheap: no powf/reflect, just dot+sqrt.
                //
                // We skip the full specular term here to avoid an extra ~10 float
                // ops per satellite at 100k scale (~1 ms CPU budget risk).
                // Flaring satellites are transiently brighter; the readout shows
                // the floor so it changes smoothly rather than flickering.
                float distFactor = kRefRange / std::max(range, kRefRange);
                distFactor *= distFactor; // 1/r²
                float crossSection = sqrtf(type.crossSectionM2 / 10.0f);
                float t = glm::clamp((glm::dot(sunDirECI, up) + 0.05f) / 0.39f, 0.0f, 1.0f);
                float dayBright = t * t;
                // kMagDiffuseFloor: zero-diffuse types (Starlink) scatter faintly from
                // structural body panels; prevents the readout from always showing "--".
                float effectiveDiffuse = std::max(type.diffuse, kMagDiffuseFloor);
                float baseFlare = effectiveDiffuse * distFactor * crossSection * kBrightnessScale / (1.0f + dayBright * kDaySuppression);
                if (baseFlare > 1e-9f)
                {
                    float mag = kMagRef - 2.5f * log10f(baseFlare / kMagRefFlare);
                    peakMagnitude = std::min(peakMagnitude, mag);
                }

                // ── Full-specular peak tracking for sky glow ──────────────
                // Mirrors the compute shader to detect transient specular
                // flares (including mirror peaks) for the sky glow effect.
                // Mirror peak computation guarded by alignment threshold so
                // the expensive pow() only fires when geometry is close.
                glm::vec3 satToObs = glm::normalize(-relPos);

                // Shadow factor (same formula as compute shader).
                glm::vec3 shadowDir = -sunDirECI;
                float proj = glm::dot(satECI_abs, shadowDir);
                glm::vec3 perpV = satECI_abs - proj * shadowDir;
                float perpLen = glm::length(perpV);
                float litF = (proj > 0.0f)
                                 ? glm::smoothstep(-kEarthRadius * 0.01f,
                                                   kEarthRadius * 0.01f,
                                                   perpLen - kEarthRadius)
                                 : 1.0f;

                // Primary surface specular.
                float irr0 = fabsf(glm::dot(sunDirECI, surfN0));
                glm::vec3 n0 = (glm::dot(sunDirECI, surfN0) >= 0.0f) ? surfN0 : -surfN0;
                glm::vec3 refl0 = glm::reflect(-sunDirECI, n0);
                float cosR0 = std::max(0.0f, glm::dot(refl0, satToObs));
                float spec0 = (type.primary.specExp < 0.01f)
                                  ? irr0 * std::max(0.0f, glm::dot(sunDirECI, satToObs))
                                  : irr0 * powf(cosR0, type.primary.specExp);

                // Mirror peak: only compute when near perfect alignment.
                if (cosR0 > 0.9f && type.mirrorFrac > 0.0f)
                {
                    float mExp = std::max(type.primary.specExp * 300.0f, 8000.0f);
                    spec0 += irr0 * powf(cosR0, mExp) * 300.0f * type.mirrorFrac;
                }

                // Secondary surface specular.
                float irr1 = fabsf(glm::dot(sunDirECI, surfN1));
                glm::vec3 n1 = (glm::dot(sunDirECI, surfN1) >= 0.0f) ? surfN1 : -surfN1;
                glm::vec3 refl1 = glm::reflect(-sunDirECI, n1);
                float cosR1 = std::max(0.0f, glm::dot(refl1, satToObs));
                float spec1 = (type.secondary.specExp < 0.01f)
                                  ? irr1 * std::max(0.0f, glm::dot(sunDirECI, satToObs))
                                  : irr1 * powf(cosR1, type.secondary.specExp);

                float specular = spec0 + spec1 * type.secondary.weight + type.diffuse;
                float flareSpec = specular * litF * distFactor * crossSection * kBrightnessScale;
                float effectSpec = flareSpec / (1.0f + dayBright * kDaySuppression);

                if (effectSpec > 0.5f)
                {
                    glm::vec4 entry{glm::normalize(glm::vec3(
                                        glm::dot(relPos, east),
                                        glm::dot(relPos, north),
                                        glm::dot(relPos, up))),
                                    effectSpec};
                    if (glowEntryCount < kMaxGlows)
                    {
                        glowEntries[glowEntryCount++] = entry;
                        if (glowEntryCount == kMaxGlows)
                        {
                            glowMinIntensity = FLT_MAX;
                            for (int gi = 0; gi < kMaxGlows; ++gi)
                                glowMinIntensity = std::min(glowMinIntensity, glowEntries[gi].w);
                        }
                    }
                    else if (effectSpec > glowMinIntensity)
                    {
                        // Replace the weakest entry.
                        int minIdx = 0;
                        for (int gi = 1; gi < kMaxGlows; ++gi)
                            if (glowEntries[gi].w < glowEntries[minIdx].w)
                                minIdx = gi;
                        glowEntries[minIdx] = entry;
                        glowMinIntensity = FLT_MAX;
                        for (int gi = 0; gi < kMaxGlows; ++gi)
                            glowMinIntensity = std::min(glowMinIntensity, glowEntries[gi].w);
                    }
                }
            }
        }
    }
}
