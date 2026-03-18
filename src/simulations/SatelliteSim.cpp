#include "SatelliteSim.h"
#include "../UIRenderer.h"
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
static constexpr float kObsLatDeg = 37;
static constexpr float kObsLat = glm::radians(kObsLatDeg);

// ── Orbital mechanics ─────────────────────────────────────────────────────────
static constexpr double kGM = 3.986004418e14; // Earth gravitational parameter (m³/s²)

static inline float computeMeanMotion(float altM)
{
    double a = (double)kEarthRadius + (double)altM;
    return (float)sqrt(kGM / (a * a * a)); // rad/s
}

// ─── init ─────────────────────────────────────────────────────────────────────
void SatelliteSim::init(VulkanContext &ctx)
{
    // Seed simTime from real UTC so solar position is accurate.
    // J2000.0 = 2000-01-01 12:00:00 UTC = Unix timestamp 946728000.
    {
        auto now = std::chrono::system_clock::now();
        auto j2000 = std::chrono::system_clock::from_time_t(946728000);
        simTime = std::chrono::duration<double>(now - j2000).count();
    }

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
    createComputePipeline(ctx);
    createSkyBgPipeline(ctx);
    createDrawPipeline(ctx);
    initConstellation();
    updatePositions(simTime);
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
    if (!timePaused)
        simTime += (double)dt * kTimeScales[timeScaleIdx] * timeDir;
    updatePositions(simTime);
    updateStars();

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

// ─── buildUI ──────────────────────────────────────────────────────────────────
void SatelliteSim::buildUI(float dt, UIRenderer &ui)
{
    // Apply camera mouse look (uses previous frame's delta, reset after).
    if (win)
        camera.update(win, dmx, dmy);
    dmx = dmy = 0.0f;

    const UIInput &inp = ui.input();

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
        time_t unixSim = (time_t)(simTime) + 946728000;
        struct tm *utc = gmtime(&unixSim);
        if (utc)
            snprintf(timeBuf, sizeof(timeBuf), "UTC %04d-%02d-%02d %02d:%02d:%02d",
                     utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
                     utc->tm_hour, utc->tm_min, utc->tm_sec);
        else
            snprintf(timeBuf, sizeof(timeBuf), "UTC --");
    }

    // ── Info panel (top-left) ─────────────────────────────────────────────────
    CLAY(CLAY_ID("SatInfoPanel"), {.layout = {
                                       .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                       .padding = {12, 12, 10, 10},
                                       .childGap = 5,
                                       .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                   .backgroundColor = {8, 10, 20, 210},
                                   .cornerRadius = CLAY_CORNER_RADIUS(6),
                                   .floating = {.offset = {12, 12}, .zIndex = 5, .attachTo = CLAY_ATTACH_TO_ROOT}})
    {
        CLAY_TEXT(CLAY_STRING("Satellite Constellation"),
                  CLAY_TEXT_CONFIG({.textColor = {200, 220, 255, 255}, .fontSize = 18}));

        Clay_String timeStr{false, (int32_t)strlen(timeBuf), timeBuf};
        CLAY_TEXT(timeStr,
                  CLAY_TEXT_CONFIG({.textColor = {160, 200, 160, 220}, .fontSize = 13}));

        static char visBuf[48];
        snprintf(visBuf, sizeof(visBuf), "%u / %u satellites visible",
                 visibleCount, activeSatCount);
        Clay_String visStr{false, (int32_t)strlen(visBuf), visBuf};
        CLAY_TEXT(visStr,
                  CLAY_TEXT_CONFIG({.textColor = {140, 170, 220, 220}, .fontSize = 13}));

        static char sunBuf[32];
        float sunElDeg = glm::degrees(asinf(glm::clamp(sunDirENU.w, -1.0f, 1.0f)));
        snprintf(sunBuf, sizeof(sunBuf), "Sun elev  %.1f°", sunElDeg);
        Clay_Color sunCol = sunElDeg > 0.0f
                                ? Clay_Color{255, 220, 100, 220}
                                : Clay_Color{100, 100, 140, 180};
        Clay_String sunStr{false, (int32_t)strlen(sunBuf), sunBuf};
        CLAY_TEXT(sunStr, CLAY_TEXT_CONFIG({.textColor = sunCol, .fontSize = 13}));

        Clay_Color camCol = camera.captured
                                ? Clay_Color{100, 255, 120, 255}
                                : Clay_Color{160, 160, 190, 220};
        CLAY_TEXT(camera.captured
                      ? CLAY_STRING("Camera: CAPTURED  (RMB to release)")
                      : CLAY_STRING("Camera: free  (Right-click to capture)"),
                  CLAY_TEXT_CONFIG({.textColor = camCol, .fontSize = 13}));

        static char fpsBuf[24];
        snprintf(fpsBuf, sizeof(fpsBuf), "%.0f fps", dt > 0.0f ? 1.0f / dt : 0.0f);
        Clay_String fpsStr{false, (int32_t)strlen(fpsBuf), fpsBuf};
        CLAY_TEXT(fpsStr,
                  CLAY_TEXT_CONFIG({.textColor = {120, 180, 120, 200}, .fontSize = 12}));
    }

    // ── Constellation toggles (top-right) ─────────────────────────────────────
    CLAY(CLAY_ID("ConstellPanel"), {.layout = {
                                        .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                        .padding = {10, 10, 8, 8},
                                        .childGap = 5,
                                        .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                    .backgroundColor = {8, 10, 20, 210},
                                    .cornerRadius = CLAY_CORNER_RADIUS(6),
                                    .floating = {.offset = {inp.screenW - 195.0f, 12}, .zIndex = 5, .attachTo = CLAY_ATTACH_TO_ROOT}})
    {
        CLAY_TEXT(CLAY_STRING("Constellations"),
                  CLAY_TEXT_CONFIG({.textColor = {200, 220, 255, 255}, .fontSize = 15}));

        static char constCntBuf[10][16];

        for (int ci = 0; ci < (int)constellations.size() && ci < 10; ++ci)
        {
            ConstellationConfig &c = constellations[ci];
            snprintf(constCntBuf[ci], sizeof(constCntBuf[ci]), "%u", c.orbitCount);

            Clay_Color rowBg = c.enabled
                                   ? Clay_Color{20, 50, 90, 180}
                                   : Clay_Color{20, 20, 35, 160};

            CLAY(CLAY_IDI("ConstRow", ci), {.layout = {
                                                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)},
                                                .padding = {4, 4, 3, 3},
                                                .childGap = 6,
                                                .childAlignment = {.y = CLAY_ALIGN_Y_CENTER},
                                                .layoutDirection = CLAY_LEFT_TO_RIGHT},
                                            .backgroundColor = rowBg,
                                            .cornerRadius = CLAY_CORNER_RADIUS(3)})
            {
                Clay_Color btnBg = c.enabled
                                       ? Clay_Color{30, 130, 60, 240}
                                       : (hovConst[ci] ? Clay_Color{80, 40, 40, 230} : Clay_Color{55, 25, 25, 210});
                CLAY(CLAY_IDI("ConstBtn", ci), {.layout = {
                                                    .sizing = {CLAY_SIZING_FIXED(30), CLAY_SIZING_FIXED(18)},
                                                    .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                .backgroundColor = btnBg,
                                                .cornerRadius = CLAY_CORNER_RADIUS(3)})
                {
                    hovConst[ci] = Clay_Hovered();
                    if (hovConst[ci] && inp.lmbPressed)
                        c.enabled = !c.enabled;
                    CLAY_TEXT(c.enabled ? CLAY_STRING("ON") : CLAY_STRING("OFF"),
                              CLAY_TEXT_CONFIG({.textColor = {220, 230, 255, 255}, .fontSize = 10}));
                }

                CLAY(CLAY_IDI("ConstName", ci), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(100), CLAY_SIZING_FIT(0)}}})
                {
                    Clay_String nameStr{false, (int32_t)strlen(c.name), c.name};
                    CLAY_TEXT(nameStr,
                              CLAY_TEXT_CONFIG({.textColor = {180, 200, 240, 220}, .fontSize = 13}));
                }

                Clay_String cntStr{false, (int32_t)strlen(constCntBuf[ci]), constCntBuf[ci]};
                CLAY_TEXT(cntStr,
                          CLAY_TEXT_CONFIG({.textColor = {110, 130, 170, 180}, .fontSize = 11}));
            }
        }
    }

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
                                .backgroundColor = {8, 10, 20, 210},
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
                      CLAY_TEXT_CONFIG({.textColor = {160, 200, 160, 220}, .fontSize = 12}));

            // Speed / direction label
            Clay_Color speedCol = timePaused       ? Clay_Color{200, 100, 60, 220}
                                  : timeDir < 0.0f ? Clay_Color{180, 120, 255, 220}
                                                   : Clay_Color{120, 200, 255, 220};
            Clay_String speedStr{false, (int32_t)strlen(speedBuf), speedBuf};
            CLAY_TEXT(speedStr, CLAY_TEXT_CONFIG({.textColor = speedCol, .fontSize = 12}));
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
            Clay_Color slowBg = hovTimeSlower ? Clay_Color{60, 60, 100, 230} : Clay_Color{30, 30, 55, 210};
            CLAY(CLAY_ID("TimeSlowerBtn"), {.layout = {
                                                .sizing = {CLAY_SIZING_FIXED(kBtnSize), CLAY_SIZING_FIXED(kBtnSize)},
                                                .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                            .backgroundColor = slowBg,
                                            .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                hovTimeSlower = Clay_Hovered();
                if (hovTimeSlower && inp.lmbPressed)
                    timeScaleIdx = std::max(0, timeScaleIdx - 1);
                CLAY(CLAY_ID("TimeSlowerIcon"), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(kIconSize), CLAY_SIZING_FIXED(kIconSize)}},
                                                 .image = {.imageData = (void *)(intptr_t)kIconAngleLeft}}) {}
            }

            // ── Pause / Play ──────────────────────────────────────────────────
            Clay_Color pauseBg = timePaused
                                     ? Clay_Color{160, 60, 30, 230}
                                     : (hovTimePause ? Clay_Color{60, 60, 100, 230} : Clay_Color{30, 30, 55, 210});
            CLAY(CLAY_ID("TimePauseBtn"), {.layout = {
                                               .sizing = {CLAY_SIZING_FIXED(kBtnSize), CLAY_SIZING_FIXED(kBtnSize)},
                                               .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                           .backgroundColor = pauseBg,
                                           .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                hovTimePause = Clay_Hovered();
                if (hovTimePause && inp.lmbPressed)
                    timePaused = !timePaused;
                int pauseIcon = timePaused ? kIconPlay : kIconPause;
                CLAY(CLAY_ID("TimePauseIcon"), {.layout = {
                                                    .sizing = {CLAY_SIZING_FIXED(kIconSize), CLAY_SIZING_FIXED(kIconSize)}},
                                                .image = {.imageData = (void *)(intptr_t)pauseIcon}}) {}
            }

            // ── Speed up ──────────────────────────────────────────────────────
            Clay_Color fastBg = hovTimeFaster ? Clay_Color{60, 60, 100, 230} : Clay_Color{30, 30, 55, 210};
            CLAY(CLAY_ID("TimeFasterBtn"), {.layout = {
                                                .sizing = {CLAY_SIZING_FIXED(kBtnSize), CLAY_SIZING_FIXED(kBtnSize)},
                                                .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                            .backgroundColor = fastBg,
                                            .cornerRadius = CLAY_CORNER_RADIUS(4)})
            {
                hovTimeFaster = Clay_Hovered();
                if (hovTimeFaster && inp.lmbPressed)
                    timeScaleIdx = std::min(kNumTimeScales - 1, timeScaleIdx + 1);
                CLAY(CLAY_ID("TimeFasterIcon"), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(kIconSize), CLAY_SIZING_FIXED(kIconSize)}},
                                                 .image = {.imageData = (void *)(intptr_t)kIconAngleRight}}) {}
            }
        }

        // Hint text
        CLAY_TEXT(CLAY_STRING(",/. = speed  Space = pause  R = reverse  Tab = hide UI"),
                  CLAY_TEXT_CONFIG({.textColor = {90, 90, 120, 160}, .fontSize = 10}));
    }

    // ── Settings icon (bottom-right) ──────────────────────────────────────────
    const float kSettingsBtnSize = 36.0f;
    const float kSettingsBtnX = inp.screenW - kSettingsBtnSize - 12.0f;
    const float kSettingsBtnY = inp.screenH - kSettingsBtnSize - 12.0f;
    Clay_Color settingsBg = hovSettings ? Clay_Color{60, 60, 100, 230} : Clay_Color{20, 20, 40, 180};

    CLAY(CLAY_ID("SettingsBtn"), {.layout = {
                                      .sizing = {CLAY_SIZING_FIXED(kSettingsBtnSize), CLAY_SIZING_FIXED(kSettingsBtnSize)},
                                      .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                  .backgroundColor = settingsBg,
                                  .cornerRadius = CLAY_CORNER_RADIUS(6),
                                  .floating = {.offset = {kSettingsBtnX, kSettingsBtnY}, .zIndex = 5, .attachTo = CLAY_ATTACH_TO_ROOT}})
    {
        hovSettings = Clay_Hovered();
        if (hovSettings && inp.lmbPressed)
            settingsOpen = !settingsOpen;
        CLAY(CLAY_ID("SettingsIcon"), {.layout = {
                                           .sizing = {CLAY_SIZING_FIXED(22), CLAY_SIZING_FIXED(22)}},
                                       .image = {.imageData = (void *)(intptr_t)kIconSettings}}) {}
    }

    // ── Settings window ───────────────────────────────────────────────────────
    if (settingsOpen)
    {
        const float kWinW = 400.0f;
        const float kWinH = 440.0f;
        const float kWinX = (inp.screenW - kWinW) * 0.5f;
        const float kWinY = (inp.screenH - kWinH) * 0.5f;

        CLAY(CLAY_ID("SettingsWin"), {.layout = {
                                          .sizing = {CLAY_SIZING_FIXED(kWinW), CLAY_SIZING_FIXED(kWinH)},
                                          .padding = {0, 0, 0, 0},
                                          .childGap = 0,
                                          .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                      .backgroundColor = {12, 14, 28, 245},
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
                                               .backgroundColor = {18, 22, 45, 255},
                                               .cornerRadius = {8, 8, 0, 0}})
            {
                CLAY_TEXT(CLAY_STRING("Settings"),
                          CLAY_TEXT_CONFIG({.textColor = {200, 220, 255, 255}, .fontSize = 16}));

                // Spacer
                CLAY(CLAY_ID("SettingsTitleSpacer"), {.layout = {
                                                          .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)}}}) {}

                // Close button [X]
                Clay_Color closeBg = hovSettingsClose ? Clay_Color{180, 40, 40, 220} : Clay_Color{60, 30, 30, 180};
                CLAY(CLAY_ID("SettingsCloseBtn"), {.layout = {
                                                       .sizing = {CLAY_SIZING_FIXED(24), CLAY_SIZING_FIXED(24)},
                                                       .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                   .backgroundColor = closeBg,
                                                   .cornerRadius = CLAY_CORNER_RADIUS(4)})
                {
                    hovSettingsClose = Clay_Hovered();
                    if (hovSettingsClose && inp.lmbPressed)
                        settingsOpen = false;
                    CLAY_TEXT(CLAY_STRING("X"),
                              CLAY_TEXT_CONFIG({.textColor = {230, 200, 200, 255}, .fontSize = 12}));
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
                CLAY_TEXT(CLAY_STRING("Controls"),
                          CLAY_TEXT_CONFIG({.textColor = {150, 170, 210, 200}, .fontSize = 14}));

                // Thin separator
                CLAY(CLAY_ID("CtrlSep"), {.layout = {
                                              .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                              .padding = {0, 0, 4, 4}},
                                          .backgroundColor = {40, 50, 80, 120}}) {}

                // One row per keybinding
                static char kbKeyBuf[8][16];
                for (int ki = 0; ki < (int)keybindings.size() && ki < 8; ++ki)
                {
                    KeyBinding &kb = keybindings[ki];
                    snprintf(kbKeyBuf[ki], sizeof(kbKeyBuf[ki]), "[%s]", keyDisplayName(kb.key));

                    Clay_Color rowBg = kb.listening
                                           ? Clay_Color{60, 40, 10, 180}
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
                                      CLAY_TEXT_CONFIG({.textColor = {180, 200, 240, 220}, .fontSize = 13}));
                        }

                        // Current key label (fixed width)
                        CLAY(CLAY_IDI("KbKey", ki), {.layout = {
                                                         .sizing = {CLAY_SIZING_FIXED(60), CLAY_SIZING_FIT(0)}}})
                        {
                            Clay_String keyStr{false, (int32_t)strlen(kbKeyBuf[ki]), kbKeyBuf[ki]};
                            Clay_Color keyCol = kb.listening
                                                    ? Clay_Color{255, 180, 60, 255}
                                                    : Clay_Color{140, 160, 200, 200};
                            CLAY_TEXT(keyStr,
                                      CLAY_TEXT_CONFIG({.textColor = keyCol, .fontSize = 13}));
                        }

                        // Rebind button
                        Clay_Color rebindBg = kb.listening
                                                  ? Clay_Color{120, 80, 0, 220}
                                                  : (hovRebind[ki] ? Clay_Color{50, 60, 100, 220} : Clay_Color{28, 30, 55, 200});
                        CLAY(CLAY_IDI("KbRebind", ki), {.layout = {
                                                            .sizing = {CLAY_SIZING_FIXED(80), CLAY_SIZING_FIXED(20)},
                                                            .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                                        .backgroundColor = rebindBg,
                                                        .cornerRadius = CLAY_CORNER_RADIUS(3)})
                        {
                            hovRebind[ki] = Clay_Hovered();
                            if (hovRebind[ki] && inp.lmbPressed)
                            {
                                // Cancel any other listening binding
                                for (auto &other : keybindings)
                                    other.listening = false;
                                kb.listening = true;
                            }
                            CLAY_TEXT(kb.listening ? CLAY_STRING("PRESS KEY") : CLAY_STRING("Rebind"),
                                      CLAY_TEXT_CONFIG({.textColor = {210, 220, 255, 255}, .fontSize = 10}));
                        }
                    }
                }

                // Camera controls section
                CLAY(CLAY_ID("CamCtrlSep"), {.layout = {
                                                 .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1)},
                                                 .padding = {0, 0, 6, 4}},
                                             .backgroundColor = {40, 50, 80, 120}}) {}
                CLAY_TEXT(CLAY_STRING("Camera"),
                          CLAY_TEXT_CONFIG({.textColor = {150, 170, 210, 200}, .fontSize = 14}));
                CLAY_TEXT(CLAY_STRING("Right-click drag   Look around"),
                          CLAY_TEXT_CONFIG({.textColor = {110, 130, 160, 180}, .fontSize = 12}));
                CLAY_TEXT(CLAY_STRING("Scroll wheel        Zoom (FOV)"),
                          CLAY_TEXT_CONFIG({.textColor = {110, 130, 160, 180}, .fontSize = 12}));
            }
        }
    }

    // ── Mouse capture rects ───────────────────────────────────────────────────
    ui.addMouseCaptureRect(12, 12, 310, 155);                   // info panel
    ui.addMouseCaptureRect(inp.screenW - 195.0f, 12, 185, 200); // constellation panel
    ui.addMouseCaptureRect(12, inp.screenH - 110.0f, 310, 100); // time panel
    ui.addMouseCaptureRect(kSettingsBtnX, kSettingsBtnY,
                           kSettingsBtnSize, kSettingsBtnSize); // settings button
    if (settingsOpen)
        ui.addMouseCaptureRect((inp.screenW - 400.0f) * 0.5f,
                               (inp.screenH - 440.0f) * 0.5f,
                               400.0f, 440.0f); // settings window
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
        float starScale = 2.0f; // tweak this to make stars bigger/smaller overall
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
    satTypes = {
        {// 0 — Starlink: flat phased-array face toward Earth, brief intense flares
         "Starlink",
         {0.80f, 0.87f, 1.00f},                      // cool blue-white
         100.0f,                                     // ~10 m² bus + visor
         {AttitudeMode::NadirPointing, 18.0f, 1.0f}, // very sharp specular (flat mirror-like face)
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},  // no significant secondary surface
         0.0f},                                      // no diffuse floor
        {                                            // 1 — LEO broadband (OneWeb/Kuiper/Xingwang/Telesat): sun-tracking panels
         "LEO Broadband",
         {1.00f, 0.92f, 0.75f}, // warm white
         120.0f,                // ~12 m² typical LEO broadband bus + panels
         {AttitudeMode::SunTracking, 6.0f, 1.0f},
         {AttitudeMode::Perpendicular, 0.0f, 0.0f},
         0.0f},
        {// 2 — GEO Comsat: large sun-tracking panels + body radiators facing away from Earth
         "GEO Comsat",
         {0.95f, 0.95f, 1.00f},                    // near-white
         50.0f,                                    // ~50 m² (large GEO body + wings)
         {AttitudeMode::SunTracking, 3.0f, 1.00f}, // broad lobe solar wings
         {AttitudeMode::AntiNadir, 2.0f, 0.10f},   // body radiators face deep space
         0.01f},                                   // slight structural glow
        {                                          // 3 — ISS: enormous truss-mounted solar arrays AND large radiator panels.
         // The PVTCS and EATCS radiators (~900 m² NH3 panels on the ITS) face away from
         // Earth for maximum view factor to cold space. From the ground: ISS at zenith shows
         // the back of the radiators (dim); ISS near the horizon shows the radiator face.
         "ISS",
         {1.00f, 0.85f, 0.70f}, // warm golden (solar array color)
         250000.0f,
         {AttitudeMode::SunTracking, 8.0f, 1.00f}, // truss-mounted solar arrays
         {AttitudeMode::AntiNadir, 4.0f, 0.35f},   // large radiator panels (ITS) face deep space
         0.04f},                                   // complex truss/module body
        {                                          // 4 — SpaceX ODC (FCC filing Jan 2026): Starlink-class bus with large radiator panels
         // for compute heat rejection. Nadir-pointing phased array + deep-space-facing radiators.
         // Flat compute/antenna face produces brief nadir flares; radiator face brightens near
         // the horizon (AntiNadir face tilts toward the observer as satellite descends).
         "SpaceX ODC",
         {0.65f, 1.00f, 0.92f},                   // cyan-teal (distinct from Starlink blue-white)
         5000000.0f,                              // ~15 m² — Starlink-class bus + extra radiator area
         {AttitudeMode::SunTracking, 8.0f, 1.0f}, // phased-array/compute face toward Earth
         {AttitudeMode::AntiNadir, 3.0f, 0.25f},  // large radiator panels face deep space
         0.001f},                                 // slight structural body scatter
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
        {"Kuiper",
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

        // Telesat Lightspeed — planned: 1,671 sats; 27 planes × 62 = 1,674
        {"Telesat",
         1'015'000.0f,         // altM:      1,015 km
         glm::radians(98.98f), // incl:      98.98° — sun-synchronous Walker
         27,                   // numPlanes: orbital planes
         62,                   // perPlane:  sats per plane (27×62 = 1,674)
         1u,                   // typeIdx:   LEO Broadband (SunTracking)
         true,                 // enabled
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
        {"SpaceX ODC SSO",
         1'250'000.0f, // altM:      1,250 km — ring centre altitude
         0.0f,         // incl:      ignored (alignTerminator=true overrides)
         200,          // numPlanes: × perPlane = total sats (200×100 = 20,000)
         100,          // perPlane:  × numPlanes = total sats
         4u,           // typeIdx:   SpaceX ODC (NadirPointing + AntiNadir radiators)
         true,         // enabled
         OrbitDistribution::Disk,
         10000.0f,    // altJitterM:      no per-satellite altitude scatter
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
                    satOrbits.push_back({raan, c.incl, u0, c.typeIdx, c.altM, 0.0f, 0.0f, {0.0f, 0.0f, 1.0f}});
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
                                     tumbleRate, tumblePhase, axis});
            }
        }
        else if (c.distribution == OrbitDistribution::Disk)
        {
            // Determine orbital plane.
            float incl_d = c.incl;
            float raan_d = c.raan;
            if (c.alignTerminator)
            {
                // Terminator plane: normal = sunDirECI.
                // orbit_normal = (sin(i)sin(Ω), -sin(i)cos(Ω), cos(i)) = sunDirECI
                incl_d = acosf(glm::clamp(sunDirECI.z, -1.0f, 1.0f));
                float sI = sinf(incl_d);
                raan_d = (sI > 1e-5f)
                             ? atan2f(sunDirECI.x / sI, -sunDirECI.y / sI)
                             : 0.0f;
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

                // Model incomplete constellation, vary number of sats per ring to fill totalSats without exceeding it.
                int satsInThisRing = glm::min(perRing, totalSats - r * perRing);

                for (int s = 0; s < satsInThisRing; ++s)
                {
                    // Evenly spaced around the ring + optional small jitter.
                    float u0 = (float)s / satsInThisRing * glm::two_pi<float>();
                    float jitter = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * c.altJitterM;
                    satOrbits.push_back({raan_d, incl_d, u0, c.typeIdx, ringAlt + jitter, 0.0f, 0.0f, {0.0f, 0.0f, 1.0f}});
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
    float theta = (float)fmod(kOmegaEarth * t, glm::two_pi<double>());
    float cosLat = cosf(kObsLat), sinLat = sinf(kObsLat);
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
    double moonAngle = fmod(2.0 * glm::pi<double>() * t / kMoonPeriodSec,
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
            float cosR = cosf(orb.raan), sinR = sinf(orb.raan);

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
                    // Solar panels rotate around the nadir body axis to track the sun's azimuth.
                    // Panel normal = sun direction projected onto the satellite's local horizontal
                    // plane (removes nadir component) to avoid the retroreflector degeneracy.
                    glm::vec3 sunHoriz = sunDirECI - glm::dot(sunDirECI, satNadir) * satNadir;
                    float horizLen = glm::length(sunHoriz);
                    return (horizLen > 1e-5f) ? sunHoriz / horizLen : satNadir;
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

            if (el > -0.01f)
                ++visibleCount;
        }
    }
}
