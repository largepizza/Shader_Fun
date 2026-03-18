#include "SatelliteSim.h"
#include "../UIRenderer.h"
#include "clay.h"
#include "star_catalog.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>

// ── Earth + observer constants ─────────────────────────────────────────────────
static constexpr float kEarthRadius = 6'371'000.0f; // mean Earth radius (m)
static constexpr double kOmegaEarth = 7.2921150e-5; // sidereal rotation rate (rad/s)
static constexpr float kObsLat = 0.785398f;         // observer geodetic latitude (45°N)

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
    simTime += (double)dt * kTimeScales[timeScaleIdx];
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

// ─── buildUI ──────────────────────────────────────────────────────────────────
void SatelliteSim::buildUI(float dt, UIRenderer &ui)
{
    // Apply camera mouse look (uses previous frame's delta, reset after).
    if (win)
        camera.update(win, dmx, dmy);
    dmx = dmy = 0.0f;

    const UIInput &inp = ui.input();

    // ── Simulated UTC time string ─────────────────────────────────────────────
    static char timeBuf[32];
    {
        // simTime = seconds since J2000.0 (2000-01-01 12:00 UTC = Unix 946728000).
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

        // UTC date/time
        Clay_String timeStr{false, (int32_t)strlen(timeBuf), timeBuf};
        CLAY_TEXT(timeStr,
                  CLAY_TEXT_CONFIG({.textColor = {160, 200, 160, 220}, .fontSize = 13}));

        // Visible satellite count
        static char visBuf[48];
        snprintf(visBuf, sizeof(visBuf), "%u / %u satellites visible",
                 visibleCount, activeSatCount);
        Clay_String visStr{false, (int32_t)strlen(visBuf), visBuf};
        CLAY_TEXT(visStr,
                  CLAY_TEXT_CONFIG({.textColor = {140, 170, 220, 220}, .fontSize = 13}));

        // Sun elevation
        static char sunBuf[32];
        float sunElDeg = glm::degrees(asinf(glm::clamp(sunDirENU.w, -1.0f, 1.0f)));
        snprintf(sunBuf, sizeof(sunBuf), "Sun elev  %.1f°", sunElDeg);
        Clay_Color sunCol = sunElDeg > 0.0f
                                ? Clay_Color{255, 220, 100, 220}
                                : Clay_Color{100, 100, 140, 180};
        Clay_String sunStr{false, (int32_t)strlen(sunBuf), sunBuf};
        CLAY_TEXT(sunStr, CLAY_TEXT_CONFIG({.textColor = sunCol, .fontSize = 13}));

        // Camera state
        Clay_Color camCol = camera.captured
                                ? Clay_Color{100, 255, 120, 255}
                                : Clay_Color{160, 160, 190, 220};
        CLAY_TEXT(camera.captured
                      ? CLAY_STRING("Camera: CAPTURED  (RMB to release)")
                      : CLAY_STRING("Camera: free  (Right-click to capture)"),
                  CLAY_TEXT_CONFIG({.textColor = camCol, .fontSize = 13}));

        // FPS
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

        static char constNameBuf[8][32];
        static char constCntBuf[8][16];

        for (int ci = 0; ci < (int)constellations.size() && ci < 8; ++ci)
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
                // Toggle button
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

                // Name (fixed-width container so row doesn't jitter on toggle)
                CLAY(CLAY_IDI("ConstName", ci), {.layout = {
                                                     .sizing = {CLAY_SIZING_FIXED(100), CLAY_SIZING_FIT(0)}}})
                {
                    Clay_String nameStr{false, (int32_t)strlen(c.name), c.name};
                    CLAY_TEXT(nameStr,
                              CLAY_TEXT_CONFIG({.textColor = {180, 200, 240, 220}, .fontSize = 13}));
                }

                // Orbit count
                Clay_String cntStr{false, (int32_t)strlen(constCntBuf[ci]), constCntBuf[ci]};
                CLAY_TEXT(cntStr,
                          CLAY_TEXT_CONFIG({.textColor = {110, 130, 170, 180}, .fontSize = 11}));
            }
        }
    }

    // ── Time controls (bottom-left) ───────────────────────────────────────────
    CLAY(CLAY_ID("TimePanel"), {.layout = {
                                    .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                    .padding = {10, 10, 8, 8},
                                    .childGap = 6,
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                .backgroundColor = {8, 10, 20, 210},
                                .cornerRadius = CLAY_CORNER_RADIUS(6),
                                .floating = {.offset = {12, inp.screenH - 100.0f}, .zIndex = 5, .attachTo = CLAY_ATTACH_TO_ROOT}})
    {
        CLAY_TEXT(CLAY_STRING("Time scale  (T = cycle)"),
                  CLAY_TEXT_CONFIG({.textColor = {160, 160, 200, 200}, .fontSize = 12}));

        // Speed buttons row
        CLAY(CLAY_ID("SpeedRow"), {.layout = {
                                       .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                       .childGap = 4,
                                       .layoutDirection = CLAY_LEFT_TO_RIGHT}})
        {
            for (int i = 0; i < kNumTimeScales; ++i)
            {
                bool active = (i == timeScaleIdx);
                Clay_Color bg = active
                                    ? Clay_Color{40, 100, 200, 240}
                                    : (hovSpeed[i] ? Clay_Color{48, 48, 75, 230} : Clay_Color{28, 28, 48, 210});
                CLAY(CLAY_IDI("SpeedBtn", i), {.layout = {
                                                   .sizing = {CLAY_SIZING_FIXED(36), CLAY_SIZING_FIXED(22)},
                                                   .childAlignment = {.x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER}},
                                               .backgroundColor = bg,
                                               .cornerRadius = CLAY_CORNER_RADIUS(3)})
                {
                    hovSpeed[i] = Clay_Hovered();
                    if (hovSpeed[i] && inp.lmbPressed)
                        timeScaleIdx = i;
                    Clay_String lbl{false, (int32_t)strlen(kTimeLabels[i]), kTimeLabels[i]};
                    CLAY_TEXT(lbl,
                              CLAY_TEXT_CONFIG({.textColor = {210, 220, 255, 255}, .fontSize = 12}));
                }
            }
        }
    }

    // Mouse capture rects for all panels
    ui.addMouseCaptureRect(12, 12, 310, 140);                   // info panel
    ui.addMouseCaptureRect(inp.screenW - 195.0f, 12, 185, 150); // constellation panel
    ui.addMouseCaptureRect(12, inp.screenH - 100.0f, 260, 90);  // time panel
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
    if (key == GLFW_KEY_T)
    {
        timeScaleIdx = (timeScaleIdx + 1) % kNumTimeScales;
    }
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
    satTypes = {
        {// 0 — Starlink: flat phased-array face toward Earth, brief intense flares
         "Starlink",
         {0.80f, 0.87f, 1.00f}, // cool blue-white
         18.0f,                 // very sharp specular (flat mirror-like face)
         AttitudeMode::NadirPointing,
         10.0f}, // ~10 m² bus + visor
        {        // 1 — OneWeb/LEO: sun-tracking panels, gentler opposition flares
         "OneWeb",
         {1.00f, 0.92f, 0.75f}, // warm white
         6.0f,
         AttitudeMode::SunTracking,
         12.0f}, // slightly larger panels
        {        // 2 — GEO Comsat: large sun-tracking panels, very broad soft flares
         "GEO Comsat",
         {0.95f, 0.95f, 1.00f}, // near-white
         3.0f,                  // very soft specular (broad panels)
         AttitudeMode::SunTracking,
         50.0f}, // ~50 m² (large GEO body + wings)
        {        // 3 — ISS: enormous solar arrays (~2 500 m²)
         "ISS",
         {1.00f, 0.85f, 0.70f}, // warm golden (solar array color)
         8.0f,
         AttitudeMode::SunTracking,
         2500.0f},
        {// 4 — Space Junk: uncontrolled tumbling debris — chaotic random flashes
         "Space Junk",
         {0.65f, 0.62f, 0.60f}, // dull metallic gray
         6.0f,
         AttitudeMode::Tumbling,
         1.0f}, // ~1 m² average debris fragment
        {       // 5 — Space Datacenter: massive nadir-facing solar arrays on the terminator.
         // NadirPointing panels at the terminator reflect anti-sunward; a nightside
         // observer at ~45° phase gets alignment≈0.4, specExp=3 → specular≈0.06.
         // The huge cross-section (50 000 m²) compensates, yielding a visible ring.
         "Space Datacenter",
         {0.85f, 0.92f, 1.00f}, // cool blue-white
         3.0f,                  // broad lobe — visible over a wide range of phase angles
         AttitudeMode::SunTracking,
         50000.0f}, // ~50 000 m² (200 m × 250 m solar array farm)
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
    constellations = {
        {"Starlink G1", 550'000.0f, glm::radians(53.0f), 72 * 10, 22, 0u, true, OrbitDistribution::Walker},
        {"OneWeb", 1'200'000.0f, glm::radians(87.9f), 18, 36, 1u, true, OrbitDistribution::Walker},
        {"GEO Belt", 35'786'000.0f, glm::radians(0.0f), 1, 50, 2u, true, OrbitDistribution::Walker},
        {"ISS", 408'000.0f, glm::radians(51.6f), 1, 1, 3u, true, OrbitDistribution::Walker},
        // Space junk: randomly distributed at LEO altitudes, full inclination range, tumbling.
        {"Space Junk LEO", 450'000.0f, glm::pi<float>(), 500, 1, 4u, true, OrbitDistribution::RandomShell, 150'000.0f},
        // Space Datacenters: single ring on the Earth-Sun terminator, isotropic glow.
        // Increase numPlanes or add more rings (.numRings) for a denser disk.
        {"Space Datacenter", 500'000.0f, 0.0f, 3000, 1, 5u, true, OrbitDistribution::Disk,
         100'000.0f, 0.0f, true}, // altJitterM=0, raan=0(ignored), alignTerminator=true
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

                float tumbleRate = 0.08f * (float)rand() / (float)RAND_MAX;
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

                for (int s = 0; s < perRing; ++s)
                {
                    // Evenly spaced around the ring + optional small jitter.
                    float u0 = (float)s / perRing * glm::two_pi<float>();
                    float jitter = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * c.altJitterM;
                    satOrbits.push_back({raan_d, incl_d, u0, c.typeIdx, ringAlt + jitter, 0.0f, 0.0f, {0.0f, 0.0f, 1.0f}});
                }
            }
        }

        c.orbitCount = (uint32_t)satOrbits.size() - c.orbitStart;
    }

    activeSatCount = (uint32_t)satOrbits.size();
    satInputData.resize(activeSatCount);
}

// ─── updatePositions ──────────────────────────────────────────────────────────
// Recomputes: observer ECI position + ECI→ENU matrix, sun direction,
// and per-satellite geometry + panel attitude (nested per-constellation).
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

            glm::vec3 panelNormal;
            if (type.attitude == AttitudeMode::NadirPointing)
            {
                panelNormal = satNadir;
            }
            else if (type.attitude == AttitudeMode::Tumbling)
            {
                // Spin around a body-fixed random axis; angle advances with simTime.
                // fmod before float cast: simTime ≈ 8e8 s gives only ~96 s float precision,
                // causing (float)t * rate to jump by ~96 rad between frames → staggering.
                float angle = orb.tumblePhase +
                              (float)fmod((double)orb.tumbleRate * t, glm::two_pi<double>());
                glm::vec3 ax = orb.tumbleAxis;
                glm::vec3 ref = (fabsf(ax.z) < 0.9f) ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
                glm::vec3 axA = glm::normalize(glm::cross(ax, ref));
                glm::vec3 axB = glm::cross(ax, axA);                 // already unit length
                panelNormal = cosf(angle) * axA + sinf(angle) * axB; // already unit length
            }
            else // SunTracking
            {
                // Solar panels rotate around the nadir body axis to track the sun's azimuth.
                // Panel normal = sun direction projected onto the satellite's local horizontal
                // plane (removes nadir component) to avoid the retroreflector degeneracy.
                glm::vec3 sunHoriz = sunDirECI - glm::dot(sunDirECI, satNadir) * satNadir;
                float horizLen = glm::length(sunHoriz);
                panelNormal = (horizLen > 1e-5f)
                                  ? sunHoriz / horizLen
                                  : satNadir;
            }

            satInputData[i].eciRelPos = relPos;
            satInputData[i].range = range;
            satInputData[i].panelNormal = panelNormal;
            satInputData[i].elevation = el;
            satInputData[i].baseColor = type.baseColor;
            satInputData[i].specExp = type.specExp;
            satInputData[i].crossSection = sqrtf(type.crossSectionM2 / 10.0f);

            if (el > -0.01f)
                ++visibleCount;
        }
    }
}
