#include "Particles.h"
#include "../UIRenderer.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <cstdio>

// clay.h is a single-header library; CLAY_IMPLEMENTATION is defined only in UIRenderer.cpp.
// Include here without the implementation define so we can use the CLAY macros.
#include "clay.h"

// ─── Layout constants — must match Clay sizing declarations exactly ────────────
static constexpr float TOOLBAR_H  = 44.0f;  // bottom toolbar height
static constexpr float BTN_W      = 92.0f;  // tool button width
static constexpr float BTN_H      = 30.0f;  // tool button height
static constexpr float SETTINGS_W = 268.0f; // settings window width
static constexpr float W_TITLE_H  = 30.0f;  // title bar height
static constexpr float W_PAD      = 10.0f;  // window inner padding
static constexpr float W_LABEL_W  = 80.0f;  // label column width
static constexpr float W_CHILD_GAP = 4.0f;  // label-to-control gap
static constexpr float SLIDER_W   = SETTINGS_W - 2*W_PAD - W_LABEL_W - W_CHILD_GAP;
static constexpr float SLIDER_H   = 14.0f;
static constexpr float W_ROW_H    = 22.0f;  // content row height
static constexpr float W_ROW_GAP  = 6.0f;   // gap between rows

// ─── Colour palette ────────────────────────────────────────────────────────────
static constexpr Clay_Color COL_WIN_BG   = { 14, 14, 24, 230 };
static constexpr Clay_Color COL_TITLE_BG = { 22, 22, 38, 255 };
static constexpr Clay_Color COL_BTN      = { 28, 28, 48, 210 };
static constexpr Clay_Color COL_BTN_HOV  = { 48, 48, 75, 230 };
static constexpr Clay_Color COL_ACTIVE   = { 40, 100, 200, 240 };
static constexpr Clay_Color COL_ACT_HOV  = { 60, 130, 230, 255 };
static constexpr Clay_Color COL_TOOLBAR  = { 12, 12, 22, 235 };
static constexpr Clay_Color COL_TRACK    = { 35, 35, 58, 255 };
static constexpr Clay_Color COL_FILL     = { 70, 120, 210, 255 };
static constexpr Clay_Color COL_TEXT_HI  = { 210, 220, 255, 255 };
static constexpr Clay_Color COL_TEXT_DIM = { 140, 140, 165, 255 };
static constexpr Clay_Color COL_CLOSE    = { 180, 50,  50, 220 };
static constexpr Clay_Color COL_CLOSE_H  = { 220, 70,  70, 255 };
static constexpr Clay_Color COL_RESET    = { 55,  40,  100, 220 };
static constexpr Clay_Color COL_RESET_H  = { 80,  60,  150, 255 };

// ─── Helper: pick between active/hovered/normal button colour ─────────────────
static inline Clay_Color btnBg(bool active, bool hov) {
    if (active) return hov ? COL_ACT_HOV : COL_ACTIVE;
    return hov ? COL_BTN_HOV : COL_BTN;
}

// ─── init ─────────────────────────────────────────────────────────────────────
void Particles::init(VulkanContext& ctx) {
    createParticleBuffer(ctx);
    createDescriptors(ctx);
    createComputePipeline(ctx);
    createDrawPipeline(ctx);
    initParticles(ctx);
}

// ─── onResize ─────────────────────────────────────────────────────────────────
void Particles::onResize(VulkanContext& ctx) {
    vkDestroyPipeline(ctx.device, drawPipeline, nullptr);
    drawPipeline = VK_NULL_HANDLE;
    createDrawPipeline(ctx);
}

// ─── recordCompute ────────────────────────────────────────────────────────────
void Particles::recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float dt) {
    // Handle pending reset (synchronous — safe before adding work to cmd)
    if (pendingReset) {
        pendingReset = false;
        initParticles(ctx);
    }

    totalTime += dt;

    ParticlePushConstants pc{};
    pc.dt            = dt;
    pc.time          = totalTime;
    pc.mouseNDC      = mouseNDC;
    pc.forceStrength = forceStrength;
    pc.viscosity     = viscosity;
    pc.colorMode     = colorMode;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        compPipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(pc), &pc);

    uint32_t groups = (PARTICLE_COUNT + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute write → vertex shader SSBO read.
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer              = buffer;
    bmb.offset              = 0;
    bmb.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
}

// ─── recordDraw ───────────────────────────────────────────────────────────────
void Particles::recordDraw(VkCommandBuffer cmd, VulkanContext& ctx, float /*dt*/) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        drawPipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdDraw(cmd, PARTICLE_COUNT, 1, 0, 0);
}

// ─── buildUI ──────────────────────────────────────────────────────────────────
// Field ordering in Clay_LayoutConfig:     sizing, padding, childGap, childAlignment, layoutDirection
// Field ordering in Clay_ElementDeclaration: layout, backgroundColor, cornerRadius, ..., floating
// Field ordering in Clay_FloatingElementConfig: offset, zIndex, pointerCaptureMode, attachTo
void Particles::buildUI(float /*dt*/, UIRenderer& ui) {
    const UIInput& inp = ui.input();
    bool overUI = ui.mouseOverUI(); // previous frame — correctly gates scene input

    // ── Compute force for this frame ─────────────────────────────────────────
    forceStrength = 0.0f;
    if (!overUI) {
        int dir = 0;
        if (inp.lmbDown) dir = (activeTool == ParticleTool::Attract) ?  1 : -1;
        if (inp.rmbDown) dir = (activeTool == ParticleTool::Attract) ? -1 :  1;
        forceStrength = (float)dir;
    }

    // ── Settings window drag ──────────────────────────────────────────────────
    if (settingsWin.dragging) {
        settingsWin.x += inp.dMouseX;
        settingsWin.y += inp.dMouseY;
        settingsWin.x = std::clamp(settingsWin.x, 0.0f, inp.screenW - SETTINGS_W);
        settingsWin.y = std::clamp(settingsWin.y, 0.0f, inp.screenH - W_TITLE_H);
        if (!inp.lmbDown) settingsWin.dragging = false;
    }

    // ── Friction slider interaction (position computed from known layout constants) ─
    if (settingsWin.open) {
        float trackX = settingsWin.x + W_PAD + W_LABEL_W + W_CHILD_GAP;
        float trackY = settingsWin.y + W_TITLE_H + W_PAD + (W_ROW_H - SLIDER_H) * 0.5f;
        bool overTrack = inp.mouseX >= trackX && inp.mouseX < trackX + SLIDER_W
                      && inp.mouseY >= trackY && inp.mouseY < trackY + SLIDER_H;
        if (inp.lmbPressed && overTrack) visSliderDrag = true;
        if (!inp.lmbDown) visSliderDrag = false;
        if (visSliderDrag) {
            float t  = std::clamp((inp.mouseX - trackX) / SLIDER_W, 0.0f, 1.0f);
            viscosity = 0.999f - t * 0.099f; // left=no friction, right=max friction
        }
    }

    // ── Register mouse-capture rects ─────────────────────────────────────────
    ui.addMouseCaptureRect(0, inp.screenH - TOOLBAR_H, inp.screenW, TOOLBAR_H);
    if (settingsWin.open)
        ui.addMouseCaptureRect(settingsWin.x, settingsWin.y, SETTINGS_W,
                               W_TITLE_H + W_PAD*2 + W_ROW_H*3 + W_ROW_GAP*2 + 36.0f);

    // ── Clay layout ──────────────────────────────────────────────────────────
    CLAY(CLAY_ID("PRoot"), {
        .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) } }
    }) {

        // ── Bottom toolbar (floating) ─────────────────────────────────────
        CLAY(CLAY_ID("Toolbar"), {
            .layout = {
                .sizing         = { CLAY_SIZING_FIXED(inp.screenW), CLAY_SIZING_FIXED(TOOLBAR_H) },
                .padding        = CLAY_PADDING_ALL(7),
                .childGap       = 6,
                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER },
                .layoutDirection = CLAY_LEFT_TO_RIGHT
            },
            .backgroundColor = COL_TOOLBAR,
            .floating = {
                .offset = { 0, inp.screenH - TOOLBAR_H },
                .zIndex = 5,
                .attachTo = CLAY_ATTACH_TO_ROOT
            }
        }) {
            // Attract tool button
            CLAY(CLAY_ID("BtnAttract"), {
                .layout = {
                    .sizing         = { CLAY_SIZING_FIXED(BTN_W), CLAY_SIZING_FIXED(BTN_H) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                },
                .backgroundColor = btnBg(activeTool == ParticleTool::Attract, hovAttract),
                .cornerRadius    = CLAY_CORNER_RADIUS(4)
            }) {
                hovAttract = Clay_Hovered();
                if (hovAttract && inp.lmbPressed) activeTool = ParticleTool::Attract;
                CLAY_TEXT(CLAY_STRING(">> Attract"),
                    CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_HI, .fontSize = 13 }));
            }

            // Repulse tool button
            CLAY(CLAY_ID("BtnRepulse"), {
                .layout = {
                    .sizing         = { CLAY_SIZING_FIXED(BTN_W), CLAY_SIZING_FIXED(BTN_H) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                },
                .backgroundColor = btnBg(activeTool == ParticleTool::Repulse, hovRepulse),
                .cornerRadius    = CLAY_CORNER_RADIUS(4)
            }) {
                hovRepulse = Clay_Hovered();
                if (hovRepulse && inp.lmbPressed) activeTool = ParticleTool::Repulse;
                CLAY_TEXT(CLAY_STRING("<< Repulse"),
                    CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_HI, .fontSize = 13 }));
            }

            CLAY(CLAY_ID("TBSep"), {
                .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1) } }
            }) {}

            CLAY_TEXT(CLAY_STRING("LMB = tool    RMB = inverse"),
                CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_DIM, .fontSize = 12 }));

            CLAY(CLAY_ID("TBSep2"), {
                .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1) } }
            }) {}

            // Settings button
            CLAY(CLAY_ID("BtnSettings"), {
                .layout = {
                    .sizing         = { CLAY_SIZING_FIXED(BTN_W), CLAY_SIZING_FIXED(BTN_H) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                },
                .backgroundColor = btnBg(settingsWin.open, hovSettings),
                .cornerRadius    = CLAY_CORNER_RADIUS(4)
            }) {
                hovSettings = Clay_Hovered();
                if (hovSettings && inp.lmbPressed) settingsWin.open = !settingsWin.open;
                CLAY_TEXT(CLAY_STRING("Settings"),
                    CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_HI, .fontSize = 13 }));
            }
        } // end Toolbar

        // ── Settings window (floating) ────────────────────────────────────
        if (settingsWin.open) {
            CLAY(CLAY_ID("SettingsWin"), {
                .layout = {
                    .sizing          = { CLAY_SIZING_FIXED(SETTINGS_W), CLAY_SIZING_FIT(0) },
                    .layoutDirection = CLAY_TOP_TO_BOTTOM
                },
                .backgroundColor = COL_WIN_BG,
                .cornerRadius    = CLAY_CORNER_RADIUS(6),
                .floating = {
                    .offset             = { settingsWin.x, settingsWin.y },
                    .zIndex             = 10,
                    .pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_CAPTURE,
                    .attachTo           = CLAY_ATTACH_TO_ROOT
                }
            }) {

                // Title bar (drag handle)
                CLAY(CLAY_ID("SettingsTitle"), {
                    .layout = {
                        .sizing         = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(W_TITLE_H) },
                        .padding        = { 10, 6, 0, 0 },
                        .childAlignment = { .y = CLAY_ALIGN_Y_CENTER },
                        .layoutDirection = CLAY_LEFT_TO_RIGHT
                    },
                    .backgroundColor = COL_TITLE_BG,
                    .cornerRadius    = { 6, 6, 0, 0 }
                }) {
                    if (Clay_Hovered() && inp.lmbPressed && !settingsWin.dragging)
                        settingsWin.dragging = true;

                    CLAY_TEXT(CLAY_STRING("Particle Settings"),
                        CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_HI, .fontSize = 14 }));

                    CLAY(CLAY_ID("TitleSpacer"), {
                        .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1) } }
                    }) {}

                    // Close button
                    CLAY(CLAY_ID("BtnClose"), {
                        .layout = {
                            .sizing         = { CLAY_SIZING_FIXED(24), CLAY_SIZING_FIXED(24) },
                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                        },
                        .backgroundColor = hovClose ? COL_CLOSE_H : COL_CLOSE,
                        .cornerRadius    = CLAY_CORNER_RADIUS(4)
                    }) {
                        hovClose = Clay_Hovered();
                        if (hovClose && inp.lmbPressed) settingsWin.open = false;
                        CLAY_TEXT(CLAY_STRING("X"),
                            CLAY_TEXT_CONFIG({ .textColor = {255,255,255,255}, .fontSize = 13 }));
                    }
                } // end title bar

                // Content area
                CLAY(CLAY_ID("SettingsContent"), {
                    .layout = {
                        .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                        .padding         = CLAY_PADDING_ALL((uint16_t)W_PAD),
                        .childGap        = (uint16_t)W_ROW_GAP,
                        .layoutDirection = CLAY_TOP_TO_BOTTOM
                    }
                }) {

                    // ── Friction slider row ────────────────────────────────
                    CLAY(CLAY_ID("ViscRow"), {
                        .layout = {
                            .sizing         = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(W_ROW_H) },
                            .childGap       = (uint16_t)W_CHILD_GAP,
                            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER },
                            .layoutDirection = CLAY_LEFT_TO_RIGHT
                        }
                    }) {
                        CLAY(CLAY_ID("ViscLabel"), {
                            .layout = {
                                .sizing         = { CLAY_SIZING_FIXED(W_LABEL_W), CLAY_SIZING_GROW(0) },
                                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
                            }
                        }) {
                            CLAY_TEXT(CLAY_STRING("Friction"),
                                CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_DIM, .fontSize = 13 }));
                        }

                        CLAY(CLAY_ID("ViscTrack"), {
                            .layout = {
                                .sizing = { CLAY_SIZING_FIXED(SLIDER_W), CLAY_SIZING_FIXED(SLIDER_H) }
                            },
                            .backgroundColor = COL_TRACK,
                            .cornerRadius    = CLAY_CORNER_RADIUS(3)
                        }) {
                            float t = (0.999f - viscosity) / 0.099f; // full bar = max friction
                            CLAY(CLAY_ID("ViscFill"), {
                                .layout = {
                                    .sizing = { CLAY_SIZING_FIXED(SLIDER_W * t), CLAY_SIZING_GROW(0) }
                                },
                                .backgroundColor = COL_FILL,
                                .cornerRadius    = CLAY_CORNER_RADIUS(3)
                            }) {}
                        }

                        // Value readout — viscBuf is a member so pointer stays valid in Clay_EndLayout
                        int pct = (int)((0.999f - viscosity) / 0.099f * 100.0f + 0.5f);
                        snprintf(viscBuf, sizeof(viscBuf), "%d%%", pct);
                        Clay_String viscStr{ false, (int32_t)strlen(viscBuf), viscBuf };
                        CLAY_TEXT(viscStr,
                            CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_DIM, .fontSize = 12 }));
                    }

                    // ── Color mode row ─────────────────────────────────────
                    CLAY(CLAY_ID("ColorRow"), {
                        .layout = {
                            .sizing         = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(W_ROW_H) },
                            .childGap       = (uint16_t)W_CHILD_GAP,
                            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER },
                            .layoutDirection = CLAY_LEFT_TO_RIGHT
                        }
                    }) {
                        CLAY_TEXT(CLAY_STRING("Color"),
                            CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_DIM, .fontSize = 13 }));

                        const Clay_String colorLabels[3] = {
                            CLAY_STRING("Rainbow"),
                            CLAY_STRING("Velocity"),
                            CLAY_STRING("Uniform")
                        };
                        for (int i = 0; i < 3; ++i) {
                            CLAY(CLAY_IDI("ColBtn", i), {
                                .layout = {
                                    .sizing         = { CLAY_SIZING_FIXED(60.0f), CLAY_SIZING_FIXED(18.0f) },
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                                },
                                .backgroundColor = btnBg(colorMode == i, hovColor[i]),
                                .cornerRadius    = CLAY_CORNER_RADIUS(3)
                            }) {
                                hovColor[i] = Clay_Hovered();
                                if (hovColor[i] && inp.lmbPressed) colorMode = i;
                                CLAY_TEXT(colorLabels[i],
                                    CLAY_TEXT_CONFIG({ .textColor = COL_TEXT_HI, .fontSize = 11 }));
                            }
                        }
                    }

                    // ── Reset row ──────────────────────────────────────────
                    CLAY(CLAY_ID("ResetRow"), {
                        .layout = {
                            .sizing         = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(30.0f) },
                            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER },
                            .layoutDirection = CLAY_LEFT_TO_RIGHT
                        }
                    }) {
                        CLAY(CLAY_ID("ResetSpacer"), {
                            .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1) } }
                        }) {}

                        CLAY(CLAY_ID("BtnReset"), {
                            .layout = {
                                .sizing         = { CLAY_SIZING_FIXED(100.0f), CLAY_SIZING_FIXED(26.0f) },
                                .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                            },
                            .backgroundColor = hovReset ? COL_RESET_H : COL_RESET,
                            .cornerRadius    = CLAY_CORNER_RADIUS(4)
                        }) {
                            hovReset = Clay_Hovered();
                            if (hovReset && inp.lmbPressed) pendingReset = true;
                            CLAY_TEXT(CLAY_STRING("Reset Particles"),
                                CLAY_TEXT_CONFIG({ .textColor = {210, 200, 255, 255}, .fontSize = 12 }));
                        }
                    }

                } // end content
            } // end SettingsWin
        } // end if settingsWin.open

    } // end PRoot
}

// ─── cleanup ──────────────────────────────────────────────────────────────────
void Particles::cleanup(VkDevice device) {
    vkDestroyPipeline(device, compPipeline, nullptr);
    vkDestroyPipeline(device, drawPipeline, nullptr);
    vkDestroyPipelineLayout(device, compPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, drawPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
}

// ─── onCursorPos ──────────────────────────────────────────────────────────────
void Particles::onCursorPos(GLFWwindow* window, double x, double y) {
    int ww, wh;
    glfwGetWindowSize(window, &ww, &wh);
    mouseNDC = {
        (float)(x / ww) * 2.0f - 1.0f,
        (float)(y / wh) * 2.0f - 1.0f
    };
}

// ─── createParticleBuffer ─────────────────────────────────────────────────────
void Particles::createParticleBuffer(VulkanContext& ctx) {
    VkDeviceSize size = sizeof(Particle) * PARTICLE_COUNT;
    ctx.createBuffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        buffer, memory);
}

// ─── createDescriptors ────────────────────────────────────────────────────────
void Particles::createDescriptors(VulkanContext& ctx) {
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 1; li.pBindings = &b;
    vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &layout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &layout;
    vkAllocateDescriptorSets(ctx.device, &ai, &descSet);

    VkDescriptorBufferInfo bi2{buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = descSet;
    w.dstBinding      = 0;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &bi2;
    vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
}

// ─── createComputePipeline ────────────────────────────────────────────────────
void Particles::createComputePipeline(VulkanContext& ctx) {
    auto mod = ctx.loadShader("shaders/particles_update.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(ParticlePushConstants);

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount         = 1;
    li.pSetLayouts            = &layout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges    = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &compPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = compPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &compPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create particle compute pipeline.");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createDrawPipeline ───────────────────────────────────────────────────────
void Particles::createDrawPipeline(VulkanContext& ctx) {
    auto vert = ctx.loadShader("shaders/particles_draw.vert.spv");
    auto frag = ctx.loadShader("shaders/particles_draw.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    // No vertex attributes: vertex shader reads from SSBO via gl_VertexIndex

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D sc{{0, 0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp;
    vps.scissorCount  = 1; vps.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Additive blending for glowing particles
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable         = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorBlendOp        = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp        = VK_BLEND_OP_ADD;
    cba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    if (drawPipeLayout == VK_NULL_HANDLE) {
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts    = &layout;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &drawPipeLayout);
    }

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState   = &ms;
    ci.pColorBlendState    = &cb;
    ci.layout              = drawPipeLayout;
    ci.renderPass          = ctx.renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &drawPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create particle draw pipeline.");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── initParticles ────────────────────────────────────────────────────────────
void Particles::initParticles(VulkanContext& ctx) {
    VkDeviceSize size = sizeof(Particle) * PARTICLE_COUNT;

    VkBuffer staging; VkDeviceMemory stagingMem;
    ctx.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem);

    void* data; vkMapMemory(ctx.device, stagingMem, 0, size, 0, &data);
    auto* particles = reinterpret_cast<Particle*>(data);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posD(-1.0f, 1.0f);
    std::uniform_real_distribution<float> velD(-0.02f, 0.02f);

    for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
        particles[i].pos   = {posD(rng), posD(rng)};
        particles[i].vel   = {velD(rng), velD(rng)};
        particles[i].color = {1.0f, 1.0f, 1.0f, 1.0f}; // compute shader overwrites every frame
    }
    vkUnmapMemory(ctx.device, stagingMem);

    auto cmd = ctx.beginOneTimeCommands();
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, staging, buffer, 1, &copy);
    ctx.endOneTimeCommands(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    totalTime = 0.0f;
}
