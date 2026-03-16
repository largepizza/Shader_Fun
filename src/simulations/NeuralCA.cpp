#include "NeuralCA.h"
#include "../UIRenderer.h"

#include <random>
#include <stdexcept>
#include <cstdio>
#include <cstring>

#include "clay.h"

// ─── init ─────────────────────────────────────────────────────────────────────
void NeuralCA::init(VulkanContext& ctx) {
    createImages(ctx);
    createSampler(ctx);
    createPongBuffer(ctx);     // must come before createDescriptors
    createDescriptors(ctx);
    createPongPipeline(ctx);
    createComputePipeline(ctx);
    createDisplayPipeline(ctx);
    resetGrid(ctx);
}

// ─── onResize ─────────────────────────────────────────────────────────────────
void NeuralCA::onResize(VulkanContext& ctx) {
    vkDestroyPipeline(ctx.device, dispPipeline, nullptr);
    dispPipeline = VK_NULL_HANDLE;
    createDisplayPipeline(ctx);
}

// ─── recordCompute ────────────────────────────────────────────────────────────
void NeuralCA::recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float dt) {
    if (pendingReset) {
        pendingReset = false;
        resetGrid(ctx);
        *pongMapped = PongState{};
    }

    totalTime += dt;

    // ── Pong step: once per visual frame ─────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pongPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pongPipeLayout, 0, 1, &pongSet[current], 0, nullptr);

    PongPushConstants ppc{};
    ppc.dt          = dt;
    ppc.paddleH     = 0.10f;
    ppc.paddleSpeed = 1.5f;
    ppc.ballSpeed   = 0.35f;
    ppc.gridW       = (int32_t)GRID_W;
    ppc.gridH       = (int32_t)GRID_H;
    vkCmdPushConstants(cmd, pongPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(ppc), &ppc);
    vkCmdDispatch(cmd, 1, 1, 1);

    // Barrier: pong SSBO write → neural compute read + fragment read
    VkBufferMemoryBarrier ssboBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    ssboBarrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    ssboBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    ssboBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ssboBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ssboBarrier.buffer              = pongBuf;
    ssboBarrier.offset              = 0;
    ssboBarrier.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 1, &ssboBarrier, 0, nullptr);

    // ── Neural update: stepsPerFrame substeps ─────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);

    for (int s = 0; s < stepsPerFrame; ++s) {
        step++;

        NeuralPushConstants pc{};
        pc.time       = totalTime;
        pc.learnRate  = learnRate;
        pc.traceDecay = traceDecay;
        pc.gridW      = (int32_t)GRID_W;
        pc.gridH      = (int32_t)GRID_H;
        pc.step       = step;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            compPipeLayout, 0, 1, &compSet[current], 0, nullptr);
        vkCmdPushConstants(cmd, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (GRID_W + 15) / 16, (GRID_H + 15) / 16, 1);

        bool lastStep = (s == stepsPerFrame - 1);
        ctx.imageBarrier(cmd, image[1 - current],
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            lastStep ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                     : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        current = 1 - current;
    }
}

// ─── recordDraw ───────────────────────────────────────────────────────────────
void NeuralCA::recordDraw(VkCommandBuffer cmd, VulkanContext& /*ctx*/, float /*dt*/) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dispPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        dispPipeLayout, 0, 1, &dispSet[current], 0, nullptr);

    NeuralDisplayPC dpc{};
    dpc.gridW    = (int32_t)GRID_W;
    dpc.gridH    = (int32_t)GRID_H;
    dpc.viewMode = viewMode;
    vkCmdPushConstants(cmd, dispPipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(dpc), &dpc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

// ─── buildUI ──────────────────────────────────────────────────────────────────
void NeuralCA::buildUI(float /*dt*/, UIRenderer& ui) {
    const UIInput& inp = ui.input();
    static const char* viewLabels[3] = { "Activity", "Trace", "Weight" };
    static char stepBuf[48];
    static char scoreBuf[48];

    snprintf(stepBuf, sizeof(stepBuf), "Step %d  (%dx/frame)", step, stepsPerFrame);
    if (pongMapped)
        snprintf(scoreBuf, sizeof(scoreBuf), "AI %d   OPP %d",
            pongMapped->ai_score, pongMapped->opp_score);

    CLAY(CLAY_ID("NCARoot"), {
        .layout = {
            .sizing  = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            .padding = CLAY_PADDING_ALL(12)
        }
    }) {
        CLAY(CLAY_ID("NCAPanel"), {
            .layout = {
                .sizing          = { CLAY_SIZING_FIXED(230), CLAY_SIZING_FIT(0) },
                .padding         = CLAY_PADDING_ALL(10),
                .childGap        = 6,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .backgroundColor = { 10, 14, 24, 215 },
            .cornerRadius    = CLAY_CORNER_RADIUS(6)
        }) {
            CLAY_TEXT(CLAY_STRING("Neural CA Pong"),
                CLAY_TEXT_CONFIG({ .textColor = {100, 200, 255, 255}, .fontSize = 17 }));

            Clay_String stepStr{ false, (int32_t)strlen(stepBuf), stepBuf };
            CLAY_TEXT(stepStr,
                CLAY_TEXT_CONFIG({ .textColor = {140, 140, 170, 255}, .fontSize = 11 }));

            if (pongMapped) {
                Clay_String scoreStr{ false, (int32_t)strlen(scoreBuf), scoreBuf };
                CLAY_TEXT(scoreStr,
                    CLAY_TEXT_CONFIG({ .textColor = {80, 220, 120, 255}, .fontSize = 13 }));
            }

            CLAY(CLAY_ID("Sep1"), {
                .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1) } },
                .backgroundColor = { 45, 50, 75, 200 }
            }) {}

            CLAY_TEXT(CLAY_STRING("Network controls AI (cyan) paddle"),
                CLAY_TEXT_CONFIG({ .textColor = {180, 200, 255, 255}, .fontSize = 12 }));
            CLAY_TEXT(CLAY_STRING("Input  : ball Y -> left col Gaussian"),
                CLAY_TEXT_CONFIG({ .textColor = {80, 220, 170, 255}, .fontSize = 11 }));
            CLAY_TEXT(CLAY_STRING("Output : right col top/bot -> up/dn"),
                CLAY_TEXT_CONFIG({ .textColor = {200, 100, 220, 255}, .fontSize = 11 }));
            CLAY_TEXT(CLAY_STRING("Reward : hit=+1  miss=-1  else=0"),
                CLAY_TEXT_CONFIG({ .textColor = {160, 160, 180, 255}, .fontSize = 11 }));

            CLAY(CLAY_ID("Sep2"), {
                .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1) } },
                .backgroundColor = { 45, 50, 75, 200 }
            }) {}

            CLAY_TEXT(CLAY_STRING("View:"),
                CLAY_TEXT_CONFIG({ .textColor = {130, 130, 155, 255}, .fontSize = 12 }));

            CLAY(CLAY_ID("ViewRow"), {
                .layout = {
                    .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                    .childGap        = 4,
                    .layoutDirection = CLAY_LEFT_TO_RIGHT
                }
            }) {
                for (int i = 0; i < 3; ++i) {
                    bool active = (viewMode == i);
                    CLAY(CLAY_IDI("VBtn", i), {
                        .layout = {
                            .sizing         = { CLAY_SIZING_FIXED(64), CLAY_SIZING_FIXED(20) },
                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                        },
                        .backgroundColor = active ? Clay_Color{40,100,200,240}
                                                   : Clay_Color{28,28,48,210},
                        .cornerRadius    = CLAY_CORNER_RADIUS(3)
                    }) {
                        if (Clay_Hovered() && inp.lmbPressed) viewMode = i;
                        Clay_String lbl{ false, (int32_t)strlen(viewLabels[i]), viewLabels[i] };
                        CLAY_TEXT(lbl,
                            CLAY_TEXT_CONFIG({ .textColor = {210,220,255,255}, .fontSize = 11 }));
                    }
                }
            }

            CLAY_TEXT(CLAY_STRING("1/2/3 = view   R = reset"),
                CLAY_TEXT_CONFIG({ .textColor = {90, 90, 115, 255}, .fontSize = 11 }));
        }
    }
}

// ─── cleanup ──────────────────────────────────────────────────────────────────
void NeuralCA::cleanup(VkDevice device) {
    vkDestroyPipeline(device, pongPipeline, nullptr);
    vkDestroyPipeline(device, compPipeline, nullptr);
    vkDestroyPipeline(device, dispPipeline, nullptr);
    vkDestroyPipelineLayout(device, pongPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, compPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, dispPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, pongLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, compLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dispLayout, nullptr);
    vkDestroySampler(device, sampler, nullptr);
    for (int i = 0; i < 2; ++i) {
        vkDestroyImageView(device, view[i], nullptr);
        vkDestroyImage(device, image[i], nullptr);
        vkFreeMemory(device, memory[i], nullptr);
    }
    if (pongMapped) vkUnmapMemory(device, pongMem);
    vkDestroyBuffer(device, pongBuf, nullptr);
    vkFreeMemory(device, pongMem, nullptr);
}

// ─── onKey ────────────────────────────────────────────────────────────────────
void NeuralCA::onKey(GLFWwindow* /*window*/, int key, int action) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_R) pendingReset = true;
    if (key == GLFW_KEY_1) viewMode = 0;
    if (key == GLFW_KEY_2) viewMode = 1;
    if (key == GLFW_KEY_3) viewMode = 2;
}

// ─── createImages ─────────────────────────────────────────────────────────────
void NeuralCA::createImages(VulkanContext& ctx) {
    constexpr VkFormat FMT = VK_FORMAT_R32G32B32A32_SFLOAT;
    constexpr VkImageUsageFlags USAGE =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    for (int i = 0; i < 2; ++i) {
        ctx.createImage(GRID_W, GRID_H, FMT, USAGE, image[i], memory[i]);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = image[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = FMT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &vci, nullptr, &view[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateImageView (NeuralCA) failed.");
    }

    auto cmd = ctx.beginOneTimeCommands();
    for (int i = 0; i < 2; ++i)
        ctx.imageBarrier(cmd, image[i],
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    ctx.endOneTimeCommands(cmd);
}

// ─── createSampler ────────────────────────────────────────────────────────────
void NeuralCA::createSampler(VulkanContext& ctx) {
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter    = VK_FILTER_NEAREST;
    ci.minFilter    = VK_FILTER_NEAREST;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx.device, &ci, nullptr, &sampler) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSampler (NeuralCA) failed.");
}

// ─── createPongBuffer ─────────────────────────────────────────────────────────
void NeuralCA::createPongBuffer(VulkanContext& ctx) {
    ctx.createBuffer(sizeof(PongState),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        pongBuf, pongMem);

    vkMapMemory(ctx.device, pongMem, 0, sizeof(PongState), 0,
        reinterpret_cast<void**>(&pongMapped));
    *pongMapped = PongState{};
}

// ─── createDescriptors ────────────────────────────────────────────────────────
void NeuralCA::createDescriptors(VulkanContext& ctx) {
    // Neural update layout: b0=src image, b1=dst image, b2=pong SSBO (readonly)
    {
        VkDescriptorSetLayoutBinding b[3] = {};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1] = b[0]; b[1].binding = 1;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 3; ci.pBindings = b;
        vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &compLayout);
    }
    // Pong step layout: b0=neural image (read), b1=pong SSBO (read/write)
    {
        VkDescriptorSetLayoutBinding b[2] = {};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2; ci.pBindings = b;
        vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &pongLayout);
    }
    // Display layout: b0=combined image sampler, b1=pong SSBO (readonly)
    {
        VkDescriptorSetLayoutBinding b[2] = {};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2; ci.pBindings = b;
        vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &dispLayout);
    }

    // Pool: 6 storage images, 2 combined samplers, 6 storage buffers, 6 sets total
    VkDescriptorPoolSize sizes[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          6},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         6},
    };
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 3; pi.pPoolSizes = sizes; pi.maxSets = 6;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descPool);

    VkDescriptorBufferInfo pongInfo{pongBuf, 0, VK_WHOLE_SIZE};

    for (int s = 0; s < 2; ++s) {
        // ── Neural update set ────────────────────────────────────────────────
        {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &compLayout;
            vkAllocateDescriptorSets(ctx.device, &ai, &compSet[s]);

            VkDescriptorImageInfo imgs[2] = {
                {VK_NULL_HANDLE, view[s],   VK_IMAGE_LAYOUT_GENERAL},
                {VK_NULL_HANDLE, view[1-s], VK_IMAGE_LAYOUT_GENERAL},
            };
            VkWriteDescriptorSet w[3] = {};
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = compSet[s]; w[0].dstBinding = 0;
            w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[0].pImageInfo = &imgs[0];
            w[1] = w[0]; w[1].dstBinding = 1; w[1].pImageInfo = &imgs[1];
            w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[2].dstSet = compSet[s]; w[2].dstBinding = 2;
            w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[2].pBufferInfo = &pongInfo;
            vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
        }
        // ── Pong step set ────────────────────────────────────────────────────
        {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &pongLayout;
            vkAllocateDescriptorSets(ctx.device, &ai, &pongSet[s]);

            VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, view[s], VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2] = {};
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = pongSet[s]; w[0].dstBinding = 0;
            w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[0].pImageInfo = &imgInfo;
            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet = pongSet[s]; w[1].dstBinding = 1;
            w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[1].pBufferInfo = &pongInfo;
            vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
        }
        // ── Display set ──────────────────────────────────────────────────────
        {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dispLayout;
            vkAllocateDescriptorSets(ctx.device, &ai, &dispSet[s]);

            VkDescriptorImageInfo sampInfo{sampler, view[s], VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2] = {};
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = dispSet[s]; w[0].dstBinding = 0;
            w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo = &sampInfo;
            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet = dispSet[s]; w[1].dstBinding = 1;
            w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[1].pBufferInfo = &pongInfo;
            vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
        }
    }
}

// ─── createPongPipeline ───────────────────────────────────────────────────────
void NeuralCA::createPongPipeline(VulkanContext& ctx) {
    auto mod = ctx.loadShader("shaders/pong_step.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = mod; stage.pName = "main";

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(PongPushConstants);

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount         = 1; li.pSetLayouts         = &pongLayout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &pongPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage; ci.layout = pongPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pongPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pong step compute pipeline.");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createComputePipeline ────────────────────────────────────────────────────
void NeuralCA::createComputePipeline(VulkanContext& ctx) {
    auto mod = ctx.loadShader("shaders/neural_update.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = mod; stage.pName = "main";

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(NeuralPushConstants);

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount         = 1; li.pSetLayouts         = &compLayout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &compPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage; ci.layout = compPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &compPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create NeuralCA compute pipeline.");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createDisplayPipeline ────────────────────────────────────────────────────
void NeuralCA::createDisplayPipeline(VulkanContext& ctx) {
    auto vert = ctx.loadShader("shaders/fullscreen.vert.spv");
    auto frag = ctx.loadShader("shaders/neural_display.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vert; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo   vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D   sc{{0, 0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp;
    vps.scissorCount  = 1; vps.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL; rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE; rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    if (dispPipeLayout == VK_NULL_HANDLE) {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.size       = sizeof(NeuralDisplayPC);

        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount         = 1; li.pSetLayouts         = &dispLayout;
        li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &dispPipeLayout);
    }

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2; ci.pStages = stages;
    ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia;
    ci.pViewportState    = &vps; ci.pRasterizationState = &rast;
    ci.pMultisampleState = &ms;  ci.pColorBlendState    = &cb;
    ci.layout = dispPipeLayout; ci.renderPass = ctx.renderPass; ci.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &dispPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create NeuralCA display pipeline.");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── resetGrid ────────────────────────────────────────────────────────────────
// Cell format: r=activation  g=weight  b=eligibility_trace  a=bias
void NeuralCA::resetGrid(VulkanContext& ctx) {
    VkDeviceSize imgSize = GRID_W * GRID_H * 4 * sizeof(float);

    VkBuffer staging; VkDeviceMemory stagingMem;
    ctx.createBuffer(imgSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem);

    void* data; vkMapMemory(ctx.device, stagingMem, 0, imgSize, 0, &data);
    auto* cells = reinterpret_cast<float*>(data);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> wNoise(-0.1f,  0.1f);
    std::uniform_real_distribution<float> bNoise( 0.0f,  0.2f);

    for (uint32_t y = 0; y < GRID_H; ++y) {
        for (uint32_t x = 0; x < GRID_W; ++x) {
            uint32_t idx = (y * GRID_W + x) * 4;
            cells[idx+0] = 0.0f;                // activation
            cells[idx+1] = 0.5f + wNoise(rng);  // weight (random initial conductance)
            cells[idx+2] = 0.0f;                // eligibility trace
            cells[idx+3] = 0.5f + bNoise(rng);  // bias: positive → cells start quiet
        }
    }
    vkUnmapMemory(ctx.device, stagingMem);

    auto cmd = ctx.beginOneTimeCommands();

    ctx.imageBarrier(cmd, image[0],
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {GRID_W, GRID_H, 1};
    vkCmdCopyBufferToImage(cmd, staging, image[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    ctx.imageBarrier(cmd, image[0],
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    ctx.imageBarrier(cmd, image[1],
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    ctx.endOneTimeCommands(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    current   = 0;
    step      = 0;
    totalTime = 0.0f;
}
