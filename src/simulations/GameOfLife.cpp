#include "GameOfLife.h"

#include <random>
#include <stdexcept>

// clay.h is a single-header library; CLAY_IMPLEMENTATION is defined only in UIRenderer.cpp.
// Include here without the implementation define so we can use the CLAY macros.
#include "clay.h"

// ─── init ─────────────────────────────────────────────────────────────────────
void GameOfLife::init(VulkanContext& ctx) {
    createImages(ctx);
    createSampler(ctx);
    createDescriptors(ctx);
    createComputePipeline(ctx);
    createDisplayPipeline(ctx);
    randomize(ctx);
}

// ─── onResize ─────────────────────────────────────────────────────────────────
void GameOfLife::onResize(VulkanContext& ctx) {
    // Only the display pipeline bakes in viewport size; compute is unaffected
    vkDestroyPipeline(ctx.device, dispPipeline, nullptr);
    dispPipeline = VK_NULL_HANDLE;
    createDisplayPipeline(ctx);
}

// ─── recordCompute ────────────────────────────────────────────────────────────
void GameOfLife::recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float /*dt*/) {
    // Handle pending randomize request — beginOneTimeCommands submits and waits
    // synchronously so it is safe here before we add any work to cmd.
    if (pendingRandomize) {
        pendingRandomize = false;
        randomize(ctx);
    }

    // GoL compute step
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        compPipeLayout, 0, 1, &compSet[current], 0, nullptr);

    uint32_t gx = (GOL_W + 15) / 16;
    uint32_t gy = (GOL_H + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Barrier: compute write → fragment read on the newly written image
    ctx.imageBarrier(cmd, image[1 - current],
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    // Advance ping-pong so recordDraw samples the just-computed image
    current = 1 - current;
}

// ─── recordDraw ───────────────────────────────────────────────────────────────
// The render pass is already open when this is called; do NOT begin/end it here.
void GameOfLife::recordDraw(VkCommandBuffer cmd, VulkanContext& ctx, float /*dt*/) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dispPipeline);
    // current was flipped in recordCompute, so image[current] is the live (just computed) image.
    // dispSet[i] samples from image[i].
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        dispPipeLayout, 0, 1, &dispSet[current], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle — no VBO needed
}

// ─── buildUI ──────────────────────────────────────────────────────────────────
void GameOfLife::buildUI(float /*dt*/, UIRenderer& /*ui*/) {
    // Full-screen invisible root container (required by Clay as the layout root)
    CLAY(CLAY_ID("GolRoot"), {
        .layout = {
            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            .padding = CLAY_PADDING_ALL(12)
        }
    }) {
        // Info panel anchored to the top-left corner
        CLAY(CLAY_ID("GolInfoPanel"), {
            .layout = {
                .sizing = {
                    .width  = CLAY_SIZING_FIXED(220),
                    .height = CLAY_SIZING_FIT(0)
                },
                .padding         = CLAY_PADDING_ALL(10),
                .childGap        = 6,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .backgroundColor = { 15, 15, 25, 200 },
            .cornerRadius    = CLAY_CORNER_RADIUS(6)
        }) {
            CLAY_TEXT(CLAY_STRING("Game of Life"),
                CLAY_TEXT_CONFIG({ .textColor = {100, 220, 150, 255}, .fontSize = 18 }));
            CLAY_TEXT(CLAY_STRING("512 x 512 grid"),
                CLAY_TEXT_CONFIG({ .textColor = {180, 180, 180, 255}, .fontSize = 14 }));
            CLAY_TEXT(CLAY_STRING("SPACE  randomize"),
                CLAY_TEXT_CONFIG({ .textColor = {120, 120, 140, 255}, .fontSize = 13 }));
        }
    }
}

// ─── cleanup ──────────────────────────────────────────────────────────────────
void GameOfLife::cleanup(VkDevice device) {
    vkDestroyPipeline(device, compPipeline, nullptr);
    vkDestroyPipeline(device, dispPipeline, nullptr);
    vkDestroyPipelineLayout(device, compPipeLayout, nullptr);
    vkDestroyPipelineLayout(device, dispPipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, compLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dispLayout, nullptr);
    vkDestroySampler(device, sampler, nullptr);
    for (int i = 0; i < 2; ++i) {
        vkDestroyImageView(device, view[i], nullptr);
        vkDestroyImage(device, image[i], nullptr);
        vkFreeMemory(device, memory[i], nullptr);
    }
}

// ─── onKey ────────────────────────────────────────────────────────────────────
void GameOfLife::onKey(GLFWwindow* /*window*/, int key, int action) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        pendingRandomize = true;
}

// ─── createImages ─────────────────────────────────────────────────────────────
void GameOfLife::createImages(VulkanContext& ctx) {
    constexpr VkFormat FMT = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkImageUsageFlags USAGE = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    for (int i = 0; i < 2; ++i) {
        ctx.createImage(GOL_W, GOL_H, FMT, USAGE, image[i], memory[i]);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = image[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = FMT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &vci, nullptr, &view[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateImageView (GoL) failed.");
    }

    // Transition both images to GENERAL layout
    auto cmd = ctx.beginOneTimeCommands();
    for (int i = 0; i < 2; ++i)
        ctx.imageBarrier(cmd, image[i],
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    ctx.endOneTimeCommands(cmd);
}

// ─── createSampler ────────────────────────────────────────────────────────────
void GameOfLife::createSampler(VulkanContext& ctx) {
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = VK_FILTER_NEAREST;
    ci.minFilter = VK_FILTER_NEAREST;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (vkCreateSampler(ctx.device, &ci, nullptr, &sampler) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSampler failed.");
}

// ─── createDescriptors ────────────────────────────────────────────────────────
void GameOfLife::createDescriptors(VulkanContext& ctx) {
    // ── Compute layout: two storage image bindings ─────────────────────────
    {
        VkDescriptorSetLayoutBinding b[2] = {};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1] = b[0]; b[1].binding = 1;

        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2; ci.pBindings = b;
        vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &compLayout);
    }
    // ── Display layout: one combined image sampler ─────────────────────────
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 1; ci.pBindings = &b;
        vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &dispLayout);
    }

    // ── Pool: 2 compute sets + 2 display sets ─────────────────────────────
    VkDescriptorPoolSize sizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
    };
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    pi.maxSets = 4;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descPool);

    // ── Allocate and write compute sets ───────────────────────────────────
    for (int s = 0; s < 2; ++s) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &compLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &compSet[s]);

        // set s: reads from image[s], writes to image[1-s]
        VkDescriptorImageInfo imgInfo[2] = {
            {VK_NULL_HANDLE, view[s],   VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, view[1-s], VK_IMAGE_LAYOUT_GENERAL},
        };
        VkWriteDescriptorSet w[2] = {};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet          = compSet[s];
        w[0].dstBinding      = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo      = &imgInfo[0];
        w[1] = w[0]; w[1].dstBinding = 1; w[1].pImageInfo = &imgInfo[1];
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    }

    // ── Allocate and write display sets ───────────────────────────────────
    for (int s = 0; s < 2; ++s) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &dispLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &dispSet[s]);

        // display set s samples from image[s]
        VkDescriptorImageInfo imgInfo{sampler, view[s], VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = dispSet[s];
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }
}

// ─── createComputePipeline ────────────────────────────────────────────────────
void GameOfLife::createComputePipeline(VulkanContext& ctx) {
    auto mod = ctx.loadShader("shaders/game_of_life.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts    = &compLayout;
    vkCreatePipelineLayout(ctx.device, &li, nullptr, &compPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = compPipeLayout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &compPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create GoL compute pipeline.");

    vkDestroyShaderModule(ctx.device, mod, nullptr);
}

// ─── createDisplayPipeline ────────────────────────────────────────────────────
void GameOfLife::createDisplayPipeline(VulkanContext& ctx) {
    auto vert = ctx.loadShader("shaders/fullscreen.vert.spv");
    auto frag = ctx.loadShader("shaders/gol_display.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo   vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

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

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    // Only create layout once
    if (dispPipeLayout == VK_NULL_HANDLE) {
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts    = &dispLayout;
        vkCreatePipelineLayout(ctx.device, &li, nullptr, &dispPipeLayout);
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
    ci.layout              = dispPipeLayout;
    ci.renderPass          = ctx.renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &dispPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create GoL display pipeline.");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── randomize ────────────────────────────────────────────────────────────────
void GameOfLife::randomize(VulkanContext& ctx) {
    // Fill image[0] with random alive/dead cells using a staging buffer
    VkDeviceSize imgSize = GOL_W * GOL_H * 4; // RGBA8
    VkBuffer staging; VkDeviceMemory stagingMem;
    ctx.createBuffer(imgSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem);

    void* data; vkMapMemory(ctx.device, stagingMem, 0, imgSize, 0, &data);
    auto* pixels = reinterpret_cast<uint8_t*>(data);
    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution alive(0.3); // 30% chance alive
    for (uint32_t i = 0; i < GOL_W * GOL_H; ++i) {
        uint8_t v = alive(rng) ? 255 : 0;
        pixels[i*4+0] = v; // R = alive flag
        pixels[i*4+1] = 0;
        pixels[i*4+2] = 0;
        pixels[i*4+3] = 255;
    }
    vkUnmapMemory(ctx.device, stagingMem);

    // Copy staging buffer → image[0]
    auto cmd = ctx.beginOneTimeCommands();
    ctx.imageBarrier(cmd, image[0],
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {GOL_W, GOL_H, 1};
    vkCmdCopyBufferToImage(cmd, staging, image[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    ctx.imageBarrier(cmd, image[0],
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Also transition image[1] to GENERAL (clear it)
    ctx.imageBarrier(cmd, image[1],
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    ctx.endOneTimeCommands(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    current = 0;
}
