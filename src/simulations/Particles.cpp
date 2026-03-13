#include "Particles.h"

#include <algorithm>
#include <random>
#include <stdexcept>

// clay.h is a single-header library; CLAY_IMPLEMENTATION is defined only in UIRenderer.cpp.
// Include here without the implementation define so we can use the CLAY macros.
#include "clay.h"

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
    // Only the draw pipeline bakes in viewport size; compute is unaffected
    vkDestroyPipeline(ctx.device, drawPipeline, nullptr);
    drawPipeline = VK_NULL_HANDLE;
    createDrawPipeline(ctx);
}

// ─── recordCompute ────────────────────────────────────────────────────────────
void Particles::recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float dt) {
    totalTime += dt;

    ParticlePushConstants pc{};
    pc.dt       = dt;
    pc.time     = totalTime;
    pc.mouseNDC = mouseNDC;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        compPipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(pc), &pc);

    uint32_t groups = (PARTICLE_COUNT + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute write → vertex shader read
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask       = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
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
// The render pass is already open when this is called; do NOT begin/end it here.
void Particles::recordDraw(VkCommandBuffer cmd, VulkanContext& ctx, float /*dt*/) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        drawPipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdDraw(cmd, PARTICLE_COUNT, 1, 0, 0);
}

// ─── buildUI ──────────────────────────────────────────────────────────────────
void Particles::buildUI(float /*dt*/) {
    // Full-screen invisible root container (required by Clay as the layout root)
    CLAY(CLAY_ID("ParticlesRoot"), {
        .layout = {
            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            .padding = CLAY_PADDING_ALL(12)
        }
    }) {
        // Info panel anchored to the top-left corner
        CLAY(CLAY_ID("ParticlesInfoPanel"), {
            .layout = {
                .sizing = {
                    .width  = CLAY_SIZING_FIXED(220),
                    .height = CLAY_SIZING_FIT(0)
                },
                .padding         = CLAY_PADDING_ALL(10),
                .childGap        = 6,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .backgroundColor = { 10, 10, 20, 200 },
            .cornerRadius    = CLAY_CORNER_RADIUS(6)
        }) {
            CLAY_TEXT(CLAY_STRING("Particles"),
                CLAY_TEXT_CONFIG({ .textColor = {100, 160, 255, 255}, .fontSize = 18 }));
            CLAY_TEXT(CLAY_STRING("500 000 particles"),
                CLAY_TEXT_CONFIG({ .textColor = {180, 180, 180, 255}, .fontSize = 14 }));
            CLAY_TEXT(CLAY_STRING("Mouse  attract"),
                CLAY_TEXT_CONFIG({ .textColor = {120, 120, 140, 255}, .fontSize = 13 }));
        }
    }
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

    // Only create layout once
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

    // Create host-visible staging buffer
    VkBuffer staging; VkDeviceMemory stagingMem;
    ctx.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem);

    void* data; vkMapMemory(ctx.device, stagingMem, 0, size, 0, &data);
    auto* particles = reinterpret_cast<Particle*>(data);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posD(-1.0f, 1.0f);
    std::uniform_real_distribution<float> velD(-0.01f, 0.01f);

    for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
        particles[i].pos   = {posD(rng), posD(rng)};
        particles[i].vel   = {velD(rng), velD(rng)};
        // Assign a rainbow hue based on index
        float h = (float)i / PARTICLE_COUNT;
        // HSV to RGB (simple version, S=1 V=1)
        float r = std::abs(h * 6.0f - 3.0f) - 1.0f;
        float g = 2.0f - std::abs(h * 6.0f - 2.0f);
        float b = 2.0f - std::abs(h * 6.0f - 4.0f);
        particles[i].color = {
            std::clamp(r, 0.0f, 1.0f) * 0.8f,
            std::clamp(g, 0.0f, 1.0f) * 0.8f,
            std::clamp(b, 0.0f, 1.0f) * 0.8f,
            1.0f
        };
    }
    vkUnmapMemory(ctx.device, stagingMem);

    // Copy to device-local buffer
    auto cmd = ctx.beginOneTimeCommands();
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, staging, buffer, 1, &copy);
    ctx.endOneTimeCommands(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
}
