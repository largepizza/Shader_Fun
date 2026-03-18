// UIRenderer.cpp
// Implements Clay UI layout + stb_truetype font rendering over the Vulkan swapchain.

// ── Single-header library implementations (one TU only) ──────────────────────
#define CLAY_IMPLEMENTATION
#include "clay.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ── Standard includes ─────────────────────────────────────────────────────────
#include "UIRenderer.h"
#include "VulkanContext.h"

#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// init
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::init(VulkanContext& ctx) {
    // ── Clay memory ──────────────────────────────────────────────────────────
    clayMemorySize = 8 * 1024 * 1024; // 8 MB
    clayMemory     = new uint8_t[clayMemorySize];

    Clay_Arena arena = Clay_CreateArenaWithCapacityAndMemory(clayMemorySize, clayMemory);

    Clay_ErrorHandler errHandler{};
    errHandler.errorHandlerFunction = [](Clay_ErrorData errorData) {
        fprintf(stderr, "[Clay] error %d: %.*s\n",
                (int)errorData.errorType,
                errorData.errorText.length, errorData.errorText.chars);
    };

    Clay_Initialize(arena,
                    Clay_Dimensions{ frameW, frameH },
                    errHandler);

    // Register measure-text callback.
    // The actual signature is:
    //   Clay_Dimensions fn(Clay_StringSlice, Clay_TextElementConfig*, void*)
    // We store a lambda wrapper that calls our static helper.
    Clay_SetMeasureTextFunction(
        [](Clay_StringSlice text, Clay_TextElementConfig* cfg, void* ud) -> Clay_Dimensions {
            UIRenderer* self = reinterpret_cast<UIRenderer*>(ud);
            stbtt_fontinfo* fi = reinterpret_cast<stbtt_fontinfo*>(self->font.fontInfo);

            // xadvance in stbtt_bakedchar is in baked-size pixels.
            // Scale to the requested fontSize by simple ratio, not by font design-unit scale.
            float renderScale = (float)cfg->fontSize / self->font.bakedSize;

            stbtt_bakedchar* bc = reinterpret_cast<stbtt_bakedchar*>(self->font.charData.data());
            float totalW = 0.0f;
            for (int i = 0; i < text.length; ++i) {
                unsigned char c = (unsigned char)text.chars[i];
                if (c < 32 || c > 126) continue;
                totalW += bc[c - 32].xadvance * renderScale;
            }

            // Height uses the font's design-unit metrics (correct — these are not in baked pixels)
            int ascent, descent, lineGap;
            stbtt_GetFontVMetrics(fi, &ascent, &descent, &lineGap);
            float fontScale = stbtt_ScaleForPixelHeight(fi, (float)cfg->fontSize);
            float height = (float)(ascent - descent + lineGap) * fontScale;

            return Clay_Dimensions{ totalW, height };
        },
        this);

    // ── Font ─────────────────────────────────────────────────────────────────
    loadFont(ctx);

    // ── GPU geometry buffers ─────────────────────────────────────────────────
    VkDeviceSize vertSize = sizeof(UIVertex)  * MAX_VERTS;
    VkDeviceSize idxSize  = sizeof(uint32_t)  * MAX_INDICES;

    ctx.createBuffer(vertSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        vertBuf, vertMem);
    vkMapMemory(ctx.device, vertMem, 0, vertSize, 0, &vertMapped);

    ctx.createBuffer(idxSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        idxBuf, idxMem);
    vkMapMemory(ctx.device, idxMem, 0, idxSize, 0, &idxMapped);

    // ── Icon placeholder (1×1 white RGBA) — must exist before descriptor writes ──
    createIconPlaceholder(ctx);

    // ── Descriptor set layout: binding 0 = font atlas, binding 1 = icon atlas ──
    {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2;
        ci.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &descLayout) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkCreateDescriptorSetLayout failed.");
    }

    // ── Descriptor pool + set ─────────────────────────────────────────────────
    {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 };
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.poolSizeCount = 1;
        pi.pPoolSizes    = &ps;
        pi.maxSets       = 1;
        if (vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descPool) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkCreateDescriptorPool failed.");

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &descLayout;
        if (vkAllocateDescriptorSets(ctx.device, &ai, &descSet) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkAllocateDescriptorSets failed.");

        VkDescriptorImageInfo fontInfo{ font.sampler, font.view,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo iconInfo{ iconSampler, iconView,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = descSet;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &fontInfo;
        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = descSet;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo      = &iconInfo;
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }

    // ── Pipeline ─────────────────────────────────────────────────────────────
    createPipeline(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// loadFont
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::loadFont(VulkanContext& ctx) {
    // Try Windows system fonts in order of preference.
    // Customize this list if deploying on non-Windows systems.
    const char* candidates[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
    };

    std::vector<uint8_t> ttfData;
    bool found = false;
    for (const char* path : candidates) {
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fclose(f); continue; }
        ttfData.resize((size_t)sz);
        fread(ttfData.data(), 1, (size_t)sz, f);
        fclose(f);
        found = true;
        break;
    }
    if (!found)
        throw std::runtime_error(
            "UIRenderer: Could not find a TTF font. Tried segoeui.ttf, arial.ttf, consola.ttf "
            "in C:/Windows/Fonts/. Please set a valid font path in UIRenderer.cpp.");

    font.fileData = std::move(ttfData);

    // Allocate and initialise stbtt_fontinfo
    font.fontInfo = new stbtt_fontinfo();
    stbtt_fontinfo* fi = reinterpret_cast<stbtt_fontinfo*>(font.fontInfo);
    if (!stbtt_InitFont(fi, font.fileData.data(),
                        stbtt_GetFontOffsetForIndex(font.fileData.data(), 0)))
        throw std::runtime_error("UIRenderer: stbtt_InitFont failed.");

    // Bake ASCII 32-126 (96 glyphs) into a 512x512 R8 atlas
    font.charData.resize(96 * sizeof(stbtt_bakedchar));
    std::vector<uint8_t> pixels(font.atlasW * font.atlasH);
    stbtt_BakeFontBitmap(font.fileData.data(), 0,
                         font.bakedSize,
                         pixels.data(), font.atlasW, font.atlasH,
                         32, 96,
                         reinterpret_cast<stbtt_bakedchar*>(font.charData.data()));

    // ── Upload atlas as VK_FORMAT_R8_UNORM ────────────────────────────────
    VkDeviceSize atlasBytes = (VkDeviceSize)(font.atlasW * font.atlasH);

    // Create staging buffer
    VkBuffer stagingBuf; VkDeviceMemory stagingMem;
    ctx.createBuffer(atlasBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuf, stagingMem);
    void* mapped;
    vkMapMemory(ctx.device, stagingMem, 0, atlasBytes, 0, &mapped);
    memcpy(mapped, pixels.data(), (size_t)atlasBytes);
    vkUnmapMemory(ctx.device, stagingMem);

    // Create atlas image
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = VK_FORMAT_R8_UNORM;
        ci.extent        = { (uint32_t)font.atlasW, (uint32_t)font.atlasH, 1 };
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(ctx.device, &ci, nullptr, &font.image) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkCreateImage (font atlas) failed.");

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(ctx.device, font.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(ctx.device, &ai, nullptr, &font.memory) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkAllocateMemory (font atlas) failed.");
        vkBindImageMemory(ctx.device, font.image, font.memory, 0);
    }

    // Upload via one-time command buffer
    {
        auto cmd = ctx.beginOneTimeCommands();

        // UNDEFINED → TRANSFER_DST
        ctx.imageBarrier(cmd, font.image,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent      = { (uint32_t)font.atlasW, (uint32_t)font.atlasH, 1 };
        vkCmdCopyBufferToImage(cmd, stagingBuf, font.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // TRANSFER_DST → SHADER_READ_ONLY_OPTIMAL (stays here permanently)
        ctx.imageBarrier(cmd, font.image,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        ctx.endOneTimeCommands(cmd);
    }

    vkDestroyBuffer(ctx.device, stagingBuf, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    // ── Image view ───────────────────────────────────────────────────────────
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = font.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = VK_FORMAT_R8_UNORM;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(ctx.device, &vci, nullptr, &font.view) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkCreateImageView (font atlas) failed.");
    }

    // ── Sampler ──────────────────────────────────────────────────────────────
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter  = VK_FILTER_LINEAR;
        sci.minFilter  = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(ctx.device, &sci, nullptr, &font.sampler) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkCreateSampler (font atlas) failed.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// createPipeline
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::createPipeline(VulkanContext& ctx) {
    auto vert = ctx.loadShader("shaders/ui.vert.spv");
    auto frag = ctx.loadShader("shaders/ui.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    // Vertex input: one binding, interleaved UIVertex
    VkVertexInputBindingDescription vbd{};
    vbd.binding   = 0;
    vbd.stride    = sizeof(UIVertex);
    vbd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vad[4] = {};
    // location 0: pos (vec2)
    vad[0].location = 0;
    vad[0].binding  = 0;
    vad[0].format   = VK_FORMAT_R32G32_SFLOAT;
    vad[0].offset   = offsetof(UIVertex, pos);
    // location 1: uv (vec2)
    vad[1].location = 1;
    vad[1].binding  = 0;
    vad[1].format   = VK_FORMAT_R32G32_SFLOAT;
    vad[1].offset   = offsetof(UIVertex, uv);
    // location 2: color (vec4)
    vad[2].location = 2;
    vad[2].binding  = 0;
    vad[2].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    vad[2].offset   = offsetof(UIVertex, color);
    // location 3: mode (float)
    vad[3].location = 3;
    vad[3].binding  = 0;
    vad[3].format   = VK_FORMAT_R32_SFLOAT;
    vad[3].offset   = offsetof(UIVertex, mode);

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vbd;
    vi.vertexAttributeDescriptionCount = 4;
    vi.pVertexAttributeDescriptions    = vad;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport baked in; scissor is dynamic
    VkViewport vp{ 0, 0,
                   (float)ctx.swapExtent.width, (float)ctx.swapExtent.height,
                   0.0f, 1.0f };
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.pViewports    = &vp;
    vps.scissorCount  = 1;
    vps.pScissors     = nullptr; // dynamic

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Standard alpha blending
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable         = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp        = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp        = VK_BLEND_OP_ADD;
    cba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    // Dynamic state: scissor only (viewport is baked)
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 1;
    dyn.pDynamicStates    = dynStates;

    // Pipeline layout: push constants + descriptor set
    if (pipeLayout == VK_NULL_HANDLE) {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(UIPushConstants);

        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount         = 1;
        li.pSetLayouts            = &descLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(ctx.device, &li, nullptr, &pipeLayout) != VK_SUCCESS)
            throw std::runtime_error("UIRenderer: vkCreatePipelineLayout failed.");
    }

    // Depth test off — UI always draws on top.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState   = &ms;
    ci.pDepthStencilState  = &ds;
    ci.pColorBlendState    = &cb;
    ci.pDynamicState       = &dyn;
    ci.layout              = pipeLayout;
    ci.renderPass          = ctx.renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("UIRenderer: vkCreateGraphicsPipelines failed.");

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// destroyPipeline
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::destroyPipeline(VkDevice device) {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// onResize
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::onResize(VulkanContext& ctx) {
    destroyPipeline(ctx.device);
    createPipeline(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// cleanup
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::cleanup(VkDevice device) {
    destroyPipeline(device);

    if (pipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeLayout, nullptr);
        pipeLayout = VK_NULL_HANDLE;
    }
    if (descPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descPool, nullptr);
        descPool = VK_NULL_HANDLE;
    }
    if (descLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        descLayout = VK_NULL_HANDLE;
    }
    if (idxBuf != VK_NULL_HANDLE) {
        vkUnmapMemory(device, idxMem);
        vkDestroyBuffer(device, idxBuf, nullptr);
        vkFreeMemory(device, idxMem, nullptr);
        idxBuf = VK_NULL_HANDLE;
    }
    if (vertBuf != VK_NULL_HANDLE) {
        vkUnmapMemory(device, vertMem);
        vkDestroyBuffer(device, vertBuf, nullptr);
        vkFreeMemory(device, vertMem, nullptr);
        vertBuf = VK_NULL_HANDLE;
    }
    if (font.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, font.sampler, nullptr);
        font.sampler = VK_NULL_HANDLE;
    }
    if (font.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, font.view, nullptr);
        font.view = VK_NULL_HANDLE;
    }
    if (font.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, font.image, nullptr);
        font.image = VK_NULL_HANDLE;
    }
    if (font.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, font.memory, nullptr);
        font.memory = VK_NULL_HANDLE;
    }
    if (font.fontInfo) {
        delete reinterpret_cast<stbtt_fontinfo*>(font.fontInfo);
        font.fontInfo = nullptr;
    }

    if (iconSampler != VK_NULL_HANDLE) { vkDestroySampler(device, iconSampler, nullptr); iconSampler = VK_NULL_HANDLE; }
    if (iconView    != VK_NULL_HANDLE) { vkDestroyImageView(device, iconView, nullptr);   iconView    = VK_NULL_HANDLE; }
    if (iconImage   != VK_NULL_HANDLE) { vkDestroyImage(device, iconImage, nullptr);      iconImage   = VK_NULL_HANDLE; }
    if (iconMemory  != VK_NULL_HANDLE) { vkFreeMemory(device, iconMemory, nullptr);       iconMemory  = VK_NULL_HANDLE; }

    delete[] reinterpret_cast<uint8_t*>(clayMemory);
    clayMemory = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// createIconPlaceholder — 1×1 white RGBA pixel so binding 1 is always valid
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::createIconPlaceholder(VulkanContext& ctx) {
    std::vector<uint8_t> pixel = { 255, 255, 255, 255 };
    uploadIconAtlas(ctx, pixel, 1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// uploadIconAtlas — create/replace the icon GPU image
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::uploadIconAtlas(VulkanContext& ctx, const std::vector<uint8_t>& rgba, int w, int h) {
    // Destroy existing icon image (safe if VK_NULL_HANDLE)
    if (iconSampler != VK_NULL_HANDLE) { vkDestroySampler(ctx.device, iconSampler, nullptr); iconSampler = VK_NULL_HANDLE; }
    if (iconView    != VK_NULL_HANDLE) { vkDestroyImageView(ctx.device, iconView, nullptr);   iconView    = VK_NULL_HANDLE; }
    if (iconImage   != VK_NULL_HANDLE) { vkDestroyImage(ctx.device, iconImage, nullptr);      iconImage   = VK_NULL_HANDLE; }
    if (iconMemory  != VK_NULL_HANDLE) { vkFreeMemory(ctx.device, iconMemory, nullptr);       iconMemory  = VK_NULL_HANDLE; }

    VkDeviceSize bytes = (VkDeviceSize)(w * h * 4);

    // Staging buffer
    VkBuffer stagingBuf; VkDeviceMemory stagingMem;
    ctx.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuf, stagingMem);
    void* mapped;
    vkMapMemory(ctx.device, stagingMem, 0, bytes, 0, &mapped);
    memcpy(mapped, rgba.data(), (size_t)bytes);
    vkUnmapMemory(ctx.device, stagingMem);

    // Image
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType   = VK_IMAGE_TYPE_2D;
    ci.format      = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent      = { (uint32_t)w, (uint32_t)h, 1 };
    ci.mipLevels   = 1;
    ci.arrayLayers = 1;
    ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ci.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.device, &ci, nullptr, &iconImage) != VK_SUCCESS)
        throw std::runtime_error("UIRenderer: vkCreateImage (icon atlas) failed.");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device, iconImage, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(ctx.device, &ai, nullptr, &iconMemory) != VK_SUCCESS)
        throw std::runtime_error("UIRenderer: vkAllocateMemory (icon atlas) failed.");
    vkBindImageMemory(ctx.device, iconImage, iconMemory, 0);

    // Upload via one-time command buffer
    {
        auto cmd = ctx.beginOneTimeCommands();
        ctx.imageBarrier(cmd, iconImage,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent      = { (uint32_t)w, (uint32_t)h, 1 };
        vkCmdCopyBufferToImage(cmd, stagingBuf, iconImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        ctx.imageBarrier(cmd, iconImage,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        ctx.endOneTimeCommands(cmd);
    }
    vkDestroyBuffer(ctx.device, stagingBuf, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    // Image view
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = iconImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(ctx.device, &vci, nullptr, &iconView) != VK_SUCCESS)
        throw std::runtime_error("UIRenderer: vkCreateImageView (icon atlas) failed.");

    // Sampler
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx.device, &sci, nullptr, &iconSampler) != VK_SUCCESS)
        throw std::runtime_error("UIRenderer: vkCreateSampler (icon atlas) failed.");
}

// ─────────────────────────────────────────────────────────────────────────────
// rebindIconDescriptor — update binding 1 after loadIcons replaces the atlas
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::rebindIconDescriptor(VkDevice device) {
    VkDescriptorImageInfo imgInfo{ iconSampler, iconView,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = descSet;
    w.dstBinding      = 1;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo      = &imgInfo;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// loadIcons — load PNG files, pack into horizontal atlas, upload to GPU
// ─────────────────────────────────────────────────────────────────────────────
int UIRenderer::loadIcons(VulkanContext& ctx, const char* const* paths, int count) {
    if (count <= 0) return 0;

    struct RawIcon { uint8_t* data; int w, h; };
    std::vector<RawIcon> icons(count);
    int totalW = 0, maxH = 0;

    for (int i = 0; i < count; ++i) {
        int nc = 0;
        icons[i].data = stbi_load(paths[i], &icons[i].w, &icons[i].h, &nc, 4);
        if (!icons[i].data) {
            // Fallback: 1×1 magenta so a missing icon is obvious
            icons[i].data = (uint8_t*)malloc(4);
            icons[i].data[0] = 255; icons[i].data[1] = 0;
            icons[i].data[2] = 255; icons[i].data[3] = 255;
            icons[i].w = icons[i].h = 1;
            fprintf(stderr, "[UIRenderer] Warning: could not load icon '%s'\n", paths[i]);
        }
        totalW += icons[i].w;
        if (icons[i].h > maxH) maxH = icons[i].h;
    }

    // Pack horizontally into an RGBA atlas
    std::vector<uint8_t> atlas((size_t)(totalW * maxH * 4), 0);
    iconEntries.resize(count);
    int x = 0;
    for (int i = 0; i < count; ++i) {
        for (int row = 0; row < icons[i].h; ++row) {
            memcpy(&atlas[((size_t)(row * totalW + x)) * 4],
                   &icons[i].data[(size_t)(row * icons[i].w) * 4],
                   (size_t)icons[i].w * 4);
        }
        iconEntries[i].u0 = (float)x           / (float)totalW;
        iconEntries[i].u1 = (float)(x + icons[i].w) / (float)totalW;
        iconEntries[i].v0 = 0.0f;
        iconEntries[i].v1 = (float)icons[i].h  / (float)maxH;
        x += icons[i].w;
        stbi_image_free(icons[i].data);
    }

    uploadIconAtlas(ctx, atlas, totalW, maxH);
    rebindIconDescriptor(ctx.device);
    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
// beginFrame
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::beginFrame(float width, float height,
                             float mouseX, float mouseY, bool lmbDown, bool rmbDown,
                             float scrollDeltaX, float scrollDeltaY,
                             float dt) {
    frameW = width;
    frameH = height;

    // Compute per-frame input state; simulations read this in buildUI().
    frameInput.screenW     = width;
    frameInput.screenH     = height;
    frameInput.mouseX      = mouseX;
    frameInput.mouseY      = mouseY;
    frameInput.dMouseX     = mouseX - prevMx;
    frameInput.dMouseY     = mouseY - prevMy;
    frameInput.lmbDown     = lmbDown;
    frameInput.lmbPressed  = lmbDown && !prevLmb;
    frameInput.lmbReleased = !lmbDown && prevLmb;
    frameInput.rmbDown     = rmbDown;
    frameInput.rmbPressed  = rmbDown && !prevRmb;
    frameInput.rmbReleased = !rmbDown && prevRmb;
    frameInput.scrollY     = scrollDeltaY;
    frameInput.dt          = dt;

    prevMx  = mouseX;
    prevMy  = mouseY;
    prevLmb = lmbDown;
    prevRmb = rmbDown;

    // Save last frame's capture result, then reset for this frame's registrations.
    prevMouseOverUI = mouseIsOverUI;
    mouseIsOverUI = false;

    Clay_SetLayoutDimensions(Clay_Dimensions{ width, height });
    Clay_SetPointerState(Clay_Vector2{ mouseX, mouseY }, lmbDown);
    Clay_UpdateScrollContainers(true,
                                Clay_Vector2{ scrollDeltaX, scrollDeltaY },
                                dt);
    Clay_BeginLayout();
}

void UIRenderer::addMouseCaptureRect(float x, float y, float w, float h) {
    if (frameInput.mouseX >= x && frameInput.mouseX < x + w &&
        frameInput.mouseY >= y && frameInput.mouseY < y + h)
        mouseIsOverUI = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// pushQuad — add two triangles for a rectangle
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::pushQuad(float x, float y, float w, float h,
                           float u0, float v0, float u1, float v1,
                           glm::vec4 color, float mode) {
    if (vertices.size() + 4 > MAX_VERTS) return;
    if (indices.size()  + 6 > MAX_INDICES) return;

    uint32_t base = (uint32_t)vertices.size();

    vertices.push_back({ {x,     y    }, {u0, v0}, color, mode });
    vertices.push_back({ {x + w, y    }, {u1, v0}, color, mode });
    vertices.push_back({ {x + w, y + h}, {u1, v1}, color, mode });
    vertices.push_back({ {x,     y + h}, {u0, v1}, color, mode });

    indices.push_back(base + 0);
    indices.push_back(base + 1);
    indices.push_back(base + 2);
    indices.push_back(base + 0);
    indices.push_back(base + 2);
    indices.push_back(base + 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// pushText — lay out a string using baked stbtt_bakedchar data
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::pushText(float x, float y, const char* text, int len,
                           float fontSize, glm::vec4 color) {
    stbtt_fontinfo* fi = reinterpret_cast<stbtt_fontinfo*>(font.fontInfo);

    // renderScale converts baked-pixel coordinates to requested-fontSize pixels.
    // bakedchar fields (xoff, yoff, x0..x1, xadvance) are in baked-size pixels,
    // NOT font design units — so use a simple ratio, not stbtt_ScaleForPixelHeight.
    float renderScale = fontSize / font.bakedSize;

    // Baseline offset uses design-unit metrics (ascent is in font units, scale converts to pixels)
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(fi, &ascent, &descent, &lineGap);
    float fontScale = stbtt_ScaleForPixelHeight(fi, fontSize);

    float penX     = x;
    float baselineY = y + (float)ascent * fontScale;

    stbtt_bakedchar* bc = reinterpret_cast<stbtt_bakedchar*>(font.charData.data());
    float atlasW = (float)font.atlasW;
    float atlasH = (float)font.atlasH;

    for (int i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c < 32 || c > 126) continue;

        stbtt_bakedchar& ch = bc[c - 32];

        float gx = penX + ch.xoff * renderScale;
        float gy = baselineY + ch.yoff * renderScale;
        float gw = (float)(ch.x1 - ch.x0) * renderScale;
        float gh = (float)(ch.y1 - ch.y0) * renderScale;

        float u0 = (float)ch.x0 / atlasW;
        float v0 = (float)ch.y0 / atlasH;
        float u1 = (float)ch.x1 / atlasW;
        float v1 = (float)ch.y1 / atlasH;

        pushQuad(gx, gy, gw, gh, u0, v0, u1, v1, color, 1.0f);

        penX += ch.xadvance * renderScale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// flushBatch — upload geometry and issue the indexed draw call
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::flushBatch(VkCommandBuffer cmd) {
    if (vertices.empty()) return;

    memcpy(vertMapped, vertices.data(), vertices.size() * sizeof(UIVertex));
    memcpy(idxMapped,  indices.data(),  indices.size()  * sizeof(uint32_t));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             pipeLayout, 0, 1, &descSet, 0, nullptr);

    UIPushConstants pc{ { frameW, frameH } };
    vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                        0, sizeof(pc), &pc);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertBuf, &offset);
    vkCmdBindIndexBuffer(cmd, idxBuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, (uint32_t)indices.size(), 1, 0, 0, 0);

    vertices.clear();
    indices.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// record — finalise Clay layout, emit draw commands
// ─────────────────────────────────────────────────────────────────────────────
void UIRenderer::record(VkCommandBuffer cmd, VulkanContext& ctx) {
    // Set initial full-viewport dynamic scissor so subsequent draws are unclipped
    VkRect2D fullScissor{ {0, 0}, ctx.swapExtent };
    vkCmdSetScissor(cmd, 0, 1, &fullScissor);

    Clay_RenderCommandArray cmds = Clay_EndLayout();

    for (int32_t i = 0; i < cmds.length; ++i) {
        Clay_RenderCommand* rc = Clay_RenderCommandArray_Get(&cmds, i);
        Clay_BoundingBox    bb = rc->boundingBox;

        switch (rc->commandType) {

        case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
            Clay_RectangleRenderData& rd = rc->renderData.rectangle;
            glm::vec4 col{
                rd.backgroundColor.r / 255.0f,
                rd.backgroundColor.g / 255.0f,
                rd.backgroundColor.b / 255.0f,
                rd.backgroundColor.a / 255.0f
            };
            pushQuad(bb.x, bb.y, bb.width, bb.height,
                     0.0f, 0.0f, 1.0f, 1.0f, col, 0.0f);
            break;
        }

        case CLAY_RENDER_COMMAND_TYPE_TEXT: {
            Clay_TextRenderData& td = rc->renderData.text;
            glm::vec4 col{
                td.textColor.r / 255.0f,
                td.textColor.g / 255.0f,
                td.textColor.b / 255.0f,
                td.textColor.a / 255.0f
            };
            pushText(bb.x, bb.y,
                     td.stringContents.chars, td.stringContents.length,
                     (float)td.fontSize, col);
            break;
        }

        case CLAY_RENDER_COMMAND_TYPE_BORDER: {
            Clay_BorderRenderData& bd = rc->renderData.border;
            glm::vec4 col{
                bd.color.r / 255.0f,
                bd.color.g / 255.0f,
                bd.color.b / 255.0f,
                bd.color.a / 255.0f
            };
            float bw;
            // Top edge
            bw = (float)bd.width.top;
            if (bw > 0.0f)
                pushQuad(bb.x, bb.y, bb.width, bw,
                         0, 0, 1, 1, col, 0.0f);
            // Bottom edge
            bw = (float)bd.width.bottom;
            if (bw > 0.0f)
                pushQuad(bb.x, bb.y + bb.height - bw, bb.width, bw,
                         0, 0, 1, 1, col, 0.0f);
            // Left edge
            bw = (float)bd.width.left;
            if (bw > 0.0f)
                pushQuad(bb.x, bb.y, bw, bb.height,
                         0, 0, 1, 1, col, 0.0f);
            // Right edge
            bw = (float)bd.width.right;
            if (bw > 0.0f)
                pushQuad(bb.x + bb.width - bw, bb.y, bw, bb.height,
                         0, 0, 1, 1, col, 0.0f);
            break;
        }

        case CLAY_RENDER_COMMAND_TYPE_SCISSOR_START: {
            flushBatch(cmd);
            VkRect2D scissor{};
            scissor.offset.x      = std::max(0, (int32_t)bb.x);
            scissor.offset.y      = std::max(0, (int32_t)bb.y);
            scissor.extent.width  = std::min((uint32_t)bb.width,
                                             ctx.swapExtent.width  - (uint32_t)scissor.offset.x);
            scissor.extent.height = std::min((uint32_t)bb.height,
                                             ctx.swapExtent.height - (uint32_t)scissor.offset.y);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            break;
        }

        case CLAY_RENDER_COMMAND_TYPE_SCISSOR_END: {
            flushBatch(cmd);
            vkCmdSetScissor(cmd, 0, 1, &fullScissor);
            break;
        }

        case CLAY_RENDER_COMMAND_TYPE_IMAGE: {
            // imageData encodes the icon index as a pointer-sized integer.
            int iconIdx = (int)(intptr_t)rc->renderData.image.imageData;
            if (iconIdx >= 0 && iconIdx < (int)iconEntries.size()) {
                const IconEntry& ie = iconEntries[iconIdx];
                pushQuad(bb.x, bb.y, bb.width, bb.height,
                         ie.u0, ie.v0, ie.u1, ie.v1,
                         {1.0f, 1.0f, 1.0f, 1.0f}, 2.0f);
            }
            break;
        }

        default:
            // NONE, CUSTOM — not handled by this renderer
            break;
        }
    }

    // Flush any remaining geometry
    flushBatch(cmd);
}
