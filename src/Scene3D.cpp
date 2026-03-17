#include "Scene3D.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <glm/gtc/matrix_inverse.hpp>

// ─── Object factory ───────────────────────────────────────────────────────────

std::shared_ptr<MeshObject3D> Scene3D::addMesh(const std::string& name,
                                                std::vector<Vertex3D> verts,
                                                std::vector<uint32_t> idxs) {
    auto obj = std::make_shared<MeshObject3D>();
    obj->name     = name;
    obj->vertices = std::move(verts);
    obj->indices  = std::move(idxs);
    objects.push_back(obj);
    return obj;
}

std::shared_ptr<SDFObject3D> Scene3D::addSDF(const std::string& name,
                                              SDFShape shape,
                                              glm::vec3 params) {
    auto obj    = std::make_shared<SDFObject3D>();
    obj->name   = name;
    obj->shape  = shape;
    obj->params = params;
    objects.push_back(obj);
    return obj;
}

// ─── Mesh generators ──────────────────────────────────────────────────────────

std::pair<std::vector<Vertex3D>, std::vector<uint32_t>>
Scene3D::makeBox(glm::vec3 h) {
    // 6 faces, 4 verts each, flat normals
    struct FaceInfo { glm::vec3 normal; glm::vec3 tangent; glm::vec3 bitangent; };
    static const FaceInfo faces[6] = {
        {{ 0, 0, 1}, { 1, 0, 0}, { 0, 1, 0}},
        {{ 0, 0,-1}, {-1, 0, 0}, { 0, 1, 0}},
        {{ 1, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
        {{-1, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},
        {{ 0, 1, 0}, { 1, 0, 0}, { 0, 0,-1}},
        {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 1}},
    };
    glm::vec2 corners[4] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    std::vector<Vertex3D> verts;
    std::vector<uint32_t> idxs;
    for (auto& f : faces) {
        uint32_t base = (uint32_t)verts.size();
        glm::vec3 he = glm::abs(f.normal) * h + glm::abs(f.tangent) * h + glm::abs(f.bitangent) * h;
        (void)he;
        float hw = glm::dot(glm::abs(f.tangent),   h);
        float hh = glm::dot(glm::abs(f.bitangent), h);
        float hd = glm::dot(glm::abs(f.normal),    h);
        for (auto& c : corners) {
            Vertex3D v;
            v.pos    = f.normal * hd + f.tangent * (c.x * hw) + f.bitangent * (c.y * hh);
            v.normal = f.normal;
            v.uv     = (c + glm::vec2(1)) * 0.5f;
            verts.push_back(v);
        }
        for (uint32_t t : {0u,1u,2u, 0u,2u,3u})
            idxs.push_back(base + t);
    }
    return {verts, idxs};
}

std::pair<std::vector<Vertex3D>, std::vector<uint32_t>>
Scene3D::makeSphere(float r, int rings, int sectors) {
    std::vector<Vertex3D> verts;
    std::vector<uint32_t> idxs;
    for (int ri = 0; ri <= rings; ++ri) {
        float phi = glm::pi<float>() * ri / rings; // 0 .. pi
        for (int si = 0; si <= sectors; ++si) {
            float theta = 2.0f * glm::pi<float>() * si / sectors;
            Vertex3D v;
            v.normal = {sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta)};
            v.pos    = v.normal * r;
            v.uv     = {(float)si / sectors, (float)ri / rings};
            verts.push_back(v);
        }
    }
    for (int ri = 0; ri < rings; ++ri) {
        for (int si = 0; si < sectors; ++si) {
            uint32_t a = ri * (sectors + 1) + si;
            uint32_t b = a + sectors + 1;
            // CCW winding (outward-facing normals) for VK_FRONT_FACE_COUNTER_CLOCKWISE
            idxs.insert(idxs.end(), {a, a+1, b, a+1, b+1, b});
        }
    }
    return {verts, idxs};
}

std::pair<std::vector<Vertex3D>, std::vector<uint32_t>>
Scene3D::makeGrid(float halfSize, int divs) {
    std::vector<Vertex3D> verts;
    std::vector<uint32_t> idxs;
    int n = divs + 1;
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            Vertex3D v;
            v.pos    = {halfSize * (2.0f * x / divs - 1), 0, halfSize * (2.0f * z / divs - 1)};
            v.normal = {0, 1, 0};
            v.uv     = {(float)x / divs, (float)z / divs};
            verts.push_back(v);
        }
    }
    for (int z = 0; z < divs; ++z) {
        for (int x = 0; x < divs; ++x) {
            uint32_t a = z * n + x;
            uint32_t b = a + n;
            idxs.insert(idxs.end(), {a, b, a+1, b, b+1, a+1});
        }
    }
    return {verts, idxs};
}

// ─── Object factory (lights) ──────────────────────────────────────────────────

DirectionalLight& Scene3D::addLight(glm::vec3 direction, glm::vec3 color, float intensity) {
    if ((int)lights.size() >= MAX_LIGHTS)
        throw std::runtime_error("Scene3D: exceeded MAX_LIGHTS");
    DirectionalLight l;
    l.direction = glm::normalize(direction);
    l.color     = color;
    l.intensity = intensity;
    lights.push_back(l);
    return lights.back();
}

// ─── Vulkan lifecycle ─────────────────────────────────────────────────────────

void Scene3D::init(VulkanContext& ctx) {
    // Camera UBO
    VkDeviceSize uboSize = sizeof(CameraUBOData);
    ctx.createBuffer(uboSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        cameraUBOBuf, cameraUBOMem);
    vkMapMemory(ctx.device, cameraUBOMem, 0, uboSize, 0, &cameraMapped);

    // SDF SSBO  (header int[4] + MAX_SDF_OBJECTS * GPUSDFObject)
    VkDeviceSize sdfSize = 4 * sizeof(int) + MAX_SDF_OBJECTS * sizeof(GPUSDFObject);
    ctx.createBuffer(sdfSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        sdfSSBOBuf, sdfSSBOMem);
    vkMapMemory(ctx.device, sdfSSBOMem, 0, sdfSize, 0, &sdfMapped);

    // Light UBO
    VkDeviceSize lightSize = sizeof(GPULightUBO);
    ctx.createBuffer(lightSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        lightUBOBuf, lightUBOMem);
    vkMapMemory(ctx.device, lightUBOMem, 0, lightSize, 0, &lightMapped);

    // Build internal indicator sphere mesh
    auto [iv, ii] = makeSphere(1.0f, 12, 16); // unit sphere, scaled by push constant
    indicatorMesh.vertices = std::move(iv);
    indicatorMesh.indices  = std::move(ii);

    createDescriptors(ctx);
    createMeshPipeline(ctx);
    createSDFPipeline(ctx);
}

void Scene3D::onResize(VulkanContext& ctx) {
    destroyPipelines(ctx.device);
    createMeshPipeline(ctx);
    createSDFPipeline(ctx);
}

static void uploadOneMesh(VulkanContext& ctx, MeshObject3D& m) {
    if (m.vertexBuf != VK_NULL_HANDLE) return; // already uploaded

    VkDeviceSize vSize = m.vertices.size() * sizeof(Vertex3D);
    VkBuffer stagV; VkDeviceMemory stagVMem;
    ctx.createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagV, stagVMem);
    void* p; vkMapMemory(ctx.device, stagVMem, 0, vSize, 0, &p);
    memcpy(p, m.vertices.data(), vSize); vkUnmapMemory(ctx.device, stagVMem);
    ctx.createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.vertexBuf, m.vertexMem);
    auto cmd = ctx.beginOneTimeCommands();
    VkBufferCopy r{0,0,vSize}; vkCmdCopyBuffer(cmd, stagV, m.vertexBuf, 1, &r);
    ctx.endOneTimeCommands(cmd);
    vkDestroyBuffer(ctx.device, stagV, nullptr); vkFreeMemory(ctx.device, stagVMem, nullptr);

    VkDeviceSize iSize = m.indices.size() * sizeof(uint32_t);
    VkBuffer stagI; VkDeviceMemory stagIMem;
    ctx.createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagI, stagIMem);
    vkMapMemory(ctx.device, stagIMem, 0, iSize, 0, &p);
    memcpy(p, m.indices.data(), iSize); vkUnmapMemory(ctx.device, stagIMem);
    ctx.createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.indexBuf, m.indexMem);
    cmd = ctx.beginOneTimeCommands();
    VkBufferCopy ir{0,0,iSize}; vkCmdCopyBuffer(cmd, stagI, m.indexBuf, 1, &ir);
    ctx.endOneTimeCommands(cmd);
    vkDestroyBuffer(ctx.device, stagI, nullptr); vkFreeMemory(ctx.device, stagIMem, nullptr);
}

void Scene3D::uploadMeshes(VulkanContext& ctx) {
    uploadOneMesh(ctx, indicatorMesh);
    for (auto& obj : objects) {
        if (!obj->isMesh()) continue;
        uploadOneMesh(ctx, *static_cast<MeshObject3D*>(obj.get()));
    }
}

static void destroyMeshBuffers(VkDevice device, MeshObject3D& m) {
    if (m.vertexBuf) { vkDestroyBuffer(device, m.vertexBuf, nullptr); m.vertexBuf = VK_NULL_HANDLE; }
    if (m.vertexMem) { vkFreeMemory(device, m.vertexMem, nullptr);    m.vertexMem = VK_NULL_HANDLE; }
    if (m.indexBuf)  { vkDestroyBuffer(device, m.indexBuf, nullptr);  m.indexBuf  = VK_NULL_HANDLE; }
    if (m.indexMem)  { vkFreeMemory(device, m.indexMem, nullptr);     m.indexMem  = VK_NULL_HANDLE; }
}

void Scene3D::cleanup(VkDevice device) {
    for (auto& obj : objects) {
        if (!obj->isMesh()) continue;
        destroyMeshBuffers(device, *static_cast<MeshObject3D*>(obj.get()));
    }
    destroyMeshBuffers(device, indicatorMesh);
    destroyPipelines(device);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkUnmapMemory(device, cameraUBOMem); vkFreeMemory(device, cameraUBOMem, nullptr);
    vkDestroyBuffer(device, cameraUBOBuf, nullptr);
    vkUnmapMemory(device, sdfSSBOMem); vkFreeMemory(device, sdfSSBOMem, nullptr);
    vkDestroyBuffer(device, sdfSSBOBuf, nullptr);
    vkUnmapMemory(device, lightUBOMem); vkFreeMemory(device, lightUBOMem, nullptr);
    vkDestroyBuffer(device, lightUBOBuf, nullptr);
}

// ─── Descriptors ──────────────────────────────────────────────────────────────

void Scene3D::createDescriptors(VulkanContext& ctx) {
    // Layout:
    //   binding 0 = Camera UBO (vertex + fragment)
    //   binding 1 = SDF SSBO  (fragment)
    //   binding 2 = Light UBO (fragment)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutCI.bindingCount = 3;
    layoutCI.pBindings    = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &layoutCI, nullptr, &descLayout);

    // Pool
    VkDescriptorPoolSize poolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},   // camera + light
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}    // SDF SSBO
    };
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.poolSizeCount = 2;
    poolCI.pPoolSizes    = poolSizes;
    poolCI.maxSets       = 1;
    vkCreateDescriptorPool(ctx.device, &poolCI, nullptr, &descPool);

    // Allocate set
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &descLayout;
    vkAllocateDescriptorSets(ctx.device, &allocInfo, &descSet);

    // Write all three bindings
    VkDescriptorBufferInfo camInfo  {cameraUBOBuf, 0, sizeof(CameraUBOData)};
    VkDeviceSize sdfBufSize = 4 * sizeof(int) + MAX_SDF_OBJECTS * sizeof(GPUSDFObject);
    VkDescriptorBufferInfo sdfInfo  {sdfSSBOBuf,   0, sdfBufSize};
    VkDescriptorBufferInfo lightInfo{lightUBOBuf,  0, sizeof(GPULightUBO)};

    VkWriteDescriptorSet writes[3] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sdfInfo, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                 descSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lightInfo, nullptr};

    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
}

// ─── Pipeline helpers ─────────────────────────────────────────────────────────

void Scene3D::destroyPipelines(VkDevice device) {
    if (meshPipeline)   { vkDestroyPipeline(device, meshPipeline, nullptr);       meshPipeline   = VK_NULL_HANDLE; }
    if (meshPipeLayout) { vkDestroyPipelineLayout(device, meshPipeLayout, nullptr); meshPipeLayout = VK_NULL_HANDLE; }
    if (sdfPipeline)    { vkDestroyPipeline(device, sdfPipeline, nullptr);        sdfPipeline    = VK_NULL_HANDLE; }
    if (sdfPipeLayout)  { vkDestroyPipelineLayout(device, sdfPipeLayout, nullptr); sdfPipeLayout  = VK_NULL_HANDLE; }
}

void Scene3D::createMeshPipeline(VulkanContext& ctx) {
    VkShaderModule vert = ctx.loadShader("shaders/scene_mesh.vert.spv");
    VkShaderModule frag = ctx.loadShader("shaders/scene_mesh.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    // Vertex input: pos(vec3) + normal(vec3) + uv(vec2)
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(Vertex3D);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribs[3] = {};
    attribs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex3D, pos)};
    attribs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex3D, normal)};
    attribs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex3D, uv)};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 1;    vi.pVertexBindingDescriptions   = &binding;
    vi.vertexAttributeDescriptionCount = 3;    vi.pVertexAttributeDescriptions = attribs;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D   sc{{0,0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1; vs.pViewports = &vp;
    vs.scissorCount  = 1; vs.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_BACK_BIT;
    // Mesh generators use CCW winding in Y-up world space.
    // GLM's proj[1][1]*=-1 Y-flip preserves CCW winding in screen space,
    // so CCW world-space → CCW screen-space → standard COUNTER_CLOCKWISE front face.
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    // Push constant: model matrix (64) + albedo (16) + roughness/metallic/emissive/pad (16) = 96 bytes
    VkPushConstantRange pcRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 96};

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount         = 1;  pli.pSetLayouts        = &descLayout;
    pli.pushConstantRangeCount = 1;  pli.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(ctx.device, &pli, nullptr, &meshPipeLayout);

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.stageCount          = 2;   pci.pStages          = stages;
    pci.pVertexInputState   = &vi; pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vs; pci.pRasterizationState = &rs;
    pci.pMultisampleState   = &ms; pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &cb;
    pci.layout              = meshPipeLayout;
    pci.renderPass          = ctx.renderPass;
    pci.subpass             = 0;
    vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr, &meshPipeline);

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

void Scene3D::createSDFPipeline(VulkanContext& ctx) {
    VkShaderModule vert = ctx.loadShader("shaders/fullscreen.vert.spv");
    VkShaderModule frag = ctx.loadShader("shaders/scene_sdf.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    // No vertex input (fullscreen triangle)
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)ctx.swapExtent.width, (float)ctx.swapExtent.height, 0, 1};
    VkRect2D   sc{{0,0}, ctx.swapExtent};
    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1; vs.pViewports = &vp;
    vs.scissorCount  = 1; vs.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth test + write enabled. SDF shader writes gl_FragDepth for correct compositing.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &descLayout;
    vkCreatePipelineLayout(ctx.device, &pli, nullptr, &sdfPipeLayout);

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.stageCount          = 2;   pci.pStages             = stages;
    pci.pVertexInputState   = &vi; pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vs; pci.pRasterizationState = &rs;
    pci.pMultisampleState   = &ms; pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &cb;
    pci.layout              = sdfPipeLayout;
    pci.renderPass          = ctx.renderPass;
    pci.subpass             = 0;
    vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr, &sdfPipeline);

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// ─── Per-frame updates ────────────────────────────────────────────────────────

void Scene3D::updateLightUBO() {
    GPULightUBO ubo{};
    ubo.numLights = (int)lights.size();
    for (int i = 0; i < (int)lights.size() && i < MAX_LIGHTS; ++i) {
        ubo.lights[i].direction = glm::vec4(glm::normalize(lights[i].direction), lights[i].intensity);
        ubo.lights[i].color     = glm::vec4(lights[i].color, 0);
    }
    memcpy(lightMapped, &ubo, sizeof(ubo));
}

void Scene3D::updateCameraUBO(VulkanContext& ctx) {
    float w = (float)ctx.swapExtent.width;
    float h = (float)ctx.swapExtent.height;
    CameraUBOData ubo{};
    ubo.view        = camera.viewMatrix();
    ubo.proj        = camera.projMatrix(w / h);
    ubo.viewProj    = ubo.proj * ubo.view;
    ubo.invViewProj = glm::inverse(ubo.viewProj);
    ubo.camPos      = glm::vec4(camera.pos, 1);
    ubo.screenParams = {w, h, camera.nearZ, camera.farZ};
    memcpy(cameraMapped, &ubo, sizeof(ubo));
}

void Scene3D::updateSdfSSBO() {
    // Pack header + objects
    int count = 0;
    std::array<GPUSDFObject, MAX_SDF_OBJECTS> gpuObjs{};
    for (auto& obj : objects) {
        if (obj->isMesh() || !obj->visible) continue;
        if (count >= MAX_SDF_OBJECTS) break;
        auto* sdf = static_cast<SDFObject3D*>(obj.get());
        GPUSDFObject& g = gpuObjs[count++];
        g.invModel  = glm::inverse(sdf->transform.matrix());
        g.params    = glm::vec4(sdf->params, 0);
        g.albedo    = glm::vec4(sdf->material.albedo, sdf->material.emissive);
        g.rm        = glm::vec4(sdf->material.roughness, sdf->material.metallic, 0, 0);
        g.shapeType = (int)sdf->shape;
    }
    // Write header (numObjects + 3 ints padding)
    int header[4] = {count, 0, 0, 0};
    uint8_t* p = static_cast<uint8_t*>(sdfMapped);
    memcpy(p, header, sizeof(header));
    memcpy(p + sizeof(header), gpuObjs.data(), count * sizeof(GPUSDFObject));
}

// ─── Light indicator rendering ────────────────────────────────────────────────

void Scene3D::renderLightIndicators(VkCommandBuffer cmd) {
    if (!indicatorMesh.vertexBuf) return;

    struct MeshPC {
        glm::mat4 model;
        glm::vec4 albedo;
        float roughness, metallic, emissive, _pad;
    } pc{};

    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &indicatorMesh.vertexBuf, &off);
    vkCmdBindIndexBuffer(cmd, indicatorMesh.indexBuf, 0, VK_INDEX_TYPE_UINT32);

    for (auto& light : lights) {
        if (!light.showDisc) continue;
        // Place disc at the light source position (toward the light, far away)
        glm::vec3 discPos = glm::normalize(light.direction) * light.discDist;
        pc.model     = glm::translate(glm::mat4(1), discPos) *
                       glm::scale(glm::mat4(1), glm::vec3(light.discRadius));
        pc.albedo    = glm::vec4(light.color, 1);
        pc.roughness = 1.0f;
        pc.metallic  = 0.0f;
        pc.emissive  = 2.5f;   // strongly emissive — visible regardless of light angle
        vkCmdPushConstants(cmd, meshPipeLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPC), &pc);
        vkCmdDrawIndexed(cmd, (uint32_t)indicatorMesh.indices.size(), 1, 0, 0, 0);
    }
}

// ─── Render ───────────────────────────────────────────────────────────────────

void Scene3D::render(VkCommandBuffer cmd, VulkanContext& ctx, float /*dt*/) {
    updateCameraUBO(ctx);
    updateSdfSSBO();
    updateLightUBO();

    struct MeshPC {
        glm::mat4 model;
        glm::vec4 albedo;
        float roughness, metallic, emissive, _pad;
    };

    // ── 1. Mesh objects (forward pass with depth) ─────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             meshPipeLayout, 0, 1, &descSet, 0, nullptr);

    for (auto& obj : objects) {
        if (!obj->isMesh() || !obj->visible) continue;
        auto* m = static_cast<MeshObject3D*>(obj.get());
        if (!m->vertexBuf) continue;

        MeshPC pc{};
        pc.model     = obj->transform.matrix();
        pc.albedo    = glm::vec4(obj->material.albedo, 1);
        pc.roughness = obj->material.roughness;
        pc.metallic  = obj->material.metallic;
        pc.emissive  = obj->material.emissive;
        vkCmdPushConstants(cmd, meshPipeLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPC), &pc);

        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m->vertexBuf, &off);
        vkCmdBindIndexBuffer(cmd, m->indexBuf, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, (uint32_t)m->indices.size(), 1, 0, 0, 0);
    }

    // ── 2. Light disc indicators ──────────────────────────────────────────
    renderLightIndicators(cmd);

    // ── 3. SDF fullscreen pass ────────────────────────────────────────────
    bool hasSDF = false;
    for (auto& obj : objects) if (!obj->isMesh() && obj->visible) { hasSDF = true; break; }
    if (hasSDF) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sdfPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 sdfPipeLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
}
