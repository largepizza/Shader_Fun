#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <memory>
#include <string>
#include <vector>

#include "VulkanContext.h"
#include "Camera3D.h"

// ─── Vertex format ─────────────────────────────────────────────────────────────
struct Vertex3D {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

// ─── Transform ─────────────────────────────────────────────────────────────────
struct Transform3D {
    glm::vec3 position = {0, 0, 0};
    glm::vec3 rotation = {0, 0, 0}; // Euler angles, degrees (X=pitch, Y=yaw, Z=roll)
    glm::vec3 scale    = {1, 1, 1};

    glm::mat4 matrix() const {
        glm::mat4 m = glm::translate(glm::mat4(1), position);
        m = glm::rotate(m, glm::radians(rotation.y), {0, 1, 0});
        m = glm::rotate(m, glm::radians(rotation.x), {1, 0, 0});
        m = glm::rotate(m, glm::radians(rotation.z), {0, 0, 1});
        m = glm::scale(m, scale);
        return m;
    }
};

// ─── Material ─────────────────────────────────────────────────────────────────
struct Material3D {
    glm::vec3 albedo    = {1, 1, 1};
    float     roughness = 0.5f;
    float     metallic  = 0.0f;
    float     emissive  = 0.0f;
};

// ─── SDF shape types ──────────────────────────────────────────────────────────
enum class SDFShape { Sphere = 0, Box = 1, Torus = 2, Capsule = 3 };

// ─── Scene object base ────────────────────────────────────────────────────────
class SceneObject3D {
public:
    std::string name;
    Transform3D transform;
    Material3D  material;
    bool        visible = true;
    virtual ~SceneObject3D() = default;
    virtual bool isMesh() const = 0;
};

// ─── Mesh object ──────────────────────────────────────────────────────────────
class MeshObject3D : public SceneObject3D {
public:
    std::vector<Vertex3D>  vertices;
    std::vector<uint32_t>  indices;

    // GPU buffers — managed by Scene3D::uploadMeshes()
    VkBuffer       vertexBuf = VK_NULL_HANDLE;
    VkDeviceMemory vertexMem = VK_NULL_HANDLE;
    VkBuffer       indexBuf  = VK_NULL_HANDLE;
    VkDeviceMemory indexMem  = VK_NULL_HANDLE;

    bool isMesh() const override { return true; }
};

// ─── SDF object ──────────────────────────────────────────────────────────────
// params usage by shape:
//   Sphere:  params.x = radius
//   Box:     params.xyz = half-extents
//   Torus:   params.x = major radius (ring), params.y = minor radius (tube)
//   Capsule: params.x = half-height (cylinder part), params.y = radius
class SDFObject3D : public SceneObject3D {
public:
    SDFShape  shape  = SDFShape::Sphere;
    glm::vec3 params = {1, 0, 0};
    bool isMesh() const override { return false; }
};

// ─── GPU-side SDF data (must match scene_sdf.frag layout exactly) ─────────────
struct GPUSDFObject {
    glm::mat4 invModel;   // inverse model matrix for object-space SDF
    glm::vec4 params;     // shape params (see SDFObject3D::params above)
    glm::vec4 albedo;     // xyz = albedo, w = emissive
    glm::vec4 rm;         // x = roughness, y = metallic
    int       shapeType;
    int       pad[3];
};

// Maximum SDF objects per scene
static constexpr int MAX_SDF_OBJECTS = 32;

// ─── Directional light ────────────────────────────────────────────────────────
// Models a sun-like source: parallel rays, infinite distance.
// 'direction' is the unit vector FROM the surface TOWARD the light (L in BRDF notation).
struct DirectionalLight {
    glm::vec3 direction  = glm::normalize(glm::vec3(0.6f, 1.0f, 0.4f));
    glm::vec3 color      = {1.0f, 0.92f, 0.78f};
    float     intensity  = 1.1f;
    // Visual indicator: a small emissive sphere placed at 'direction * discDist'
    bool      showDisc   = true;
    float     discDist   = 28.0f;  // world-space units from origin
    float     discRadius = 0.4f;   // sphere radius for the indicator
};

static constexpr int MAX_LIGHTS = 4;

// GPU layout (std140, must match both 3D shaders)
struct GPULight {
    glm::vec4 direction;  // xyz = toward-light direction, w = intensity
    glm::vec4 color;      // xyz = color, w = unused
};
struct GPULightUBO {
    int      numLights;
    int      pad[3];
    GPULight lights[MAX_LIGHTS];
};

// ─── Camera UBO (must match all 3D shaders) ───────────────────────────────────
struct CameraUBOData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 invViewProj;
    glm::vec4 camPos;       // xyz = world position
    glm::vec4 screenParams; // x = width, y = height, z = near, w = far
};

// ─── Scene3D ──────────────────────────────────────────────────────────────────
// Framework class: manages a list of SceneObject3Ds and their Vulkan resources.
// Use addMesh() / addSDF() to populate the scene, then call init() once and
// render() every frame from your simulation's recordDraw().
class Scene3D {
public:
    Camera3D camera;
    std::vector<std::shared_ptr<SceneObject3D>> objects;

    // Directional lights (up to MAX_LIGHTS)
    std::vector<DirectionalLight> lights;

    // ── Object factory ────────────────────────────────────────────────────
    std::shared_ptr<MeshObject3D> addMesh(const std::string& name,
                                          std::vector<Vertex3D> verts,
                                          std::vector<uint32_t> idxs);

    std::shared_ptr<SDFObject3D>  addSDF(const std::string& name,
                                         SDFShape shape,
                                         glm::vec3 params);

    DirectionalLight& addLight(glm::vec3 direction,
                               glm::vec3 color    = {1.0f, 0.92f, 0.78f},
                               float     intensity = 1.1f);

    // ── Mesh generators (static helpers) ─────────────────────────────────
    static std::pair<std::vector<Vertex3D>, std::vector<uint32_t>>
        makeBox(glm::vec3 half);
    static std::pair<std::vector<Vertex3D>, std::vector<uint32_t>>
        makeSphere(float radius, int rings = 24, int sectors = 32);
    static std::pair<std::vector<Vertex3D>, std::vector<uint32_t>>
        makeGrid(float halfSize, int divisions);

    // ── Vulkan lifecycle ──────────────────────────────────────────────────
    void init(VulkanContext& ctx);       // call once after Vulkan is ready
    void onResize(VulkanContext& ctx);   // call on swapchain recreation
    void uploadMeshes(VulkanContext& ctx); // upload any un-uploaded mesh objects
    void render(VkCommandBuffer cmd, VulkanContext& ctx, float dt);
    void cleanup(VkDevice device);

private:
    // ── Descriptor layout shared by mesh + SDF pipelines ─────────────────
    //   Binding 0: Camera UBO   (vertex + fragment)
    //   Binding 1: SDF SSBO     (fragment only)
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descPool   = VK_NULL_HANDLE;
    VkDescriptorSet       descSet    = VK_NULL_HANDLE;

    // ── Camera UBO (persistently mapped) ─────────────────────────────────
    VkBuffer       cameraUBOBuf = VK_NULL_HANDLE;
    VkDeviceMemory cameraUBOMem = VK_NULL_HANDLE;
    void*          cameraMapped = nullptr;

    // ── SDF SSBO (persistently mapped) ───────────────────────────────────
    VkBuffer       sdfSSBOBuf = VK_NULL_HANDLE;
    VkDeviceMemory sdfSSBOMem = VK_NULL_HANDLE;
    void*          sdfMapped  = nullptr;

    // ── Light UBO (persistently mapped) ──────────────────────────────────
    VkBuffer       lightUBOBuf = VK_NULL_HANDLE;
    VkDeviceMemory lightUBOMem = VK_NULL_HANDLE;
    void*          lightMapped = nullptr;

    // ── Internal indicator sphere (reused for all light discs) ───────────
    MeshObject3D indicatorMesh;

    // ── Mesh pipeline ─────────────────────────────────────────────────────
    VkPipelineLayout meshPipeLayout = VK_NULL_HANDLE;
    VkPipeline       meshPipeline   = VK_NULL_HANDLE;

    // ── SDF fullscreen pipeline ───────────────────────────────────────────
    VkPipelineLayout sdfPipeLayout = VK_NULL_HANDLE;
    VkPipeline       sdfPipeline   = VK_NULL_HANDLE;

    void createDescriptors(VulkanContext& ctx);
    void createMeshPipeline(VulkanContext& ctx);
    void createSDFPipeline(VulkanContext& ctx);
    void destroyPipelines(VkDevice device);
    void updateCameraUBO(VulkanContext& ctx);
    void updateSdfSSBO();
    void updateLightUBO();
    void renderLightIndicators(VkCommandBuffer cmd);
};
