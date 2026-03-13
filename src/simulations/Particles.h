#pragma once
#include "../Simulation.h"
#include <glm/glm.hpp>

struct Particle {
    glm::vec2 pos;
    glm::vec2 vel;
    glm::vec4 color; // set fresh every compute frame based on colorMode
};

// Must match the push_constant block in particles_update.comp exactly.
struct ParticlePushConstants {
    float     dt;
    float     time;
    glm::vec2 mouseNDC;
    float     forceStrength; // 0 = no force, +1 = attract, -1 = repulse
    float     viscosity;     // velocity damping per second (0.9..0.999)
    int32_t   colorMode;     // 0 = rainbow, 1 = velocity, 2 = uniform
};

enum class ParticleTool { Attract = 0, Repulse = 1 };

// Draggable / closeable window state.
struct UIWindowState {
    bool  open    = false;
    float x       = 60.0f;
    float y       = 60.0f;
    bool  dragging = false;
};

class Particles : public Simulation {
public:
    const char* name() const override { return "Particles"; }

    void init(VulkanContext& ctx) override;
    void onResize(VulkanContext& ctx) override;
    void recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float dt) override;
    void recordDraw   (VkCommandBuffer cmd, VulkanContext& ctx, float dt) override;
    void buildUI(float dt, UIRenderer& ui) override;
    VkClearValue clearColor() const override { return {{{0.01f, 0.01f, 0.02f, 1.0f}}}; }
    void cleanup(VkDevice device) override;

    void onCursorPos(GLFWwindow* window, double x, double y) override;

private:
    // ── Vulkan resources ──────────────────────────────────────────────────
    VkBuffer              buffer      = VK_NULL_HANDLE;
    VkDeviceMemory        memory      = VK_NULL_HANDLE;

    VkDescriptorSetLayout layout      = VK_NULL_HANDLE;
    VkDescriptorPool      descPool    = VK_NULL_HANDLE;
    VkDescriptorSet       descSet     = VK_NULL_HANDLE;

    VkPipelineLayout compPipeLayout   = VK_NULL_HANDLE;
    VkPipeline       compPipeline     = VK_NULL_HANDLE;
    VkPipelineLayout drawPipeLayout   = VK_NULL_HANDLE;
    VkPipeline       drawPipeline     = VK_NULL_HANDLE;

    // ── Simulation state ──────────────────────────────────────────────────
    float     totalTime    = 0.0f;
    glm::vec2 mouseNDC     = {0.0f, 0.0f};
    bool      pendingReset = false;

    // ── Per-frame values computed in buildUI, consumed in recordCompute ───
    float   forceStrength = 0.0f; // +1=attract, -1=repulse, 0=none
    float   viscosity     = 0.985f;
    int32_t colorMode     = 0;

    // ── Toolbar / tool selection ──────────────────────────────────────────
    ParticleTool activeTool = ParticleTool::Attract;

    // ── Settings window ───────────────────────────────────────────────────
    UIWindowState settingsWin;

    // Slider drag state (viscosity slider)
    bool visSliderDrag = false;

    // Small string buffer for viscosity value display — must outlive buildUI()
    // because Clay stores a raw pointer and reads it in record() via Clay_EndLayout.
    char viscBuf[8] = "0.985";

    // One-frame-lag hover state for dynamic button background colours
    bool hovAttract  = false;
    bool hovRepulse  = false;
    bool hovSettings = false;
    bool hovClose    = false;
    bool hovReset    = false;
    bool hovColor[3] = {};

    // ── Private helpers ───────────────────────────────────────────────────
    void createParticleBuffer(VulkanContext& ctx);
    void createDescriptors(VulkanContext& ctx);
    void createComputePipeline(VulkanContext& ctx);
    void createDrawPipeline(VulkanContext& ctx);
    void initParticles(VulkanContext& ctx);
};
