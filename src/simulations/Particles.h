#pragma once
#include "../Simulation.h"
#include <glm/glm.hpp>

struct Particle {
    glm::vec2 pos;
    glm::vec2 vel;
    glm::vec4 color;
};

struct ParticlePushConstants {
    float     dt;
    float     time;
    glm::vec2 mouseNDC;
};

class Particles : public Simulation {
public:
    const char* name() const override { return "Particles  [mouse = attract]"; }

    void init(VulkanContext& ctx) override;
    void onResize(VulkanContext& ctx) override;
    void recordFrame(VkCommandBuffer cmd, VkFramebuffer fb,
                     VulkanContext& ctx, float dt) override;
    void cleanup(VkDevice device) override;

    void onCursorPos(GLFWwindow* window, double x, double y) override;

private:
    VkBuffer              buffer      = VK_NULL_HANDLE;
    VkDeviceMemory        memory      = VK_NULL_HANDLE;

    VkDescriptorSetLayout layout      = VK_NULL_HANDLE;
    VkDescriptorPool      descPool    = VK_NULL_HANDLE;
    VkDescriptorSet       descSet     = VK_NULL_HANDLE;

    VkPipelineLayout compPipeLayout   = VK_NULL_HANDLE;
    VkPipeline       compPipeline     = VK_NULL_HANDLE;
    VkPipelineLayout drawPipeLayout   = VK_NULL_HANDLE;
    VkPipeline       drawPipeline     = VK_NULL_HANDLE;

    float     totalTime  = 0.0f;
    glm::vec2 mouseNDC   = {0.0f, 0.0f};

    void createParticleBuffer(VulkanContext& ctx);
    void createDescriptors(VulkanContext& ctx);
    void createComputePipeline(VulkanContext& ctx);
    void createDrawPipeline(VulkanContext& ctx);
    void initParticles(VulkanContext& ctx);
};
