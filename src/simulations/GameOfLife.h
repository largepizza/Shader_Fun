#pragma once
#include "../Simulation.h"

class GameOfLife : public Simulation {
public:
    const char* name() const override { return "Game of Life  [SPACE = randomize]"; }

    void init(VulkanContext& ctx) override;
    void onResize(VulkanContext& ctx) override;
    void recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float dt) override;
    void recordDraw   (VkCommandBuffer cmd, VulkanContext& ctx, float dt) override;
    void buildUI(float dt, UIRenderer& ui) override;
    VkClearValue clearColor() const override { return {{{0.02f, 0.02f, 0.05f, 1.0f}}}; }
    void cleanup(VkDevice device) override;

    void onKey(GLFWwindow* window, int key, int action) override;

private:
    // Two rgba8 storage images for ping-pong (GENERAL layout throughout)
    VkImage        image[2]    = {};
    VkDeviceMemory memory[2]   = {};
    VkImageView    view[2]     = {};
    int            current     = 0;   // image[current] = live state

    VkSampler             sampler         = VK_NULL_HANDLE;
    VkDescriptorSetLayout compLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout dispLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      descPool        = VK_NULL_HANDLE;
    VkDescriptorSet       compSet[2]      = {};  // [i]: read img[i], write img[1-i]
    VkDescriptorSet       dispSet[2]      = {};  // [i]: sample from img[i]

    VkPipelineLayout  compPipeLayout  = VK_NULL_HANDLE;
    VkPipeline        compPipeline    = VK_NULL_HANDLE;
    VkPipelineLayout  dispPipeLayout  = VK_NULL_HANDLE;
    VkPipeline        dispPipeline    = VK_NULL_HANDLE;

    void createImages(VulkanContext& ctx);
    void createSampler(VulkanContext& ctx);
    void createDescriptors(VulkanContext& ctx);
    void createComputePipeline(VulkanContext& ctx);
    void createDisplayPipeline(VulkanContext& ctx);
    void randomize(VulkanContext& ctx);

    bool pendingRandomize = false;
};
