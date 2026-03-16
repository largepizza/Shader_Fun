#pragma once
#include "../Simulation.h"
#include <glm/glm.hpp>
#include <cstdint>

// Must match push_constant block in neural_update.comp exactly.
struct NeuralPushConstants
{
    float time;
    float learnRate;
    float traceDecay; // eligibility trace decay
    int32_t gridW;
    int32_t gridH;
    int32_t step;
};

// Must match push_constant block in pong_step.comp exactly.
struct PongPushConstants
{
    float dt;
    float paddleH;
    float paddleSpeed;
    float ballSpeed;
    int32_t gridW;
    int32_t gridH;
};

// Must match push_constant block in neural_display.frag exactly.
struct NeuralDisplayPC
{
    int32_t gridW;
    int32_t gridH;
    int32_t viewMode;
};

// Must match SSBO layout in pong_step.comp / neural_update.comp / neural_display.frag.
struct PongState
{
    float ball_x = 0.5f;
    float ball_y = 0.5f;
    float ball_vx = -0.35f;
    float ball_vy = 0.0f;
    float ai_paddle_y = 0.5f;
    float opp_paddle_y = 0.5f;
    float reward = 0.0f;
    int32_t ai_score = 0;
    int32_t opp_score = 0;
    float rewardDisplay = 0.0f;
};

class NeuralCA : public Simulation
{
public:
    const char *name() const override { return "Neural CA Pong  [R=reset  1/2/3=view]"; }

    void init(VulkanContext &ctx) override;
    void onResize(VulkanContext &ctx) override;
    void recordCompute(VkCommandBuffer cmd, VulkanContext &ctx, float dt) override;
    void recordDraw(VkCommandBuffer cmd, VulkanContext &ctx, float dt) override;
    void buildUI(float dt, UIRenderer &ui) override;
    VkClearValue clearColor() const override { return {{{0.0f, 0.0f, 0.02f, 1.0f}}}; }
    void cleanup(VkDevice device) override;

    void onKey(GLFWwindow *window, int key, int action) override;

private:
    static constexpr uint32_t GRID_W = 10;
    static constexpr uint32_t GRID_H = 64;

    // Ping-pong images (RGBA32F: r=activation, g=weight, b=trace, a=bias)
    VkImage image[2] = {};
    VkDeviceMemory memory[2] = {};
    VkImageView view[2] = {};
    VkSampler sampler = VK_NULL_HANDLE;

    // Pong state SSBO (host-visible so scores can be read for UI)
    VkBuffer pongBuf = VK_NULL_HANDLE;
    VkDeviceMemory pongMem = VK_NULL_HANDLE;
    PongState *pongMapped = nullptr;

    // Pong step compute pipeline
    VkDescriptorSetLayout pongLayout = VK_NULL_HANDLE;
    VkDescriptorSet pongSet[2] = {};
    VkPipelineLayout pongPipeLayout = VK_NULL_HANDLE;
    VkPipeline pongPipeline = VK_NULL_HANDLE;

    // Neural update compute pipeline
    VkDescriptorSetLayout compLayout = VK_NULL_HANDLE;
    VkDescriptorSet compSet[2] = {};
    VkPipelineLayout compPipeLayout = VK_NULL_HANDLE;
    VkPipeline compPipeline = VK_NULL_HANDLE;

    // Display pipeline
    VkDescriptorSetLayout dispLayout = VK_NULL_HANDLE;
    VkDescriptorSet dispSet[2] = {};
    VkPipelineLayout dispPipeLayout = VK_NULL_HANDLE;
    VkPipeline dispPipeline = VK_NULL_HANDLE;

    VkDescriptorPool descPool = VK_NULL_HANDLE;

    int current = 0;
    int32_t step = 0;
    float totalTime = 0.0f;
    bool pendingReset = false;

    // UI-controlled parameters
    float learnRate = 0.02f;
    float traceDecay = 0.998f;
    int32_t viewMode = 0;
    int32_t stepsPerFrame = 4;

    void createImages(VulkanContext &ctx);
    void createSampler(VulkanContext &ctx);
    void createPongBuffer(VulkanContext &ctx);
    void createDescriptors(VulkanContext &ctx);
    void createPongPipeline(VulkanContext &ctx);
    void createComputePipeline(VulkanContext &ctx);
    void createDisplayPipeline(VulkanContext &ctx);
    void resetGrid(VulkanContext &ctx);
};
