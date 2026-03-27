#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "VulkanContext.h"

class UIRenderer;   // forward declare — simulations include UIRenderer.h in their .cpp
class AudioSystem;  // forward declare — simulations include AudioSystem.h in their .cpp

// Abstract base class for all simulations.
// To add a new simulation:
//   1. Create MySimulation.h/.cpp in src/simulations/
//   2. Inherit from Simulation and implement all pure virtual methods
//   3. In main.cpp, construct App with std::make_unique<MySimulation>()
class Simulation {
public:
    virtual ~Simulation() = default;

    // Called once after Vulkan context is ready.
    // Create all GPU resources (images, buffers, pipelines, descriptors) here.
    virtual void init(VulkanContext& ctx) = 0;

    // Called when the swapchain is recreated (e.g. window resize).
    // Recreate any pipeline that bakes in the viewport size.
    // Compute pipelines are viewport-independent and do NOT need recreation.
    virtual void onResize(VulkanContext& ctx) = 0;

    // Record compute work into an already-begun command buffer, before the render pass.
    // Default implementation does nothing (simulations with no compute override this).
    // dt = seconds since last frame.
    virtual void recordCompute(VkCommandBuffer cmd, VulkanContext& ctx, float dt) {}

    // Record draw calls into an already-begun command buffer inside an open render pass.
    // The render pass is begun and ended by App; do NOT call vkCmdBeginRenderPass here.
    // dt = seconds since last frame.
    virtual void recordDraw(VkCommandBuffer cmd, VulkanContext& ctx, float dt) = 0;

    // Declare Clay UI layout elements for this frame.
    // Called between UIRenderer::beginFrame() and UIRenderer::record().
    // ui provides per-frame input state (mouse, buttons) and the mouse-capture API.
    // Use CLAY() and CLAY_TEXT() macros to emit layout elements.
    // Default implementation declares no UI.
    virtual void buildUI(float dt, UIRenderer& ui) {}

    // Clear color used when App begins the render pass.
    virtual VkClearValue clearColor() const { return {{{0.0f, 0.0f, 0.0f, 1.0f}}}; }

    // Called once before the device is destroyed. Release all Vulkan resources.
    virtual void cleanup(VkDevice device) = 0;

    // Optional input handlers — default implementations do nothing.
    virtual void onKey(GLFWwindow* window, int key, int action) {}
    virtual void onCursorPos(GLFWwindow* window, double x, double y) {}

    // Called by App after both the simulation and AudioSystem are initialised.
    // Override to configure the playlist and store the pointer for use in buildUI.
    // Default: no-op (simulations without audio ignore this).
    virtual void setAudio(AudioSystem* /*audio*/) {}

    // Window title shown while this simulation runs
    virtual const char* name() const { return "ShaderFun"; }
};
