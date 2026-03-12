#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "VulkanContext.h"

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

    // Record all GPU work for this frame into an already-begun command buffer.
    // Responsible for: compute dispatch, pipeline barriers, begin/end render pass, draw calls.
    // dt = seconds since last frame.
    virtual void recordFrame(VkCommandBuffer cmd,
                              VkFramebuffer  framebuffer,
                              VulkanContext& ctx,
                              float          dt) = 0;

    // Called once before the device is destroyed. Release all Vulkan resources.
    virtual void cleanup(VkDevice device) = 0;

    // Optional input handlers — default implementations do nothing.
    virtual void onKey(GLFWwindow* window, int key, int action) {}
    virtual void onCursorPos(GLFWwindow* window, double x, double y) {}

    // Window title shown while this simulation runs
    virtual const char* name() const { return "ShaderFun"; }
};
