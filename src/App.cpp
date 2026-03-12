#include "App.h"
#include <stdexcept>
#include <algorithm>

App::App(std::unique_ptr<Simulation> s) : sim(std::move(s)) {}

void App::run() {
    initWindow();
    ctx.init(window);
    sim->init(ctx);
    mainLoop();
    // Wait for GPU idle before tearing down
    vkDeviceWaitIdle(ctx.device);
    sim->cleanup(ctx.device);
    ctx.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void App::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIN_W, WIN_H, sim->name(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, cbResize);
    glfwSetKeyCallback(window, cbKey);
    glfwSetCursorPosCallback(window, cbCursorPos);
}

void App::mainLoop() {
    lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
}

void App::drawFrame() {
    vkWaitForFences(ctx.device, 1, &ctx.fenceFrame, VK_TRUE, UINT64_MAX);

    uint32_t imgIdx;
    VkResult res = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
                                          ctx.semImageAvailable, VK_NULL_HANDLE, &imgIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        ctx.recreateSwapchain(window);
        sim->onResize(ctx);
        return;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR failed.");

    // Compute dt here so it reflects the true frame gap
    double now = glfwGetTime();
    float  dt  = std::min((float)(now - lastTime), 0.05f); // clamp to 50ms max
    lastTime   = now;

    vkResetFences(ctx.device, 1, &ctx.fenceFrame);
    vkResetCommandBuffer(ctx.commandBuffer, 0);

    // Begin the command buffer, let the simulation fill it, then end it
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(ctx.commandBuffer, &bi);
    sim->recordFrame(ctx.commandBuffer, ctx.framebuffers[imgIdx], ctx, dt);
    vkEndCommandBuffer(ctx.commandBuffer);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &ctx.semImageAvailable;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &ctx.commandBuffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &ctx.semRenderDone[imgIdx];
    if (vkQueueSubmit(ctx.graphicsQueue, 1, &si, ctx.fenceFrame) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed.");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &ctx.semRenderDone[imgIdx];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &ctx.swapchain;
    pi.pImageIndices      = &imgIdx;
    res = vkQueuePresentKHR(ctx.graphicsQueue, &pi);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || resized) {
        resized = false;
        ctx.recreateSwapchain(window);
        sim->onResize(ctx);
    }
}

void App::cbResize(GLFWwindow* w, int, int) {
    reinterpret_cast<App*>(glfwGetWindowUserPointer(w))->resized = true;
}

void App::cbKey(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    auto* app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
    app->sim->onKey(w, key, action);
}

void App::cbCursorPos(GLFWwindow* w, double x, double y) {
    reinterpret_cast<App*>(glfwGetWindowUserPointer(w))->sim->onCursorPos(w, x, y);
}
