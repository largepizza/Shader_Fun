#include "App.h"
#include <stdexcept>
#include <algorithm>

App::App(std::unique_ptr<Simulation> s) : sim(std::move(s)) {}

void App::run() {
    initWindow();
    ctx.init(window);
    sim->init(ctx);
    audio.init();
    sim->setAudio(&audio);  // let the simulation configure its playlist
    ui.init(ctx);
    mainLoop();
    // Wait for GPU idle before tearing down
    vkDeviceWaitIdle(ctx.device);
    audio.cleanup();
    ui.cleanup(ctx.device);
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
    glfwSetScrollCallback(window, cbScroll);
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
        ui.onResize(ctx);
        return;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR failed.");

    // Compute dt
    double now = glfwGetTime();
    float  dt  = std::min((float)(now - lastTime), 0.05f);
    lastTime   = now;

    // Get current mouse state
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    int  ww, wh;
    glfwGetWindowSize(window, &ww, &wh);

    // Prepare Clay layout for this frame — simulation may call CLAY() in buildUI()
    ui.beginFrame((float)ww, (float)wh,
                  (float)mx, (float)my, lmb, rmb,
                  scrollX, scrollY, dt);
    scrollX = scrollY = 0.0f; // consumed

    // Advance music playlist (detects track end, starts next track).
    audio.update(dt);

    // Let the simulation declare its UI elements and read input state via ui.
    sim->buildUI(dt, ui);

    // Record GPU commands
    vkResetFences(ctx.device, 1, &ctx.fenceFrame);
    vkResetCommandBuffer(ctx.commandBuffer, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(ctx.commandBuffer, &bi);

    // 1. Simulation compute work (before render pass)
    sim->recordCompute(ctx.commandBuffer, ctx, dt);

    // 2. Begin render pass — App now owns this
    VkClearValue clearValues[2];
    clearValues[0] = sim->clearColor();
    clearValues[1].depthStencil = {1.0f, 0};   // far depth = 1.0
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = ctx.renderPass;
    rbi.framebuffer     = ctx.framebuffers[imgIdx];
    rbi.renderArea      = {{0, 0}, ctx.swapExtent};
    rbi.clearValueCount = 2;
    rbi.pClearValues    = clearValues;
    vkCmdBeginRenderPass(ctx.commandBuffer, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    // 3. Simulation draw calls (render pass is already open)
    sim->recordDraw(ctx.commandBuffer, ctx, dt);

    // 4. UI draws on top of the simulation
    ui.record(ctx.commandBuffer, ctx);

    vkCmdEndRenderPass(ctx.commandBuffer);
    vkEndCommandBuffer(ctx.commandBuffer);

    // Submit
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

    // Present
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
        ui.onResize(ctx);
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

void App::cbScroll(GLFWwindow* w, double dx, double dy) {
    auto* app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
    app->scrollX += (float)dx;
    app->scrollY += (float)dy;
}
