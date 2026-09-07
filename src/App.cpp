#include "App.h"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

// SATLIGHTSIM_FRAME_TRACE=1 → print a per-phase CPU wall-clock breakdown of drawFrame() every
// 60 frames. Pins down whether the frame is lost in vkWaitForFences (GPU busy), vkQueueSubmit
// (MoltenVK synchronous encode+submit), vkQueuePresentKHR (compositor/vsync stall), or the
// command recording itself.
namespace {
bool frameTraceEnabled() {
    static int v = -1;
    if (v < 0) { const char *e = std::getenv("SATLIGHTSIM_FRAME_TRACE"); v = (e && e[0] == '1') ? 1 : 0; }
    return v == 1;
}
}

App::App(std::unique_ptr<Simulation> s) : sim(std::move(s)) {}

void App::run() {
    initWindow();
    ctx.init(window);
    sim->init(ctx);
    sim->setWindow(window);  // give sim access to window handle (e.g. fullscreen toggle)
    audio.init();
    sim->setAudio(&audio);  // let the simulation configure its playlist
    ui.init(ctx, window);
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
    // Boot maximized (windowed, not exclusive fullscreen) rather than at the fixed WIN_W x WIN_H
    // default — a small window the player has to manually enlarge undercuts the intro cinematic's
    // impact and invites fiddling with the window instead of watching it. WIN_W/WIN_H are still
    // passed as the restore size for whenever the player un-maximizes later.
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    // macOS: render at logical (point) resolution, not the 2x Retina backing size. On the
    // integrated/older discrete GPUs these machines have, a maximized Retina framebuffer is
    // ~4x the pixels and the volumetric passes can't keep up. Also makes the fixed-bitmap
    // font atlas pixel-exact instead of upscaled. No-op on non-Apple platforms.
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
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
        double frameStart = glfwGetTime();
        glfwPollEvents();
        drawFrame();

        // NEW-7: manual pacing for numeric FPS caps (see Simulation::targetFpsCap comment —
        // FIFO/V-Sync already paces itself, this only fires for the 30/60/120 caps).
        float capHz = sim->targetFpsCap();
        if (capHz > 0.0f) {
            double budget = 1.0 / (double)capHz;
            double remaining = budget - (glfwGetTime() - frameStart);
            if (remaining > 0.0)
                std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
        }
    }
}

void App::drawFrame() {
    const bool ft = frameTraceEnabled();
    static uint64_t ftFrame = 0;
    const bool ftPrint = ft && (ftFrame % 60 == 0);
    double ftT0 = ft ? glfwGetTime() : 0.0, ftPrev = ftT0;
    auto ftMark = [&](const char *name) {
        if (!ftPrint) return;
        double now = glfwGetTime();
        std::printf("  %-14s %7.1f ms\n", name, (now - ftPrev) * 1000.0);
        ftPrev = now;
    };
    int ftRecreated = 0;

    vkWaitForFences(ctx.device, 1, &ctx.fenceFrame, VK_TRUE, UINT64_MAX);
    ftMark("waitFences");

    // Resolve last frame's GPU timestamp queries now that the fence proves the GPU
    // is done with them. Skipped on the very first call: the fence starts pre-signaled
    // so nothing has actually been submitted yet, and there'd be no query data to read.
    if (submittedOnce) {
        ctx.resolveTimestamps();
        // UC6: same reasoning — a screenshot copy recorded last frame is only safe to read back
        // once this fence wait proves the GPU has finished writing it.
        sim->finalizeScreenshot();
    }

    uint32_t imgIdx;
    VkResult res = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
                                          ctx.semImageAvailable, VK_NULL_HANDLE, &imgIdx);
    ftMark("acquire");
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        ctx.recreateSwapchain(window);
        sim->onResize(ctx);
        ui.onResize(ctx);
        if (ft) { ++ftFrame; std::printf("[frame-trace] acquire OUT_OF_DATE -> full swapchain recreate\n"); }
        return;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR failed.");

    // Compute dt. Clamped only against genuine multi-second hitches (first-frame shader
    // compile, window resize, breakpoint) — NOT against a merely slow GPU. The old 0.05s
    // (20 fps) ceiling silently pinned the HUD fps badge and every perf snapshot at 20 on
    // hardware actually running slower, and ran the sim in slow-motion to match. 1.0s (1 fps)
    // still bounds a pathological stall while letting true frame times through even on very
    // slow setups (MoltenVK translation on old GPUs can genuinely land here).
    double now = glfwGetTime();
    float  dt  = std::min((float)(now - lastTime), 1.0f);
    lastTime   = now;

    // Get current mouse state
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    int  ww, wh;
    glfwGetWindowSize(window, &ww, &wh);

    // While the camera has the cursor captured (GLFW_CURSOR_DISABLED, used for
    // RMB mouse-look), GLFW reports an unbounded "virtual" position that free-drifts
    // far outside the window as the user pans. Feeding that raw value to the UI made
    // Clay's hit-testing register phantom hovers/clicks (rollover blips, accidental
    // keybind rebinds) on whatever always-visible panel the drifted coordinate landed
    // on. Freeze the UI-facing cursor at its last known on-screen position instead.
    if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
        mx = uiMouseX;
        my = uiMouseY;
    } else {
        uiMouseX = mx;
        uiMouseY = my;
    }

    // UC4: gamepad virtual cursor — when active, overrides the real mouse position/click state
    // for this frame's UI pass so every existing Clay_Hovered()/click handler works unmodified.
    float vx, vy;
    bool  vClick;
    if (sim->virtualCursor(vx, vy, vClick)) {
        mx  = vx;
        my  = vy;
        lmb = vClick;
    }

    // Prepare Clay layout for this frame — simulation may call CLAY() in buildUI()
    ui.beginFrame((float)ww, (float)wh,
                  (float)mx, (float)my, lmb, rmb,
                  scrollX, scrollY, dt);
    scrollX = scrollY = 0.0f; // consumed

    // Advance music playlist (detects track end, starts next track).
    audio.update(dt);

    // Let the simulation declare its UI elements and read input state via ui.
    sim->buildUI(dt, ui);
    ftMark("buildUI");

    // Record GPU commands
    vkResetFences(ctx.device, 1, &ctx.fenceFrame);
    vkResetCommandBuffer(ctx.commandBuffer, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(ctx.commandBuffer, &bi);

    // GPU timestamp profiling: slot 0 marks frame start. Slots 1-5 are written inside
    // sim->recordCompute() (compute-pass breakdown: scene depth, beam cloud block, orbit compute,
    // cloud march, flare compute); slot 6 is written inside sim->recordPrePass() or
    // sim->recordDraw() (end of the sky background pass, whichever path rendered it).
    // Slots 7-8 mark the end of the satellite+star draw and the UI overlay respectively.
    // See VulkanContext::kTimestampCount for the authoritative slot table.
    ctx.resetTimestamps(ctx.commandBuffer);
    ctx.writeTimestamp(ctx.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);

    // 1. Simulation compute work (before render pass)
    sim->recordCompute(ctx.commandBuffer, ctx, dt);
    ftMark("recordCompute");

    // 1b. Optional offscreen pre-pass (e.g. a low-res background blitted into the swapchain
    // image ahead of time) — must run before the main render pass begins. Default: no-op.
    sim->recordPrePass(ctx.commandBuffer, ctx, dt, imgIdx);

    // 2. Begin render pass — App now owns this
    VkClearValue clearValues[2];
    clearValues[0] = sim->clearColor();
    clearValues[1].depthStencil = {1.0f, 0};   // far depth = 1.0
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = sim->activeRenderPass(ctx);
    rbi.framebuffer     = ctx.framebuffers[imgIdx];
    rbi.renderArea      = {{0, 0}, ctx.swapExtent};
    rbi.clearValueCount = 2;
    rbi.pClearValues    = clearValues;
    vkCmdBeginRenderPass(ctx.commandBuffer, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    // 3. Simulation draw calls (render pass is already open)
    sim->recordDraw(ctx.commandBuffer, ctx, dt);
    ctx.writeTimestamp(ctx.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 7);

    // 4. UI draws on top of the simulation — UC6 clean-shot mode skips this for a captured frame
    // (nobody shares a screenshot with a settings panel in it).
    if (!sim->wantsCleanScreenshot())
        ui.record(ctx.commandBuffer, ctx);
    ctx.writeTimestamp(ctx.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 8);

    vkCmdEndRenderPass(ctx.commandBuffer);

    // UC6: record the screenshot copy (a no-op unless one is actually pending) now that the
    // swapchain image holds the fully composited frame — must happen after the render pass ends
    // (the image is back in a layout a transfer op can act on) and before the command buffer is
    // ended, since this project has one command buffer / one frame in flight.
    sim->recordScreenshotCopy(ctx.commandBuffer, ctx, ctx.swapImages[imgIdx]);

    vkEndCommandBuffer(ctx.commandBuffer);
    ftMark("record rest");

    // Submit
    // Includes TRANSFER now (not just COLOR_ATTACHMENT_OUTPUT): a simulation's recordPrePass may
    // blit directly into the swapchain image before the main render pass opens (resolution
    // scaling) — that write must also wait for the presentation engine to be done with this
    // image, the same guarantee the render pass itself already got from this semaphore.
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
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
    submittedOnce = true;
    ftMark("queueSubmit");

    // Present
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &ctx.semRenderDone[imgIdx];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &ctx.swapchain;
    pi.pImageIndices      = &imgIdx;
    res = vkQueuePresentKHR(ctx.graphicsQueue, &pi);
    ftMark("queuePresent");
    // NOTE: VK_SUBOPTIMAL_KHR is deliberately NOT a trigger here. MoltenVK returns it on almost
    // every present on Retina/scaled displays, and recreateSwapchain() rebuilds every graphics
    // pipeline (viewport is baked in) — on MoltenVK that re-converts SPIR-V→MSL and recompiles
    // sat_sky.frag et al. into fresh MTLRenderPipelineStates, ~hundreds of ms. Doing that every
    // frame pinned the whole app at a few fps. SUBOPTIMAL still presents correctly; only a real
    // size change (resized flag) or OUT_OF_DATE actually needs a rebuild. The acquire path above
    // already ignores SUBOPTIMAL for the same reason.
    if (res == VK_ERROR_OUT_OF_DATE_KHR || resized || sim->consumeSwapchainRebuildRequest()) {
        resized = false;
        ctx.recreateSwapchain(window);
        sim->onResize(ctx);
        ui.onResize(ctx);
        ftRecreated = 1;
    }
    ftMark("post-present");

    if (ftPrint) {
        std::printf("[frame-trace] frame %llu  present_res=%d  recreated=%d  TOTAL %.1f ms\n\n",
                    (unsigned long long)ftFrame, (int)res, ftRecreated,
                    (glfwGetTime() - ftT0) * 1000.0);
        std::fflush(stdout);
    }
    if (ft) ++ftFrame;
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
