#pragma once
#define GLFW_INCLUDE_VULKAN
// GLFW_INCLUDE_VULKAN only adds <vulkan/vulkan.h> — glfw3.h still pulls in the platform's
// default OpenGL header (<GL/gl.h> on Linux, <OpenGL/gl.h> on macOS) unless told not to.
// This project is Vulkan-only and never touches OpenGL; Windows/macOS SDKs happen to ship
// that header so this was silently masked there, but it broke the Linux CI build (no
// GL/gl.h without an extra Mesa dev package) — GLFW_INCLUDE_NONE is the real fix, not
// papering over it with an apt-get install.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <optional>
#include <string>
#include <vector>

// All constants live here so simulations can reference them
constexpr uint32_t WIN_W          = 1280;
constexpr uint32_t WIN_H          = 720;
constexpr uint32_t GOL_W          = 512;
constexpr uint32_t GOL_H          = 512;
constexpr uint32_t PARTICLE_COUNT = 500'000;

// ── Internal helper types ─────────────────────────────────────────────────────
// Declared here so member function return types compile cleanly under MSVC.
// Treat these as private implementation details of VulkanContext.
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> compute;
    bool complete() const { return graphics.has_value() && compute.has_value(); }
};
struct SwapchainDetails {
    VkSurfaceCapabilitiesKHR         caps;
    std::vector<VkSurfaceFormatKHR>  formats;
    std::vector<VkPresentModeKHR>    modes;
};

// ─────────────────────────────────────────────────────────────────────────────
struct VulkanContext {
    // ── Core objects (read by simulations) ─────────────────────────────────
    VkInstance               instance        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger  = VK_NULL_HANDLE;
    VkSurfaceKHR             surface         = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice  = VK_NULL_HANDLE;
    VkDevice                 device          = VK_NULL_HANDLE;
    VkQueue                  graphicsQueue   = VK_NULL_HANDLE;
    VkQueue                  computeQueue    = VK_NULL_HANDLE;
    uint32_t                 graphicsFamily  = 0;
    uint32_t                 computeFamily   = 0;

    // ── Swapchain ──────────────────────────────────────────────────────────
    VkSwapchainKHR           swapchain    = VK_NULL_HANDLE;
    std::vector<VkImage>     swapImages;
    std::vector<VkImageView> swapViews;
    VkFormat                 swapFormat   = VK_FORMAT_UNDEFINED;
    VkExtent2D               swapExtent{};
    // NEW-7 (RELEASE_v1_1_PLAN.md): Settings > Display "Frame limiter" preference, read by
    // createSwapchain() at both initial creation and every recreateSwapchain() call. Default
    // FIFO (V-Sync) — MAILBOX was the old unconditional default, which lets the GPU run flat out
    // forever on a laptop (fans/heat/battery, exactly the low-end audience this release targets).
    // createSwapchain() falls back to FIFO (spec-guaranteed present) if the requested mode isn't
    // in the surface's supported list. A simulation changes this and then reports true from
    // Simulation::consumeSwapchainRebuildRequest() to have App rebuild the swapchain with it.
    VkPresentModeKHR         presentModePreference = VK_PRESENT_MODE_FIFO_KHR;
    // UC6 (RELEASE_v1_1_PLAN.md): true if the surface advertised TRANSFER_SRC support for
    // swapchain images (near-universal but not spec-guaranteed) — createSwapchain() only adds
    // VK_IMAGE_USAGE_TRANSFER_SRC_BIT to ci.imageUsage when this would be true, since requesting
    // an unsupported usage bit is invalid and would fail swapchain creation outright. Simulations
    // must gate screenshot capture on this rather than assuming the copy will always succeed.
    bool                     screenshotSupported = false;

    // ── Render pass & framebuffers ─────────────────────────────────────────
    VkRenderPass                renderPass = VK_NULL_HANDLE;
    // Same attachment formats/sample-counts as renderPass (so it stays compatible with the same
    // framebuffers below), but LOAD instead of CLEAR for the color attachment — for simulations
    // that pre-populate the swapchain image via a blit before the main pass begins (see
    // Simulation::recordPrePass/activeRenderPass). Depth still CLEARs normally either way.
    VkRenderPass                renderPassLoad = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>  framebuffers;

    // ── Depth buffer (shared; recreated on resize) ─────────────────────────
    VkImage        depthImage  = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView    depthView   = VK_NULL_HANDLE;
    VkFormat       depthFormat = VK_FORMAT_D32_SFLOAT;

    // ── Commands & sync ────────────────────────────────────────────────────
    VkCommandPool            commandPool        = VK_NULL_HANDLE;
    VkCommandBuffer          commandBuffer      = VK_NULL_HANDLE;
    VkSemaphore              semImageAvailable  = VK_NULL_HANDLE;
    std::vector<VkSemaphore> semRenderDone;   // one per swapchain image
    VkFence                  fenceFrame         = VK_NULL_HANDLE;

    // ── GPU timestamp profiling ────────────────────────────────────────────
    // Single frame in flight, so it's safe to resolve the previous frame's
    // query results right after the fence wait in App::drawFrame, before the
    // command buffer is reset and re-recorded for the new frame. Slot layout
    // is a shared contract between App.cpp (writes 0, 7, 8) and SatelliteSim
    // (writes 1-5 in recordCompute; 6 in recordPrePass XOR recordDraw).
    //
    // Authoritative slot layout (keep in sync with SatelliteSim::updateGpuTimingStats,
    // kPerfLabels[] in SatelliteSimUI.cpp, and the JSON keys in savePerfSnapshot):
    //   0 frame start                 App.cpp
    //   1 scene_depth.comp done       SatelliteSim::recordCompute
    //   2 (vestigial — see below)     SatelliteSim::recordCompute
    //   3 sat_orbit.comp AND          SatelliteSim::recordCompute
    //     beam_self_march.comp done
    //   4 cloud_march.comp done       SatelliteSim::recordCompute
    //   5 sat_flare.comp done         SatelliteSim::recordCompute
    //   6 sky background draw done    SatelliteSim::recordPrePass XOR ::recordDraw
    //   7 satellite + star draw done  App.cpp
    //   8 UI overlay done             App.cpp
    //
    // History: cloud_shadow.comp (C12) held a slot here until the pipeline-unification pass
    // folded its work into cloud_march.comp; beam_cloud_block.comp was added when its cost was
    // found to be hiding inside the orbit bucket, taking slot 2 ("beam_cloud_block.comp done");
    // scene_depth.comp then took slot 1 at the head of the pass. 2026-08-09: beam_cloud_block.comp
    // itself was retired (replaced by beam_self_march.comp, a per-BEAM march dispatched AFTER
    // sat_orbit.comp instead of before it — see BEAM_CLOUD_PLAN.md) — rather than renumber every
    // downstream slot again, slot 2 was left in place as a vestigial marker (written right before
    // sat_orbit.comp now starts, so its own bucket, delta(1,2), reads ~0 every frame) and slot 3
    // was pushed out to also cover beam_self_march.comp's dispatch, so delta(2,3) — labeled
    // "orbit_compute" in kPerfLabels[]/the JSON keys — now measures both passes combined. Every
    // downstream slot NUMBER is unchanged from before this pass existed. Slot numbers are
    // documented here and nowhere else authoritative.
    static constexpr uint32_t kTimestampCount = 9;
    VkQueryPool queryPool         = VK_NULL_HANDLE;
    double      timestampPeriodNs = 0.0;   // ns/tick, from device limits; 0 = unsupported
    bool        timestampsReady   = false; // false until one full frame has been resolved
    double      timestampMs[kTimestampCount] = {}; // resolved, ms relative to slot 0

    void resetTimestamps(VkCommandBuffer cmd);
    void writeTimestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage, uint32_t slot);
    void resolveTimestamps();

    // ── Lifecycle ──────────────────────────────────────────────────────────
    void init(GLFWwindow* window);
    void recreateSwapchain(GLFWwindow* window);
    void cleanup();

    // ── Helpers exposed to simulations ─────────────────────────────────────
    VkShaderModule loadShader(const std::string& spvPath);
    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props,
                      VkBuffer& buf, VkDeviceMemory& mem);
    void createImage(uint32_t w, uint32_t h, VkFormat fmt,
                     VkImageUsageFlags usage,
                     VkImage& img, VkDeviceMemory& mem,
                     uint32_t mipLevels = 1,
                     uint32_t depth = 1);
    // Generates mip levels 1..mipLevels-1 from mip 0 via linear blit.
    // Mip 0 must be in TRANSFER_DST_OPTIMAL on entry; all mips will be
    // in SHADER_READ_ONLY_OPTIMAL on return.
    void generateMipmaps(VkCommandBuffer cmd, VkImage img, VkFormat fmt,
                         uint32_t w, uint32_t h, uint32_t mipLevels);
    VkCommandBuffer beginOneTimeCommands();
    void endOneTimeCommands(VkCommandBuffer cmd);
    // Linear blit filtering is a per-format capability, not a given — see the definition.
    VkFilter bestBlitFilter(VkFormat fmt) const;

    void imageBarrier(VkCommandBuffer cmd, VkImage image,
                      VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                      VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage);

private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    // Logs every device limit this project exceeds the Vulkan guaranteed minimum for, and
    // throws a named error if one is actually short. See the definition for why each matters.
    void logDeviceLimits(const VkPhysicalDeviceLimits &lim);
    void createDevice();
    void createSwapchain(GLFWwindow* window);
    void createRenderPass();
    void createRenderPassLoad();
    void createDepthResources();
    void destroyDepthResources();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();
    void createQueryPool();
    void cleanupSwapchain();
    VkFormat findDepthFormat();

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice pd);
    SwapchainDetails   querySwapchainDetails(VkPhysicalDevice pd);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* userData);
};
