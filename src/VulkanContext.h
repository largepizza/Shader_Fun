#pragma once
#define GLFW_INCLUDE_VULKAN
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

    // ── Render pass & framebuffers ─────────────────────────────────────────
    VkRenderPass                renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>  framebuffers;

    // ── Commands & sync ────────────────────────────────────────────────────
    VkCommandPool            commandPool        = VK_NULL_HANDLE;
    VkCommandBuffer          commandBuffer      = VK_NULL_HANDLE;
    VkSemaphore              semImageAvailable  = VK_NULL_HANDLE;
    std::vector<VkSemaphore> semRenderDone;   // one per swapchain image
    VkFence                  fenceFrame         = VK_NULL_HANDLE;

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
                     VkImage& img, VkDeviceMemory& mem);
    VkCommandBuffer beginOneTimeCommands();
    void endOneTimeCommands(VkCommandBuffer cmd);
    void imageBarrier(VkCommandBuffer cmd, VkImage image,
                      VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                      VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage);

private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain(GLFWwindow* window);
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();
    void cleanupSwapchain();

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice pd);
    SwapchainDetails   querySwapchainDetails(VkPhysicalDevice pd);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* userData);
};
