#pragma once

// Standard Vulkan + GLFW (GLFW includes Vulkan when GLFW_INCLUDE_VULKAN is defined)
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// ─── Simulation dimensions ────────────────────────────────────────────────────
constexpr uint32_t WIN_W = 1280;
constexpr uint32_t WIN_H = 720;

constexpr uint32_t GOL_W = 512;   // Game of Life grid width  (cells)
constexpr uint32_t GOL_H = 512;   // Game of Life grid height (cells)

constexpr uint32_t PARTICLE_COUNT = 500'000;  // Number of GPU particles

// ─── Data structures shared with shaders ─────────────────────────────────────
// Must match the layout in particles_update.comp and particles_draw.vert
struct Particle {
    glm::vec2 pos;   // NDC position (-1..1)
    glm::vec2 vel;   // velocity
    glm::vec4 color; // RGBA
};

// Push constants for the particle compute shader
struct ParticlePushConstants {
    float dt;
    float time;
    glm::vec2 mouseNDC; // mouse position in NDC (-1..1)
};

// ─── Internal helper structs ──────────────────────────────────────────────────
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> compute;
    bool complete() const { return graphics.has_value() && compute.has_value(); }
};

struct SwapchainDetails {
    VkSurfaceCapabilitiesKHR caps;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> modes;
};

// ─── Simulation mode ──────────────────────────────────────────────────────────
enum class SimMode { GameOfLife, Particles };

// ─── App class ────────────────────────────────────────────────────────────────
class App {
public:
    void run();

private:
    // ── Window ─────────────────────────────────────────────────────────────
    GLFWwindow* window = nullptr;
    bool resized = false;
    SimMode mode = SimMode::GameOfLife;
    double lastTime = 0.0;
    float simTime = 0.0f;

    // ── Vulkan core ────────────────────────────────────────────────────────
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue gfxQueue = VK_NULL_HANDLE;    // graphics queue
    VkQueue compQueue = VK_NULL_HANDLE;   // compute queue (may be same handle)
    uint32_t gfxFamily = 0;
    uint32_t compFamily = 0;

    // ── Swapchain ──────────────────────────────────────────────────────────
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    VkFormat swapFmt = VK_FORMAT_UNDEFINED;
    VkExtent2D swapExt{};

    // ── Render pass & framebuffers ─────────────────────────────────────────
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    // ── Commands & sync ────────────────────────────────────────────────────
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkSemaphore semImageAvailable = VK_NULL_HANDLE;
    VkSemaphore semRenderDone = VK_NULL_HANDLE;
    VkFence fenceFrame = VK_NULL_HANDLE;

    // ══ Game of Life ════════════════════════════════════════════════════════
    // Two R8G8B8A8 storage images, ping-ponged each frame.
    // golCurrent=0 means image[0] holds the "live" state; compute reads 0, writes 1.
    VkImage       golImg[2]  = {};
    VkDeviceMemory golMem[2] = {};
    VkImageView   golView[2] = {};
    int golCurrent = 0;

    VkSampler            golSampler       = VK_NULL_HANDLE;
    VkDescriptorSetLayout golCompLayout   = VK_NULL_HANDLE;  // storage images
    VkDescriptorSetLayout golDispLayout   = VK_NULL_HANDLE;  // combined image sampler
    VkDescriptorPool      golDescPool     = VK_NULL_HANDLE;
    VkDescriptorSet       golCompSet[2]   = {};  // [0]: read img0 write img1, [1]: vice versa
    VkDescriptorSet       golDispSet[2]   = {};  // [i]: sample from img[i]
    VkPipelineLayout      golCompPipeLayout = VK_NULL_HANDLE;
    VkPipeline            golCompPipe       = VK_NULL_HANDLE;
    VkPipelineLayout      golDispPipeLayout = VK_NULL_HANDLE;
    VkPipeline            golDispPipe       = VK_NULL_HANDLE;

    // ══ Particles ═══════════════════════════════════════════════════════════
    VkBuffer              partBuf       = VK_NULL_HANDLE;
    VkDeviceMemory        partMem       = VK_NULL_HANDLE;
    VkDescriptorSetLayout partLayout    = VK_NULL_HANDLE;
    VkDescriptorPool      partDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet       partSet       = VK_NULL_HANDLE;
    VkPipelineLayout      partCompLayout = VK_NULL_HANDLE;
    VkPipeline            partCompPipe   = VK_NULL_HANDLE;
    VkPipelineLayout      partDrawLayout = VK_NULL_HANDLE;
    VkPipeline            partDrawPipe   = VK_NULL_HANDLE;

    // ── Initialisation ─────────────────────────────────────────────────────
    void initWindow();
    void initVulkan();
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();

    // GoL
    void createGoLImages();
    void createGoLSampler();
    void createGoLDescriptors();
    void createGoLComputePipeline();
    void createGoLDisplayPipeline();
    void initGoLState();

    // Particles
    void createParticleBuffer();
    void createParticleDescriptors();
    void createParticleComputePipeline();
    void createParticleDrawPipeline();
    void initParticles();

    // ── Loop ──────────────────────────────────────────────────────────────
    void mainLoop();
    void drawFrame();
    void recordGoL(uint32_t imgIdx);
    void recordParticles(uint32_t imgIdx);
    void recreateSwapchain();

    // ── Cleanup ───────────────────────────────────────────────────────────
    void cleanupSwapchain();
    void cleanup();

    // ── Helpers ───────────────────────────────────────────────────────────
    VkShaderModule loadShader(const std::string& spvPath);
    uint32_t findMemType(uint32_t filter, VkMemoryPropertyFlags props);
    void createImage(uint32_t w, uint32_t h, VkFormat fmt,
                     VkImageUsageFlags usage, VkImage& img, VkDeviceMemory& mem);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);
    VkCommandBuffer beginOneTime();
    void endOneTime(VkCommandBuffer cmd);
    void imageBarrier(VkCommandBuffer cmd, VkImage img,
                      VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                      VkImageLayout oldLay, VkImageLayout newLay,
                      VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice pd);
    SwapchainDetails   querySwapchainDetails(VkPhysicalDevice pd);

    // ── Static callbacks ──────────────────────────────────────────────────
    static void cbResize(GLFWwindow* w, int, int);
    static void cbKey(GLFWwindow* w, int key, int, int action, int);
    static VKAPI_ATTR VkBool32 VKAPI_CALL cbDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT*,
        void*);
};
