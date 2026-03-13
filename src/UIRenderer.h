#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

struct VulkanContext; // forward declare

// Single vertex format for all UI geometry (rectangles and text glyphs).
// The fragment shader switches between solid-color and texture-sampled based on `mode`.
struct UIVertex {
    glm::vec2 pos;   // screen-space pixels, origin top-left
    glm::vec2 uv;    // font atlas UV (ignored for solid rects)
    glm::vec4 color; // RGBA [0,1]
    float     mode;  // 0.0 = solid rectangle, 1.0 = text glyph
};

struct UIPushConstants {
    glm::vec2 screenSize; // viewport size in pixels, used by vertex shader for NDC conversion
};

// Per-frame mouse and button state — read by simulations in buildUI().
struct UIInput {
    float screenW  = 0, screenH  = 0; // window dimensions in pixels
    float mouseX   = 0, mouseY   = 0; // current cursor position
    float dMouseX  = 0, dMouseY  = 0; // cursor delta since last frame
    bool  lmbDown     = false; // left mouse button held
    bool  lmbPressed  = false; // went down this frame
    bool  lmbReleased = false; // went up this frame
    bool  rmbDown     = false; // right mouse button held
    bool  rmbPressed  = false;
    bool  rmbReleased = false;
    float dt = 0;
};

class UIRenderer {
public:
    // Call after VulkanContext is initialized.
    void init(VulkanContext& ctx);

    // Call when swapchain is recreated (window resize).
    void onResize(VulkanContext& ctx);

    // Release all Vulkan resources.
    void cleanup(VkDevice device);

    // Call once at the start of each frame, before the simulation's buildUI().
    // Accepts both left and right mouse button states.
    void beginFrame(float width, float height,
                    float mouseX, float mouseY, bool lmbDown, bool rmbDown,
                    float scrollDeltaX, float scrollDeltaY,
                    float dt);

    // Call inside the render pass, after sim->recordDraw() and before vkCmdEndRenderPass.
    // Finalizes Clay layout, processes render commands, draws all UI geometry.
    void record(VkCommandBuffer cmd, VulkanContext& ctx);

    // Per-frame input state — read by simulations in buildUI().
    const UIInput& input() const { return frameInput; }

    // Register a screen rect that should absorb mouse events (toolbar, open windows, etc.).
    // Call in buildUI() for each visible UI panel. Resets every frame in beginFrame().
    void addMouseCaptureRect(float x, float y, float w, float h);

    // True if mouse is currently over any registered capture rect.
    // Simulations read this to suppress scene interaction while hovering over UI.
    bool mouseOverUI() const { return prevMouseOverUI; }

    // Font IDs for CLAY_TEXT_CONFIG — load additional fonts if needed.
    uint16_t defaultFontId() const { return 0; }

private:
    // ── Clay state ────────────────────────────────────────────────────────
    void*    clayMemory     = nullptr;
    uint32_t clayMemorySize = 0;

    // ── Font (stb_truetype) ───────────────────────────────────────────────
    struct FontAtlas {
        VkImage        image   = VK_NULL_HANDLE;
        VkDeviceMemory memory  = VK_NULL_HANDLE;
        VkImageView    view    = VK_NULL_HANDLE;
        VkSampler      sampler = VK_NULL_HANDLE;
        int            atlasW  = 512;
        int            atlasH  = 512;
        float          bakedSize = 32.0f; // pixels the atlas was baked at
        std::vector<uint8_t> charData;    // stbtt_bakedchar array, cast when used
        std::vector<uint8_t> fileData;    // keep TTF bytes alive for stbtt_fontinfo
        void*                fontInfo = nullptr; // stbtt_fontinfo*, heap-allocated
    } font;

    // ── GPU geometry buffers (persistently mapped host-visible) ───────────
    static constexpr uint32_t MAX_VERTS   = 65536;
    static constexpr uint32_t MAX_INDICES = MAX_VERTS * 3;

    VkBuffer       vertBuf = VK_NULL_HANDLE;
    VkDeviceMemory vertMem = VK_NULL_HANDLE;
    void*          vertMapped = nullptr;

    VkBuffer       idxBuf  = VK_NULL_HANDLE;
    VkDeviceMemory idxMem  = VK_NULL_HANDLE;
    void*          idxMapped = nullptr;

    // ── Descriptors ───────────────────────────────────────────────────────
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descPool   = VK_NULL_HANDLE;
    VkDescriptorSet       descSet    = VK_NULL_HANDLE;

    // ── Pipeline ──────────────────────────────────────────────────────────
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline       pipeline   = VK_NULL_HANDLE;

    // ── Per-frame CPU-side geometry ───────────────────────────────────────
    std::vector<UIVertex>  vertices;
    std::vector<uint32_t>  indices;
    float                  frameW = 800.0f, frameH = 600.0f;

    // ── Input tracking ────────────────────────────────────────────────────
    UIInput frameInput;
    bool    prevLmb = false, prevRmb = false;
    float   prevMx  = 0,     prevMy  = 0;
    bool    mouseIsOverUI = false;
    bool    prevMouseOverUI = false; // value from the end of last frame; read by mouseOverUI()

    // ── Private helpers ───────────────────────────────────────────────────
    void loadFont(VulkanContext& ctx);
    void createPipeline(VulkanContext& ctx);
    void destroyPipeline(VkDevice device);
    void flushBatch(VkCommandBuffer cmd);

    void pushQuad(float x, float y, float w, float h,
                  float u0, float v0, float u1, float v1,
                  glm::vec4 color, float mode);
    void pushText(float x, float y, const char* text, int len,
                  float fontSize, glm::vec4 color);

    // (No static callbacks needed — lambdas are used directly in init())
};
