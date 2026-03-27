#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

struct VulkanContext; // forward declare

// Single vertex format for all UI geometry (rectangles, text glyphs, icons).
// The fragment shader switches rendering mode based on `mode`.
struct UIVertex {
    glm::vec2 pos;   // screen-space pixels, origin top-left
    glm::vec2 uv;    // atlas UV (font atlas for mode=1, icon atlas for mode=2)
    glm::vec4 color; // RGBA [0,1]
    float     mode;  // 0.0 = solid rectangle, 1.0 = text glyph, 2.0 = icon sprite
};

struct UIPushConstants {
    glm::vec2 screenSize; // viewport size in pixels, used by vertex shader for NDC conversion
};

// Per-frame mouse, button, and scroll state — read by simulations in buildUI().
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
    float scrollY  = 0.0f;    // vertical scroll delta this frame (positive = scroll up)
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
    void beginFrame(float width, float height,
                    float mouseX, float mouseY, bool lmbDown, bool rmbDown,
                    float scrollDeltaX, float scrollDeltaY,
                    float dt);

    // Call inside the render pass, after sim->recordDraw() and before vkCmdEndRenderPass.
    void record(VkCommandBuffer cmd, VulkanContext& ctx);

    // Per-frame input state — read by simulations in buildUI().
    const UIInput& input() const { return frameInput; }

    // Register a screen rect that should absorb mouse events (toolbar, open windows, etc.).
    void addMouseCaptureRect(float x, float y, float w, float h);

    // True if mouse is currently over any registered capture rect (one-frame lag).
    bool mouseOverUI() const { return prevMouseOverUI; }

    // Font IDs for CLAY_TEXT_CONFIG.
    uint16_t defaultFontId() const { return 0; }

    // ── Icon atlas ────────────────────────────────────────────────────────────
    // Load PNG icons from disk and pack into a horizontal GPU atlas.
    // Call from buildUI() on first frame (lazy init); safe to call once.
    // Returns the number of icons successfully loaded.
    int loadIcons(VulkanContext& ctx, const char* const* paths, int count);

    // Number of icons currently loaded.
    int iconCount() const { return (int)iconEntries.size(); }

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
        float          bakedSize = 32.0f;
        std::vector<uint8_t> charData;
        std::vector<uint8_t> fileData;
        void*                fontInfo = nullptr;
    } font;

    // ── Icon atlas (stb_image, RGBA8) ─────────────────────────────────────
    struct IconEntry {
        float u0, v0, u1, v1; // UV rectangle in the packed atlas
    };
    VkImage        iconImage   = VK_NULL_HANDLE;
    VkDeviceMemory iconMemory  = VK_NULL_HANDLE;
    VkImageView    iconView    = VK_NULL_HANDLE;
    VkSampler      iconSampler = VK_NULL_HANDLE;
    std::vector<IconEntry> iconEntries;

    void createIconPlaceholder(VulkanContext& ctx);
    void uploadIconAtlas(VulkanContext& ctx, const std::vector<uint8_t>& rgba, int w, int h);
    void rebindIconDescriptor(VkDevice device);

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
    uint32_t               batchVertOffset = 0; // running write offset in vertBuf (in #vertices)
    uint32_t               batchIdxOffset  = 0; // running write offset in idxBuf  (in #indices)
    float                  frameW = 800.0f, frameH = 600.0f;

    // ── Input tracking ────────────────────────────────────────────────────
    UIInput frameInput;
    bool    prevLmb = false, prevRmb = false;
    float   prevMx  = 0,     prevMy  = 0;
    bool    mouseIsOverUI = false;
    bool    prevMouseOverUI = false;

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
};
