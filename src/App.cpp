#include "App.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

// ─── Validation layers (debug only) ──────────────────────────────────────────
#ifdef NDEBUG
    constexpr bool VALIDATION = false;
#else
    constexpr bool VALIDATION = true;
#endif

static const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};
static const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// ═════════════════════════════════════════════════════════════════════════════
// Public entry
// ═════════════════════════════════════════════════════════════════════════════
void App::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

// ═════════════════════════════════════════════════════════════════════════════
// Window
// ═════════════════════════════════════════════════════════════════════════════
void App::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIN_W, WIN_H,
        "ShaderFun | TAB = switch sim | SPACE = reset GoL | ESC = quit", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, cbResize);
    glfwSetKeyCallback(window, cbKey);
}

void App::cbResize(GLFWwindow* w, int, int) {
    auto* app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
    app->resized = true;
}

void App::cbKey(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* app = reinterpret_cast<App*>(glfwGetWindowUserPointer(w));
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_TAB) {
        vkDeviceWaitIdle(app->dev);
        app->mode = (app->mode == SimMode::GameOfLife) ? SimMode::Particles : SimMode::GameOfLife;
        std::cout << "Switched to: "
                  << (app->mode == SimMode::GameOfLife ? "Game of Life" : "Particles") << "\n";
    }
    if (key == GLFW_KEY_SPACE && app->mode == SimMode::GameOfLife) {
        vkDeviceWaitIdle(app->dev);
        app->initGoLState();
        std::cout << "GoL state randomized.\n";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Vulkan init
// ═════════════════════════════════════════════════════════════════════════════
void App::initVulkan() {
    createInstance();
    if (VALIDATION) setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createDevice();
    createSwapchain();
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();

    // Game of Life resources
    createGoLImages();
    createGoLSampler();
    createGoLDescriptors();
    createGoLComputePipeline();
    createGoLDisplayPipeline();
    initGoLState();

    // Particle resources
    createParticleBuffer();
    createParticleDescriptors();
    createParticleComputePipeline();
    createParticleDrawPipeline();
    initParticles();
}

// ─── Instance ─────────────────────────────────────────────────────────────────
static bool checkValidationSupport() {
    uint32_t n; vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> layers(n);
    vkEnumerateInstanceLayerProperties(&n, layers.data());
    for (auto* name : VALIDATION_LAYERS) {
        bool found = false;
        for (auto& l : layers) if (strcmp(l.layerName, name) == 0) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

void App::createInstance() {
    if (VALIDATION && !checkValidationSupport())
        throw std::runtime_error("Validation layers requested but not available.");

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "ShaderFun";
    ai.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    ai.apiVersion = VK_API_VERSION_1_2;

    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> exts(glfwExts, glfwExts + glfwExtCount);
    if (VALIDATION) exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    if (VALIDATION) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed.");
}

// ─── Debug messenger ──────────────────────────────────────────────────────────
VKAPI_ATTR VkBool32 VKAPI_CALL App::cbDebug(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

void App::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                   | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                   | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = cbDebug;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn || fn(instance, &ci, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to set up debug messenger.");
}

// ─── Surface ──────────────────────────────────────────────────────────────────
void App::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed.");
}

// ─── Physical device ──────────────────────────────────────────────────────────
QueueFamilyIndices App::findQueueFamilies(VkPhysicalDevice pd) {
    QueueFamilyIndices idx;
    uint32_t n; vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    for (uint32_t i = 0; i < n; ++i) {
        if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) idx.graphics = i;
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)  idx.compute  = i;
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
        // Prefer the graphics queue to also present
        if (present && !idx.graphics.has_value()) idx.graphics = i;
        if (idx.complete()) break;
    }
    // If graphics doesn't support present, find one that does
    if (idx.complete()) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, *idx.graphics, surface, &present);
        if (!present) {
            for (uint32_t i = 0; i < n; ++i) {
                VkBool32 p = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &p);
                if (p) { idx.graphics = i; break; }
            }
        }
    }
    return idx;
}

SwapchainDetails App::querySwapchainDetails(VkPhysicalDevice pd) {
    SwapchainDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &d.caps);
    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, nullptr);
    d.formats.resize(n); vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, d.formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, nullptr);
    d.modes.resize(n); vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, d.modes.data());
    return d;
}

static bool isDeviceSuitable(VkPhysicalDevice pd, VkSurfaceKHR surface,
    const std::function<QueueFamilyIndices(VkPhysicalDevice)>& getQ,
    const std::function<SwapchainDetails(VkPhysicalDevice)>& getSC)
{
    // Check extensions
    uint32_t n; vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, exts.data());
    for (auto* req : DEVICE_EXTENSIONS) {
        bool found = false;
        for (auto& e : exts) if (strcmp(e.extensionName, req) == 0) { found = true; break; }
        if (!found) return false;
    }
    auto q  = getQ(pd);
    auto sc = getSC(pd);
    return q.complete() && !sc.formats.empty() && !sc.modes.empty();
}

void App::pickPhysicalDevice() {
    uint32_t n; vkEnumeratePhysicalDevices(instance, &n, nullptr);
    if (!n) throw std::runtime_error("No GPUs with Vulkan support.");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance, &n, devs.data());

    for (auto pd : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        if (!isDeviceSuitable(pd, surface,
            [this](VkPhysicalDevice d){ return findQueueFamilies(d); },
            [this](VkPhysicalDevice d){ return querySwapchainDetails(d); }))
            continue;
        // Prefer discrete GPU
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physDev = pd;
            std::cout << "GPU: " << p.deviceName << " (discrete)\n";
            break;
        }
        if (!physDev) {
            physDev = pd;
            std::cout << "GPU: " << p.deviceName << "\n";
        }
    }
    if (!physDev) throw std::runtime_error("Failed to find a suitable GPU.");
}

// ─── Logical device ───────────────────────────────────────────────────────────
void App::createDevice() {
    auto qi = findQueueFamilies(physDev);
    gfxFamily  = *qi.graphics;
    compFamily = *qi.compute;

    std::set<uint32_t> uniqueFamilies = {gfxFamily, compFamily};
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qCIs;
    for (uint32_t fam : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = fam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        qCIs.push_back(qci);
    }

    VkPhysicalDeviceFeatures features{};
    features.shaderStorageImageExtendedFormats = VK_TRUE; // for rgba8 storage images
    features.largePoints = VK_TRUE; // for gl_PointSize in particles

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = (uint32_t)qCIs.size();
    ci.pQueueCreateInfos    = qCIs.data();
    ci.enabledExtensionCount   = (uint32_t)DEVICE_EXTENSIONS.size();
    ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    ci.pEnabledFeatures = &features;
    if (VALIDATION) {
        ci.enabledLayerCount   = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateDevice(physDev, &ci, nullptr, &dev) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed.");

    vkGetDeviceQueue(dev, gfxFamily,  0, &gfxQueue);
    vkGetDeviceQueue(dev, compFamily, 0, &compQueue);
}

// ─── Swapchain ────────────────────────────────────────────────────────────────
void App::createSwapchain() {
    auto sc = querySwapchainDetails(physDev);

    // Pick format (prefer sRGB B8G8R8A8)
    VkSurfaceFormatKHR fmt = sc.formats[0];
    for (auto& f : sc.formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { fmt = f; break; }

    // Pick present mode (prefer mailbox, fall back to FIFO)
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    for (auto& m : sc.modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) { pm = m; break; }

    // Extent
    VkExtent2D ext;
    if (sc.caps.currentExtent.width != UINT32_MAX) {
        ext = sc.caps.currentExtent;
    } else {
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        ext = { std::clamp((uint32_t)w, sc.caps.minImageExtent.width, sc.caps.maxImageExtent.width),
                std::clamp((uint32_t)h, sc.caps.minImageExtent.height, sc.caps.maxImageExtent.height) };
    }

    uint32_t imgCount = sc.caps.minImageCount + 1;
    if (sc.caps.maxImageCount && imgCount > sc.caps.maxImageCount)
        imgCount = sc.caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface          = surface;
    ci.minImageCount    = imgCount;
    ci.imageFormat      = fmt.format;
    ci.imageColorSpace  = fmt.colorSpace;
    ci.imageExtent      = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    uint32_t qfams[] = {gfxFamily, compFamily};
    if (gfxFamily != compFamily) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = qfams;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform   = sc.caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = pm;
    ci.clipped        = VK_TRUE;

    if (vkCreateSwapchainKHR(dev, &ci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed.");

    swapFmt = fmt.format;
    swapExt = ext;

    uint32_t cnt; vkGetSwapchainImagesKHR(dev, swapchain, &cnt, nullptr);
    swapImages.resize(cnt); vkGetSwapchainImagesKHR(dev, swapchain, &cnt, swapImages.data());

    swapViews.resize(cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = swapImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = swapFmt;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(dev, &vci, nullptr, &swapViews[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateImageView failed.");
    }
}

// ─── Render pass ──────────────────────────────────────────────────────────────
void App::createRenderPass() {
    VkAttachmentDescription color{};
    color.format         = swapFmt;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &colorRef;

    // Subpass dependency: ensure compute finishes before fragment reads
    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 1; ci.pAttachments = &color;
    ci.subpassCount    = 1; ci.pSubpasses   = &sub;
    ci.dependencyCount = 1; ci.pDependencies = &dep;

    if (vkCreateRenderPass(dev, &ci, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed.");
}

// ─── Framebuffers ─────────────────────────────────────────────────────────────
void App::createFramebuffers() {
    framebuffers.resize(swapViews.size());
    for (size_t i = 0; i < swapViews.size(); ++i) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = renderPass;
        ci.attachmentCount = 1;
        ci.pAttachments    = &swapViews[i];
        ci.width  = swapExt.width;
        ci.height = swapExt.height;
        ci.layers = 1;
        if (vkCreateFramebuffer(dev, &ci, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer failed.");
    }
}

// ─── Command pool & buffer ────────────────────────────────────────────────────
void App::createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = gfxFamily;
    if (vkCreateCommandPool(dev, &ci, nullptr, &cmdPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateCommandPool failed.");
}

void App::createCommandBuffer() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev, &ai, &cmdBuf) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers failed.");
}

// ─── Sync objects ─────────────────────────────────────────────────────────────
void App::createSyncObjects() {
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first frame doesn't hang

    if (vkCreateSemaphore(dev, &si, nullptr, &semImageAvailable) != VK_SUCCESS ||
        vkCreateSemaphore(dev, &si, nullptr, &semRenderDone)      != VK_SUCCESS ||
        vkCreateFence    (dev, &fi, nullptr, &fenceFrame)         != VK_SUCCESS)
        throw std::runtime_error("Failed to create sync objects.");
}

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════
VkShaderModule App::loadShader(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open shader: " + path);
    size_t sz = f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), sz);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(dev, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed for: " + path);
    return mod;
}

uint32_t App::findMemType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("Failed to find suitable memory type.");
}

void App::createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                      VkImage& img, VkDeviceMemory& mem) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType   = VK_IMAGE_TYPE_2D;
    ci.format      = fmt;
    ci.extent      = {w, h, 1};
    ci.mipLevels   = 1;
    ci.arrayLayers = 1;
    ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &ci, nullptr, &img) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImage failed.");

    VkMemoryRequirements req; vkGetImageMemoryRequirements(dev, img, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (image) failed.");
    vkBindImageMemory(dev, img, mem, 0);
}

void App::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &ci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed.");

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, props);
    if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (buffer) failed.");
    vkBindBufferMemory(dev, buf, mem, 0);
}

VkCommandBuffer App::beginOneTime() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void App::endOneTime(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(gfxQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(gfxQueue);
    vkFreeCommandBuffers(dev, cmdPool, 1, &cmd);
}

void App::imageBarrier(VkCommandBuffer cmd, VkImage img,
                       VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                       VkImageLayout oldLay, VkImageLayout newLay,
                       VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask       = srcAccess;
    b.dstAccessMask       = dstAccess;
    b.oldLayout           = oldLay;
    b.newLayout           = newLay;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = img;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ═════════════════════════════════════════════════════════════════════════════
// Game of Life
// ═════════════════════════════════════════════════════════════════════════════
void App::createGoLImages() {
    constexpr VkFormat FMT = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkImageUsageFlags USAGE = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    for (int i = 0; i < 2; ++i) {
        createImage(GOL_W, GOL_H, FMT, USAGE, golImg[i], golMem[i]);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = golImg[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = FMT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(dev, &vci, nullptr, &golView[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateImageView (GoL) failed.");
    }

    // Transition both images to GENERAL layout
    auto cmd = beginOneTime();
    for (int i = 0; i < 2; ++i)
        imageBarrier(cmd, golImg[i],
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    endOneTime(cmd);
}

void App::createGoLSampler() {
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = VK_FILTER_NEAREST;
    ci.minFilter = VK_FILTER_NEAREST;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (vkCreateSampler(dev, &ci, nullptr, &golSampler) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSampler failed.");
}

void App::createGoLDescriptors() {
    // ── Compute layout: two storage image bindings ────────────────────────
    {
        VkDescriptorSetLayoutBinding b[2] = {};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1] = b[0]; b[1].binding = 1;

        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2; ci.pBindings = b;
        vkCreateDescriptorSetLayout(dev, &ci, nullptr, &golCompLayout);
    }
    // ── Display layout: one combined image sampler ────────────────────────
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 1; ci.pBindings = &b;
        vkCreateDescriptorSetLayout(dev, &ci, nullptr, &golDispLayout);
    }

    // ── Pool: 2 compute sets + 2 display sets ─────────────────────────────
    VkDescriptorPoolSize sizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
    };
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    pi.maxSets = 4;
    vkCreateDescriptorPool(dev, &pi, nullptr, &golDescPool);

    // ── Allocate and write compute sets ──────────────────────────────────
    for (int s = 0; s < 2; ++s) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = golDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &golCompLayout;
        vkAllocateDescriptorSets(dev, &ai, &golCompSet[s]);

        // set s: reads from golImg[s], writes to golImg[1-s]
        VkDescriptorImageInfo imgInfo[2] = {
            {VK_NULL_HANDLE, golView[s],   VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, golView[1-s], VK_IMAGE_LAYOUT_GENERAL},
        };
        VkWriteDescriptorSet w[2] = {};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet          = golCompSet[s];
        w[0].dstBinding      = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo      = &imgInfo[0];
        w[1] = w[0]; w[1].dstBinding = 1; w[1].pImageInfo = &imgInfo[1];
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    }

    // ── Allocate and write display sets ──────────────────────────────────
    for (int s = 0; s < 2; ++s) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = golDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &golDispLayout;
        vkAllocateDescriptorSets(dev, &ai, &golDispSet[s]);

        // display set s samples from golImg[s]
        VkDescriptorImageInfo imgInfo{golSampler, golView[s], VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = golDispSet[s];
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    }
}

void App::createGoLComputePipeline() {
    auto mod = loadShader("shaders/game_of_life.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts    = &golCompLayout;
    vkCreatePipelineLayout(dev, &li, nullptr, &golCompPipeLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = golCompPipeLayout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &golCompPipe) != VK_SUCCESS)
        throw std::runtime_error("Failed to create GoL compute pipeline.");

    vkDestroyShaderModule(dev, mod, nullptr);
}

void App::createGoLDisplayPipeline() {
    auto vert = loadShader("shaders/fullscreen.vert.spv");
    auto frag = loadShader("shaders/gol_display.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo   vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, (float)swapExt.width, (float)swapExt.height, 0, 1};
    VkRect2D sc{{0,0}, swapExt};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp;
    vps.scissorCount  = 1; vps.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts    = &golDispLayout;
    vkCreatePipelineLayout(dev, &li, nullptr, &golDispPipeLayout);

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState   = &ms;
    ci.pColorBlendState    = &cb;
    ci.layout              = golDispPipeLayout;
    ci.renderPass          = renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &golDispPipe) != VK_SUCCESS)
        throw std::runtime_error("Failed to create GoL display pipeline.");

    vkDestroyShaderModule(dev, vert, nullptr);
    vkDestroyShaderModule(dev, frag, nullptr);
}

void App::initGoLState() {
    // Fill image[0] with random alive/dead cells using a staging buffer
    VkDeviceSize imgSize = GOL_W * GOL_H * 4; // RGBA8
    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(imgSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem);

    void* data; vkMapMemory(dev, stagingMem, 0, imgSize, 0, &data);
    auto* pixels = reinterpret_cast<uint8_t*>(data);
    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution alive(0.3); // 30% chance alive
    for (uint32_t i = 0; i < GOL_W * GOL_H; ++i) {
        uint8_t v = alive(rng) ? 255 : 0;
        pixels[i*4+0] = v; // R = alive flag
        pixels[i*4+1] = 0;
        pixels[i*4+2] = 0;
        pixels[i*4+3] = 255;
    }
    vkUnmapMemory(dev, stagingMem);

    // Copy staging buffer → golImg[0]
    auto cmd = beginOneTime();
    imageBarrier(cmd, golImg[0],
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {GOL_W, GOL_H, 1};
    vkCmdCopyBufferToImage(cmd, staging, golImg[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    imageBarrier(cmd, golImg[0],
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Also transition golImg[1] to GENERAL (clear it)
    imageBarrier(cmd, golImg[1],
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    endOneTime(cmd);

    vkDestroyBuffer(dev, staging, nullptr);
    vkFreeMemory(dev, stagingMem, nullptr);

    golCurrent = 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// Particles
// ═════════════════════════════════════════════════════════════════════════════
void App::createParticleBuffer() {
    VkDeviceSize size = sizeof(Particle) * PARTICLE_COUNT;
    createBuffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        partBuf, partMem);
}

void App::createParticleDescriptors() {
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 1; li.pBindings = &b;
    vkCreateDescriptorSetLayout(dev, &li, nullptr, &partLayout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    pi.maxSets = 1;
    vkCreateDescriptorPool(dev, &pi, nullptr, &partDescPool);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = partDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &partLayout;
    vkAllocateDescriptorSets(dev, &ai, &partSet);

    VkDescriptorBufferInfo bi2{partBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = partSet;
    w.dstBinding      = 0;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &bi2;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

void App::createParticleComputePipeline() {
    auto mod = loadShader("shaders/particles_update.comp.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(ParticlePushConstants);

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount         = 1;
    li.pSetLayouts            = &partLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges    = &pcr;
    vkCreatePipelineLayout(dev, &li, nullptr, &partCompLayout);

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = partCompLayout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &partCompPipe) != VK_SUCCESS)
        throw std::runtime_error("Failed to create particle compute pipeline.");

    vkDestroyShaderModule(dev, mod, nullptr);
}

void App::createParticleDrawPipeline() {
    auto vert = loadShader("shaders/particles_draw.vert.spv");
    auto frag = loadShader("shaders/particles_draw.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    // No vertex attributes: vertex shader reads from SSBO via gl_VertexIndex

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp{0, 0, (float)swapExt.width, (float)swapExt.height, 0, 1};
    VkRect2D sc{{0,0}, swapExt};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp;
    vps.scissorCount  = 1; vps.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Additive blending for glowing particles
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable         = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorBlendOp        = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp        = VK_BLEND_OP_ADD;
    cba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts    = &partLayout;
    vkCreatePipelineLayout(dev, &li, nullptr, &partDrawLayout);

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vps;
    ci.pRasterizationState = &rast;
    ci.pMultisampleState   = &ms;
    ci.pColorBlendState    = &cb;
    ci.layout              = partDrawLayout;
    ci.renderPass          = renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &partDrawPipe) != VK_SUCCESS)
        throw std::runtime_error("Failed to create particle draw pipeline.");

    vkDestroyShaderModule(dev, vert, nullptr);
    vkDestroyShaderModule(dev, frag, nullptr);
}

void App::initParticles() {
    VkDeviceSize size = sizeof(Particle) * PARTICLE_COUNT;

    // Create host-visible staging buffer
    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem);

    void* data; vkMapMemory(dev, stagingMem, 0, size, 0, &data);
    auto* particles = reinterpret_cast<Particle*>(data);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posD(-1.0f, 1.0f);
    std::uniform_real_distribution<float> velD(-0.01f, 0.01f);

    for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
        particles[i].pos   = {posD(rng), posD(rng)};
        particles[i].vel   = {velD(rng), velD(rng)};
        // Assign a rainbow hue based on index
        float h = (float)i / PARTICLE_COUNT;
        // HSV to RGB (simple version, S=1 V=1)
        float r = std::abs(h * 6.0f - 3.0f) - 1.0f;
        float g = 2.0f - std::abs(h * 6.0f - 2.0f);
        float b = 2.0f - std::abs(h * 6.0f - 4.0f);
        particles[i].color = {
            std::clamp(r, 0.0f, 1.0f) * 0.8f,
            std::clamp(g, 0.0f, 1.0f) * 0.8f,
            std::clamp(b, 0.0f, 1.0f) * 0.8f,
            1.0f
        };
    }
    vkUnmapMemory(dev, stagingMem);

    // Copy to device-local buffer
    auto cmd = beginOneTime();
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, staging, partBuf, 1, &copy);
    endOneTime(cmd);

    vkDestroyBuffer(dev, staging, nullptr);
    vkFreeMemory(dev, stagingMem, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Main loop
// ═════════════════════════════════════════════════════════════════════════════
void App::mainLoop() {
    lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        double now = glfwGetTime();
        float dt   = (float)(now - lastTime);
        lastTime   = now;
        simTime   += dt;
        drawFrame();
    }
    vkDeviceWaitIdle(dev);
}

void App::drawFrame() {
    // Wait for previous frame
    vkWaitForFences(dev, 1, &fenceFrame, VK_TRUE, UINT64_MAX);

    uint32_t imgIdx;
    VkResult res = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX,
                                          semImageAvailable, VK_NULL_HANDLE, &imgIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR failed.");

    vkResetFences(dev, 1, &fenceFrame);
    vkResetCommandBuffer(cmdBuf, 0);

    // Record the right commands for the current simulation mode
    if (mode == SimMode::GameOfLife)
        recordGoL(imgIdx);
    else
        recordParticles(imgIdx);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &semImageAvailable;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmdBuf;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &semRenderDone;
    if (vkQueueSubmit(gfxQueue, 1, &si, fenceFrame) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed.");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &semRenderDone;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain;
    pi.pImageIndices      = &imgIdx;
    res = vkQueuePresentKHR(gfxQueue, &pi);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || resized) {
        resized = false;
        recreateSwapchain();
    }
}

void App::recordGoL(uint32_t imgIdx) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmdBuf, &bi);

    // ── Compute pass: GoL step ────────────────────────────────────────────
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, golCompPipe);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
        golCompPipeLayout, 0, 1, &golCompSet[golCurrent], 0, nullptr);

    uint32_t gx = (GOL_W + 15) / 16;
    uint32_t gy = (GOL_H + 15) / 16;
    vkCmdDispatch(cmdBuf, gx, gy, 1);

    // ── Barrier: compute write → fragment read ────────────────────────────
    // The "next" image (1-golCurrent) was just written, now needs to be readable
    imageBarrier(cmdBuf, golImg[1-golCurrent],
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    // ── Render pass: display result ───────────────────────────────────────
    VkClearValue clear{{{0.02f, 0.02f, 0.05f, 1.0f}}};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass      = renderPass;
    rbi.framebuffer     = framebuffers[imgIdx];
    rbi.renderArea      = {{0,0}, swapExt};
    rbi.clearValueCount = 1;
    rbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cmdBuf, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, golDispPipe);
    // Display the "next" image (just computed), which is golImg[1-golCurrent]
    // golDispSet[i] samples from golImg[i], so we want golDispSet[1-golCurrent]
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
        golDispPipeLayout, 0, 1, &golDispSet[1-golCurrent], 0, nullptr);
    vkCmdDraw(cmdBuf, 3, 1, 0, 0); // fullscreen triangle

    vkCmdEndRenderPass(cmdBuf);
    vkEndCommandBuffer(cmdBuf);

    // Advance ping-pong
    golCurrent = 1 - golCurrent;
}

void App::recordParticles(uint32_t imgIdx) {
    // Get mouse position in NDC
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    int ww, wh; glfwGetWindowSize(window, &ww, &wh);
    ParticlePushConstants pc{};
    pc.dt       = std::min((float)(glfwGetTime() - lastTime + 1.0/60.0), 0.05f);
    pc.time     = simTime;
    pc.mouseNDC = { (float)(mx / ww) * 2.0f - 1.0f,
                    (float)(my / wh) * 2.0f - 1.0f };

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmdBuf, &bi);

    // ── Compute pass: update particles ────────────────────────────────────
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, partCompPipe);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
        partCompLayout, 0, 1, &partSet, 0, nullptr);
    vkCmdPushConstants(cmdBuf, partCompLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(pc), &pc);
    uint32_t groups = (PARTICLE_COUNT + 255) / 256;
    vkCmdDispatch(cmdBuf, groups, 1, 1);

    // ── Barrier: compute write → vertex read ─────────────────────────────
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask       = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer              = partBuf;
    bmb.offset              = 0;
    bmb.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);

    // ── Render pass: draw particles ───────────────────────────────────────
    VkClearValue clear{{{0.01f, 0.01f, 0.02f, 1.0f}}};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass      = renderPass;
    rbi.framebuffer     = framebuffers[imgIdx];
    rbi.renderArea      = {{0,0}, swapExt};
    rbi.clearValueCount = 1;
    rbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cmdBuf, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, partDrawPipe);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
        partDrawLayout, 0, 1, &partSet, 0, nullptr);
    vkCmdDraw(cmdBuf, PARTICLE_COUNT, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuf);
    vkEndCommandBuffer(cmdBuf);
}

// ═════════════════════════════════════════════════════════════════════════════
// Swapchain recreation (on resize)
// ═════════════════════════════════════════════════════════════════════════════
void App::cleanupSwapchain() {
    for (auto fb : framebuffers) vkDestroyFramebuffer(dev, fb, nullptr);
    framebuffers.clear();
    for (auto v : swapViews)    vkDestroyImageView(dev, v, nullptr);
    swapViews.clear();
    vkDestroySwapchainKHR(dev, swapchain, nullptr);
}

void App::recreateSwapchain() {
    // Wait if minimized
    int w = 0, h = 0;
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(dev);
    cleanupSwapchain();
    createSwapchain();
    createFramebuffers();
    // Pipelines bake in the viewport size, so recreate the graphics pipelines.
    // In a production engine you would use VK_DYNAMIC_STATE_VIEWPORT instead.
    vkDestroyPipeline(dev, golDispPipe, nullptr);
    vkDestroyPipeline(dev, partDrawPipe, nullptr);
    createGoLDisplayPipeline();
    createParticleDrawPipeline();
}

// ═════════════════════════════════════════════════════════════════════════════
// Cleanup
// ═════════════════════════════════════════════════════════════════════════════
void App::cleanup() {
    cleanupSwapchain();
    vkDestroyRenderPass(dev, renderPass, nullptr);

    // GoL
    vkDestroyPipeline(dev, golCompPipe, nullptr);
    vkDestroyPipeline(dev, golDispPipe, nullptr);
    vkDestroyPipelineLayout(dev, golCompPipeLayout, nullptr);
    vkDestroyPipelineLayout(dev, golDispPipeLayout, nullptr);
    vkDestroyDescriptorPool(dev, golDescPool, nullptr);
    vkDestroyDescriptorSetLayout(dev, golCompLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, golDispLayout, nullptr);
    vkDestroySampler(dev, golSampler, nullptr);
    for (int i = 0; i < 2; ++i) {
        vkDestroyImageView(dev, golView[i], nullptr);
        vkDestroyImage(dev, golImg[i], nullptr);
        vkFreeMemory(dev, golMem[i], nullptr);
    }

    // Particles
    vkDestroyPipeline(dev, partCompPipe, nullptr);
    vkDestroyPipeline(dev, partDrawPipe, nullptr);
    vkDestroyPipelineLayout(dev, partCompLayout, nullptr);
    vkDestroyPipelineLayout(dev, partDrawLayout, nullptr);
    vkDestroyDescriptorPool(dev, partDescPool, nullptr);
    vkDestroyDescriptorSetLayout(dev, partLayout, nullptr);
    vkDestroyBuffer(dev, partBuf, nullptr);
    vkFreeMemory(dev, partMem, nullptr);

    // Core
    vkDestroySemaphore(dev, semImageAvailable, nullptr);
    vkDestroySemaphore(dev, semRenderDone, nullptr);
    vkDestroyFence(dev, fenceFrame, nullptr);
    vkDestroyCommandPool(dev, cmdPool, nullptr);
    vkDestroyDevice(dev, nullptr);
    if (VALIDATION) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance, debugMessenger, nullptr);
    }
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}
