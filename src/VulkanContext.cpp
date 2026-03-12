#include "VulkanContext.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
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
// Lifecycle
// ═════════════════════════════════════════════════════════════════════════════
void VulkanContext::init(GLFWwindow* window) {
    createInstance();
    if (VALIDATION) setupDebugMessenger();
    createSurface(window);
    pickPhysicalDevice();
    createDevice();
    createSwapchain(window);
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
}

void VulkanContext::recreateSwapchain(GLFWwindow* window) {
    int w = 0, h = 0;
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(device);
    cleanupSwapchain();
    createSwapchain(window);
    createFramebuffers();
    // Recreate per-image semaphores to match new image count
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semRenderDone.resize(swapImages.size());
    for (auto& sem : semRenderDone)
        vkCreateSemaphore(device, &si, nullptr, &sem);
}

void VulkanContext::cleanup() {
    cleanupSwapchain();
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroySemaphore(device, semImageAvailable, nullptr);
    vkDestroyFence(device, fenceFrame, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    if (VALIDATION) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance, debugMessenger, nullptr);
    }
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

void VulkanContext::cleanupSwapchain() {
    for (auto fb  : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();
    for (auto v   : swapViews)    vkDestroyImageView(device, v, nullptr);
    swapViews.clear();
    for (auto sem : semRenderDone) vkDestroySemaphore(device, sem, nullptr);
    semRenderDone.clear();
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Instance
// ═════════════════════════════════════════════════════════════════════════════
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

void VulkanContext::createInstance() {
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
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

void VulkanContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                   | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                   | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn || fn(instance, &ci, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to set up debug messenger.");
}

// ─── Surface ──────────────────────────────────────────────────────────────────
void VulkanContext::createSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed.");
}

// ─── Physical device ──────────────────────────────────────────────────────────
QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice pd) {
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

SwapchainDetails VulkanContext::querySwapchainDetails(VkPhysicalDevice pd) {
    SwapchainDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &d.caps);
    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, nullptr);
    d.formats.resize(n); vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, d.formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, nullptr);
    d.modes.resize(n); vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, d.modes.data());
    return d;
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t n; vkEnumeratePhysicalDevices(instance, &n, nullptr);
    if (!n) throw std::runtime_error("No GPUs with Vulkan support.");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance, &n, devs.data());

    for (auto pd : devs) {
        // Check required device extensions
        uint32_t extCount; vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
        bool allExtsFound = true;
        for (auto* req : DEVICE_EXTENSIONS) {
            bool found = false;
            for (auto& e : exts) if (strcmp(e.extensionName, req) == 0) { found = true; break; }
            if (!found) { allExtsFound = false; break; }
        }
        if (!allExtsFound) continue;

        // Check queue families and swapchain support
        auto qi = findQueueFamilies(pd);
        auto sc = querySwapchainDetails(pd);
        if (!qi.complete() || sc.formats.empty() || sc.modes.empty()) continue;

        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        // Prefer discrete GPU; accept any suitable GPU as fallback
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice = pd;
            std::cout << "GPU: " << p.deviceName << " (discrete)\n";
            break;
        }
        if (!physicalDevice) {
            physicalDevice = pd;
            std::cout << "GPU: " << p.deviceName << "\n";
        }
    }
    if (!physicalDevice) throw std::runtime_error("Failed to find a suitable GPU.");
}

// ─── Logical device ───────────────────────────────────────────────────────────
void VulkanContext::createDevice() {
    auto qi = findQueueFamilies(physicalDevice);
    graphicsFamily = *qi.graphics;
    computeFamily  = *qi.compute;

    std::set<uint32_t> uniqueFamilies = {graphicsFamily, computeFamily};
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

    if (vkCreateDevice(physicalDevice, &ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed.");

    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, computeFamily,  0, &computeQueue);
}

// ─── Swapchain ────────────────────────────────────────────────────────────────
void VulkanContext::createSwapchain(GLFWwindow* window) {
    auto sc = querySwapchainDetails(physicalDevice);

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
    uint32_t qfams[] = {graphicsFamily, computeFamily};
    if (graphicsFamily != computeFamily) {
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

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed.");

    swapFormat = fmt.format;
    swapExtent = ext;

    uint32_t cnt; vkGetSwapchainImagesKHR(device, swapchain, &cnt, nullptr);
    swapImages.resize(cnt); vkGetSwapchainImagesKHR(device, swapchain, &cnt, swapImages.data());

    swapViews.resize(cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = swapImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = swapFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &vci, nullptr, &swapViews[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateImageView failed.");
    }
}

// ─── Render pass ──────────────────────────────────────────────────────────────
void VulkanContext::createRenderPass() {
    VkAttachmentDescription color{};
    color.format         = swapFormat;
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

    if (vkCreateRenderPass(device, &ci, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed.");
}

// ─── Framebuffers ─────────────────────────────────────────────────────────────
void VulkanContext::createFramebuffers() {
    framebuffers.resize(swapViews.size());
    for (size_t i = 0; i < swapViews.size(); ++i) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = renderPass;
        ci.attachmentCount = 1;
        ci.pAttachments    = &swapViews[i];
        ci.width  = swapExtent.width;
        ci.height = swapExtent.height;
        ci.layers = 1;
        if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer failed.");
    }
}

// ─── Command pool & buffer ────────────────────────────────────────────────────
void VulkanContext::createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphicsFamily;
    if (vkCreateCommandPool(device, &ci, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateCommandPool failed.");
}

void VulkanContext::createCommandBuffer() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &ai, &commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers failed.");
}

// ─── Sync objects ─────────────────────────────────────────────────────────────
void VulkanContext::createSyncObjects() {
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first frame doesn't hang

    if (vkCreateSemaphore(device, &si, nullptr, &semImageAvailable) != VK_SUCCESS ||
        vkCreateFence    (device, &fi, nullptr, &fenceFrame)         != VK_SUCCESS)
        throw std::runtime_error("Failed to create sync objects.");

    // Create one render-done semaphore per swapchain image.
    semRenderDone.resize(swapImages.size());
    for (auto& sem : semRenderDone)
        if (vkCreateSemaphore(device, &si, nullptr, &sem) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render-done semaphore.");
}

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════
VkShaderModule VulkanContext::loadShader(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open shader: " + path);
    size_t sz = f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), sz);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed for: " + path);
    return mod;
}

uint32_t VulkanContext::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("Failed to find suitable memory type.");
}

void VulkanContext::createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
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
    if (vkCreateImage(device, &ci, nullptr, &img) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImage failed.");

    VkMemoryRequirements req; vkGetImageMemoryRequirements(device, img, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (image) failed.");
    vkBindImageMemory(device, img, mem, 0);
}

void VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &ci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed.");

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (buffer) failed.");
    vkBindBufferMemory(device, buf, mem, 0);
}

VkCommandBuffer VulkanContext::beginOneTimeCommands() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endOneTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

void VulkanContext::imageBarrier(VkCommandBuffer cmd, VkImage image,
                                 VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                 VkImageLayout oldLayout, VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask       = srcAccess;
    b.dstAccessMask       = dstAccess;
    b.oldLayout           = oldLayout;
    b.newLayout           = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = image;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}
