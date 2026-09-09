#include "VulkanContext.h"
#include "Log.h"
#include "Paths.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ─── Validation layers (debug only) ──────────────────────────────────────────
#ifdef NDEBUG
constexpr bool VALIDATION = false;
#else
constexpr bool VALIDATION = true;
#endif

static const std::vector<const char *> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"};
static const std::vector<const char *> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// ═════════════════════════════════════════════════════════════════════════════
// Lifecycle
// ═════════════════════════════════════════════════════════════════════════════
void VulkanContext::init(GLFWwindow *window)
{
    createInstance();
    if (VALIDATION)
        setupDebugMessenger();
    createSurface(window);
    pickPhysicalDevice();
    createDevice();
    createSwapchain(window);
    createRenderPass();
    createRenderPassLoad();
    createDepthResources();
    createFramebuffers();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
    createQueryPool();
}

void VulkanContext::recreateSwapchain(GLFWwindow *window)
{
    int w = 0, h = 0;
    while (w == 0 || h == 0)
    {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(device);
    cleanupSwapchain();
    createSwapchain(window);
    createDepthResources();
    createFramebuffers();
    // Recreate per-image semaphores to match new image count
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semRenderDone.resize(swapImages.size());
    for (auto &sem : semRenderDone)
        vkCreateSemaphore(device, &si, nullptr, &sem);
}

void VulkanContext::cleanup()
{
    cleanupSwapchain();
    if (queryPool != VK_NULL_HANDLE)
        vkDestroyQueryPool(device, queryPool, nullptr);
    vkDestroyRenderPass(device, renderPassLoad, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroySemaphore(device, semImageAvailable, nullptr);
    vkDestroyFence(device, fenceFrame, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    if (VALIDATION)
    {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn)
            fn(instance, debugMessenger, nullptr);
    }
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

void VulkanContext::cleanupSwapchain()
{
    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();
    destroyDepthResources();
    for (auto v : swapViews)
        vkDestroyImageView(device, v, nullptr);
    swapViews.clear();
    for (auto sem : semRenderDone)
        vkDestroySemaphore(device, sem, nullptr);
    semRenderDone.clear();
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Instance
// ═════════════════════════════════════════════════════════════════════════════
static bool checkValidationSupport()
{
    uint32_t n;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> layers(n);
    vkEnumerateInstanceLayerProperties(&n, layers.data());
    for (auto *name : VALIDATION_LAYERS)
    {
        bool found = false;
        for (auto &l : layers)
            if (strcmp(l.layerName, name) == 0)
            {
                found = true;
                break;
            }
        if (!found)
            return false;
    }
    return true;
}

void VulkanContext::createInstance()
{
    if (VALIDATION && !checkValidationSupport())
        throw std::runtime_error("Validation layers requested but not available.");

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "SAT LIGHT SIM";
    ai.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    ai.apiVersion = VK_API_VERSION_1_2;

    uint32_t glfwExtCount = 0;
    const char **glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char *> exts(glfwExts, glfwExts + glfwExtCount);
    if (VALIDATION)
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // MoltenVK (macOS) is a non-conformant Vulkan ICD; the loader refuses to enumerate it
    // during vkCreateInstance unless VK_KHR_portability_enumeration is requested and the
    // matching instance flag is set. Checked at runtime rather than #ifdef __APPLE__ so this
    // is a true no-op on Windows/Linux (the extension simply won't be in the supported list).
    uint32_t availExtCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availExtCount, nullptr);
    std::vector<VkExtensionProperties> availExts(availExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availExtCount, availExts.data());
    bool portabilityEnumeration = false;
    for (auto &e : availExts)
        if (strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0)
        {
            portabilityEnumeration = true;
            break;
        }
    if (portabilityEnumeration)
        exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    ci.flags = portabilityEnumeration ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
    if (VALIDATION)
    {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed. No Vulkan-capable driver found — "
                                  "install/update your GPU driver from your GPU vendor's website.");
    Log::line("Vulkan instance created.");
}

// ─── Debug messenger ──────────────────────────────────────────────────────────
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

void VulkanContext::setupDebugMessenger()
{
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn || fn(instance, &ci, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("Failed to set up debug messenger.");
}

// ─── Surface ──────────────────────────────────────────────────────────────────
void VulkanContext::createSurface(GLFWwindow *window)
{
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed.");
}

// ─── Physical device ──────────────────────────────────────────────────────────
QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice pd)
{
    QueueFamilyIndices idx;
    uint32_t n;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    for (uint32_t i = 0; i < n; ++i)
    {
        if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            idx.graphics = i;
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            idx.compute = i;
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
        // Prefer the graphics queue to also present
        if (present && !idx.graphics.has_value())
            idx.graphics = i;
        if (idx.complete())
            break;
    }
    // If graphics doesn't support present, find one that does
    if (idx.complete())
    {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, *idx.graphics, surface, &present);
        if (!present)
        {
            for (uint32_t i = 0; i < n; ++i)
            {
                VkBool32 p = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &p);
                if (p)
                {
                    idx.graphics = i;
                    break;
                }
            }
        }
    }
    return idx;
}

SwapchainDetails VulkanContext::querySwapchainDetails(VkPhysicalDevice pd)
{
    SwapchainDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &d.caps);
    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, nullptr);
    d.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, d.formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, nullptr);
    d.modes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &n, d.modes.data());
    return d;
}

void VulkanContext::pickPhysicalDevice()
{
    uint32_t n;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    if (!n)
        throw std::runtime_error("No GPUs with Vulkan support.");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance, &n, devs.data());

    for (auto pd : devs)
    {
        // Check required device extensions
        uint32_t extCount;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
        bool allExtsFound = true;
        for (auto *req : DEVICE_EXTENSIONS)
        {
            bool found = false;
            for (auto &e : exts)
                if (strcmp(e.extensionName, req) == 0)
                {
                    found = true;
                    break;
                }
            if (!found)
            {
                allExtsFound = false;
                break;
            }
        }
        if (!allExtsFound)
            continue;

        // Check queue families and swapchain support
        auto qi = findQueueFamilies(pd);
        auto sc = querySwapchainDetails(pd);
        if (!qi.complete() || sc.formats.empty() || sc.modes.empty())
            continue;

        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        // Prefer discrete GPU; accept any suitable GPU as fallback
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            physicalDevice = pd;
            std::cout << "GPU: " << p.deviceName << " (discrete)\n";
            break;
        }
        if (!physicalDevice)
        {
            physicalDevice = pd;
            std::cout << "GPU: " << p.deviceName << "\n";
        }
    }
    if (!physicalDevice)
        throw std::runtime_error("Failed to find a suitable GPU. Your graphics driver may be missing "
                                  "Vulkan support — install/update it from your GPU vendor's website.");

    VkPhysicalDeviceProperties chosen;
    vkGetPhysicalDeviceProperties(physicalDevice, &chosen);
    {
        std::ostringstream oss;
        oss << "GPU selected: " << chosen.deviceName
            << " (driver " << VK_VERSION_MAJOR(chosen.driverVersion) << "."
            << VK_VERSION_MINOR(chosen.driverVersion) << "." << VK_VERSION_PATCH(chosen.driverVersion)
            << ", Vulkan API " << VK_VERSION_MAJOR(chosen.apiVersion) << "."
            << VK_VERSION_MINOR(chosen.apiVersion) << "." << VK_VERSION_PATCH(chosen.apiVersion) << ")";
        Log::line(oss.str());
    }

    // The Vulkan spec only guarantees 128 bytes of push-constant space, and every push-constant
    // struct in this project now fits exactly that (SatDrawPC / PointDrawPC / CloudMarchPC /
    // SatOrbitPC / SatFlarePC all static_assert == 128 in SatelliteSim.h).
    //
    // This gate read 144 until 2026-09-08 — the pre-trim SatDrawPC size. That made it reject
    // precisely the hardware the 128-byte trim was performed FOR: an old AMD integrated part
    // reporting the guaranteed minimum would fail here with "requires 144" even though the app
    // no longer needs more than 128. Keep this number equal to the largest push-constant
    // struct; if one ever grows past 128 the correct fix is to move fields into the CloudParams
    // UBO, not to raise this (see "Push-constant relief" in GpuCloudParams).
    constexpr uint32_t kRequiredPushConstantsSize = 128;
    if (chosen.limits.maxPushConstantsSize < kRequiredPushConstantsSize)
    {
        std::ostringstream oss;
        oss << "GPU " << chosen.deviceName << " only supports " << chosen.limits.maxPushConstantsSize
            << " bytes of push constants; this app requires " << kRequiredPushConstantsSize << ".";
        Log::line("FATAL: " + oss.str());
        throw std::runtime_error(oss.str());
    }

    logDeviceLimits(chosen.limits);
}

// ─── Device limit / capability audit ──────────────────────────────────────────
// Every value here is something this project exceeds the Vulkan *guaranteed minimum* for, so
// each is a real portability cliff on old or mobile-class hardware rather than a theoretical
// one. They are logged unconditionally (a bug report from a machine nobody has access to is
// otherwise unanswerable) and the genuinely fatal ones throw with a message naming the asset
// or shader responsible, instead of surfacing as a vkCreateImage/vkCreateShaderModule failure
// with no context.
void VulkanContext::logDeviceLimits(const VkPhysicalDeviceLimits &lim)
{
    // Largest 2D texture the app loads: assets/textures/earth_elevation.png at 14999x7500.
    // Guaranteed minimum is 4096. GCN 1.0 and Metal both allow 16384, so the shipped DEM fits
    // with little headroom — anything larger needs downscaling, not a bigger number here.
    constexpr uint32_t kNeeded2D = 14999;
    // aurora_noise.comp bakes a 1024x16x256 volume; cloud noise is 128^3. Guaranteed minimum
    // for maxImageDimension3D is only 256, i.e. the aurora volume is 4x over the floor.
    constexpr uint32_t kNeeded3D = 1024;
    // cloud_march.comp / scene_depth.comp / flare_blur.comp all dispatch local_size 16x16 = 256
    // invocations. Guaranteed minimum is 128.
    constexpr uint32_t kNeededWorkgroupInvocations = 256;
    // cloud_march.comp's two tile-cull lists: 384 beams x 3 arrays + 128 light indices + counters.
    constexpr uint32_t kNeededSharedMemory = 6 * 1024;

    {
        std::ostringstream oss;
        oss << "Device limits: maxImageDimension2D=" << lim.maxImageDimension2D
            << " maxImageDimension3D=" << lim.maxImageDimension3D
            << " maxComputeWorkGroupInvocations=" << lim.maxComputeWorkGroupInvocations
            << " maxComputeSharedMemorySize=" << lim.maxComputeSharedMemorySize
            << " maxPerStageDescriptorSampledImages=" << lim.maxPerStageDescriptorSampledImages
            << " maxPerStageDescriptorStorageBuffers=" << lim.maxPerStageDescriptorStorageBuffers
            << " maxPerStageDescriptorUniformBuffers=" << lim.maxPerStageDescriptorUniformBuffers
            << " maxPushConstantsSize=" << lim.maxPushConstantsSize;
        Log::line(oss.str());
    }

    struct Requirement
    {
        const char *what;
        uint32_t need;
        uint32_t have;
    };
    const Requirement reqs[] = {
        {"2D texture size (earth_elevation.png is 14999x7500)", kNeeded2D, lim.maxImageDimension2D},
        {"3D texture size (aurora noise volume is 1024 wide)", kNeeded3D, lim.maxImageDimension3D},
        {"compute workgroup invocations (shaders use local_size 16x16)",
         kNeededWorkgroupInvocations, lim.maxComputeWorkGroupInvocations},
        {"compute shared memory (cloud_march.comp tile culling)",
         kNeededSharedMemory, lim.maxComputeSharedMemorySize},
        // sat_sky.frag's descriptor set binds 15 combined image samplers and 6 storage buffers
        // in one stage. Guaranteed minimums are 16 and 4 respectively — the storage-buffer count
        // is the one actually over the floor, and MoltenVK is the realistic place to hit it,
        // since it maps SSBOs, UBOs and vertex buffers into Metal's 31 per-stage buffer slots.
        {"per-stage sampled images (sat_sky.frag binds 15)", 15,
         lim.maxPerStageDescriptorSampledImages},
        {"per-stage storage buffers (sat_sky.frag binds 6)", 6,
         lim.maxPerStageDescriptorStorageBuffers},
    };

    std::ostringstream fail;
    bool anyFail = false;
    for (const auto &r : reqs)
    {
        if (r.have < r.need)
        {
            anyFail = true;
            fail << "\n  - " << r.what << ": needs " << r.need << ", GPU reports " << r.have;
        }
    }
    if (anyFail)
    {
        std::string msg = "This GPU is below the limits this app requires:" + fail.str();
        Log::line("FATAL: " + msg);
        throw std::runtime_error(msg);
    }
}

// ─── Logical device ───────────────────────────────────────────────────────────
void VulkanContext::createDevice()
{
    auto qi = findQueueFamilies(physicalDevice);
    graphicsFamily = *qi.graphics;
    computeFamily = *qi.compute;

    std::set<uint32_t> uniqueFamilies = {graphicsFamily, computeFamily};
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qCIs;
    for (uint32_t fam : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = fam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        qCIs.push_back(qci);
    }

    // Requesting a feature the device does not advertise makes vkCreateDevice fail outright with
    // VK_ERROR_FEATURE_NOT_PRESENT — no indication of WHICH feature, on a code path that runs
    // before any window content appears. Both of these are near-universal but neither is
    // guaranteed by the spec, so check availability and report by name.
    VkPhysicalDeviceFeatures available{};
    vkGetPhysicalDeviceFeatures(physicalDevice, &available);

    VkPhysicalDeviceFeatures features{};
    features.shaderStorageImageExtendedFormats = VK_TRUE; // for rgba8 storage images
    features.largePoints = VK_TRUE;                       // for gl_PointSize in particles

    {
        const std::pair<const char *, VkBool32> required[] = {
            {"shaderStorageImageExtendedFormats", available.shaderStorageImageExtendedFormats},
            {"largePoints", available.largePoints},
        };
        std::string missing;
        for (const auto &[name, have] : required)
            if (!have)
                missing += std::string(missing.empty() ? "" : ", ") + name;
        if (!missing.empty())
        {
            std::string msg = "GPU does not support required Vulkan feature(s): " + missing;
            Log::line("FATAL: " + msg);
            throw std::runtime_error(msg);
        }
    }

    // On MoltenVK, VK_KHR_portability_subset MUST be enabled if the device supports it (spec
    // requirement, not optional). Its name macro lives behind vulkan_beta.h/VK_ENABLE_BETA_EXTENSIONS
    // on most SDK versions, so it's matched by literal string here rather than pulling that header
    // in project-wide. No-op on Windows/Linux: the device never reports this extension there.
    std::vector<const char *> deviceExts = DEVICE_EXTENSIONS;
    {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());
        for (auto &e : exts)
            if (strcmp(e.extensionName, "VK_KHR_portability_subset") == 0)
            {
                deviceExts.push_back("VK_KHR_portability_subset");
                break;
            }
    }

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = (uint32_t)qCIs.size();
    ci.pQueueCreateInfos = qCIs.data();
    ci.enabledExtensionCount = (uint32_t)deviceExts.size();
    ci.ppEnabledExtensionNames = deviceExts.data();
    ci.pEnabledFeatures = &features;
    if (VALIDATION)
    {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateDevice(physicalDevice, &ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed.");
    Log::line("Logical device created.");

    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);
}

// ─── Swapchain ────────────────────────────────────────────────────────────────
void VulkanContext::createSwapchain(GLFWwindow *window)
{
    auto sc = querySwapchainDetails(physicalDevice);

    // Pick format (prefer sRGB B8G8R8A8)
    VkSurfaceFormatKHR fmt = sc.formats[0];
    for (auto &f : sc.formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            fmt = f;
            break;
        }

    // Pick present mode: honor presentModePreference (NEW-7 frame limiter) if the surface
    // actually supports it, otherwise fall back to FIFO, which every Vulkan implementation
    // guarantees.
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    for (auto &m : sc.modes)
        if (m == presentModePreference)
        {
            pm = m;
            break;
        }

    // Extent
    VkExtent2D ext;
    if (sc.caps.currentExtent.width != UINT32_MAX)
    {
        ext = sc.caps.currentExtent;
    }
    else
    {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        ext = {std::clamp((uint32_t)w, sc.caps.minImageExtent.width, sc.caps.maxImageExtent.width),
               std::clamp((uint32_t)h, sc.caps.minImageExtent.height, sc.caps.maxImageExtent.height)};
    }

    uint32_t imgCount = sc.caps.minImageCount + 1;
    if (sc.caps.maxImageCount && imgCount > sc.caps.maxImageCount)
        imgCount = sc.caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = imgCount;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // UC6: screenshots need to vkCmdCopyImageToBuffer straight off the swapchain image, which
    // requires TRANSFER_SRC usage — only request it when the surface actually advertises support
    // (near-universal, but not guaranteed by the spec); requesting an unsupported bit would fail
    // swapchain creation outright, so this degrades to "no screenshots" instead.
    screenshotSupported = (sc.caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    if (screenshotSupported)
        ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    uint32_t qfams[] = {graphicsFamily, computeFamily};
    if (graphicsFamily != computeFamily)
    {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = qfams;
    }
    else
    {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = sc.caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed.");

    swapFormat = fmt.format;
    swapExtent = ext;
    {
        std::ostringstream oss;
        oss << "Swapchain created: " << swapExtent.width << "x" << swapExtent.height
            << ", present mode " << pm << ", " << imgCount << " images.";
        Log::line(oss.str());
    }

    uint32_t cnt;
    vkGetSwapchainImagesKHR(device, swapchain, &cnt, nullptr);
    swapImages.resize(cnt);
    vkGetSwapchainImagesKHR(device, swapchain, &cnt, swapImages.data());

    swapViews.resize(cnt);
    for (uint32_t i = 0; i < cnt; ++i)
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = swapImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = swapFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &vci, nullptr, &swapViews[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateImageView failed.");
    }
}

// ─── Depth format ─────────────────────────────────────────────────────────────
VkFormat VulkanContext::findDepthFormat()
{
    for (VkFormat fmt : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT})
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, fmt, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return fmt;
    }
    throw std::runtime_error("Failed to find supported depth format.");
}

// ─── Render pass ──────────────────────────────────────────────────────────────
void VulkanContext::createRenderPass()
{
    depthFormat = findDepthFormat();

    VkAttachmentDescription color{};
    color.format = swapFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format = depthFormat;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    // Subpass dependency: ensure compute finishes before fragment reads
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkAttachmentDescription attachments[] = {color, depth};
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 2;
    ci.pAttachments = attachments;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;

    if (vkCreateRenderPass(device, &ci, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed.");
}

// ─── Load-based render pass (resolution scaling support) ──────────────────────
// Identical to createRenderPass() except: color attachment uses LOAD instead of CLEAR (the
// simulation is expected to have pre-filled it via a blit in recordPrePass, see Simulation.h),
// with initialLayout TRANSFER_DST_OPTIMAL to match the state a blit destination is left in
// (rather than UNDEFINED, which LOAD would otherwise read as garbage). Depth still CLEARs
// normally — this app does not blit depth (see SatelliteSim's resolution-scaling comments for
// why: depth-format blit support isn't spec-guaranteed, a real portability concern specifically
// on the lower-end hardware this feature targets), so depth keeps its normal UNDEFINED/CLEAR
// behavior unchanged. Reuses the SAME ctx.framebuffers as renderPass — render pass compatibility
// only requires matching attachment format/sample-count, not matching load/store ops or layouts.
void VulkanContext::createRenderPassLoad()
{
    VkAttachmentDescription color{};
    color.format = swapFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format = depthFormat;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    // Two dependencies: the usual compute->fragment one (unchanged from renderPass), plus a new
    // TRANSFER->color-attachment one so the render pass's automatic initialLayout transition
    // waits for recordPrePass's blit to actually finish writing before this pass reads/writes it.
    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].dstSubpass = 0;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = {color, depth};
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 2;
    ci.pAttachments = attachments;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 2;
    ci.pDependencies = deps;

    if (vkCreateRenderPass(device, &ci, nullptr, &renderPassLoad) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass (load variant) failed.");
}

// ─── Depth resources ──────────────────────────────────────────────────────────
void VulkanContext::createDepthResources()
{
    createImage(swapExtent.width, swapExtent.height, depthFormat,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                depthImage, depthMemory);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = depthImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = depthFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &vci, nullptr, &depthView) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImageView (depth) failed.");
}

void VulkanContext::destroyDepthResources()
{
    if (depthView)
    {
        vkDestroyImageView(device, depthView, nullptr);
        depthView = VK_NULL_HANDLE;
    }
    if (depthImage)
    {
        vkDestroyImage(device, depthImage, nullptr);
        depthImage = VK_NULL_HANDLE;
    }
    if (depthMemory)
    {
        vkFreeMemory(device, depthMemory, nullptr);
        depthMemory = VK_NULL_HANDLE;
    }
}

// ─── Framebuffers ─────────────────────────────────────────────────────────────
void VulkanContext::createFramebuffers()
{
    framebuffers.resize(swapViews.size());
    for (size_t i = 0; i < swapViews.size(); ++i)
    {
        VkImageView attachments[] = {swapViews[i], depthView};
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass = renderPass;
        ci.attachmentCount = 2;
        ci.pAttachments = attachments;
        ci.width = swapExtent.width;
        ci.height = swapExtent.height;
        ci.layers = 1;
        if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer failed.");
    }
}

// ─── Command pool & buffer ────────────────────────────────────────────────────
void VulkanContext::createCommandPool()
{
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphicsFamily;
    if (vkCreateCommandPool(device, &ci, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateCommandPool failed.");
}

void VulkanContext::createCommandBuffer()
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &ai, &commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers failed.");
}

// ─── Sync objects ─────────────────────────────────────────────────────────────
void VulkanContext::createSyncObjects()
{
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first frame doesn't hang

    if (vkCreateSemaphore(device, &si, nullptr, &semImageAvailable) != VK_SUCCESS ||
        vkCreateFence(device, &fi, nullptr, &fenceFrame) != VK_SUCCESS)
        throw std::runtime_error("Failed to create sync objects.");

    // Create one render-done semaphore per swapchain image.
    semRenderDone.resize(swapImages.size());
    for (auto &sem : semRenderDone)
        if (vkCreateSemaphore(device, &si, nullptr, &sem) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render-done semaphore.");
}

// ─── GPU timestamp query pool ──────────────────────────────────────────────────
void VulkanContext::createQueryPool()
{
    // Opt-out escape hatch. On MoltenVK over a Metal GPU without MTLCounterSamplingPoint
    // support (2015-era AMD/Intel), each vkCmdWriteTimestamp forces a command-encoder split;
    // ~9 per frame can dominate the frame time on that path. SATLIGHTSIM_NO_GPU_TIMERS=1
    // leaves queryPool null, so every writeTimestamp/resetTimestamps/resolveTimestamps call
    // below no-ops and the "GPU FRAME BREAKDOWN" HUD simply reads zero.
    if (const char *e = std::getenv("SATLIGHTSIM_NO_GPU_TIMERS"); e && e[0] == '1')
    {
        Log::line("GPU timestamp queries disabled via SATLIGHTSIM_NO_GPU_TIMERS.");
        return;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    // A zero period means the device reports no usable timestamp resolution;
    // resolveTimestamps() checks this and no-ops rather than dividing by zero.
    timestampPeriodNs = (double)props.limits.timestampPeriod;
    if (timestampPeriodNs <= 0.0)
        return;

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = kTimestampCount;
    if (vkCreateQueryPool(device, &qpci, nullptr, &queryPool) != VK_SUCCESS)
        queryPool = VK_NULL_HANDLE; // profiling is best-effort; app must run without it
}

void VulkanContext::resetTimestamps(VkCommandBuffer cmd)
{
    if (queryPool != VK_NULL_HANDLE)
        vkCmdResetQueryPool(cmd, queryPool, 0, kTimestampCount);
}

void VulkanContext::writeTimestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage, uint32_t slot)
{
    if (queryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(cmd, stage, queryPool, slot);
}

// Called once per frame in App::drawFrame, right after the fence wait — with a
// single frame in flight, the fence signaling guarantees the previous frame's
// queries have already completed, so VK_QUERY_RESULT_WAIT_BIT never actually
// blocks here (it's there for correctness, not as a stall).
void VulkanContext::resolveTimestamps()
{
    if (queryPool == VK_NULL_HANDLE)
        return;
    uint64_t raw[kTimestampCount];
    VkResult r = vkGetQueryPoolResults(device, queryPool, 0, kTimestampCount,
                                       sizeof(raw), raw, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS)
        return;
    for (uint32_t i = 0; i < kTimestampCount; ++i)
        timestampMs[i] = (double)(raw[i] - raw[0]) * timestampPeriodNs / 1.0e6;
    timestampsReady = true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════
VkShaderModule VulkanContext::loadShader(const std::string &path)
{
    // Resolve relative paths against the exe directory so the app can be run from any CWD.
    std::filesystem::path resolved = path;
    if (resolved.is_relative())
        resolved = std::filesystem::path(Paths::exeDir()) / resolved;
    std::ifstream f(resolved, std::ios::ate | std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open shader: " + resolved.string());
    size_t sz = f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), sz);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz;
    ci.pCode = reinterpret_cast<const uint32_t *>(buf.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed for: " + path);
    return mod;
}

uint32_t VulkanContext::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("Failed to find suitable memory type.");
}

void VulkanContext::createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                                VkImage &img, VkDeviceMemory &mem, uint32_t mipLevels,
                                uint32_t depth)
{
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = (depth > 1) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, depth};
    ci.mipLevels = mipLevels;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ci, nullptr, &img) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImage failed.");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, img, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (image) failed.");
    vkBindImageMemory(device, img, mem, 0);
}

// Linear-filtered vkCmdBlitImage is NOT guaranteed for every format: it requires the source
// format to advertise VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT for its tiling. Desktop
// drivers support it for every format this project blits, but the spec allows otherwise and the
// result of ignoring it is undefined behaviour, not a clean error — so callers pick their filter
// through here. Results are cached: this is called per texture upload and per scaled frame.
VkFilter VulkanContext::bestBlitFilter(VkFormat fmt) const
{
    static std::map<VkFormat, VkFilter> cache;
    if (auto it = cache.find(fmt); it != cache.end())
        return it->second;

    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, fmt, &props);
    const bool linear =
        (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    if (!linear)
    {
        std::ostringstream oss;
        oss << "WARN: format " << (int)fmt << " does not support linear blit filtering; "
            << "falling back to NEAREST (mipmaps and scaled-resolution upscale will look coarser).";
        Log::line(oss.str());
    }
    VkFilter f = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    cache[fmt] = f;
    return f;
}

void VulkanContext::generateMipmaps(VkCommandBuffer cmd, VkImage img, VkFormat fmt,
                                     uint32_t w, uint32_t h, uint32_t mipLevels)
{
    const VkFilter blitFilter = bestBlitFilter(fmt);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.image = img;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    b.subresourceRange.levelCount = 1;

    int32_t mipW = (int32_t)w;
    int32_t mipH = (int32_t)h;

    for (uint32_t i = 1; i < mipLevels; ++i) {
        // Transition previous mip TRANSFER_DST → TRANSFER_SRC
        b.subresourceRange.baseMipLevel = i - 1;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        // Blit mip i-1 → mip i
        int32_t nextW = mipW > 1 ? mipW / 2 : 1;
        int32_t nextH = mipH > 1 ? mipH / 2 : 1;
        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nextW, nextH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        vkCmdBlitImage(cmd,
            img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, blitFilter);

        // Transition previous mip TRANSFER_SRC → SHADER_READ_ONLY
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        mipW = nextW;
        mipH = nextH;
    }

    // Transition the last mip TRANSFER_DST → SHADER_READ_ONLY
    b.subresourceRange.baseMipLevel = mipLevels - 1;
    b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);
}

void VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem)
{
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &ci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed.");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (buffer) failed.");
    vkBindBufferMemory(device, buf, mem, 0);
}

VkCommandBuffer VulkanContext::beginOneTimeCommands()
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endOneTimeCommands(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

void VulkanContext::imageBarrier(VkCommandBuffer cmd, VkImage image,
                                 VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                 VkImageLayout oldLayout, VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}
