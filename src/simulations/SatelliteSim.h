#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../Simulation.h"

#include <vector>
#include <cstdint>
#include <cmath>

// ── Maximum satellites per frame ──────────────────────────────────────────────
static constexpr uint32_t MAX_SATELLITES = 100'000;

// ── Satellite attitude model ───────────────────────────────────────────────────
enum class AttitudeMode
{
    NadirPointing, // flat face toward Earth (Starlink bus/antenna) — brief intense flares
    SunTracking,   // panel normal tracks sun for power — opposition flares
    Tumbling,      // uncontrolled random tumble — chaotic flashes (debris)
};

// ── Orbit distribution type ────────────────────────────────────────────────────
enum class OrbitDistribution
{
    Walker,      // regular Walker constellation: numPlanes planes × perPlane satellites
    RandomShell, // randomly distributed: random RAAN, random incl in [0, incl], jittered alt
    Disk,        // ring or concentric disk in a fixed orbital plane (incl + raan)
                 // Set alignTerminator=true to auto-derive the plane from sunDirECI
};

// ── Per-type satellite parameters (CPU-side, drives GpuSatInput fields) ───────
struct SatelliteType
{
    const char *name;
    glm::vec3 baseColor;      // visual tint (also base reflectance)
    float specExp;            // specular exponent: higher = sharper peak flares
    AttitudeMode attitude;
    float crossSectionM2;     // effective reflective cross-section (m²); drives brightness
};

// ── Constellation descriptor ───────────────────────────────────────────────────
// One entry per shell. Multiple shells can share the same SatelliteType.
// orbitStart/Count are filled by initConstellation() — do not set manually.
struct ConstellationConfig
{
    const char *name;
    float altM;       // orbital altitude above surface (meters)
    float incl;       // Walker: fixed inclination; RandomShell: max inclination (radians)
    int numPlanes;    // Walker: number of planes; other: total satellite count
    int perPlane;     // Walker: satellites per plane; other: ignored (use numPlanes as total)
    uint32_t typeIdx; // index into satTypes[]
    bool enabled;     // visibility toggle (hot-swappable)
    OrbitDistribution distribution = OrbitDistribution::Walker;
    float altJitterM = 0.0f;       // RandomShell: ±altitude jitter; Disk: ±per-satellite alt scatter
    float raan = 0.0f;             // Disk: orbital plane RAAN (radians); ignored if alignTerminator
    bool alignTerminator = false;  // Disk: derive incl+raan from sunDirECI at init time
    int numRings = 1;              // Disk: number of concentric rings (1 = single ring)
    float ringSpacingM = 0.0f;    // Disk: altitude step between consecutive rings (meters)
    // Populated by initConstellation():
    uint32_t orbitStart = 0; // first index into satOrbits[]
    uint32_t orbitCount = 0; // number of orbits belonging to this constellation
};

// ── GPU data structures ───────────────────────────────────────────────────────
// std430 packing: vec3 alignment=16 size=12, so vec3+float = 16 bytes.
// Three vec3+float pairs = 48 bytes total.

struct GpuSatInput
{
    glm::vec3 eciRelPos;   // observer-relative ECI position (meters)
    float range;           // distance (meters)
    glm::vec3 panelNormal; // panel normal in ECI (unit vector), attitude-dependent
    float elevation;       // elevation above local horizon (radians), pre-computed on CPU
    glm::vec3 baseColor;   // satellite tint from SatelliteType
    float specExp;         // specular exponent from SatelliteType
    float crossSection;    // sqrt(crossSectionM2 / 10.0): area brightness scale (~1 = 10 m²)
    float _pad[3];         // pad to 64 bytes (std430 alignment)
};
static_assert(sizeof(GpuSatInput) == 64, "GpuSatInput layout mismatch");

struct GpuSatVisible
{
    glm::vec3 skyDir;     // unit vector in ENU (x=East, y=North, z=Up)
    float flareIntensity; // [0, 1+]
    glm::vec3 baseColor;  // satellite tint
    float angularSize;    // point sprite size hint (pixels)
};
static_assert(sizeof(GpuSatVisible) == 32, "GpuSatVisible layout mismatch");

// Compute push constants (must match sat_flare.comp push_constant block exactly).
// GLSL std430 layout, vec3 aligned to 16 bytes:
//   enuX/Y/Z (vec4): offsets 0,16,32
//   sunDirECI (vec3): offset 48  (48 is 16-aligned ✓)
//   satCount  (uint): offset 60
//   obsECI    (vec3): offset 64  (64 is 16-aligned ✓)
//   pad       (float): offset 76
//   total: 80 bytes
struct SatFlarePC
{
    glm::vec4 enuX;      // East  basis in ECI (w unused)
    glm::vec4 enuY;      // North basis in ECI (w unused)
    glm::vec4 enuZ;      // Up    basis in ECI (w unused)
    glm::vec3 sunDirECI; // unit vector toward sun in ECI
    uint32_t satCount;
    glm::vec3 obsECI; // observer ECI position (meters) for shadow test
    float pad;
}; // total: 80 bytes
static_assert(sizeof(SatFlarePC) == 80, "SatFlarePC layout mismatch");

// Draw push constants (passed to sat_point.vert and both sky shaders).
// GLSL std430 layout:
//   skyView    (mat4):  offset 0
//   fovYRad    (float): offset 64
//   aspect     (float): offset 68
//   pad[2]     (float[2]): offsets 72, 76
//   sunDirENU  (vec4):  offset 80   xyz=direction, w=sin(elevation)
//   moonDirENU (vec4):  offset 96   xyz=moon dir in ENU, w=illuminated fraction
//   total: 112 bytes
struct SatDrawPC
{
    glm::mat4 skyView;    // ENU → camera space
    float fovYRad;        // vertical field of view (radians)
    float aspect;         // viewport width / height
    float pad[2];         // pad to 16-byte boundary before sunDirENU
    glm::vec4 sunDirENU;  // sun direction in ENU (xyz unit vec, w = sin(elevation))
    glm::vec4 moonDirENU; // moon direction in ENU (xyz unit vec, w = illuminated fraction)
}; // total: 112 bytes
static_assert(sizeof(SatDrawPC) == 112, "SatDrawPC layout mismatch");

// ── Sky camera ────────────────────────────────────────────────────────────────
// Azimuth/elevation look direction in the local ENU frame.
// Right-click to capture mouse; WASD-style look via mouse deltas.
struct SkyCamera
{
    float azDeg = 0.0f;    // azimuth of look direction (0=North, 90=East), degrees
    float elDeg = 30.0f;   // elevation of look direction, degrees
    float fovYDeg = 70.0f; // vertical field of view, degrees
    float sens = 0.12f;    // mouse sensitivity (degrees per pixel)
    bool captured = false;

    // Returns a view matrix that transforms ENU directions into camera space.
    // Camera convention: +X=right, +Y=up, -Z=forward (standard OpenGL).
    glm::mat4 viewMatrix() const
    {
        float az = glm::radians(azDeg);
        float el = glm::radians(elDeg);
        // Forward vector in ENU (x=East, y=North, z=Up)
        glm::vec3 fwd{sinf(az) * cosf(el), cosf(az) * cosf(el), sinf(el)};
        // World up = ENU Up. Fall back to North when near zenith/nadir.
        glm::vec3 worldUp = (fabsf(elDeg) > 88.0f)
                                ? glm::vec3{0.0f, 1.0f, 0.0f}  // North when near zenith
                                : glm::vec3{0.0f, 0.0f, 1.0f}; // ENU Up otherwise
        return glm::lookAt(glm::vec3(0.0f), fwd, worldUp);
    }

    void update(GLFWwindow *win, float dmx, float dmy)
    {
        if (!captured)
        {
            if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            {
                captured = true;
                glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            return;
        }
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
        {
            captured = false;
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        azDeg += dmx * sens;
        elDeg -= dmy * sens; // screen Y down → mouse up = negative dmy = increase el
        elDeg = glm::clamp(elDeg, -89.0f, 89.0f);
    }
};

// ── Fixed orbital parameters (one per satellite, computed once at init) ───────
struct SatOrbit
{
    float raan;           // right ascension of ascending node (radians)
    float incl;           // inclination (radians) — per-satellite so RandomShell can vary
    float u0;             // initial mean argument of latitude (radians)
    uint32_t typeIdx;     // index into satTypes[]
    float altM;           // orbital altitude above surface (meters) — per-satellite
    float tumbleRate;     // rotation rate (rad/s); 0 = not tumbling
    float tumblePhase;    // initial rotation angle (radians)
    glm::vec3 tumbleAxis; // fixed body tumble axis (unit vector in ECI)
};

// ── SatelliteSim ──────────────────────────────────────────────────────────────
class SatelliteSim : public Simulation
{
public:
    const char *name() const override { return "Satellite Constellation"; }

    void init(VulkanContext &ctx) override;
    void onResize(VulkanContext &ctx) override;
    void recordCompute(VkCommandBuffer cmd, VulkanContext &ctx, float dt) override;
    void recordDraw(VkCommandBuffer cmd, VulkanContext &ctx, float dt) override;
    void buildUI(float dt, UIRenderer &ui) override;
    VkClearValue clearColor() const override { return {{{0.0f, 0.0f, 0.015f, 1.0f}}}; }
    void cleanup(VkDevice device) override;
    void onKey(GLFWwindow *w, int key, int action) override;
    void onCursorPos(GLFWwindow *w, double x, double y) override;

private:
    // ── SSBOs ─────────────────────────────────────────────────────────────────
    VkBuffer satInputBuf = VK_NULL_HANDLE; // host-visible, CPU writes
    VkDeviceMemory satInputMem = VK_NULL_HANDLE;
    void *satInputMapped = nullptr;
    VkBuffer satVisibleBuf = VK_NULL_HANDLE; // device-local, compute→vertex
    VkDeviceMemory satVisibleMem = VK_NULL_HANDLE;

    // ── Descriptors ───────────────────────────────────────────────────────────
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    // ── Pipelines ─────────────────────────────────────────────────────────────
    VkPipelineLayout compPipeLayout = VK_NULL_HANDLE;
    VkPipeline compPipeline = VK_NULL_HANDLE;
    VkPipelineLayout skyBgPipeLayout = VK_NULL_HANDLE; // sky/ground background
    VkPipeline skyBgPipeline = VK_NULL_HANDLE;
    VkPipelineLayout drawPipeLayout = VK_NULL_HANDLE;
    VkPipeline drawPipeline = VK_NULL_HANDLE;

    // Moon state (updated each frame in updatePositions)
    glm::vec3 moonDirECI{1, 0, 0}; // unit vector toward moon in ECI (equatorial orbit)
    glm::vec4 moonDirENU{0, 1, 0, 0}; // xyz = moon dir in ENU, w = illuminated fraction

    // ── Stars ─────────────────────────────────────────────────────────────────
    struct StarRecord
    {
        glm::vec3 eciDir;      // unit vector toward star in ECI (J2000)
        float rawIntensity;    // magnitude-derived brightness (no night factor)
        glm::vec3 color;       // spectral color from B-V index
        float angSize;         // point sprite size in pixels
    };
    std::vector<StarRecord> starRecords;
    VkBuffer starBuf = VK_NULL_HANDLE;
    VkDeviceMemory starMem = VK_NULL_HANDLE;
    void *starMapped = nullptr;
    VkDescriptorSetLayout starDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool starDescPool = VK_NULL_HANDLE;
    VkDescriptorSet starDescSet = VK_NULL_HANDLE;
    VkPipelineLayout starPipeLayout = VK_NULL_HANDLE;
    VkPipeline starPipeline = VK_NULL_HANDLE;
    uint32_t starCount = 0;

    // ── Simulation state ──────────────────────────────────────────────────────
    SkyCamera camera;
    double simTime = 0.0; // accumulated simulation seconds
    int timeScaleIdx = 1; // index into kTimeScales[]
    uint32_t activeSatCount = 0;
    uint32_t visibleCount = 0; // above-horizon satellites this frame

    // ── ECI → ENU rotation (updated each frame in updatePositions) ────────────
    // Encodes the surface-fixed observer's local frame in ECI coordinates.
    glm::vec4 eci2enuX{1, 0, 0, 0}; // East  basis in ECI
    glm::vec4 eci2enuY{0, 1, 0, 0}; // North basis in ECI
    glm::vec4 eci2enuZ{0, 0, 1, 0}; // Up    basis in ECI

    // ── Sun + observer state (updated each frame in updatePositions) ──────────
    glm::vec3 sunDirECI{1, 0, 0};    // unit vector from Earth toward Sun in ECI
    glm::vec4 sunDirENU{0, 1, 0, 0}; // sun direction in ENU (xyz), w = sin(elevation)
    glm::vec3 obsECI{0, 0, 6371000}; // observer ECI position (meters)

    // ── Satellite type catalogue (defined once in initConstellation) ──────────
    std::vector<SatelliteType> satTypes;

    // ── Orbital parameters (fixed at init, positions updated each frame) ──────
    std::vector<ConstellationConfig> constellations;
    std::vector<SatOrbit> satOrbits;
    std::vector<GpuSatInput> satInputData;

    // ── Mouse state ───────────────────────────────────────────────────────────
    GLFWwindow *win = nullptr;
    bool firstMouse = true;
    double prevX = 0, prevY = 0;
    float dmx = 0, dmy = 0; // accumulated delta for this frame

    // ── UI hover state (one-frame lag) ────────────────────────────────────────
    bool hovSpeed[5] = {};
    bool hovConst[8] = {}; // constellation toggle buttons

    // ── Private helpers ───────────────────────────────────────────────────────
    void createBuffers(VulkanContext &ctx);
    void createDescriptors(VulkanContext &ctx);
    void createComputePipeline(VulkanContext &ctx);
    void createSkyBgPipeline(VulkanContext &ctx);
    void createDrawPipeline(VulkanContext &ctx);
    void initStars(VulkanContext &ctx);
    void createStarPipeline(VulkanContext &ctx);
    void updateStars();
    void initConstellation();       // called once: populates satOrbits
    void updatePositions(double t); // called each frame: fills satInputData + eci2enu
};

// Time scale options (simulated seconds per real second)
static constexpr float kTimeScales[] = {1.0f, 60.0f, 300.0f, 3600.0f, 86400.0f};
static constexpr const char *kTimeLabels[] = {"1x", "1m", "5m", "1h", "1d"};
static constexpr int kNumTimeScales = 5;
