# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.
The primary simulation is **SatelliteSim** (`src/simulations/SatelliteSim.h/.cpp`).
All other simulations (GameOfLife, Particles, Scene3DDemo) are legacy and rarely touched.

---

## Build Commands

```bash
cmake -B build -S .                           # configure (downloads deps via FetchContent)
cmake --build build                           # build + compile shaders + copy SPVs next to exe
cmake --build build --config Release          # release build
```

Run: `build/Debug/ShaderFun.exe`

Shaders: auto-detected glob (`shaders/*.vert|.frag|.comp`), compiled by `glslc`, copied as `shaders/*.spv`. New shader files are picked up automatically on next build.

**Requirements**: Vulkan SDK + `VULKAN_SDK` env var, CMake 3.20+, MSVC C++20.

---

## Architecture

Three layers (stable → frequently changed):

| Layer | Files | Role |
|-------|-------|------|
| Platform | `VulkanContext.h/.cpp` | All Vulkan boilerplate; exposes helpers |
| Framework | `App.h/.cpp`, `Simulation.h`, `UIRenderer.h/.cpp`, `AudioSystem.h/.cpp` | Window, frame loop, UI, audio |
| Simulation | `src/simulations/SatelliteSim.h/.cpp` | All active development |

### Frame Loop Order (`App::drawFrame`)
```
ui.beginFrame()          → resets Clay, saves prevMouseOverUI
sim->buildUI(dt, ui)     → Clay layout; camera look; mouse capture rects
sim->recordCompute(cmd)  → WASD movement; simTime advance; updatePositions(); compute dispatch + barriers
vkCmdBeginRenderPass     → owned by App
sim->recordDraw(cmd)     → sky pass → satellite points → stars
ui.record(cmd)           → Clay → Vulkan quads/text/icons on top
vkCmdEndRenderPass       → owned by App
```

---

## Subsystem: UIRenderer / Clay

- `#define CLAY_IMPLEMENTATION` only in `UIRenderer.cpp`. All other files `#include "clay.h"` without it.
- `ui.input()` → `UIInput`: per-frame mouse/scroll/button state. `scrollY` positive = scroll up. `screenW/H` = window dims.
- `ui.mouseOverUI()` → **previous frame's** capture result. Read in `buildUI` to gate scene interaction.
- `ui.addMouseCaptureRect(x, y, w, h)` — call for every visible panel in `buildUI`.
- `Clay_Hovered()` only valid **inside** a `CLAY()` element body, not in the config struct.
- **One-frame hover lag**: store `Clay_Hovered()` in member bools; use those bools for colors the next frame.
- `CLAY_STRING(x)` requires a **string literal**. For runtime strings: `Clay_String{ false, (int32_t)strlen(buf), buf }` with a **member variable** buffer (Clay stores raw pointers read after `buildUI` returns).
- **Clip rule**: never put `.clip` on a floating container that also has `backgroundColor` — SCISSOR_START fires before RECTANGLE, hiding the background.
- Scrollable containers: `.clip = {.vertical = true, .childOffset = Clay_GetScrollOffset()}` on the content div.

### Icon Atlas
- `ui.loadIcons(ctx, paths, count)` — loads PNGs, packs into RGBA GPU atlas, rebinds descriptor. Call once on first frame (lazy init). Store `VulkanContext*` in your sim.
- Icons: `.image = {.imageData = (void*)(intptr_t)iconIdx}`. Renderer samples the atlas UV range for that index.
- Shader `mode`: `0.0` = solid rect, `1.0` = text glyph, `2.0` = icon sprite. Binding 1 is always valid (1×1 white placeholder at init).

### MSVC C++20 Designated Initializer Ordering
MSVC requires designators in declaration order:
- `Clay_LayoutConfig`: `sizing` → `padding` → `childGap` → `childAlignment` → `layoutDirection`
- `Clay_ElementDeclaration`: `layout` → `backgroundColor` → `cornerRadius` → `aspectRatio` → `image` → `floating` → `custom` → `clip` → `border` → `userData`
- `Clay_FloatingElementConfig`: `offset` → `zIndex` → `pointerCaptureMode` → `attachTo`

### Manual Hit-Testing
Clay does not expose element positions post-layout. Compute absolute positions from constants that exactly match Clay sizing declarations. Wrap labels in `CLAY_SIZING_FIXED` containers so layout width matches hit-test math.

---

## Subsystem: Controls / Keybinding Pipeline

**All interactive keys go through the `keybindings` vector.** The settings window and rebind UI are driven entirely from this vector — no extra wiring needed.

### `KeyBinding` struct
```cpp
struct KeyBinding {
    const char *action;  // display name in settings
    int  key;            // GLFW_KEY_*
    bool held;           // true = polled; false = event (pressed once)
    bool listening;      // true = waiting for rebind input
};
```

### `KB` enum (canonical indices)
```cpp
enum KB {
    KB_TOGGLE_UI  = 0,   // Tab    — event
    KB_PAUSE      = 1,   // Space  — event
    KB_SLOWER     = 2,   // ,      — event
    KB_FASTER     = 3,   // .      — event
    KB_REVERSE    = 4,   // R      — event
    KB_MOVE_BOOST = 5,   // LShift — held
    KB_MOVE_FINE  = 6,   // LCtrl  — held
    KB_COUNT      = 7,
};
```

### Adding a new control (complete checklist)
1. Add `KB_NEWNAME` before `KB_COUNT` in the enum
2. Add one line to `keybindings` in `init()`: `{"Display Name", GLFW_KEY_X, held, false}`
3. Bump `static_assert(KB_COUNT == N)` to the new count
4. Wire the action:
   - **Event** (`held=false`): `if (pressed(KB_NEWNAME)) { ... }` in `onKey()`
   - **Held** (`held=true`): `glfwGetKey(win, keybindings[KB_NEWNAME].key) == GLFW_PRESS` in `recordCompute()`

Settings display, rebinding, hover state, and `keyDisplayName()` all work automatically. `hovRebind[KB_COUNT]` and `kbKeyBuf[KB_COUNT]` are sized by the enum so no array changes are needed.

`keyDisplayName()` handles: letters, digits, Space, Tab, Enter, Esc, Bksp, modifier keys (LShift/RShift/LCtrl/RCtrl/LAlt/RAlt), F-keys (F1–F12), arrow keys, nav cluster (PgUp/PgDn/Home/End/Ins/Del), and common punctuation.

---

## Subsystem: Satellite Types

Each `SatelliteType` composes two surfaces + a diffuse floor:
- `primary` (`SurfaceSpec`) — always active
- `secondary` (`SurfaceSpec`) — optional; `weight=0` disables
- `diffuse` — constant Lambertian floor (always visible)
- `mirrorFrac` — fraction of primary that is near-perfect mirror; adds ultra-narrow spike on top of Phong lobe (MIRROR_BOOST=300×)

`SurfaceSpec`: `{AttitudeMode, specExp, weight}`

### AttitudeMode values
| Mode | surfN | Use case |
|------|-------|----------|
| `NadirPointing` | satNadir | Antenna/array face toward Earth (Starlink) |
| `SunTracking` | sunDirECI | Solar panels track sun (LEO Broadband, ISS) |
| `Tumbling` | spinning around random body axis | Debris, uncontrolled objects |
| `Perpendicular` | cross(surfN0, satNadir) | Secondary only — along orbital track |
| `AntiNadir` | -satNadir | Radiators facing deep space; brighter near horizon |
| `FlatMirror45` | normalize(sunDir + satNadir) | Flat mirror reflecting sunlight straight down |
| `TargetedReflector` | normalize(sunDir + toTarget) | Mirror aimed at nearest valid night-side ground target |

### Satellite type catalogue (typeIdx)
| Idx | Name | Area (m²) | Primary attitude | mirrorFrac |
|-----|------|-----------|-----------------|------------|
| 0 | Starlink | 10 | NadirPointing, spec=18 | 0.05 |
| 1 | LEO Broadband | 5 | SunTracking, spec=18 | 0.02 |
| 2 | GEO Comsat | 50 | SunTracking, spec=3 | 0.10 |
| 3 | ISS | 250 | SunTracking, spec=12 | 0.05 |
| 4 | SpaceX AI Sats | ~600 | SunTracking, spec=18 | 0.01 |
| 5 | Reflect Mirror | 2376 | TargetedReflector, spec=200 | 0.97 |

`crossSection = sqrt(crossSectionM2 / 10.0)` — so 10 m² → 1.0, 2376 m² → ~15.4.

### Adding a new satellite type
1. Add a `SatelliteType` entry to `satTypes` in `initConstellation()` — new typeIdx = last index + 1
2. No GPU struct changes needed; all fields map to existing `GpuSatInput` members
3. Reference the new typeIdx in a `ConstellationConfig`

---

## Subsystem: Orbital Mechanics / Constellations

### Orbit distributions
- **Walker** — `numPlanes × perPlane` satellites, evenly spaced RAAN, random phase per plane
- **RandomShell** — random RAAN, random incl in [0, c.incl], jittered altitude, random tumble axis
- **Disk** — concentric rings in a single orbital plane (incl + raan). `alignTerminator=true` derives incl/raan from sunDirECI at J2000 epoch and precesses RAAN at SSO rate (kSSOPrecRate = 2π/year)

### ConstellationConfig field order
```cpp
// Walker:
{ name, altM, incl, numPlanes, perPlane, typeIdx, enabled, OrbitDistribution::Walker }

// Disk (extra trailing fields):
{ name, altM, incl, numPlanes, perPlane, typeIdx, enabled, OrbitDistribution::Disk,
  altJitterM, raan, alignTerminator, numRings, ringSpacingM }
```
- `perPlane` is **never** ignored — total = `numPlanes × perPlane` for all distributions
- `incl` is ignored when `alignTerminator=true`

### Adding a new constellation
1. Add a `ConstellationConfig` entry to `constellations` in `initConstellation()`
2. Add a hover bool to `hovConst[]` and bump the loop cap if needed (currently `ci < 10`)
3. Check total satellite count against `MAX_SATELLITES` (currently 200,000)

### Current constellation roster (11 total)
| Name | Sats | Alt (km) | Incl | Dist |
|------|------|----------|------|------|
| Starlink Gen1 | 4,392 | 550 | 53° | Walker |
| Starlink Gen2 | 30,480 | 525 | 53.2° | Walker |
| OneWeb | 648 | 1,200 | 87.9° | Walker |
| Amazon LEO | 7,742 | 630 | 51.9° | Walker |
| Guowang | 13,920 | 508 | 85° | Walker |
| ISS | 1 | 408 | 51.6° | Walker |
| SpaceX AI Sat | 20,000 | 575–1,925 km | SSO | Disk+terminator |
| Reflect Orbital | 1,000 | 500 | SSO | Disk+terminator, **disabled** |

`updatePositions()` is O(N) CPU every frame. ~10 ms at 100k sats.

### SSO precession model (alignTerminator=true)
Inclination from J2 formula: `cos(i) = -kSSOPrecRate / (1.5 × n × kJ2 × (Re/a)²)`
RAAN anchored at J2000 epoch (sunDirECIAtJ2000), precesses as `raan_j2000 + kSSOPrecRate × t`.
Call `updatePositions(simTime)` **before** `initConstellation()` so sunDirECI is populated.

---

## Subsystem: TargetedReflector / Mirror Ground Targets

Mirrors in `TargetedReflector` mode aim at the nearest valid night-side ground target.

### Target generation (once at init)
`kNumReflectorTargets = 201` random ECEF unit vectors, uniformly distributed on sphere.
Last slot (index 200) is fixed at the observer spawn point (67°S, 67°W) — always aimed here when in darkness.

### Per-frame update (updatePositions, before satellite loop)
ECEF → ECI: rotate by GMST = `kOmegaEarth × t` around Z axis.
Valid = night side only: `dot(normalize(targetECI), sunDirECI) < 0`.

### Per-satellite normal computation
Scans all 201 targets, picks nearest by `dot(satZenith, normalize(targetECI))`.
Mirror normal = `normalize(sunDirECI + toTarget)` — reflects sunlight toward target by half-vector identity.
Falls back to FlatMirror45 (straight down) if no valid targets exist.

### Mirror slew rate
`kMirrorRotRateDegPerSec = 1.0f` degrees per **simulated** second (not real-time — consistent across all time warp levels).
`satMirrorNormals[i]` stores current physical orientation per satellite.
Zero-vector = uninitialized; snaps to target on first frame, then slews.

---

## Subsystem: Photometry / Shader Constants

**CPU–GPU constant mirror pairs** (must stay in sync):
| C++ constant | GLSL constant | Value |
|---|---|---|
| `kBrightnessScale = 6.0f` | `BRIGHTNESS_SCALE = 4.0` | global flux multiplier |
| `kDaySuppression = 150.0f` | `DAY_SUPPRESSION = 50.0` | sky background ratio |

Note: these are intentionally different — the C++ value is used only for the magnitude readout UI; the GLSL value drives actual rendering.

`effectFlare = flare / (1 + dayBright × DAY_SUPPRESSION)`
`magnitude = kMagRef - 2.5 × log10(effectFlare / kMagRefFlare)` where `kMagRef=6.0`, `kMagRefFlare=0.008`

`MIRROR_BOOST = 300` — peak multiplier for near-perfect mirror alignment. `mirrorExp = max(specExp0 × 300, 8000)` gives sub-degree angular width (matches solar disc ~0.26°).

---

## Subsystem: GpuSatInput Layout (80 bytes, std430)

```
[  0] eciRelPos (vec3) + range (float)
[ 16] surfN0    (vec3) + elevation (float)   — primary surface normal
[ 32] surfN1    (vec3) + specExp0 (float)    — secondary surface normal
[ 48] baseColor (vec3) + specExp1 (float)
[ 64] crossSection + w1 + diffuse + mirrorFrac (float×4)
```

`static_assert(sizeof(GpuSatInput) == 80)` — do not change field order without updating both the C++ struct and the GLSL `SatInput` struct in `sat_flare.comp`.

---

## Subsystem: Sky Glow SSBO

Top-64 brightest flares per frame → host-coherent SSBO (`glowBuf`) → sky fragment shader.

`GpuGlowBuf` (std430): `int count; float pad[3]; vec4 entries[64];` (xyz=ENU dir, w=effectFlare)
`kMaxGlows = 64` must match array size in both `SatelliteSim.h` and `sat_sky.frag`.

CPU top-N: threshold > 0.5; linear scan replaces minimum when array is full.
Shader: `if (e.z < -0.08) continue;` skips below-horizon; log2 intensity scale.

---

## Subsystem: VulkanContext Helpers

```cpp
ctx.device, ctx.physicalDevice, ctx.renderPass, ctx.swapExtent, ctx.swapFormat
ctx.graphicsQueue, ctx.commandPool
ctx.loadShader("shaders/foo.spv")
ctx.createBuffer(size, usage, props, buf, mem)
ctx.createImage(w, h, fmt, usage, img, mem)
ctx.beginOneTimeCommands() / ctx.endOneTimeCommands(cmd)
ctx.imageBarrier(cmd, img, srcAccess, dstAccess, oldLayout, newLayout, srcStage, dstStage)
ctx.findMemoryType(filter, props)
```

Key Vulkan design decisions:
- Single command buffer, single frame in flight
- `VK_ACCESS_SHADER_READ_BIT` + `VK_PIPELINE_STAGE_VERTEX_SHADER_BIT` for compute→vertex SSBO barriers (not `VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT`)
- `onResize` must recreate graphics pipelines (viewport baked in); compute pipelines are viewport-independent

---

## Fixed Simulation State

**Start epoch**: UTC 2026-03-30 05:53:58 → J2000 seconds = `1774849038 - 946728000 + 11×3600 + 20×60 = 828121038`
**Observer**: 67°S 67°W → ECEF `obsDir = {0.1527, -0.3596, -0.9205}`, facing north
**Moon phase offset**: `kMoonPhaseOffsetRad = 3.916 rad` → ~91% waxing gibbous at start epoch
