# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cmake -B build -S .          # configure (first time; downloads GLFW, GLM, Clay, stb via FetchContent)
cmake --build build          # build + compile shaders + copy SPVs next to exe
cmake --build build --config Debug    # explicit debug build
cmake --build build --config Release  # release build
```

There are no tests. Run the output binary directly: `build/Debug/ShaderFun.exe`

To switch simulations, edit `src/main.cpp` (one line), then rebuild. No CMake reconfigure needed.

Shaders are auto-detected by CMake via glob (`shaders/*.vert|.frag|.comp`), compiled by `glslc`, and copied next to the exe as `shaders/*.spv`. Adding a new shader file is picked up automatically on the next build.

**Requirements**: Vulkan SDK installed, `VULKAN_SDK` env var set, CMake 3.20+, MSVC (C++20).

## Architecture

Three layers, from stable to frequently-changed:

| Layer | Files | Role |
|-------|-------|------|
| Platform | `VulkanContext.h/.cpp` | All Vulkan boilerplate; exposes helpers |
| Framework | `App.h/.cpp`, `Simulation.h`, `UIRenderer.h/.cpp` | Window, frame loop, UI rendering |
| Simulations | `src/simulations/*.h/.cpp` | Self-contained GPU experiments |

### Frame Loop Order (App::drawFrame)
```
beginFrame(ui)          → resets Clay, saves prevMouseOverUI, computes UIInput deltas
sim->buildUI(dt, ui)    → declares Clay layout; calls addMouseCaptureRect(); computes forceStrength
recordCompute(cmd)      → compute dispatches + pipeline barriers (before render pass)
vkCmdBeginRenderPass    → owned by App, not simulations
sim->recordDraw(cmd)    → draw calls only, render pass already open
ui.record(cmd)          → Clay → Vulkan quads/text on top of scene
vkCmdEndRenderPass      → owned by App
```

### Simulation Interface (`Simulation.h`)
To add a new simulation: create `src/simulations/MyFoo.h/.cpp`, inherit from `Simulation`, implement the pure virtuals, add one line to `main.cpp`. CMake picks up the `.cpp` automatically.

Key contracts:
- `recordDraw` is called inside an already-open render pass — do **not** call `vkCmdBeginRenderPass`/`vkCmdEndRenderPass`
- `recordCompute` runs before the render pass — put barriers here
- `buildUI` runs between `Clay_BeginLayout` and `Clay_EndLayout` — use CLAY() macros here
- `onResize` must recreate any pipeline that bakes in the viewport size (compute pipelines are viewport-independent and can be skipped)

### UIRenderer / Clay Integration
- `#define CLAY_IMPLEMENTATION` only in `UIRenderer.cpp`. All other files `#include "clay.h"` without it.
- `ui.input()` returns `UIInput` — per-frame mouse/scroll/button state. `UIInput.scrollY` is vertical scroll delta (positive = scroll up). `UIInput.screenW/H` are window dimensions.
- `ui.mouseOverUI()` returns **previous frame's** capture result (saved before reset in `beginFrame`). Read this in `buildUI` to gate scene interaction.
- `ui.addMouseCaptureRect(x, y, w, h)` — call in `buildUI` for every visible UI panel (toolbar, windows). Updates current frame's capture state.
- `Clay_Hovered()` is only valid **inside** a `CLAY()` element body (not in the config struct).
- One-frame-lag hover state: store `Clay_Hovered()` results in member bools, use those bools for background colors the next frame.
- `CLAY_STRING(x)` requires a **string literal** (uses `sizeof`). For runtime strings, use `Clay_String{ false, (int32_t)strlen(buf), buf }` with a **member variable** buffer (Clay stores raw pointers read after `buildUI` returns).

### Icon Atlas (UIRenderer)
- `ui.loadIcons(ctx, paths, count)` — loads PNG files (via stb_image), packs them horizontally into a single RGBA GPU atlas, and rebinds the icon descriptor. Safe to call once from `buildUI` on the first frame (lazy init). Requires storing `VulkanContext*` in your simulation and passing it through.
- Icons render as Clay IMAGE elements: `.image = {.imageData = (void*)(intptr_t)iconIdx}`. The renderer extracts the index via `(int)(intptr_t)rc->renderData.image.imageData` and samples the atlas UV range for that icon.
- The fragment shader uses `mode`: `0.0` = solid rect, `1.0` = text glyph (font atlas R channel), `2.0` = icon sprite (icon atlas RGBA × tint). Icon elements get `mode=2.0` and white tint `{1,1,1,1}` by default.
- Binding 1 (icon atlas) is always valid: a 1×1 white placeholder is created at `UIRenderer::init()` so descriptors are never unbound.
- Scrollable Clay containers: `.clip = {.vertical = true, .childOffset = Clay_GetScrollOffset()}` on the content div.

### MSVC C++20 Designated Initializer Ordering
MSVC requires designators in declaration order. Key Clay struct orderings:
- `Clay_LayoutConfig`: `sizing` → `padding` → `childGap` → `childAlignment` → `layoutDirection`
- `Clay_ElementDeclaration`: `layout` → `backgroundColor` → `cornerRadius` → `aspectRatio` → `image` → `floating` → `custom` → `clip` → `border` → `userData`
- `Clay_FloatingElementConfig`: `offset` → `zIndex` → `pointerCaptureMode` → `attachTo`

### VulkanContext Helpers
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

### Vulkan Design Decisions
- Single command buffer, single frame in flight.
- One `semRenderDone` semaphore per swapchain image (avoids reuse race VUID-00067).
- SSBO data read by the vertex shader via `gl_VertexIndex` — use `VK_ACCESS_SHADER_READ_BIT` + `VK_PIPELINE_STAGE_VERTEX_SHADER_BIT` in the compute→graphics barrier, **not** `VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT`.
- Validation layers enabled in debug builds (`#ifndef NDEBUG`).
- `onResize` must recreate graphics pipelines (viewport baked in); compute pipelines are viewport-independent and can be skipped.

### Manual Hit-Testing Pattern
Clay does not expose element positions after layout. When a widget needs cursor hit-testing, compute the absolute position from constants that exactly match the Clay sizing declarations. Any mismatch causes offset errors. Wrap labels in fixed-width containers (`CLAY_SIZING_FIXED`) so their layout width matches the constant used in hit-test math.

## SatelliteSim

### Satellite Type System
Each `SatelliteType` composes up to two reflective surfaces plus an isotropic diffuse floor:
- `primary` (`SurfaceSpec`) — always active; dominant surface (solar panels, antenna face)
- `secondary` (`SurfaceSpec`) — optional; set `weight=0` to disable
- `diffuse` — constant Lambertian floor for structural body scatter (always faintly visible)

`SurfaceSpec` fields: `attitude` (how the normal is oriented), `specExp` (0 = Lambertian diffuse), `weight` (contribution relative to primary).

`AttitudeMode` values:
- `NadirPointing` — flat face toward Earth; brief intense flares (Starlink bus/antenna)
- `SunTracking` — panel normal tracks sun for power; opposition flares
- `Tumbling` — uncontrolled random tumble; chaotic flashes
- `Perpendicular` — secondary only; normal = `cross(surfN0, satNadir)`
- `AntiNadir` — secondary only; normal = `-satNadir`; radiator panels facing deep space; brightens near the horizon, nearly invisible when satellite is at zenith

### GpuSatInput Layout (80 bytes, std430)
```
[ 0] eciRelPos (vec3) + range (float)
[16] surfN0    (vec3) + elevation (float)   — primary surface normal
[32] surfN1    (vec3) + specExp0 (float)    — secondary surface normal
[48] baseColor (vec3) + specExp1 (float)
[64] crossSection + w1 + diffuse + _pad
```
`crossSection = sqrt(crossSectionM2 / 10.0)` — so 10 m² → scale 1.0.

### ConstellationConfig Field Order
```cpp
// Walker:
{ name, altM, incl, numPlanes, perPlane, typeIdx, enabled, OrbitDistribution::Walker }
//   total sats = numPlanes × perPlane

// Disk (extra trailing args):
{ name, altM, incl, numPlanes, perPlane, typeIdx, enabled, OrbitDistribution::Disk,
  altJitterM, raan, alignTerminator, numRings, ringSpacingM }
//   total sats = numPlanes × perPlane, spread across numRings concentric rings
//   alignTerminator=true: overrides incl+raan to track sunDirECI (orbital plane = terminator plane)
```
- `incl` is ignored when `alignTerminator=true`
- `perPlane` is **not** ignored for Disk — total = `numPlanes × perPlane`

### Constellation UI Limits
- `hovConst[10]` — max 10 constellations in the toggle panel (loop cap in `buildUI`)
- `MAX_SATELLITES = 100'000` — hard cap; `initConstellation()` truncates and warns to stderr if exceeded
- `updatePositions()` is O(N) on CPU every frame; ~10 ms at 100k sats. Move to GPU compute if pushing the cap.

### Current Constellations (10 total, ~98,907 sats when all enabled)
Source: planet4589.org/space/con/conlist.html + FCC filings

| Name | Sats | Alt (km) | Incl | Distribution |
|------|------|----------|------|--------------|
| Starlink Gen1 | 4,392 | 550 | 53° | Walker |
| Starlink Gen2 | 30,480 | 525 | 53.2° | Walker |
| OneWeb | 648 | 1,200 | 87.9° | Walker |
| Kuiper | 7,742 | 630 | 51.9° | Walker |
| Xingwang | 13,920 | 508 | 85° | Walker |
| Telesat | 1,674 | 1,015 | 99° | Walker |
| GEO Belt | 50 | 35,786 | 0° | Walker |
| ISS | 1 | 408 | 51.6° | Walker |
| SpaceX ODC 30° | 20,000 | 1,000 | 30° | Walker |
| SpaceX ODC SSO | 20,000 | 1,250 center | terminator | Disk, alignTerminator |

SpaceX ODC is based on the Jan 2026 FCC filing for up to 1M satellites (Orbital Data Center system). The SSO Disk shell uses `alignTerminator=true` so the ring tracks the Earth-Sun terminator plane.
