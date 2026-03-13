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
- `ui.input()` returns `UIInput` — per-frame mouse position, deltas, button press/release/held state.
- `ui.mouseOverUI()` returns **previous frame's** capture result (saved before reset in `beginFrame`). Read this in `buildUI` to gate scene interaction.
- `ui.addMouseCaptureRect(x, y, w, h)` — call in `buildUI` for every visible UI panel (toolbar, windows). Updates current frame's capture state.
- `Clay_Hovered()` is only valid **inside** a `CLAY()` element body (not in the config struct).
- One-frame-lag hover state: store `Clay_Hovered()` results in member bools, use those bools for background colors the next frame.
- `CLAY_STRING(x)` requires a **string literal** (uses `sizeof`). For runtime strings, use `Clay_String{ false, (int32_t)strlen(buf), buf }` with a **member variable** buffer (Clay stores raw pointers read after `buildUI` returns).

### MSVC C++20 Designated Initializer Ordering
MSVC requires designators in declaration order. Key Clay struct orderings:
- `Clay_LayoutConfig`: `sizing` → `padding` → `childGap` → `childAlignment` → `layoutDirection`
- `Clay_ElementDeclaration`: `layout` → `backgroundColor` → `cornerRadius` → `floating`
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
- GoL storage images stay in `VK_IMAGE_LAYOUT_GENERAL` throughout (valid for both storage and sampled).
- One `semRenderDone` semaphore per swapchain image (avoids reuse race VUID-00067).
- Particle SSBO is read via `gl_VertexIndex` in the vertex shader — use `VK_ACCESS_SHADER_READ_BIT` with `VK_PIPELINE_STAGE_VERTEX_SHADER_BIT`, **not** `VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT`.
- Validation layers enabled in debug builds (`#ifndef NDEBUG`).

### Manual Slider Hit-Testing Pattern
Clay does not expose element positions after layout. When a custom widget needs cursor hit-testing (e.g., the viscosity slider), compute the absolute position from known constants that match the Clay sizing declarations exactly. Any mismatch causes offset errors. Wrap labels in fixed-width containers (`CLAY_SIZING_FIXED`) so their layout width matches the constant used in hit-test math.
