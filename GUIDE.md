# ShaderFun — Vulkan Project Guide

A playground for GPU-driven simulations using Vulkan compute and graphics shaders.
This document explains **what** the code does, **why** each piece exists, and **how to experiment** with it.

---

## Table of Contents
1. [Project Layout](#1-project-layout)
2. [The Vulkan Mental Model](#2-the-vulkan-mental-model)
3. [Initialization Pipeline](#3-initialization-pipeline)
4. [The Frame Loop](#4-the-frame-loop)
5. [Simulation: Game of Life](#5-simulation-game-of-life)
6. [Simulation: Particles](#6-simulation-particles)
7. [Shader Reference](#7-shader-reference)
8. [Build & Run](#8-build--run)
9. [How to Experiment](#9-how-to-experiment)

---

## 1. Project Layout

```
Shader_Fun/
├── CMakeLists.txt          # Build system — fetches deps, compiles shaders
├── GUIDE.md                # This file
├── .vscode/                # VS Code workspace config (CMake Tools, IntelliSense)
│   ├── settings.json
│   ├── launch.json         # F5 = build + run with debugger
│   └── c_cpp_properties.json
├── src/
│   ├── main.cpp            # Entry point — picks one simulation, runs App
│   ├── App.h / App.cpp     # Thin loop: window + frame submit (~100 lines)
│   ├── VulkanContext.h/.cpp# All Vulkan boilerplate + helpers (~550 lines, never changes)
│   ├── Simulation.h        # Abstract base class — the interface every sim implements
│   └── simulations/
│       ├── GameOfLife.h/.cpp  # Self-contained Conway's Game of Life
│       └── Particles.h/.cpp   # Self-contained 500k particle system
└── shaders/
    ├── fullscreen.vert        # Draws a fullscreen triangle (no vertex buffer needed)
    ├── gol_display.frag       # Colors Game of Life pixels with glow
    ├── game_of_life.comp      # Conway's Game of Life rules on the GPU
    ├── particles_update.comp  # Physics update for 500k particles
    ├── particles_draw.vert    # Reads particle positions from GPU buffer
    └── particles_draw.frag    # Circular point sprite with brightness falloff
```

### The layered design

The codebase is split into three stable layers:

| Layer | Files | Changes when |
|-------|-------|-------------|
| **Platform** | `VulkanContext.h/.cpp` | Almost never — only if you add a Vulkan feature |
| **Framework** | `App.h/.cpp`, `Simulation.h` | Rarely — only if the frame loop changes |
| **Simulations** | `simulations/*.h/.cpp` | Constantly — this is where experiments live |

Adding a new simulation means creating two new files and changing one line in `main.cpp`.
The platform and framework layers never need to be touched.

---

## 2. The Vulkan Mental Model

Coming from OpenGL or Direct3D 11, Vulkan feels extremely verbose. Here's why
it's designed that way, and why that matters for GPU simulations.

### Vulkan is explicit by design

In older graphics APIs the driver makes thousands of decisions per frame on your
behalf — flushing caches, synchronizing resources, picking memory types. This is
convenient but unpredictable. Vulkan moves every decision to your code so that the
GPU's behaviour is **deterministic and fully controlled**. That's critical for
compute shaders where you're doing real parallel computation.

### The object hierarchy

```
VkInstance          ← the Vulkan library itself
  └─ VkPhysicalDevice   ← a specific GPU (or iGPU)
       └─ VkDevice       ← your logical connection to that GPU
            ├─ VkQueue        ← streams of GPU work (graphics, compute, transfer)
            ├─ VkSwapchainKHR ← the sequence of images displayed on screen
            ├─ VkBuffer       ← raw GPU memory (SSBOs, vertex data)
            ├─ VkImage        ← structured GPU memory (textures, render targets)
            ├─ VkShaderModule ← compiled SPIR-V shader bytecode
            ├─ VkPipeline     ← the full GPU programme (shaders + fixed state)
            └─ VkCommandBuffer← a recorded list of GPU commands
```

Each object you create stays alive until you explicitly destroy it. Vulkan never
destroys anything for you. This makes cleanup order important.

### Why SPIR-V?

GLSL shaders are compiled to **SPIR-V** bytecode (by `glslc`, part of the Vulkan
SDK) before being given to the driver. SPIR-V is an intermediate format — the
driver compiles it again to native GPU instructions. This separation means:

- Shaders are portable across GPU vendors
- You can cross-compile from HLSL, GLSL, or WGSL
- Compilation errors are caught at build time, not at runtime

---

## 3. Initialization Pipeline

`App::initVulkan()` calls these functions in order. Each one depends on the previous.

```
createInstance()          Creates the Vulkan library connection.
  setupDebugMessenger()   Hooks validation layer warnings to stderr (debug builds only).
createSurface()           Ties Vulkan to the GLFW window (platform-specific).
pickPhysicalDevice()      Selects a GPU (prefers discrete; checks for required features).
createDevice()            Opens a logical connection to that GPU and retrieves queues.
createSwapchain()         Creates a ring of images that get displayed to the screen.
createRenderPass()        Declares what attachments a render operation will write to.
createFramebuffers()      Binds each swapchain image to the render pass.
createCommandPool()       A memory pool for allocating command buffers.
createCommandBuffer()     Allocates one reusable command buffer.
createSyncObjects()       Creates semaphores and a fence for GPU/CPU sync.
```

Then simulation-specific setup runs for both GoL and particles so switching between
them mid-run costs nothing (resources are already on the GPU).

### Key concepts

**VkQueue** — A queue is a submission channel to the GPU. Graphics and compute work
can share a queue (most desktop GPUs expose one queue family that supports both). We
grab both `gfxQueue` and `compQueue` from `createDevice()`; they may refer to the
same underlying queue, which is fine.

**VkRenderPass** — A render pass is a description of what you're drawing *into*.
It specifies how many color/depth buffers there are, what happens to them at the
start (clear, load), and what happens at the end (store, discard). Vulkan uses this
description at pipeline compile time to optimize memory access on tile-based GPUs.

**VkPipeline** — Compiling a pipeline is expensive (~50–200ms). It bakes together:
- Shader stages (vertex + fragment, or just compute)
- Vertex input layout
- Rasterizer settings (cull mode, polygon fill)
- Blend state (how colors mix)
- Viewport and scissor (baked in unless you use dynamic state)

We compile all pipelines at startup. The only time we recompile is on window resize,
because the viewport size is baked into the graphics pipelines.

**Swapchain** — The swapchain is a queue of images. You acquire one, draw into it,
and present it. The display subsystem reads from the presented image. We use
`VK_PRESENT_MODE_MAILBOX_KHR` (triple-buffering) if available, falling back to
`VK_PRESENT_MODE_FIFO_KHR` (vsync). For simulation purposes, mailbox is ideal since
it lets the GPU run as fast as possible.

---

## 4. The Frame Loop

Each call to `drawFrame()` follows this sequence:

```
1. vkWaitForFences()           ← CPU blocks until the GPU finished the last frame.
2. vkAcquireNextImageKHR()     ← Ask the swapchain "which image can I draw into?"
3. vkResetCommandBuffer()      ← Wipe the previous frame's commands.
4. Record commands:
     a. Compute dispatch       ← Run the simulation shader (GoL step or particle update).
     b. Pipeline barrier       ← Tell the GPU "wait for compute to finish before reading."
     c. Begin render pass      ← Clear the screen, set up the framebuffer.
     d. Draw call              ← Draw the fullscreen quad (GoL) or points (particles).
     e. End render pass
5. vkQueueSubmit()             ← Hand the command buffer to the GPU.
6. vkQueuePresentKHR()         ← Display the result.
```

### Synchronization

Vulkan's synchronization is the hardest part for beginners. Two mechanisms matter here:

**Semaphores** — GPU-to-GPU signals. We use two:
- `semImageAvailable`: signaled when the swapchain has given us an image to draw into.
  The GPU waits on this before writing color output.
- `semRenderDone`: signaled when drawing is finished. The present engine waits on
  this before showing the image on screen.

**Fences** — GPU-to-CPU signals. `fenceFrame` is signaled when the GPU finishes
the whole frame. The CPU waits on it at the start of the *next* frame (step 1).
Without this, the CPU would race ahead, overwriting command buffers the GPU is
still reading.

**Pipeline barriers** (`imageBarrier`, `vkCmdPipelineBarrier`) — Ordered flushes
*within* a command buffer. Example: after the compute shader writes to a storage
image, the fragment shader can't immediately read it — the GPU's caches might not
be flushed. A barrier says: "wait until `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` has
finished writing (`VK_ACCESS_SHADER_WRITE_BIT`) before `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT`
reads (`VK_ACCESS_SHADER_READ_BIT`)."

---

## 5. Simulation: Game of Life

Conway's Game of Life is an ideal first GPU simulation: embarrassingly parallel
(each cell is independent), simple rules, visually interesting.

### Data representation

Two `VkImage` objects: `golImg[0]` and `golImg[1]`, format `VK_FORMAT_R8G8B8A8_UNORM`.
- The **R channel** stores the cell state: `255` = alive, `0` = dead.
- Both images stay in `VK_IMAGE_LAYOUT_GENERAL` permanently. This layout is valid
  for both storage image access (compute) and sampled access (fragment shader with a
  sampler). It avoids image layout transitions every frame.

### Ping-pong pattern

The compute shader cannot read and write the same image simultaneously (race
condition — a cell's neighbor might be updated before you read it). The solution is
two images that alternate roles each frame:

```
Frame N:  compute reads golImg[0]  →  writes golImg[1]
Frame N+1: compute reads golImg[1] →  writes golImg[0]
...
```

`golCurrent` (0 or 1) tracks which image holds the "current" state. After recording
the compute dispatch, it flips: `golCurrent = 1 - golCurrent`.

### Descriptor sets for ping-pong

We pre-create **two compute descriptor sets**:
- `golCompSet[0]`: binding 0 = `golImg[0]` (read), binding 1 = `golImg[1]` (write)
- `golCompSet[1]`: binding 0 = `golImg[1]` (read), binding 1 = `golImg[0]` (write)

And **two display descriptor sets**:
- `golDispSet[0]`: samples from `golImg[0]`
- `golDispSet[1]`: samples from `golImg[1]`

Each frame we bind `golCompSet[golCurrent]` for compute and `golDispSet[1-golCurrent]`
for display (showing the newly written image).

### Compute dispatch size

```
// Shader uses 16×16 local workgroups
uint32_t gx = (GOL_W + 15) / 16;  // = 32 groups for 512-wide grid
uint32_t gy = (GOL_H + 15) / 16;  // = 32 groups for 512-high grid
vkCmdDispatch(cmd, gx, gy, 1);    // Total: 32×32 = 1024 workgroups
                                   //        1024 × 256 = 262,144 threads
```

The shader guards against out-of-bounds access in case grid dimensions aren't
exact multiples of 16.

---

## 6. Simulation: Particles

Half a million particles simulated and rendered entirely on the GPU.

### Data representation

One `VkBuffer` (`partBuf`) holds a flat array of `Particle` structs:

```cpp
struct Particle {
    glm::vec2 pos;   // position in NDC space (-1 to +1)
    glm::vec2 vel;   // velocity (NDC units per second)
    glm::vec4 color; // RGBA
};
// Total: 32 bytes × 500,000 = 16 MB on the GPU
```

The buffer has three usage flags:
- `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` — compute shader reads and writes it
- `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` — the driver knows it might be used as vertex data
- `VK_BUFFER_USAGE_TRANSFER_DST_BIT` — lets us copy initial data from a staging buffer

### Why `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`?

Device-local memory lives on the GPU's VRAM. The CPU cannot directly map/write it,
but the GPU accesses it at full bandwidth (~400 GB/s on modern discrete GPUs).
We copy initial data there once via a staging buffer (host-visible → device-local copy).

### Push constants

Instead of a descriptor set for per-frame values (delta time, total time, mouse
position), we use **push constants** — a small (128-byte minimum) block of data
embedded directly in the command buffer. It's faster than a uniform buffer for tiny,
frequently-changing values.

```glsl
layout(push_constant) uniform PC {
    float dt;
    float time;
    vec2  mouseNDC;
} pc;
```

### Rendering without vertex buffers

The vertex shader reads directly from the SSBO using `gl_VertexIndex`:

```glsl
Particle p = particles[gl_VertexIndex];
gl_Position  = vec4(p.pos, 0.0, 1.0);
gl_PointSize = 2.0;
```

`vkCmdDraw(cmd, PARTICLE_COUNT, 1, 0, 0)` issues 500,000 vertices — the GPU
calls the vertex shader 500k times with `gl_VertexIndex` ranging from 0 to 499,999.
No vertex buffer is bound because the data comes from the SSBO.

### Additive blending

Particles use `VK_BLEND_OP_ADD` so overlapping particles brighten each other
rather than occluding each other. This creates the "glowing nebula" look for dense
clusters. The blend equation for each fragment is:

```
outColor = srcColor × 1  +  dstColor × 1
         = particle_color + whatever_was_already_in_the_framebuffer
```

---

## 7. Shader Reference

| File | Stage | Purpose |
|------|-------|---------|
| `fullscreen.vert` | Vertex | Generates a fullscreen triangle from `gl_VertexIndex` (0,1,2). No vertex buffer. |
| `gol_display.frag` | Fragment | Samples the GoL image, colors alive=green, dead=dark, adds a 3×3 glow. |
| `game_of_life.comp` | Compute | Reads `currentGrid`, applies Conway rules, writes `nextGrid`. 16×16 thread groups. |
| `particles_update.comp` | Compute | Updates `pos` and `vel` for each particle using mouse gravity + center attractor. 256 threads/group. |
| `particles_draw.vert` | Vertex | Reads `Particle` from SSBO via `gl_VertexIndex`, outputs position and color. |
| `particles_draw.frag` | Fragment | Makes circular point sprites from `gl_PointCoord`; discards corners. |

### The fullscreen triangle trick

Drawing a fullscreen quad normally requires 2 triangles (6 vertices or 4+index buffer).
A single oversized triangle is cheaper: 3 vertices, no index buffer, and modern GPUs
clip the excess efficiently. The UVs also go from 0→2, so the shader uses only the
[0,1] region — outside that, the rasterizer doesn't generate fragments.

```
(-1,3)
  │ ╲
  │   ╲
(-1,-1)─────(3,-1)
```

---

## 8. Build & Run

### Prerequisites
- [Vulkan SDK](https://vulkan.lunarg.com/) — sets the `VULKAN_SDK` environment variable
- CMake 3.20+
- Visual Studio 2022 (MSVC) or Clang/GCC with C++20

### Using VS Code (recommended)

1. Open the `Shader_Fun` folder in VS Code
2. CMake Tools will prompt to configure — click **Yes** (or it auto-configures)
3. Select a kit in the status bar (bottom): e.g., `Visual Studio Community 2022 - amd64`
4. Press **F5** to build and launch with the debugger
   - Or click the ▶ (play) button in the CMake Tools status bar to run without debugger
   - Or press **F7** to just build

On first run, CMake will download GLFW and GLM and compile all shaders. Subsequent
builds only recompile changed files.

### Command line

```bash
cmake -B build -S .
cmake --build build --config Debug
./build/Debug/ShaderFun.exe
```

### Controls

| Key | Action |
|-----|--------|
| `Tab` | Switch between Game of Life and Particles |
| `Space` | Randomize the Game of Life grid |
| `Mouse` | Attracts particles (in Particles mode) |
| `Esc` | Quit |

---

## 9. How to Experiment

### Change GoL grid size

In `src/App.h`:
```cpp
constexpr uint32_t GOL_W = 1024;  // was 512
constexpr uint32_t GOL_H = 1024;
```
Rebuild. The compute dispatch and image allocation adjust automatically.

### Implement a different cellular automaton

Copy `shaders/game_of_life.comp` to `shaders/my_rule.comp`. The only part to
change is the `main()` logic. Example — Brian's Brain (3-state automaton):

```glsl
// 0.0 = dead, 0.5 = dying, 1.0 = alive
float state = imageLoad(currentGrid, coord).r;
float aliveNeighbors = ...; // count neighbors where r > 0.8

float next;
if (state > 0.8) {
    next = 0.5;                          // alive → dying
} else if (state > 0.3) {
    next = 0.0;                          // dying → dead
} else {
    next = (aliveNeighbors == 2.0) ? 1.0 : 0.0;  // dead → alive with 2 alive neighbors
}
```

Then point CMakeLists to compile `my_rule.comp` and update `createGoLComputePipeline()`
to load `"shaders/my_rule.comp.spv"`.

### Change particle behaviour

Everything is in `shaders/particles_update.comp`. Try:

- **Gravity**: replace the mouse attractor with `p.vel.y += 0.5 * pc.dt;`
- **Vortex**: apply a perpendicular force `p.vel += vec2(-toCenter.y, toCenter.x) * k;`
- **Multiple attractors**: add more `vec2` positions to the push constants
- **Flocking**: this requires reading neighbor velocities — look at Compute Shader
  shared memory (`shared` keyword) to average neighbors within a workgroup

### Add a new simulation mode

1. Add `NewSim` to the `SimMode` enum in `App.h`
2. Add GPU resources (images or buffers) in `App.h`
3. Implement `createNewSimResources()`, `createNewSimPipeline()`, `initNewSim()` in `App.cpp`
4. Call them from `initVulkan()`
5. Add a `recordNewSim(uint32_t imgIdx)` function
6. Handle `SimMode::NewSim` in `drawFrame()` and the `Tab` key handler

### Understanding validation layer output

When built in Debug mode, the Khronos validation layer checks every Vulkan call and
prints errors to stderr. Common messages:

- **"VUID-vkCmdDraw-None-02698"** — a descriptor wasn't bound before drawing.
  Check `vkCmdBindDescriptorSets` calls.
- **"VUID-VkSubmitInfo-pWaitDstStageMask"** — wrong pipeline stage in a semaphore
  wait. The stage must cover all the work that produces the resource you're waiting for.
- **"Hazard WRITE_AFTER_WRITE"** (with `VK_LAYER_KHRONOS_synchronization2`) — two
  compute dispatches write the same image without a barrier between them.

Validation is disabled in Release builds (`-DCMAKE_BUILD_TYPE=Release`) for full speed.
