# ShaderFun

A Vulkan GPU playground for cellular automata, particle simulations, and shader experiments.
Built to learn and experiment with Vulkan compute + graphics pipelines.

## Features

- **Game of Life** — Conway's Game of Life on a 512×512 grid, computed entirely on the GPU via a compute shader with ping-pong storage images
- **Particles** — 500k particles simulated and rendered on the GPU; mouse attracts particles
- **Modular** — each simulation is a self-contained class; adding a new one is two new files + one line in `main.cpp`

## Prerequisites

| Dependency | Notes |
|------------|-------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Sets `VULKAN_SDK` environment variable |
| CMake 3.20+ | `winget install Kitware.CMake` |
| Visual Studio 2022 | C++20 compiler + MSBuild |

GLFW and GLM are fetched automatically by CMake at configure time.

## Build

```bash
cmake -B build -S .
cmake --build build --config Debug
```

Or open the folder in **VS Code** with the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension:
- Select a kit in the status bar (e.g. `Visual Studio Community 2022 - amd64`)
- Press **F5** to build and run with the debugger attached
- Press **F7** to build only

## Switching simulations

Edit [`src/main.cpp`](src/main.cpp) and uncomment the simulation you want:

```cpp
App app(std::make_unique<GameOfLife>());
// App app(std::make_unique<Particles>());
```

Then rebuild. Only one simulation is compiled into the run at a time.

## Controls

| Key / Input | Action |
|-------------|--------|
| `Space`     | Randomize Game of Life grid |
| `Mouse`     | Attracts particles (Particles mode) |
| `Esc`       | Quit |

## Project structure

```
src/
├── main.cpp                  ← pick your simulation here
├── App.h / App.cpp           ← window + frame loop (~100 lines, rarely changes)
├── VulkanContext.h / .cpp    ← all Vulkan boilerplate + helper utilities
├── Simulation.h              ← abstract base class every simulation inherits
└── simulations/
    ├── GameOfLife.h / .cpp   ← Conway's Game of Life
    └── Particles.h / .cpp    ← GPU particle system
shaders/                      ← GLSL sources (compiled to SPIR-V at build time)
```

See [`GUIDE.md`](GUIDE.md) for a full walkthrough of the Vulkan architecture, how each simulation works, and how to add your own.

## Adding a new simulation

1. Create `src/simulations/MySim.h` and `MySim.cpp`
2. Inherit from `Simulation` and implement: `init`, `onResize`, `recordFrame`, `cleanup`
3. In `main.cpp`: `App app(std::make_unique<MySim>());`
4. Rebuild — CMake picks up the new `.cpp` automatically

The `VulkanContext&` passed to `init` and `recordFrame` gives you everything needed:
device, physical device, render pass, swapchain extent, and helpers like
`ctx.createBuffer()`, `ctx.loadShader()`, `ctx.imageBarrier()`.
