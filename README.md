# ShaderFun

A Vulkan GPU playground for shader experiments, particle simulations, and astronomical visualization.
Built to learn and experiment with Vulkan compute + graphics pipelines.

## Simulations

- **Game of Life** — Conway's Game of Life on a 512×512 grid via GPU compute, ping-pong storage images
- **Particles** — 500k particles simulated and rendered on the GPU; mouse attracts particles
- **Scene3DDemo** — 3D mesh with SDF-based rendering
- **StarCatalog** — real star catalog renderer with spectral colors and atmospheric scattering
- **SatelliteSim** — satellite constellation flare visualizer *(active development)*

### SatelliteSim

Real-time visualization of satellite constellation flares as seen from any point on Earth.

- **~99k satellites** across 10 real constellations (Starlink Gen1/2, OneWeb, Kuiper, Xingwang, Telesat, GEO belt, ISS, SpaceX ODC)
- **Multi-surface photometry** — each satellite type has primary + secondary reflective surfaces with specular/Lambertian BRDFs, mirror peak flash model, and diffuse floor
- **Physically-based sky** — Rayleigh + Mie atmospheric scattering with analytic auto-exposure; sun, moon disc + halo, star catalog
- **Satellite sky glow** — up to 16 brightest flares contribute Gaussian sky illumination with proper atmospheric extinction (orange near horizon)
- **Daytime suppression** — flares fade across the terminator with a realistic atmospheric ramp
- **SSO precession** — SpaceX ODC terminator-aligned disk shell precesses at 360°/year to stay sun-synchronous
- **Audio** — spatial flare sound events via miniaudio; volume controls in Settings panel
- **Time controls** — pause, reverse, and warp from real-time to 1000× speed
- **Floating UI** — constellation toggles, camera/display settings, time panel, status bar

## Prerequisites

| Dependency | Notes |
|------------|-------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Sets `VULKAN_SDK` environment variable |
| CMake 3.20+ | `winget install Kitware.CMake` |
| Visual Studio 2022 | C++20 compiler + MSBuild |

GLFW, GLM, Clay (UI), stb (fonts/images), and miniaudio are fetched automatically by CMake at configure time.

## Build

```bash
cmake -B build -S .
cmake --build build
```

Or open the folder in **VS Code** with the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension — press **F5** to build and debug, **F7** to build only.

Shaders are auto-detected by CMake, compiled by `glslc`, and copied next to the executable.

## Switching simulations

Edit [`src/main.cpp`](src/main.cpp) and replace the simulation being constructed:

```cpp
App app(std::make_unique<SatelliteSim>());
// App app(std::make_unique<GameOfLife>());
// App app(std::make_unique<Particles>());
```

Then rebuild. CMake picks up new `.cpp` files automatically — no reconfigure needed.

## SatelliteSim Controls

| Input | Action |
|-------|--------|
| Right-click drag | Look around |
| WASD | Move observer position along Earth's surface |
| `,` / `.` | Decrease / increase time scale |
| `R` | Reverse time |
| `Space` | Pause / resume |
| `S` | Toggle settings panel |
| `Esc` | Quit |

## Project structure

```
src/
├── main.cpp                    ← pick your simulation here
├── App.h / App.cpp             ← window + frame loop
├── VulkanContext.h / .cpp      ← Vulkan boilerplate + helpers
├── Simulation.h                ← abstract base class
├── UIRenderer.h / .cpp         ← Clay UI → Vulkan pipeline
├── AudioSystem.h / .cpp        ← miniaudio wrapper
└── simulations/
    ├── SatelliteSim.h / .cpp   ← satellite constellation visualizer
    ├── StarCatalog.h / .cpp    ← star catalog renderer
    ├── GameOfLife.h / .cpp     ← Conway's Game of Life
    ├── Particles.h / .cpp      ← GPU particle system
    └── Scene3DDemo.h / .cpp    ← 3D mesh + SDF rendering
shaders/                        ← GLSL sources (compiled to SPIR-V at build time)
assets/
└── sound/                      ← audio samples for SatelliteSim
```

## Adding a new simulation

1. Create `src/simulations/MySim.h` and `MySim.cpp`
2. Inherit from `Simulation` and implement: `init`, `onResize`, `recordDraw`, `cleanup`
3. In `main.cpp`: `App app(std::make_unique<MySim>());`
4. Rebuild — CMake picks up the new `.cpp` automatically

The `VulkanContext&` passed to `init` gives you everything needed: device, physical device, render pass, swapchain extent, and helpers like `ctx.createBuffer()`, `ctx.loadShader()`, `ctx.imageBarrier()`.
