# Satellite Constellation Visualizer

Real-time GPU visualization of Earth's satellite megaconstellations and speculative space infrastructure, rendered from any point on the surface with physically-based photometry, atmospheric scattering, and accurate orbital mechanics.

Built on a Vulkan compute + graphics pipeline. Other simulations (Game of Life, Particles, Scene3D) remain in the codebase but SatelliteSim is the primary focus.

---

## What it shows

You are standing on Earth's surface looking up at the night sky. Every bright point is a real satellite from a real constellation, reflecting sunlight with a physically modeled BRDF. The intensity, color, and flash pattern depend on the satellite's type, orientation, range, and your position relative to the terminator.

- **~100k satellites** across 11 constellations (Starlink, OneWeb, Kuiper, Xingwang, ISS, SpaceX ODC, and more)
- **Multi-surface photometry** — primary + secondary reflective surfaces, Phong specular lobes, mirror-peak flash model (Iridium-class flares), and isotropic diffuse floor
- **Physically-based sky** — Rayleigh + Mie atmospheric scattering, sun disc + corona, moon disc + phase, star catalog with spectral colors
- **Satellite sky glow** — top-64 brightest flares per frame contribute Gaussian sky illumination with atmospheric extinction
- **Daytime suppression** — realistic terminator ramp; flares survive daylight only above a brightness threshold
- **SSO precession** — sun-synchronous orbits precess at 360°/year via J2 nodal formula
- **Reflect Orbital mirrors** — speculative 55 m flat mirror constellation with FlatMirror45 or TargetedReflector attitude modes; mirrors physically slew toward ground targets at a configurable rate
- **Audio** — spatial flare sound events, music playlist, UI sounds via miniaudio
- **Time controls** — pause, reverse, 8 warp levels from 1× to 1 year/s
- **Rebindable controls** — all keyboard bindings editable in the Settings panel at runtime

---

## Prerequisites

| Dependency | Notes |
|------------|-------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Sets `VULKAN_SDK` env var |
| CMake 3.20+ | `winget install Kitware.CMake` |
| Visual Studio 2022 | C++20 + MSBuild |

GLFW, GLM, Clay (UI), stb (fonts/images), and miniaudio are fetched automatically at configure time.

---

## Build

```bash
cmake -B build -S .
cmake --build build
```

Or open the folder in **VS Code** with CMake Tools — **F5** to build + debug, **F7** to build only.

Shaders are auto-detected by CMake, compiled by `glslc`, and copied next to the executable. No manual shader step needed.

---

## Controls

All keybindings except right-click and WASD are rebindable in the Settings panel.

| Input | Action |
|-------|--------|
| Right-click drag | Look around (camera) |
| WASD | Move observer along Earth's surface |
| Shift + WASD | Move fast |
| Ctrl + WASD | Move fine (precise placement) |
| `,` / `.` | Decrease / increase time warp |
| `Space` | Pause / resume simulation |
| `R` | Reverse time direction |
| `Tab` | Toggle UI visibility |
| `F11` | Toggle fullscreen |
| `Esc` | Quit |

---

## Constellations

| Name | Satellites | Altitude | Inclination | Type |
|------|-----------|----------|-------------|------|
| Starlink Gen1 | 4,392 | 550 km | 53° | Walker |
| Starlink Gen2 | 30,480 | 525 km | 53.2° | Walker |
| OneWeb | 648 | 1,200 km | 87.9° | Walker |
| Amazon LEO (Kuiper) | 7,742 | 630 km | 51.9° | Walker |
| Guowang (GW) | 13,920 | 508 km | 85° | Walker |
| ISS | 1 | 408 km | 51.6° | Walker |
| SpaceX AI Sat (ODC) | 20,000 | 575–1,925 km | SSO | Disk, alignTerminator |
| Reflect Orbital | 1,000 | 500 km | SSO | Disk, alignTerminator |


Data sources: planet4589.org/space/con/conlist.html, FCC filings, public orbital data.

---

## Project structure

```
src/
├── main.cpp                    ← pick simulation here (one line)
├── App.h / App.cpp             ← window + frame loop
├── VulkanContext.h / .cpp      ← Vulkan boilerplate + helpers
├── Simulation.h                ← abstract base class
├── UIRenderer.h / .cpp         ← Clay UI → Vulkan pipeline
├── AudioSystem.h / .cpp        ← miniaudio wrapper
└── simulations/
    ├── SatelliteSim.h / .cpp   ← primary simulation (this project)
    ├── StarCatalog.h / .cpp    ← star catalog renderer (precursor)
    ├── GameOfLife.h / .cpp     ← Conway's Game of Life
    ├── Particles.h / .cpp      ← GPU particle system
    └── Scene3DDemo.h / .cpp    ← 3D mesh + SDF rendering
shaders/
    sat_flare.comp              ← photometry compute: CPU positions → GPU visibility records
    sat_point.vert/frag         ← satellite point sprites (additive blend)
    sat_sky.vert/frag           ← sky background: atmosphere + sun + moon + glow
    star_point.vert/frag        ← star catalog points
    ui.vert/frag                ← Clay UI quads + text + icons
assets/
    sound/                      ← audio: music tracks, flare SFX, UI sounds
    icons/ui/                   ← PNG icon sprites packed into GPU atlas
```

---

## Switching simulations

Edit `src/main.cpp`:
```cpp
App app(std::make_unique<SatelliteSim>());
// App app(std::make_unique<GameOfLife>());
// App app(std::make_unique<Particles>());
```

Rebuild. No CMake reconfigure needed.
