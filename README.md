# SAT LIGHT SIM

A real-time GPU visualization of future satellite megaconstellations, seen from any point near Earth. Physically-based photometry, atmospheric scattering, volumetric clouds, terrain, ocean,
aurora, all rendered through Vulkan.


---


  ![SAT LIGHT SIM](docs/screenshots/title.png)



---

## Prerequisites

| Dependency | Notes |
|------------|-------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Sets `VULKAN_SDK`. On macOS/Linux run its `setup-env.sh` |
| CMake 3.20+ | `winget install Kitware.CMake` / `brew install cmake` / distro package |
| Visual Studio 2022 (Windows) | C++20 + MSBuild |
| Xcode Command Line Tools (macOS) | Clang + the macOS SDK |

GLFW, GLM, Clay (UI), stb (fonts/images), miniaudio, and nlohmann/json are fetched automatically at
configure time via CMake FetchContent.

Nothing machine-specific is committed: `VULKAN_SDK` and `cmake` are found through the environment
and `PATH`. Per-developer overrides belong in `CMakeUserPresets.json` (gitignored), never in
`CMakePresets.json` or `.vscode/`.

---

## Build

```bash
cmake -B build -S .
cmake --build build
```

Or open the folder in **VS Code** with the CMake Tools extension — **F5** to build + debug, **F7**
to build only.

Shaders are auto-detected by CMake, compiled by `glslc`, and copied next to the executable. No
manual shader step needed. The built executable is named `SAT_LIGHT_SIM_V_<version>` (tracks the
`VERSION` file), e.g. `build/Debug/SAT_LIGHT_SIM_V_1_1_0.exe`.

### Presets (CMake 3.21+)

`CMakePresets.json` carries the per-platform configurations, so no flags need remembering:

```bash
cmake --preset windows                  # or: linux / macos
cmake --build --preset windows
```

### Release packaging

```bash
cmake --preset windows-release
cmake --build --preset windows-release --parallel
cmake --build --preset windows-package     # → dist/SAT_LIGHT_SIM_v<ver>_Windows.zip
```

`release.bat` wraps the Windows and (via WSL) Linux legs of that. The `package-release` target
stages exactly what ships — the copy list lives once in `cmake/PackageRelease.cmake`, and CI uses
the same target, so a local archive and a tagged release have identical layouts.

macOS builds are produced by GitHub Actions as a **universal binary** (arm64 + x86_64, deployment
target 11.0) — push a `vX.Y.Z` tag. Packaging on macOS also bundles the Vulkan loader + MoltenVK
and writes a launcher `.command`, since macOS has no system Vulkan.

---


## Constellations

Fully moddable via `constellations.json` next to the executable — see `CONSTELLATION_MODDING.md`.

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
    ├── SatelliteSim.h/.cpp/SatelliteSimUI.cpp  ← primary simulation (this project)
    ├── StarCatalog.h / .cpp    ← star catalog renderer (precursor, legacy)
    ├── GameOfLife.h / .cpp     ← Conway's Game of Life (legacy)
    ├── Particles.h / .cpp      ← GPU particle system (legacy)
    └── Scene3DDemo.h / .cpp    ← 3D mesh + SDF rendering (legacy)
shaders/
    sat_orbit.comp               ← GPU orbital mechanics + attitude
    sat_flare.comp               ← photometry compute: visibility + glow histogram
    scene_depth.comp             ← shared terrain/ocean depth buffer
    cloud_march.comp             ← half-res volumetric clouds/cirrus/aurora/airglow
    beam_cloud_block.comp        ← Reflect Orbital target illumination
    sat_point.vert/frag          ← satellite point sprites (additive blend)
    sat_sky.vert/frag            ← sky background: atmosphere + terrain + ocean + sun + moon
    star_point.vert/frag         ← star catalog + planet points
    flare_source/blur/composite  ← render-to-texture lens flare pipeline
    ui.vert/frag                 ← Clay UI quads + text + icons
    include/                     ← shared GLSL headers (common/terrain/cloud_params/reflect_beam)
data/
    constellations.json           ← satellite types + constellation definitions (moddable)
    reflector_targets.json        ← real solar-farm sites for Reflect Orbital mirrors
assets/
    textures/                    ← Earth day/night/elevation/clouds, moon, Milky Way
    sound/                       ← music tracks, flare SFX, UI sounds
    icons/ui/                    ← PNG icon sprites packed into GPU atlas
```

AI Code was used in this project.
See `CLAUDE.md` for the full architecture writeup (frame loop order, GPU buffer layouts, subsystem
design notes) and `THIRD_PARTY_NOTICES.txt` for third-party licenses.

---



