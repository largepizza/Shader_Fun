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

Run: `build/Debug/SAT_LIGHT_SIM_V_<version>.exe` (e.g. `build/Debug/SAT_LIGHT_SIM_V_1_1_0.exe` — the
exact name tracks `VERSION`; see `CMakeLists.txt`'s `EXE_BASENAME`).

Shaders: auto-detected glob (`shaders/*.vert|.frag|.comp`), compiled by `glslc`, copied as `shaders/*.spv`. New shader files are picked up automatically on next build.

**Do not launch or run the app yourself** (no `run` skill, no invoking the exe) to verify changes, especially UI behavior. Verifying UI/UX changes (opening the program, panning the camera, clicking through menus, etc.) is an involved manual process — the user runs and audits the app themselves after you build. Just build (and typecheck/compile-check) and report what changed; let the user test it.

**Requirements**: Vulkan SDK + `VULKAN_SDK` env var, CMake 3.20+, MSVC C++20.

### Presets and release packaging

`CMakePresets.json` holds every per-platform configuration (`windows` / `linux` / `macos`, plus
`*-release` variants and `macos-universal-release`). **It is committed and must stay free of
absolute paths** — machine-specific values go in `CMakeUserPresets.json`, which is gitignored.
`.vscode/settings.json` and `launch.json` are committed too and are under the same rule; a
hardcoded `cmake.cmakePath` and `VULKAN_SDK` under a developer's home directory shipped there once
(2026-09, macOS-potato branch) and reached the public repo's history.

`cmake --build <dir> --target package-release` stages and archives the distributable into `dist/`.
`cmake/PackageRelease.cmake` holds the copy list — **the only copy of it**. `.github/workflows/
release.yml` and `release.bat` both drive that target rather than repeating the list, which they
previously did five times over (3 CI jobs + 2 batch branches) and had already drifted. On macOS the
same script bundles the Vulkan loader + MoltenVK into `lib/` and writes the launcher `.command`
(there is no system Vulkan on macOS; the exe's baked-in `$VULKAN_SDK` path exists only on the build
machine).

**`CMAKE_POLICY_VERSION_MINIMUM 3.5` at the top of CMakeLists.txt is load-bearing, not cosmetic.**
CMake 4.0 turned `cmake_minimum_required(VERSION <3.5)` from a deprecation warning into a hard
configure error, and two FetchContent deps still declare one (glfw 3.4 → `3.4...3.28`,
nlohmann/json v3.11.3 → `3.1...3.14`). Without it, a stock CMake 4 — Homebrew's default, and
increasingly Windows'/Linux's — cannot configure this project at all. That failure is what the
committed absolute path to a portable CMake 3.31 was working around.

### macOS release architecture

The macOS CI job builds a **universal binary** (`CMAKE_OSX_ARCHITECTURES="arm64;x86_64"`) with
`CMAKE_OSX_DEPLOYMENT_TARGET=11.0`. Both are required and they fix *different* failures:
- **Architecture.** `macos-14` runners are Apple Silicon, so a default build is arm64-only and an
  Intel Mac rejects it with **"bad CPU type in executable"** — the kernel refusing to exec a Mach-O
  with no slice for the host, before any of this app's code runs. Universal is preferred over a
  second `macos-13` Intel runner job because GitHub is retiring those images, and it ships one
  download instead of making players choose.
- **Deployment target.** Without it clang stamps the *build* machine's OS version, and a
  macOS 14-targeted binary is refused on Monterey (12.x) with "not supported on this version of
  macOS" — the error that surfaces *next* once the CPU-type one is fixed.

The workflow's `lipo -archs` step fails the build if either slice is missing from the exe **or from
the bundled dylibs** — LunarG ships universal dylibs, but that is a property of their build, not
something this workflow controls.

### Old / low-end hardware floor

The 128-byte push-constant trim (see "Subsystem: Weak-Hardware Sky Tiers") is one of several places
this project sits above a Vulkan *guaranteed minimum*. `VulkanContext::logDeviceLimits()` logs all
of them every launch and throws a named error when one is short, so a report from a machine nobody
has access to is answerable. Current margins:

| Limit | Guaranteed min | This app needs | Why |
|---|---|---|---|
| `maxPushConstantsSize` | 128 | **128** | every PC struct static_asserts to exactly 128 |
| `maxImageDimension2D` | 4096 | **14999** | `earth_elevation.png` is 14999×7500 (GCN1/Metal cap is 16384 — little headroom) |
| `maxImageDimension3D` | 256 | **1024** | `aurora_noise.comp` bakes a 1024×16×256 volume |
| `maxComputeWorkGroupInvocations` | 128 | **256** | `local_size 16×16` in cloud_march / scene_depth / flare_blur |
| `maxComputeSharedMemorySize` | 16 KB | ~5.2 KB | tile-cull lists — comfortable |
| `maxPerStageDescriptorStorageBuffers` | 4 | **6** | `sat_sky.frag`'s set; MoltenVK is the realistic place to hit it, since it maps SSBOs + UBOs + vertex buffers into Metal's 31 per-stage buffer slots |
| `maxPerStageDescriptorSampledImages` | 16 | 15 | `sat_sky.frag` — one binding from the floor |

**The push-constant gate in `pickPhysicalDevice()` read 144 until 2026-09-08** — the pre-trim
`SatDrawPC` size — so it rejected precisely the hardware the trim was performed for. Keep that
constant equal to the largest PC struct; if one grows past 128 the fix is the CloudParams UBO, not
a bigger number there.

Two more non-guaranteed things are now checked rather than assumed: device **features** are
verified present before `vkCreateDevice` (requesting an unsupported one fails with
`VK_ERROR_FEATURE_NOT_PRESENT` and no indication of which), and **linear blit filtering** is a
per-format optional feature, so `VulkanContext::bestBlitFilter()` picks LINEAR/NEAREST per format
for both mipmap generation and the `renderScale < 1.0` upscale — the latter being the path that
exists *for* weak hardware, the least safe place to assume the optional feature.

VRAM is the untested one: the shipped textures are roughly 500 MB of GPU memory before mips
(8K day/night/clouds/specular/Milky Way + the 14999×7500 R8 DEM ≈ 112 MB on its own), against
2 GB on a 2015 MacBook Pro R9 M370X. Nothing streams or downsamples them.

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
sim->recordCompute(cmd)  → WASD movement; simTime advance;
                           CPU updatePositions() — sun/moon/obsECI/eci2enu/reflector targets only;
                           orbit rebake check (every 7 sim-days);
                           dispatch 1: scene_depth.comp   (half-res shared terrain/ocean depth)
                           dispatch 2: sat_orbit.comp     (orbital mechanics + attitude + beam list)
                           dispatch 3: beam_self_march.comp (per-beam cloud occlusion, up to 2048
                                                            beams — replaced beam_cloud_block.comp's
                                                            201-target version 2026-08-09, see
                                                            "Subsystem: Reflect-Orbital Beam Cloud
                                                            Occlusion" below)
                           dispatch 4: cloud_march.comp   (half-res clouds/cirrus/aurora/airglow-red/
                                                            beam pointing ray + volumetric glow +
                                                            per-pixel cloud shadow)
                           dispatch 5: sat_flare.comp     (lighting + visibility culling)
                           barriers between each (see recordCompute for exact stage/access pairs)
sim->recordPrePass(cmd)  → renderScale < 1.0 only: low-res sky → vkCmdBlitImage into swapchain
vkCmdBeginRenderPass     → owned by App
sim->recordDraw(cmd)     → sky/ground background → satellite points → stars
ui.record(cmd)           → Clay → Vulkan quads/text/icons on top
vkCmdEndRenderPass       → owned by App
```

### Occlusion: one shared depth buffer

`scene_depth.comp` writes `sceneDepthImg` — half `ctx.swapExtent`, **`R32_SFLOAT`**, linear metres
along each view ray to the first terrain/ocean surface, or `kNoSurfaceT` (1e30) for sky. Every
later pass tests against it instead of deriving its own answer.

This replaced three separate mechanisms that existed only because `cloud_march.comp` had no
terrain data: `beamTerrainVisibility()` (an 8-32 sample DEM march run *per beam per pixel* — the
single most expensive thing in that shader), `tEnterCombined` (one entry distance fused across
cirrus+cloud+beam+aurora, so none could be occluded independently), and the opacity-gated
`tCloudOcclude` for terrain purposes. Each volumetric march now clamps to `tScene` at march time,
which also gives real partial truncation where a ridge pokes into a shell.

**Sizing is deliberate and load-bearing:** half of the SWAP extent, independent of `renderScale`,
exactly matching `cloudMarchTargetA/B`. That is what lets `cloud_march.comp` read it 1:1 with
`texelFetch` at its own `gl_GlobalInvocationID` — no UV math to get wrong. Fragment consumers use
`gl_FragCoord.xy / pc.screenSizePx`. **Consequence:** at `renderScale < 1.0` the depth pass does
not shrink, so its relative cost rises sharply (at 50% it marches terrain at the same pixel count
as the sky pass itself). Knockout bit 1024 disables it — the buffer fills with `kNoSurfaceT`, which
reproduces pre-unification occlusion behaviour, making the whole architecture one A/B checkbox.

**Never store a distance in a half-float.** `tEnterCombined` lived in an `RGBA16F` alpha, whose
65504 ceiling every near-horizon cloud entry overflowed to `+inf` — silently suppressing the entire
composite on any ray that also hit the sea-level sphere. That bug is why `sceneDepthImg` is R32.

### Shader `#include`

`shaders/include/*.glsl`, compiled with `glslc -I`. CMake globs the headers and makes every shader
depend on all of them — coarse, but `add_custom_command(DEPFILE)` needs CMake 3.27 for VS
generators and this project requires 3.20.

| Header | Holds |
|---|---|
| `common.glsl` | PI, R_EARTH/R_ATMOS, BETA_R/M, H_R/H_M, G_MIE, SUN_INTENSITY, cloud noise freqs, `kNoSurfaceT`, `raySphere`, `rotateZ`, `remap`, phase functions |
| `terrain.glsl` | DEM decode constants, `dirToUV`/`posToUV`, `terrainHeightAtUV/AtDir`, `enuBasis`, `observerEffHeight`, `observerPos` |
| `cloud_params.glsl` | the `CloudParams` UBO block + `CloudLayer` (`#define CLOUD_PARAMS_BINDING` first) |

**`GpuCloudParams` in `SatelliteSim.h` is a hand-maintained mirror of `cloud_params.glsl`** — GLSL
and C++ cannot share a declaration, so that pairing is the one place a CloudParams mismatch can
still hide. **Run `python tools/check_cloud_params.py` after touching either.**

The failure mode is a *permutation*, not a size change, which is why the `static_assert` does not
catch it: append a field in a different position in each file and the total size is unchanged, so
everything compiles and every field from the divergence point onward silently reads its
neighbour's value. This shipped once — `flatSunGainScale` read a pad (0, so 2D clouds rendered
black) while `flatCoverageScale` read 4.0 (so coverage quadrupled and swallowed the Earth).
Prefer appending at the end of both files.
| `reflect_beam.glsl` | `ReflectBeam` + `BEAM_MAX_ACTIVE` |

**`observerEffHeight` must be used by every pass that produces or consumes a distance.**
`cloud_march.comp` previously took a CPU-computed `obsEffH` while `sat_sky.frag` did its own GPU
lookup; harmless while each only compared against itself, wrong the moment one produces a depth
buffer the other reads.

**`optDepth` is the cautionary tale — and NOT in the direction you would expect.** Its two copies
look like classic drift: `cloud_march.comp` hardcodes `N_LIGHT=12` while `sat_sky.frag` reads
`cloud.lightSamples`, so the "Light samples" slider affects atmosphere and not clouds/aurora.
Sharing them was tried and **reverted after a measured performance regression**.

The reason is that de-duplication and specialization are in tension here. `cloud_march.comp`'s
trip count is a compile-time constant, so its loop unrolls; a shared function that must also serve
a settings-tunable count necessarily takes the count as a parameter, making the bound runtime.
glslc emits it as a real function (SPIR-V confirmed non-inlined, 5 call sites in that shader), so
the driver has to both inline and prove the constant to recover the unroll. It did not recover it.

**The false assumption was that extracting byte-identical code into a header is codegen-neutral.**
It is for pure declarations (`CloudParams`, `ReflectBeam` — no risk, real value, keep doing that)
and for code that was already a function with an unchanged signature. It is NOT when the signature
change turns a constant into a parameter. If you unify these anyway, measure `cloud_march` before
and after, at altitude and near the surface — do not assume.

**Still hand-duplicated, by choice:** `optDepth` (see above — sharing it was measurably slower),
and the aurora function set (`auroraFrame`, `auroraCoverage`,
`auroraOvalMask`, `auroraCurtainNoise`, `auroraSampleAt`/`auroraCurtainSample`) in `sat_sky.frag`
and `cloud_march.comp`. Verified byte-identical (comments stripped) as of this pass, so there is no
active bug — but they bind `auroraNoiseTex` at different indices and read different PC structs, so
sharing them needs sampler parameters threaded through all five. Keep both copies in sync until
someone does that work. Same applies to the cloud-column sample body, which appears in
`cloud_march.comp` (view march + sun cone + terrain shadow) and `beam_self_march.comp` (per-beam
cloud occlusion march — see "Subsystem: Reflect-Orbital Beam Cloud Occlusion" below).

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
- **Pointer-capture rule**: any `.floating` element with no explicit `pointerCaptureMode` defaults to `CLAY_POINTER_CAPTURE_MODE_CAPTURE` — Clay's root hit-test DFS stops dead the instant it finds the pointer inside that element, so nothing below it (in z-order) gets hover/click at all. Harmless for a normal panel (you want it to swallow clicks meant for it), but fatal for any floating element that is deliberately drawn *at* the pointer's own position every frame — the pointer is then *always* "inside" it, so it permanently blackholes every panel/button underneath. This shipped once as the gamepad virtual cursor dot (`buildUI`'s `VirtualCursor`): hover/click on every real button looked totally dead while the pad cursor sat still over it (the dot's own stale hitbox from last frame permanently overlapped the current test point), but worked in brief "blips" while the cursor was moving (the one-frame-stale dot briefly lagged behind the live position, leaving a gap for the real element underneath to get tested that frame) — a pattern that reads like a coordinate or deadzone bug but isn't. Any purely-visual floating element positioned at/following the pointer (cursor dots, drag ghosts, custom tooltips) MUST set `.pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_PASSTHROUGH`.
- Scrollable containers: `.clip = {.vertical = true, .childOffset = Clay_GetScrollOffset()}` on the content div.

### Icon Atlas
- `ui.loadIcons(ctx, paths, count)` — loads PNGs, packs into RGBA GPU atlas, rebinds descriptor. Call once on first frame (lazy init). Store `VulkanContext*` in your sim.
- Icons: `.image = {.imageData = (void*)(intptr_t)iconIdx}`. Renderer samples the atlas UV range for that index.
- Shader `mode`: `0.0` = solid rect, `1.0` = text glyph, `2.0` = icon sprite. Binding 1 is always valid (1×1 white placeholder at init).

### Font atlas is a fixed-size bitmap, not resolution-independent
`loadFont()` bakes ASCII 32-126 once via `stbtt_BakeFontBitmap` at a single pixel height
(`font.bakedSize`) into a fixed `atlasW`x`atlasH` R8 atlas. Every requested `CLAY_TEXT_CONFIG`
`fontSize` (via `fs(base) = base * uiScale`, `SatelliteSim.h`) just scales that ONE baked bitmap by
`fontSize / bakedSize` (`pushText`'s `renderScale`) — there is no per-size re-rasterization, so any
requested size well above `bakedSize` visibly upscales/blocks, most noticeably on large one-off
text like the intro's title captions (`fs(48)`/`fs(34)`, which can request 68-96px at high
`uiScale`). Bumped `bakedSize` 32→48 (session follow-up) with `atlasW/H` scaled by the same
`(48/32)²` factor (512→768) to preserve `stbtt_BakeFontBitmap`'s packing headroom — it's "a very
crappy packing" (its own doc comment) that silently drops/omits glyphs past whatever fits, so
`loadFont()` now checks its return value and logs a warning if that ever happens again. This is a
mitigation, not a fix: text is still a raster upscale, just from a less-coarse source. True
resolution-independence at arbitrary `uiScale`/title sizes would need an SDF font atlas instead — a
separate, larger change (new bake step, new glyph metadata, a distance-based alpha threshold in
the text fragment shader) not undertaken here.

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
    KB_CINEMATIC  = 7,   // LAlt   — event (toggle cinematic pan mode while RMB held)
    KB_COUNT      = 8,
};
```

### Adding a new control (complete checklist)
1. Add `KB_NEWNAME` before `KB_COUNT` in the enum
2. Add one line to `keybindings` in `init()`: `{"Display Name", GLFW_KEY_X, held, false}`
3. Bump `static_assert(KB_COUNT == N)` to the new count
4. Wire the action:
   - **Event** (`held=false`): `if (pressed(KB_NEWNAME)) { ... }` in `onKey()`
   - **Held** (`held=true`): `glfwGetKey(win, keybindings[KB_NEWNAME].key) == GLFW_PRESS` in `recordCompute()`

Settings display, rebinding, hover state, and `keyDisplayName()` all work automatically. `hovRebind[KB_COUNT]` is sized by the enum so no array changes are needed.

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
| `Perpendicular` | cross(surfN0, satNadir) | Secondary only — derived from primary normal |
| `AntiNadir` | -satNadir | Radiators facing deep space; brighter near horizon |
| `FlatMirror45` | normalize(sunDir + satNadir) | Flat mirror reflecting sunlight straight down |
| `TargetedReflector` | normalize(sunDir + toTarget) | Mirror aimed at nearest valid night-side ground target |
| `KnifeEdge` | roll around velHat; clamped ±80° | Starlink post-2020 roll-angle policy (Mallama 2023) |
| `SunPerp` | normalize(cross(sunDirECI, satNadir)) | Thermal radiator edge-on to sun; irr=0 always (correct thermal design — never receives direct sunlight). Visual contribution via diffuse. Used for AI1 datacenter radiators. |

`velHat` is computed in `sat_orbit.comp` from the orbital trig already in scope: `{-sinU·cosR - cosU·cosI·sinR, -sinU·sinR + cosU·cosI·cosR, cosU·sinI}` — already unit length for circular orbits.

### Satellite type catalogue (typeIdx)
| Idx | Name | Area (m²) | Primary attitude | Secondary | mirrorFrac |
|-----|------|-----------|-----------------|-----------|------------|
| 0 | Starlink | 10 | NadirPointing, spec=18 | — | 0.05 |
| 1 | LEO Broadband | 5 | SunTracking, spec=18 | — | 0.02 |
| 2 | GEO Comsat | 50 | SunTracking, spec=3 | AntiNadir, w=0.10 | 0.10 |
| 3 | ISS | 250 | SunTracking, spec=12 | AntiNadir, w=0.35 | 0.05 |
| 4 | SpaceX AI Sats | 600 | SunTracking, spec=25 | SunPerp, w=0.18 (radiators) | 0.01 |
| 5 | Reflect Mirror | 2376 | TargetedReflector, spec=200 | — | 0.97 |
| 6 | Debris | 1 | Tumbling, spec=6 | — | 0.03 |
| 7 | Starlink KE | 10 | KnifeEdge, spec=18 | — | 0.05 |

`crossSection = sqrt(crossSectionM2 / 10.0)` — so 10 m² → 1.0, 2376 m² → ~15.4.

### Satellite type data source
Types and constellations are loaded from `constellations.json` next to the exe. If the file is missing or malformed, `loadHardcoded()` provides the catalogue above as a fallback. The JSON schema is in `constellations.schema.json`.

### Adding a new satellite type
1. Add to `satTypes` in `constellations.json` (or `loadHardcoded()` as fallback)
2. No GPU struct changes needed; all fields map to existing `GpuSatInput` members
3. Reference the new typeIdx in a constellation entry

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
1. Add a `ConstellationConfig` entry to `constellations.json` (or `loadHardcoded()`)
2. `hovConst` and `hovHighlightConst` are `std::vector<bool>` and auto-size to `constellations.size()` — no manual hover bool management needed
3. `MAX_SATELLITES = 10,000,000` — cap is generous; only relevant for very large test configs

### Current constellation roster (9 total, hardcoded fallback)
| Name | Sats | Alt (km) | Incl | Dist | TypeIdx |
|------|------|----------|------|------|---------|
| Starlink Gen1 | 4,392 | 550 | 53° | Walker | 0 |
| Starlink Gen2 | 30,480 | 525 | 53.2° | Walker | 7 (KnifeEdge) |
| OneWeb | 648 | 1,200 | 87.9° | Walker | 1 |
| Amazon LEO | 7,742 | 630 | 51.9° | Walker | 1 |
| Guowang | 13,920 | 508 | 85° | Walker | 1 |
| ISS | 1 | 408 | 51.6° | Walker | 3 |
| SpaceX AI Sat | 20,000 | 575–1,925 | SSO | Disk+terminator, 10 rings | 4 |
| Reflect Orbital | 1,000 | 500 | SSO | Disk+terminator, 10 rings | 5 |
| Space Junk | 3,000 | ~1,000 | random 0–180° | RandomShell | 6 |

CPU `updatePositions()` is now **O(1)** — it only updates sun/moon/obsECI/eci2enu and uploads `reflectorTargetsBuf`. All orbital mechanics run on GPU via `sat_orbit.comp`.

### SSO precession model (alignTerminator=true)
Inclination from J2 formula: `cos(i) = -kSSOPrecRate / (1.5 × n × kJ2 × (Re/a)²)`
RAAN anchored at **sim-start** using `sunDirECI` (set by `updatePositions()` before `initConstellation()`): `raan_start = atan2(sunDirECI.x, -sunDirECI.y)`. GPU formula: `liveRaan = raan_start + kSSOPrecRate × (simTime − t_start)`. Anchoring at sim-start avoids the ~3° obliquity-driven phase error that accumulates when extrapolating from J2000 to a solstice epoch.

---

## Subsystem: GPU Orbital Pipeline

All per-satellite orbital mechanics and attitude computation runs on the GPU. The CPU only manages the small reflector targets buffer and triggers a rebake when needed.

### Two-dispatch pattern (recordCompute)
```
sat_orbit.comp dispatch   → reads satOrbitBuf + reflectorTargetsECEFBuf; writes satInputBuf
barrier satInputBuf       → SHADER_WRITE → SHADER_READ, compute→compute
vkCmdFillBuffer(glowBuf)  → zeros the glow histogram for this frame
barrier glowBuf           → TRANSFER_WRITE → SHADER_READ|SHADER_WRITE, transfer→compute
sat_flare.comp dispatch   → reads satInputBuf; writes satVisibleBuf + glowBuf (atomicMax)
barrier satVisibleBuf     → SHADER_WRITE → SHADER_READ, compute→vertex
```

### Buffers
| Buffer | Memory | Lifetime | Updated by |
|--------|--------|----------|------------|
| `satOrbitBuf` | device-local | uploaded once; rebaked every 7 sim-days | `uploadSatOrbits()` |
| `satInputBuf` | device-local | per-frame | `sat_orbit.comp` |
| `satVisibleBuf` | device-local | per-frame | `sat_flare.comp` |
| `reflectorTargetsECEFBuf` | host-visible, mapped | uploaded once at target-generation time | `loadReflectorTargets()`/fallback |
| `glowBuf` | host-coherent, mapped | per-frame | `sat_flare.comp` write; App reads back |

`mirrorNormalsBuf` (persistent per-satellite mirror lock/slew state) and the old per-frame
CPU-compacted `reflectorTargetsBuf` were both removed 2026-08-06 — see "Subsystem: TargetedReflector
/ Mirror Ground Targets" below. `sat_orbit.comp` now reads target data from a static
`reflectorTargetsECEFBuf` (uploaded once at target-generation time), and carries no persisted GPU
state of its own: every frame's TargetedReflector selection and orientation is a pure function of
that frame's push constants.

### Orbit rebake
`kOrbitRebakeDays = 7`. Each `GpuSatOrbit` bakes `u0 = fmod(orig_u0 + meanMot × epochT0, 2π)` so the shader only adds `meanMot × deltaT` where deltaT < 7×86400 s. Float ULP at that scale ≈ 0.07 s, well within tolerable orbital error. `uploadSatOrbits()` auto-triggers in `recordCompute()` when `|simDayJ2000 - orbitEpochDay| >= 7`.

### simTime representation
Split into `simDayJ2000` (int64_t days) + `simSecInDay` (double, re-based to [0, 86400) each frame). Avoids accumulated float precision loss when a large J2000 base is added to a small per-frame delta. The shader receives `deltaT = float((dDays × 86400) + dSec)` where dDays < 7 (ensured by rebake).

### GpuSatOrbit layout (112 bytes, std430)
All plain floats/uints — no vec3 — so C++ struct packing matches GLSL std430 with no padding.
Must match `SatOrbit` in `sat_orbit.comp` exactly.
```
[ 0] raan, u0, R_sat, meanMot
[16] cosI, sinI, cosRaan, sinRaan
[32] tumbleRate, tumblePhase, alignTerminator, tumbleAxisX
[48] tumbleAxisY, tumbleAxisZ, primaryAttitude (uint), secondaryAttitude (uint)
[64] baseColorR, baseColorG, baseColorB, crossSection
[80] specExp0, specExp1, w1, diffuse
[96] mirrorFrac, constIdx (uint), pad0, pad1
```
`static_assert(sizeof(GpuSatOrbit) == 112)` — do not change field order without updating both structs.

### GpuSatVisible layout (32 bytes, std430)
Output of `sat_flare.comp`; read by `sat_point.vert`.
```
[ 0] skyDir (vec3) + flareIntensity (float)  — ENU unit vector + intensity [0,1+]
[16] baseColor (vec3) + angularSize (float)  — tint + point sprite size hint (pixels)
```
`static_assert(sizeof(GpuSatVisible) == 32)`

### Push constants

**SatOrbitPC** (128 bytes) — sat_orbit.comp:
```
enuX (vec4), enuY (vec4), enuZ (vec4)  — ECI→ENU basis, offsets 0/16/32
sunDirECI (vec3), deltaT (float)       — offset 48/60
obsECI (vec3), satCount (uint)         — offset 64/76
highlightMask (uint), enabledMask (uint), simDt (float), elevCutoff (float) — offset 80/84/88/92
beamGain (float), reflectorLockWindowS (float), targetCount (uint),
minBeamElevSin (float)                 — offsets 96/100/104/108
gmstNow (float), windowFrac (float), mirrorMaxRateDegPerSec (float), pad2 — offset 112/116/120,
padded to 128
```
2026-08-06 reversibility rework repurposed the fields at offset 100-124 in place (same total size,
no growth): `mirrorSlewDegPerSec` → `reflectorLockWindowS` (a duration instead of a rate — see
below), `activeTargetCount` → `targetCount` (now the full loaded count, not a per-frame
night-side-compacted subset), `mirrorSnap` → `gmstNow` (current-frame GMST, for rotating a target's
static ECEF entry to its live ECI position), `minBeamElevSinRelease` → `windowFrac` (fractional
position within the current lock window, `fract(simTimeAbs / reflectorLockWindowS)`), and (same-day
follow-up) `pad1` → `mirrorMaxRateDegPerSec` — a real angular-rate cap, added once the original
fixed-fraction-of-window crossfade turned out to read as satellites snapping to target (it only
covered one of several transition cases and wasn't derived from actual angular distance — see
"Orientation" below). All of `deltaT`, `gmstNow`, and `windowFrac` are pure functions of absolute
sim time, computed on the CPU in double precision and narrowed to float only after the relevant
periodic reduction — so `sat_orbit.comp` can extrapolate exactly to a lock window's start instant
(see below) with no persisted GPU state anywhere in the pipeline.

**SatFlarePC** (128 bytes) — sat_flare.comp:
```
enuX (vec4), enuY (vec4), enuZ (vec4)  — offsets 0/16/32
sunDirECI (vec3), satCount (uint)      — offset 48/60
obsECI (vec3), elevCutoff (float)      — offset 64/76
brightnessScale, daySuppression, mirrorBoost, visThresh, highlightFlare,
pad2, moonSuppression, pad0            — offsets 80–108
moonDirECI (vec3), pad1 (float)        — offset 112/124
```
`pad2` was `lightPollution` — superseded (session 26) by the directional `lightDomeBuf` SSBO
(binding 3 in the sat_flare.comp descriptor set, 8 floats, host-visible/mapped), which doesn't
need push-constant space. See "Subsystem: Light Pollution Dome" below.

**SatDrawPC** (128 bytes) — `sat_sky.vert`/`.frag` (+ `_lite`/`_minimal`) via `skyBgPipeLayout`:
```
skyView (mat4)                          — offset 0
fovYRad, aspect, gmst, waveTime         — offsets 64/68/72/76
sunDirENU (vec4) — xyz=dir, w=sin(el)  — offset 80
moonDirENU (vec4) — xyz=dir, w=illum   — offset 96
obsECEFDir (vec4) — xyz=ECEF, w=obsHeightOffset — offset 112
```
Exactly the 128-byte `maxPushConstantsSize` floor (oldest AMD integrated parts). Everything that
used to trail past offset 128 — `debugDisableMask`, `screenSizePx`, `skyGlareVisibility`, the four
`beam*` scalars, `mwSuppressEased` — was per-frame-uniform and moved into the **CloudParams UBO**
(`cloud_params.glsl` / `GpuCloudParams` "Push-constant relief" block): `sat_sky.frag` reads them as
`cloud.dbgDisableMask` / `vec2(cloud.skyScreenW, cloud.skyScreenH)` / `cloud.skyGlareVisibility` /
`cloud.beam*` / `cloud.mwSuppressEased`. `buildSkyDrawPC()` fills it.

**PointDrawPC** (128 bytes) — `sat_point.vert`/`.frag` (`drawPipeLayout`) + `star_point.vert`/`.frag`
(`starPipeLayout`, also planets/trail):
```
skyView (mat4)                          — offset 0
fovYRad, aspect, waveTime, noTwinkle    — offsets 64/68/72/76
moonDirENU (vec4)                       — offset 80
obsECEFDir (vec4) — w=obsHeightOffset   — offset 96
screenSizePx (vec2)                     — offset 112   (always ctx.swapExtent — point draws never scale)
debugDisableMask (uint)                 — offset 120   (only sat_point.frag's knockout bit 4096)
manualTerrainTest (float)               — offset 124   (1 on the trail draws only)
```
Split from `SatDrawPC` so both point pipeline layouts fit 128 bytes while still carrying the two
per-draw flags that genuinely can't be frame-UBO'd: `noTwinkle`=1 on the planet draw, and
`manualTerrainTest`=1 on the trail draws — both differ between draws within one frame.
`buildPointDrawPC()` fills it; callers set the two flags. Each point shader declares only the
prefix it reads.

---

## Subsystem: TargetedReflector / Mirror Ground Targets

Mirrors in `TargetedReflector` mode aim at a ground target chosen by each satellite. All
per-satellite selection and orientation computation runs in `sat_orbit.comp`.

**2026-08-06 reversibility rework.** The previous design (lock + acquire/release hysteresis in
`mirrorNormalsBuf`, rate-limited slew integrated frame-by-frame) was persistent, history-dependent
GPU state: which target a satellite held and how far its mirror had slewed toward it were both a
function of the *sequence of frames* used to reach the current sim time, not of that sim time
alone. Playing forward to an instant and reaching the same instant by reversing time therefore
accumulated different lock/slew histories and could show different satellite/target pairings —
structural, not a tunable-away bug, since hysteresis is deliberately not time-symmetric. Separately,
the per-(satellite,target) preference hash (`hash11(float) `on a large combined index) silently
collapsed to a constant `0.0` for every candidate once a satellite's global dispatch index exceeded
~15,000 float32 mantissa range — Reflect Orbital satellites sit at index ~57,000+ given the
constellation upload order, so EVERY score was tied and argmax degenerated to "first eligible
candidate in scan order," meaning only the lowest-original-index site among a satellite's
simultaneously-eligible set ever won and other co-eligible sites were never picked.

Both are fixed together: `mirrorNormalsBuf` and the old per-frame CPU-compacted
`reflectorTargetsBuf` are gone. TargetedReflector selection and orientation are now pure functions
of the current frame's push constants (themselves pure functions of absolute sim time), so forward
and reverse playback of the same instant produce bit-identical results, and the hash is an
all-integer mix (`pairScore`, `sat_orbit.comp`) immune to magnitude collapse.

### Target generation (once at init) — S1, RELEASE_v1_1_PLAN.md
`SatelliteSim::loadReflectorTargets()` reads `reflector_targets.json` (next to the exe, moddable
exactly like `constellations.json`) — a hand-curated list of ~50 real, publicly-known solar
installations (`{name, lat, lon, capacity_mw}`), with the first entry flagged
`"observer_spawn": true` at the exact fixed spawn point (67°S, 67°W — see "Fixed Simulation
State"). Falls back to `generateReflectorTargetsRandomFallback()` (uniformly-random ECEF points,
still with a real fixed entry at index 0 for the observer-spawn pin) if the file is missing,
malformed, or empty. `kNumReflectorTargets = 201` is a **capacity** (buffer sizing), not the real
count — `reflectorTargetCount` (≤ capacity) holds how many actually loaded.
`reflectorObserverSpawnIdx` records which loaded index is the pin (informational/logging).

Per-target ground radius (`reflectorTargetsRadiusM[]`, real terrain elevation via a 3×3-max
`earthElevCpu` lookup) is computed by the shared `computeReflectorTargetElevationRadius(ti)`
helper — used by both the JSON path and the fallback. Both xyz (unit ECEF direction) and this
radius are uploaded ONCE, at generation time, into `reflectorTargetsECEFBuf` (host-visible,
`vec4` per target: xyz=ECEF dir, w=radius) — read by `sat_orbit.comp` for TargetedReflector target
search. There is no per-frame CPU rotation step any more; `sat_orbit.comp` rotates ECEF→ECI itself,
on demand, for whichever instant it needs (see below).

### Per-satellite selection (GPU, sat_orbit.comp) — deterministic lock windows
Target IDENTITY is chosen per fixed-width **sim-time window**
(`SatOrbitPC::reflectorLockWindowS`, default 90s, settings-window "Target lock window (s)"), not by
a persisted per-satellite lock.

**Per-satellite phase offset (2026-08-06 same-day follow-up).** `SatOrbitPC::windowFrac` is one
GLOBAL value, identical for every satellite dispatched this frame — without correction, every
TargetedReflector satellite's window boundary lands at the exact same sim-time instant, so all of
them ease toward a new target simultaneously: reported as one large synchronized wave of motion
regardless of how `reflectorLockWindowS`/`mirrorMaxRateDegPerSec` were tuned (a short window reads
as constant chaos, a long one as periodic mass movement). Fixed with a per-satellite hash offset —
`windowFracI = fract(pc.windowFrac + hashU(i × 0x2545F491u)/2³²)` — which is exactly the fractional
part of `(simTimeAbs + offsetSeconds_i) / W` (dropping the integer part doesn't care which multiple
of `W` `offsetSeconds_i` came from), i.e. this satellite's OWN window fraction, needing no extra
CPU-side work or push-constant fields. Every use of `windowFrac` in this section below is really
`windowFracI` in the shader; still a pure function of (sim time, satellite index), so still fully
reversible.

`sat_orbit.comp` extrapolates this frame's `deltaT`/`gmstNow` back to the CURRENT window's own
START instant — `toWinStart = -windowFracI × reflectorLockWindowS`, then
`evalDeltaT = deltaT + toWinStart` and `gmstEval = gmstNow + K_OMEGA_EARTH × toWinStart` (and one
more `reflectorLockWindowS` further back for the PREVIOUS window's start). **This extrapolation is
exact, not approximate**, for both quantities in this sim's model: orbital phase
(`u = u0 + meanMot×deltaT`) and GMST are both exactly linear in time, so evaluating at a shifted
`deltaT` is algebraically identical to the CPU having computed `deltaT` relative to a different
reference instant — no precision or physical approximation beyond what `deltaT`/`gmstNow` already
carry. The one real approximation is treating `sunDirECI` as constant across one window (true to
within the sun's ~1°/day drift against a ~90s window).

At each eval instant, `findWinner()` re-derives the satellite's own ECI position (`satEciAt`, same
closed-form math as the real position, evaluated at `evalDeltaT`) and scans **every** loaded target
(`pc.targetCount`, not a pre-filtered subset — the old per-frame CPU night-side compaction is gone),
rotating each target's static ECEF entry by `gmstEval`, rejecting day-side-at-that-instant and
anything below `pc.minBeamElevSin` (local elevation of the satellite *as seen from the target*), and
taking the `pairScore(satIdx, ORIGINAL target index)` argmax among survivors. Called once for the
CURRENT window's start (`bestIdx`) and once for the PREVIOUS window's (`bestIdxPrev`) — both
constant across their own whole window, since they only depend on that window's own start instant,
not on which frame within it asks. `pairScore` doesn't depend on the eval instant at all, only on
which targets are eligible — so a satellite's top-scoring target keeps winning unprompted across
consecutive windows as long as it stays eligible; a window boundary only actually changes the
winner when the incumbent drops out (day-side, or below `minBeamElevSin`).

`pairScore` is a pure function of the `(satellite, target)` pair (all-integer hash, see the block
comment above `hashU`/`pairScore` in `sat_orbit.comp`), independent of scan order and of how many
other candidates exist — the same load-spreading property the old `hash11`-based score was designed
to have, minus the magnitude-collapse bug.

### Orientation — a rate-limited ease, not integrated slew
2026-08-06 same-day follow-up: the first cut of this rework smoothed only ONE transition case (both
the current and a look-AHEAD "next window" valid and different) via a fixed-fraction-of-window
crossfade, unrelated to how far the mirror actually had to swing. Every other transition (acquiring
a target from nothing, losing one, falling back to the nearest night-valid site) snapped instantly,
and even the smoothed case could look abrupt for a wide swing compressed into a fixed ~13.5s —
reported as satellites visibly snapping to target. Replaced with a genuine angular-rate cap
(`SatOrbitPC::mirrorMaxRateDegPerSec`, settings-window "Mirror max slew rate (deg/s)") applied via a
closed-form ease, covering every transition uniformly:
- `bestIdxPrev` (previous window's winner — a one-window lookback, not unbounded history, so still
  a pure function of current sim time) gives `startAim`: the ideal aim direction toward whatever the
  mirror was presumably doing right as the CURRENT window began, evaluated AT the current window's
  own start instant (`idealTowards(satEciCur, targetPosAt(bestIdxPrev, gmstEvalCur), ...)`).
  `nearFallbackIdeal()` (nearest night-valid target, no elevation gate, or `FlatMirror45` if truly
  nothing night-valid exists) substitutes whenever the relevant index is `< 0`, used identically for
  "previous window had nothing" and "current window has nothing."
- `destAtStart` is the same computation for `bestIdx` (current window's own winner) at that same
  start instant — directly comparable to `startAim` since both are evaluated at the identical
  moment, only the target differs. `angle0 = acos(dot(startAim, destAtStart))` is therefore a
  single stable angle for the whole window, not something that recomputes every frame.
- `slewDuration = clamp(angle0 / radians(mirrorMaxRateDegPerSec), ~0, reflectorLockWindowS)` — the
  real time a slew at the configured max rate would take, capped at one window so a huge swing
  can't bleed into the NEXT window's own (independently computed) ease.
- The mirror eases from `startAim` toward `liveIdeal` (the target's true LIVE, unquantized position
  right now — this is what makes it track exactly once caught up, not toward a stale window-start
  snapshot) via `smoothstep(0, slewDuration, windowFrac × reflectorLockWindowS)`. When
  `bestIdxPrev == bestIdx` (the common case — nothing actually changed), `angle0` is exactly 0, the
  ease completes within a clamped instant, and the mirror simply tracks `liveIdeal` for the whole
  window, matching the previous design's "once slew has caught up" steady state.
- `bestIdx` (for beam bookkeeping — footprint, `beamCloudBlock` lookup) is always the CURRENT
  window's winner, using its live position, regardless of ease progress — only ORIENTATION lags
  during a transition, never where the beam is drawn. `aimErrorRad` is the residual angle between
  the eased orientation and `liveIdeal` (0 once caught up) — same field, same downstream consumer
  (`cloud_march.comp`'s beam debug ray fade) as before, just driven by the rate-limited ease instead
  of window-crossfade progress.

Because every one of these quantities — `windowFrac`, `evalDeltaT`/`gmstEval` (current AND
previous), `bestIdx`/`bestIdxPrev`, `angle0`, the ease itself — is a pure function of the current
frame's push constants (themselves pure functions of absolute sim time computed in double precision
on the CPU), the whole pipeline remains reversible by construction: there is nothing to accumulate
differently depending on playback direction, even though the visual result now has a genuine
physically-motivated slew rate.

---

## Subsystem: Reflect-Orbital Beam Cloud Occlusion

See `.plans/BEAM_CLOUD_PLAN.md` for the full session-by-session history — this section is the
current-architecture summary only.

**The visible beam is not called "the debug ray" despite its name.** `showBeamDebugRays`
(`SatelliteSim.h`) started as a literal debug-only visualization (a green line, C12 follow-up #12)
back when a separate volumetric "tube" was the real beam visual. Once that tube was thrown out for
graphics/performance reasons, this ray was reworked in a later session (realistic color, altitude
attenuation) into the actual production beam visual — it defaults to `true` and is what a player
sees, not a diagnostic overlay. The shader-side header comment in `cloud_march.comp` calling it
"Opt-in and off by default" is itself the stale artifact of that history, not a bug — a same-session
pass mistakenly "fixed" the default to `false` reasoning from that comment before this was
clarified (2026-08-09); do not repeat that mistake. This ray was, until 2026-08-09, occluded by
terrain ONLY — genuinely no cloud awareness at all — which is almost certainly what a string of
"beams pass through clouds no matter what" reports were actually seeing, independent of whatever
was fixed in the occlusion math underneath it.

**`beam_self_march.comp`** (2026-08-09) computes real per-BEAM cloud occlusion, replacing
`beam_cloud_block.comp`'s per-TARGET vertical-column approximation (deleted). Dispatched over
`BEAM_MAX_ACTIVE` (2048) threads right after `sat_orbit.comp` (needs the `satENU`/`reflectDirENU`
that shader just wrote), fixed-size every frame (`beamCount` is a GPU atomic counter, not known at
command-buffer record time — inactive slots return immediately). For each active beam it
reconstructs the satellite's true ECEF position from `ReflectBeamsBuf`'s observer-relative ENU
offset (via `terrain.glsl`'s `observerPos()`/`enuBasis()`), then marches the SEGMENT from the
satellite to the mirror's **actual current ground intersection** — `reflectDirENU` traced to
`R_EARTH` via `raySphere`, NOT the chosen target's fixed `targetENU` position — through the cloud
shell, overwriting `blockAltM`/`blockOpacity` on that same `ReflectBeam` entry in place.

**Marching toward `targetENU` instead of the real ray was a second, subtler bug**, found in-app
the same day after the ground-intersection fix above already shipped: occlusion looked correct for
a beam locked onto its target (where `reflectDirENU` and the target direction coincide) but did
nothing for a beam still slewing between two targets (where they genuinely diverge — see
`aimErrorRad`/the TargetedReflector orientation section above). The march was checking cloud along
a path the beam wasn't physically following. Fixed by using `raySphere(satECEF, reflectDirECEF,
R_EARTH)`'s hit point as the near endpoint instead of `targetECEF` — same shell-crossing math
otherwise (the hit point is still guaranteed inside both cloud-base/cloud-top spheres, same as
before), just anchored to reality instead of intent.

**Why per-beam, when "Deliberately NOT per-satellite" was the standing design rule** (see
`beam_cloud_block.comp`'s own retired header, and `TERRAIN_PLAN.md` follow-ups #14-#16): that rule
was about a DIFFERENT failure shape. Follow-up #14's cost blowup was a real `cloudDensity()` march
evaluated per SCREEN PIXEL near each beam's line, once per satellite — cost multiplied by
(satellites × pixels). Follow-up #16's flicker came from a dedup fix that picked one "winning"
satellite's geometry per target per frame, and the winner's identity flipped frame-to-frame among
near-equal candidates. `beam_self_march.comp` is neither: a single bounded march per beam (same
shape `beam_cloud_block.comp` itself already proved cheap at 201 threads — beam_self_march is
~10x the thread count along a segment instead of straight up, still well under 1ms against
`cloud_march.comp`'s own ~16ms), with zero arbitration — every beam computes its own value from its
own stable geometry, so there is nothing to flicker between.

**`blockAltM`/`blockOpacity` keep their pre-existing meaning** (altitude where the path's
transmittance first drops below 50%, and overall 0-1 opacity — including the weighted-mean-
absorption-altitude fallback for a column that never crosses that threshold, so a thin/scattered
cloud doesn't produce a fake cutoff edge pinned to the shell's nominal ceiling). `sat_sky.frag`'s
ground-spot `shadowAtten = 1.0 - groundBeams[bi].blockOpacity` needed zero changes and automatically
reads more physically-accurate per-beam data.

**The visible pointing ray and the volumetric cloud glow were reworked together, same day, per
explicit user direction, once the march above was marching the real path.** Both previously read
NOTHING from `blockAltM`/`blockOpacity` (the ray) or read it through a per-TARGET CPU aggregation
with no directionality at all (the old `beamCloudLighting()`/`BeamCloudLightBuf` glow — a purely
horizontal Gaussian around the target's ground position, gated by a height cutoff averaged/argmaxed
across every satellite servicing that site). The visible ray and the cloud glow ended up on two
DIFFERENT architectures, after an intermediate design that didn't work out:
- **The ray** stayed in `cloud_march.comp`'s `main()`, per-pixel, folded into its existing
  closest-approach loop (`showBeamDebugRays`-gated). It applies a height-cutoff `smoothstep`/`mix`
  shape at its own closest-approach altitude (`qAltM`, already computed there for the vacuum/
  altitude fade), ADDITIONALLY multiplied with the pre-existing `cloudGate` term — the two answer
  different questions and neither substitutes for the other: `cloudGate` gates whether the
  CAMERA's own line of sight to this point is blocked by unrelated cloud; the height-cutoff term
  gates whether the BEAM's own sunlight survives to reach this point along its real path. Both
  must be open for the ray to render there.
- **The cloud glow went through a second design that shipped and was reverted the same day.**
  First attempt folded it into that same per-pixel ray loop, reusing `perpDist` (the CAMERA's
  view-ray-to-beam-line distance) for proximity — cheap, but geometrically wrong: that distance's
  iso-contours are not soft blobs, they're hyperbola-like curves, and it rendered as large white
  rings intersecting the beams. Explicit user correction: this needs to work exactly like the
  sun/moon terms already do, evaluated PER CLOUD SAMPLE inside the volumetric march, not as a
  screen-space effect. Reverted outright (not patched) back to the ray-only per-pixel loop.
- **The glow's current (third) design restores the per-sample shape** (`beamCloudLighting()`,
  called from `cloudMarchCS`'s own loop and added into `inScatter` right alongside
  `sunColorCloud`/`moonContrib` — the "feed forward" the user asked for), fed from a small
  CPU-built list (`SatelliteSim.cpp`, `kMaxCloudBeamLights`=**512**, `GpuBeamCloudLights`; this
  doc said 16 for a long time and was simply wrong — the cap was raised for coverage and the
  number here was never updated, which is exactly why that function's real cost went unnoticed) — the only
  shape proven affordable at per-march-sample call frequency (TERRAIN_PLAN.md follow-ups #14/#16).
  Each list entry is now ONE individual real beam (no per-target aggregation, so nothing to
  average/argmax and nothing to flicker between): `posENU` is that beam's REAL ground intersection
  (`satENU + reflectDirENU` traced to `R_EARTH`, via the same rotation-invariant local-frame
  `raySphere` trick the GPU shaders use — valid on the CPU too, no ECEF conversion needed), and
  `dirToSource` is its REAL direction (ground toward satellite), driving `phaseCloud()` per light
  instead of a shared `normalize(p)` local-zenith stand-in the old per-target version used.
  (**"ONE individual real beam per entry" is history, not current behaviour** — clustering was added
  2026-08-09 and reworked 2026-08-12. The per-beam GEOMETRY described here is still exactly what
  feeds the list; what an entry represents is not. See "Cloud-light identity and cross-frame easing"
  below.)
- `sat_sky.frag`'s ground spot separately anchors at the same real ray-ground intersection
  (`raySphere(satWorldPos, reflectDirENU, R_EARTH)`) instead of `targetENU` — this part of the fix
  was correct from the start (not implicated in the ring bug, a genuinely different computation)
  and was left alone through the glow's redesign. The range-cutoff fade (`beamMaxRangeM`)
  deliberately stays keyed to the fixed target site position, not the transient ray-ground point —
  that's "is the observer close enough to this SITE," which shouldn't flicker as a mid-slew ray
  briefly touches down elsewhere.
- That pre-existing gap — knockout bit 128 gates `sat_sky.frag`'s ground-spot loop and
  `beamCloudLighting()`'s per-sample glow, but NOT `cloud_march.comp`'s per-pixel ray loop, which
  had only ever been gated by `showBeamDebugRays` — **was closed 2026-08-10 by knockout bit 8192**,
  and measuring it is what found the cost below.

### Cloud-light identity and cross-frame easing (2026-08-12)

`GpuBeamCloudLights`, the GPU-side struct and its 512 cap, is **unchanged** — so is every consumer
(`beamCloudLighting()`, `cullCloudLightsForTile`). What changed is how the CPU builds it. Full
session history in `.plans/BEAM_CLOUD_PLAN.md`; this is the architecture summary.

The list used to be rebuilt from scratch every frame, and a cluster's identity was **emergent**: it
was seeded by whichever beam the scan reached first at a target (a 2 m `targetENU` epsilon match)
and admitted members by comparing them against a running partial average that changed as members
were added. That partition is discontinuous in its own inputs — one satellite dropping out
repartitions the survivors, changing cluster count, every direction and every summed intensity in a
single frame. Since `posENU`/`dirToSource`/`blockAltM`/`blockOpacity` are all intensity-weighted
means and `hgG≈0.99`, that reads as lights popping and re-aiming instantly. **A 2026-08-11 attempt
to fix this by adding a fade on top was reverted** (still flickery, 20 FPS): easing a slot whose
*meaning* changes underneath it doesn't help, and recovering identity by proximity-searching a pool
cost O(rawClusters × 256) per frame.

Identity is now **declared**, which is possible because `sat_orbit.comp` carries its `bestIdx`
through as `ReflectBeam::targetIdx` (**this is why the record grew 64 → 80 bytes** — the three pads
next to it are load-bearing, see below):

| | key | why it's stable |
|---|---|---|
| cluster | `(targetIdx, direction bucket)` | bucket is a FIXED quantization of the beam's direction in the **target site's own** local ENU frame (`reflectorSiteEnu*[]`, computed once in `computeReflectorTargetElevationRadius()`) — a pure function of that beam's geometry, independent of scan order, of the cluster's contents, and of the observer |
| individual (transiting) | originating satellite dispatch index (`debugPad`) | already guaranteed stable by `sat_orbit.comp`, unlike the atomic-append slot `s` |

`TrackedBeamLight` (`SatelliteSim.h`) holds the persistent state: two pools with the same reserved
budgets as before (256 + 256 = `kMaxCloudBeamLights`, so the emit can't truncate), each with a
power-of-two open-addressed key→slot index **rebuilt from live slots every frame** — O(live) ≤ 256,
and it avoids tombstones entirely. Values are eased with the `1 - exp(-dt/τ)` idiom
(`mwSuppressEased`'s): intensity asymmetric (`beamClusterFadeInS`/`OutS`, Beams tab), geometry on a
shorter fixed `kTrackedLightGeomEaseS`.

**Three invariants:**
1. **Geometry is stored in Earth-fixed ECEF, in `glm::dvec3` — never observer-relative ENU.** A
   ground site is stationary in ECEF, so an entry that goes unmatched for its whole fade-out cannot
   drift however far or fast the observer moves. `rebase()` still applies to the raw per-beam fields
   (genuinely one frame stale) and must NOT be applied to tracked state: it is rotation-only, built
   for exactly one frame of lag, and the 2026-08-11 revert burned three rounds relearning that.
2. **`ReflectBeam`'s three explicit `uint` pads must exist in BOTH `reflect_beam.glsl` and
   `GpuReflectBeam`.** std430 rounds the GLSL struct up to its 16-byte alignment; C++ does not,
   because `glm::vec3` is 4-aligned. The pre-`targetIdx` total agreed at 64 only by luck. Same
   silent-permutation hazard as `GpuCloudParams`, and the `static_assert` only catches a size change.
3. **Nothing in the readback loop may depend on scan order again.** The `std::sort` by `debugPad`
   that used to enforce determinism is deleted — the partition is order-independent now (sums and a
   max), `groundTopK` ranks on intensity, and `nearest`/the opacity diagnostics are commutative.

`beamClusterDirThresholdDeg` kept its slider, label, range and settings key, but its **meaning
changed**: it was the merge tolerance against a running-average direction, it is now the angular
size of a fixed bucket. Settings → Beams also shows live pool occupancy, which is the instrument for
the failure mode that killed the previous attempt — a count pinned at 256 means entries are
respawning instead of matching; a count near the real active-site count means keying is working.

Accepted: the eased list is history-dependent, so it is **not** bit-reversible under time reversal
the way the orbital pipeline is. Same class and precedent as `skyGlareEased`/`mwSuppressEased`.

### Beam pointing-ray tile culling (2026-08-10)

The Anchorage worst-case sweep measured that per-pixel ray loop at **7.54 ms at Medium — 28% of the
whole frame**, and 4.72 ms of Planetarium's ~10.9 ms (43%). It ran `min(beamCount, 2048)` iterations
on every half-res texel (571 beams × 484,800 texels = **277M iterations/frame**) and rejected on
distance only at the END, after ~7 SSBO loads, a `raySphere` with a sqrt and the full
closest-approach solve. Beam rendering in total (this loop + bit 128's two consumers + the
`beam_self_march.comp` dispatch) was **14.0 ms, 52% of the Medium frame** — more than the volumetric
cloud march.

`cullBeamsForTile()` is the standard Forward+ light-cull answer, and it fits for free because this
shader is already dispatched at `local_size 16x16`: one workgroup owns a 16×16-texel screen tile, so
its 256 threads cooperatively test every beam ONCE against the tile's bounding view cone and leave a
short shared-memory list (`sTileBeamIdx`/`RayLen`/`Fade`, cap `kTileBeamMax`=384, ~4.6 KB shared)
that each thread walks. Per-thread iteration count becomes "beams crossing this tile" rather than
"beams that exist", so **cost stops scaling with the observer's location** — Anchorage concentrating
beams was the entire problem.

It also does the **Tier-1 hoist**: `rayLen` (a `raySphere` against `R_EARTH`), `targetDistM`/
`rangeFade` and `aimFade` are per-beam constants the old loop recomputed once per texel — 484,800
times each. They are now computed once per workgroup and passed through shared memory. Doing the
hoist here rather than in `beam_self_march.comp` needs no new `ReflectBeam` fields and no
producer-side plumbing, and — the deciding reason — does **not** make bit 512 (that dispatch's
knockout) an unsafe fallback.

**Three invariants, all easy to break:**
1. **`kDebugRayRadiusM`/`kDebugRayMinAngRad`/`kDebugRayMaxLenM`/`kDebugRayAimMaxRad`/`kSkyBeamFadeM`
   moved to file scope** precisely so the cull and the per-texel loop cannot disagree. If they
   diverge the cull stops being conservative and beams pop at tile boundaries.
2. **The conservatism bound.** Two rays sharing an origin and diverging by at most the tile
   half-angle are separated by at most `tFar * tileHalfAngle` anywhere within `tFar`; the closest
   point on the view ray to any point of the beam segment lies within `tFar` of the origin. So the
   accept test uses `radiusMax*4 + tFar*tileHalfAngle` with `radiusMax` derived from `tFar` (not the
   centre ray's own `t`). The failure mode is unmistakable: beams appearing/disappearing on a
   16×16-texel grid.
3. **The barriers must stay in uniform control flow.** `cullBeamsForTile` is called BEFORE `main()`'s
   out-of-bounds early return, so every thread of an edge workgroup reaches both `barrier()` calls;
   the scan itself sits in a push-constant-only (workgroup-uniform) branch, and the barriers are at
   the function's top level, outside it. `obsEffH`/`obsPos` are therefore also resolved above that
   early return — safe, since `observerEffHeight` depends only on `pc.obsECEFDir`, not on `coord`.

Overflow past `kTileBeamMax` sets `sTileOverflow` and falls back to scanning the whole buffer,
recomputing exactly what the cull would have supplied — a pathological tile gets slow, never wrong.
**Knockout bit 131072 forces that same fallback**, which makes it the cull's correctness A/B (image
must be pixel-identical with it on) and the way to measure what the cull bought — its sweep
`cost_ms` is the SAVING, reported with the opposite sign to every other row.

**Measured:** beam pointing rays 7.54 ms → **1.08 ms** at Medium; the in-sweep A/B says the cull
saves **6.78 ms** there and **3.29 ms** at Planetarium (which dropped 10.9 → 6.85 ms total).

### Cloud-light tile culling (`cullCloudLightsForTile`, 2026-08-10)

Same workgroup, same pattern, different list. `beamCloudLighting()` is called from `cloudMarchCS`'s
innermost loop — once per in-cloud SAMPLE — and walked all `beamLightCount` entries every time,
against a cap of **512**. That was 3.48 ms of the `cloud_march` bucket at Medium. The cull reduces
512 lights to the handful whose influence cylinder can reach any ray in the tile; the per-sample
loop walks that shared list instead.

Its conservatism argument differs from the ray cull's in one place worth understanding:
`tRangeMax` is the tile-centre ray's own far crossing of the **cloud-top sphere**, which
upper-bounds where `cloudMarchCS` can march (`tExit` is min'd against exactly that shell exit, and
both `tScene` and `maxRenderDistM` only shorten it), scaled by `kTileRangeMargin` = 1.25 so a
tile-edge ray whose own shell crossing runs slightly longer is still covered. `kBeamCutoffSigma`
moved to file scope for the same reason the ray constants did — the cull and the per-sample test
must bound against the identical radius. Bit 131072 disables this cull too.

### Beam ground-spot CPU hoist (2026-08-10)

`sat_sky.frag`'s ground-spot loop measured 1.59 ms at Medium, on a `GroundBeamsBuf` sitting at its
full `kMaxGroundBeams` = 256 cap, so every ground-hit pixel paid all 256 iterations at full
resolution. Almost the entire body was view-INDEPENDENT: a `length(targetENU)` + smoothstep range
fade, an `obsPos + satENU` and a `raySphere` (two sqrts) for the real ray/ground intersection, the
elevation fade, and the shadow attenuation. Only the horizontal distance to the landing spot and
the two Gaussians built from it genuinely vary per pixel.

All of it moved to the CPU loop that already builds this buffer each frame, into a new packed
`GpuGroundBeam` (32 bytes, mirrored as `GroundBeam` in `sat_sky.frag` — hand-mirrored, same
convention and same hazard as `GpuCloudParams`): `weight` (= intensity × rangeFade × elevFade ×
shadowAtten) plus `invFootprintSq`/`invCoreSq`/`cutoffSq`. The shader's per-beam work is now a 2D
subtract, a dot, a squared-distance reject and two exps — **with the reject first rather than
last**, and no sqrt anywhere. `intensity` is still carried, unread by the shader, purely so the
CPU top-K eviction ranks on exactly the quantity it did before (ranking stability is load-bearing —
see the flicker history at the insertion site). Entries that fail the elevation/range/ground-hit
tests keep their slot with `weight = 0` rather than being skipped, so top-K membership stays a
function of intensity alone and doesn't churn frame to frame.

---

## Subsystem: Photometry / Shader Constants

Photometry values are **runtime members** on `SatelliteSim`, synced to `SatFlarePC` each frame. They are persisted in `settings.json` and adjustable in the settings window.

| Member | Default | Description |
|--------|---------|-------------|
| `brightnessScale` | 1.0 | global flux multiplier |
| `daySuppression` | 500.0 | sky background suppression ratio (sun) |
| `mirrorBoost` | 300.0 | mirror peak multiplier (MIRROR_BOOST) |
| `visThresh` | 0.0 | visibility cull threshold |
| `highlightFlare` | 0.05 | fixed flare for highlight/census mode |
| `moonSuppression` | 4.0 | sky background suppression ratio (moon) |
| `lightPollutionGain` | 1.0 | multiplies the light-pollution dome at its source — see "Subsystem: Light Pollution Dome" |
| `extinctionCoeff` | 0.25 | atmospheric extinction, magnitudes per airmass — see "Subsystem: Atmospheric Extinction" |

`effectFlare = flare / (1 + (dayBright × daySuppression + moonBright × moonSuppression) × atmFrac)`,
then `×= extinction` (airmass), then `×= (1 − domeVal × 0.85)` (light pollution)
`magnitude = kMagRef - 2.5 × log10(effectFlare / kMagRefFlare)` where `kMagRef=6.0`, `kMagRefFlare=0.008`

`dayBright`/`moonBright` are elevation-ramp scalars (squared linear, sun/moon dot observer-zenith)
computed once per frame — **uniform across the sky, not per-satellite-direction**. This is an
accepted simplification for both (unlike light pollution below, neither has been made directional).
`moonBright` additionally omits the near-moon sky-brightening halo (real moonlight scatters more
strongly close to the moon's disc) — not built, no current plan to.

Stars (`SatelliteSim::updateStars`, CPU-side) apply the same three suppression sources
independently, with their own fixed (non-slider) response caps: `kStarPollutionMaxDim=0.85`,
`kStarMoonMaxDim=0.9`. Day suppression for stars is `nightFactorEff` (sun-elevation ramp), not
`dayBright`/`daySuppression` — a separate, older formula; the two were never unified.

## Subsystem: Light Pollution Dome

Session 26 replaced a single scalar (city brightness at the *observer's own* lat/lon — correct
about moving with the observer, wrong about being uniform across every direction of the sky) with
a 16-azimuth-sector dome, interpolated between sector centers, brighter near the horizon toward
nearby cities and fainter elsewhere — consumed identically by both satellites and stars.

**`SatelliteSim::updateLightPollutionDome()`** (CPU, called each frame in `recordCompute()` right
before `updateStars()`): for each of 16 sectors (22.5° each, bearing clockwise from North —
independent of `sat_flare.comp`'s unrelated `GlowBuf` 8-sector `azBin`, decoupled on purpose),
samples `earthNightCpu` at 4 radii (2/8/20/45 km) along that bearing using a flat-Earth
tangent-plane lat/lon offset (adequate at this scale), combined via **weighted max** (a single
nearby bright city should dominate that direction, not get averaged down by darker samples at
other radii in the same sector) with `exp(-D/20000)` distance weighting. The 2 km near sample
exists because the observer's own position can sit inside a bright pixel while every 8+ km ring
around it is already dark countryside (small/isolated towns) — without it the dome could miss the
pollution source entirely, the direct analog of the old scalar's distance-0 sample. Response curve
(`kNightFloor`/`kCityCompressK`) and the observer's own altitude falloff (`exp(-obsHeight/3000)`)
match the pre-session-26 scalar's constants exactly — only the sampling geometry changed. Result
scaled by `lightPollutionGain` (settings-window slider "Pollution gain", default 1.0, user-widened
range) applied once here at the source — **intentionally left unclamped**, not `clamp`ed to `[0,1]`
— so satellites and stars stay coherently scaled by construction (same array). A 5-tap circular
blur (`[0.1, 0.2, 0.4, 0.2, 0.1]`, ~±45°) then smooths the 16 raw per-sector values before storing:
each sector is a single bearing ray, so a real city's edge (which doesn't line up with 22.5° sector
boundaries) could put a bright sector directly next to a dark one — sampling noise, not genuine
geography, and the direct cause of "stars/satellites suddenly get much brighter" pops reported
when panning across a sector boundary near the horizon (worst there because `elevFalloff` is
largest at the horizon, fully exposing the noise). Result: `lightDomeAz[16]`, a CPU member array.

**Delivery:** `lightDomeAz` is memcpy'd into `lightDomeBuf` (host-visible/coherent, 16 floats,
binding 3 in the sat_flare.comp descriptor set — same `reflectorTargetsBuf`-style "CPU writes,
GPU reads this frame, single frame in flight" pattern, no barrier needed) for `sat_flare.comp`.
`updateStars()` reads the `lightDomeAz` CPU array directly, no upload round-trip needed.

**Per-consumer lookup** (both `sat_flare.comp` and `updateStars()` compute this the same way, GLSL
and C++ mirrors of each other): rather than a hard `azBin` lookup, interpolates between the two
nearest sector *centers* — `secF = bearing/22.5° - 0.5`, `sec0 = floor(secF)`,
`domeAz = mix(lightDomeAz[sec0], lightDomeAz[sec0+1], frac(secF))` (both indices wrapped mod 16).
Hard-binning (even at 16 sectors) showed visible blocky transitions over wide, fairly uniform
bright regions (e.g. flying over Europe) — the interpolation, not the sector count, is what fixes
that. Then `elevFalloff = 0.35 / (max(skyDir.z, 0) + 0.35)` (1.0 at the horizon, ~0.26 at zenith —
city glow hangs low in the sky, not overhead). **The only clamp is here**, after `elevFalloff`:
`domeVal = clamp(domeAz * elevFalloff, 0, 1)` — clamping `lightDomeAz` itself upstream was a real
bug (fixed same session): it let `elevFalloff` (≤1 off the horizon) silently cap the *effective*
max well below 1.0 at every non-horizon angle, no matter how high `lightPollutionGain` went, so
gain past the point where it first saturated the pre-`elevFalloff` value (~5) looked identical to
gain=500. `domeVal` feeds the existing `1 - domeVal × kPollutionMaxDim` dimming multiplier
unchanged (`kSatPollutionMaxDim` = 0.85 in `sat_flare.comp`, `kStarPollutionMaxDim` = 0.99 in
`updateStars()`, both user-tuned — still a hard ceiling on max dimming regardless of gain).

**S2c isotropic floor (RELEASE_v1_1_PLAN.md, session 30):** `elevFalloff` alone bottoms out at
~0.26 at zenith, so no `kPollutionMaxDim` could ever dim a straight-overhead target — satellite,
star, or the Milky Way — by more than ~26%, regardless of how bright the city or how high
`lightPollutionGain` went. This is why the Milky Way stayed visible near cities: it's a large,
mostly-high-in-the-sky feature living almost entirely in the region `elevFalloff` can't reach. Real
urban skyglow raises zenith brightness far more (Bortle 8 zenith ≈ 50× Bortle 1) via isotropically
scattered light, not just the horizon-hugging direct glow `elevFalloff` models. Fix, applied
identically at all four consumers (`sat_flare.comp`, `sat_sky.frag`'s Milky Way, `cloud_march.comp`'s
aurora, `updateStars()`) — `beamDomeVal`/beam-glow dome copies are deliberately left alone, since a
Reflect-Orbital beam flash is a genuinely horizon-hugging point source, not city skyglow:
```
domeVal = clamp(domeAz * (kIsotropicFrac + (1 - kIsotropicFrac) * elevFalloff), 0, 1)
```
`kIsotropicFrac = 0.4` (hand-duplicated at each site, same convention as `elevFalloff` itself).
Horizon behaviour (`elevFalloff≈1`) is unchanged; zenith now floors at `domeAz * kIsotropicFrac`
instead of `domeAz * 0.26`. Expect `lightPollutionGain` to need re-tuning after this — it now reaches
brightness levels near cities it structurally could not reach before.

**Not built:** the elevation falloff shape is a fixed analytic curve, not itself sampled/measured —
a true 2D (azimuth × elevation) dome would need real atmospheric-scattering-height modeling, judged
not worth the complexity over the fixed-curve approximation.

## Subsystem: Atmospheric Extinction

Session 26 follow-up: the light-pollution dome's `elevFalloff` was, until this was added, the
*only* term anywhere that varied a star's or satellite's brightness by its own viewing elevation —
there was no real horizon-dimming baseline, which is part of why the dome's directional noise (see
above) read as unsubtle: nothing else was smoothly dimming things toward the horizon for it to
modulate on top of.

**Formula** (identical in `sat_flare.comp` and `updateStars()` — a star and a satellite at the same
elevation must dim by the same amount, since this represents real atmospheric transmission, not a
stylized brightness knob): Kasten & Young 1989 airmass approximation,
`airmass = 1 / (sin(el) + 0.50572 × (elDeg + 6.07995)^-1.6364)` — stays finite down to the true
horizon (elDeg=0 → airmass≈38), unlike the naive `1/sin(el)` which diverges to infinity. Then
`extinctMag = extinctionCoeff × (airmass - 1) × atmFrac` (magnitudes of dimming beyond the zenith
baseline; `atmFrac`-gated since an orbiting observer has no atmospheric column along the line of
sight regardless of apparent "elevation" in their local frame) and
`extinction = 10^(-0.4 × extinctMag)`, multiplied directly into `effectFlare`/star `intensity`.

**Tunable:** `extinctionCoeff` (magnitudes per airmass; ~0.2-0.3 is typical clear-sky sea-level;
default 0.25), settings slider "Extinction". Reuses `SatFlarePC`'s `pad2` slot (the one freed by
`lightPollution`'s move to `lightDomeBuf`) rather than growing the struct — stars read the same
`extinctionCoeff` C++ member directly, no separate push-constant path needed for the CPU side.

`MIRROR_BOOST = 300` — peak multiplier for near-perfect mirror alignment. `mirrorExp = max(specExp0 × 300, 8000)` gives sub-degree angular width (matches solar disc ~0.26°).

---

## Subsystem: GpuSatInput Layout (80 bytes, std430)

Written by `sat_orbit.comp`, read by `sat_flare.comp`.

```
[  0] eciRelPos (vec3) + range (float)
[ 16] surfN0    (vec3) + elevation (float)   — primary surface normal; elevation = -π/2 for below-horizon/disabled
[ 32] surfN1    (vec3) + specExp0 (float)    — secondary surface normal
[ 48] baseColor (vec3) + specExp1 (float)
[ 64] crossSection + w1 + diffuse + mirrorFrac (float×4)
```

`static_assert(sizeof(GpuSatInput) == 80)` — do not change field order without updating both the C++ struct and the GLSL `SatInput` struct in `sat_flare.comp`.

Below-horizon and disabled satellites write `elevation = -π/2` and return early. `sat_flare.comp`'s horizon cull (`elevation < -0.01 rad`) discards them at zero cost.

---

## Subsystem: Sky Glow SSBO

`sat_flare.comp` writes a spatial histogram + per-satellite flare list each frame → `sat_sky.frag` reads them.

### GpuGlowBuf layout (std430)
```cpp
static constexpr int kGlowBins  = 64;   // 8 azimuth × 8 elevation cells (45° × 11.25°)
static constexpr int kMaxFlares = 8;    // per-satellite lens-flare slots

struct GpuGlowBuf {
    uint32_t bins[kGlowBins];           // atomicMax(floatBitsToUint(effectFlare)) per bin — wide Gaussian glow
    uint32_t flareCount;                // number of entries claimed (capped at kMaxFlares)
    uint32_t flarePad[3];
    glm::vec4 flareEntries[kMaxFlares]; // xyz=ENU dir, w=effectFlare — spiky corona + lens artifacts
};
// sizeof = kGlowBins*4 + 16 + kMaxFlares*16
```
`static_assert(sizeof(GpuGlowBuf) == kGlowBins * 4 + 16 + kMaxFlares * 16)`

`kGlowBins` and `kMaxFlares` must match constants in `sat_sky.frag`. `glowBuf` must be zeroed with `vkCmdFillBuffer` before each `sat_flare.comp` dispatch (floatBitsToUint(0.0) == 0u, so fill value 0 correctly marks bins empty).

---

## Subsystem: Planets

See `PLANETS_PLAN.md` for the session log and forward-looking next-steps list (in-app QA still
outstanding, known simplifications, ring rendering, attribution follow-up) — this section is the
architecture/design writeup only.

Session 30. Mercury, Venus, Mars, Jupiter, Saturn, Uranus (`enum PlanetId`, `kPlanetCount = 6` —
Neptune excluded, never naked-eye at ~mag 7.8) with real Keplerian-approximation orbital positions,
rendered as clickable points of light. Deliberately **not** built on the satellite orbital-compute
pipeline (`GpuSatOrbit`/`sat_orbit.comp`) — that solves near-field Earth-relative geometry (shadow,
attitude, specular surfaces) planets don't have. Instead: closed-form CPU math in the sun/moon
pattern (`updatePositions()`), rendered through the star pipeline's shape (direction + magnitude-
driven brightness/size, no near-field 3D position needed at render time).

**Ephemeris** (`SatelliteSim.cpp`, top-of-file constants block, `keplerEclipticPos()`): low-precision
Keplerian elements + linear centurial rates (JPL/Standish, valid 1800-2050 —
https://ssd.jpl.nasa.gov/planets/approx_pos.html), one `KeplerElements` row each for Earth
(`kEarthElements`, the table's EM Bary row) and the six planets (`kPlanetElements[]`). Computed every
frame in `updatePositions()`, right after the Sun/Moon block, using the same `Tcent` (Julian
centuries since J2000, derived from the same `dJ2000` the Sun calc already computes) and the same
`epsR` obliquity rotation — no separate time base. Standard Newton-Raphson Kepler-equation solve +
3-1-3 (ω,i,Ω) orbital-plane-to-ecliptic rotation; geocentric vector = `helio_planet - helio_earth`.
Results land in `planetStates[kPlanetCount]` (`PlanetState`: `eciDir`/`distanceAU`/`sunDistAU`/
`phaseAngleDeg`), a plain ephemeris record — distinct from the render-ready `GpuSatVisible` entries
`updatePlanets()` derives from it each frame.

**The Moon's direction was also fixed in this session**, same block, same pattern: it was a circular
equatorial orbit with a phase constant (`kMoonPhaseOffsetRad`) hand-calibrated for a single epoch
(2026-03-30) that had already drifted stale for the sim's actual fixed epoch (2036-06-21) — see
"Fixed Simulation State" above. Replaced with `kMoonElements`, a two-body Keplerian fit to the linear
terms of the Moon's real ELP2000-82B mean elements (Meeus ch. 47) — real inclination (~5.145°) and
eccentricity, still an approximation (no evection/variation/other periodic perturbations) but a real
improvement, run through the exact same `keplerEclipticPos()` used for the planets (geocentric
directly, no Earth-subtraction step). `moonIllum`'s formula is unchanged — only the direction's
derivation changed, so `sat_sky.frag`'s Moon disc rendering needed zero changes.

**Brightness** (`updatePlanets()`, mirrors `updateStars()`): apparent magnitude via
`planetApparentMagnitude()` — Paul Schlyter's standard formulas
(stjarnhimlen.se/comp/ppcomp.html), `V = V0 + 5*log10(r*Δ) + phase-angle polynomial`. Saturn's ring
brightness is deliberately omitted (needs Saturnicentric ring-plane geometry, not just phase angle —
accepted simplification, dims slightly near ring-plane-open oppositions). Converted to
`rawIntensity = 10^(-V/2.5)` — **the same convention `initStars()` uses** — so a planet's brightness
runs through the exact same suppression chain stars already have (day/moon/pollution-dome/
extinction), hand-duplicated into `updatePlanets()` per this codebase's established per-consumer-
duplication convention for that formula (see "Subsystem: Light Pollution Dome" above). Point-sprite
size reuses `initStars()`'s `0.25 + 2.5*sqrt(rawIntensity)` curve so a planet reads at the same
visual weight as an equally-bright star. Color (`kPlanetColor[kPlanetCount]`, same-day follow-up):
hand-picked approximate true colors, not computed — planets have no B-V spectral index to derive
one from the way stars do. Mars is the one that actually reads as visibly colored at naked-eye
scale (rust/salmon); the others stay close to near-white/pale by design, matching their real subtle
cloud-top/regolith colors. No shader change needed — `star_point.frag`'s existing intensity-driven
desaturation (bright = full tint, faint = fades toward white) was already written generically
against `fragColor`/`fragIntensity` and applies correctly to planets for free.

**Rendering**: a second tiny host-mapped `planetBuf` (`GpuSatVisible`-shaped, 6 entries) + a second
descriptor set (`planetDescSet`, reusing `starDescLayout`/`starDescPool`'s shape via its own tiny
`planetDescPool` — `starDescPool` itself is sized `maxSets=1`) — but the **same** `starPipeline`/
`starPipeLayout`/shaders (`star_point.vert`/`.frag`), just a second `vkCmdDraw` in `recordDraw()`
right after the star draw. Reusing the pipeline object means `onResize()`'s existing
`createStarPipeline()` recreation covers planets for free — no separate resize handling needed.
One real shader difference was necessary: `star_point.vert`'s atmospheric scintillation/twinkle is
physically wrong for planets (small resolved discs, not point sources) — gated behind a new
`noTwinkle` field, set to 1 only on the planet draw's own copy of the push constant. (Originally
`SatDrawPC.noTwinkle` at offset 164; now `PointDrawPC.noTwinkle` at offset 76 after the
128-byte push-constant split — see the **PointDrawPC** entry under "Subsystem: GPU Orbital
Pipeline".)
The pre-existing Moon-disc occlusion cull in the same shader stays active for planets unchanged
(correct — a planet behind the Moon's disc should still be culled).

**Picking**: `pickPlanetAt()` is cheaper than `pickSatelliteAt()` — `planetBuf` is already
HOST_VISIBLE/COHERENT (`updatePlanets()` writes it directly from the CPU), so unlike satellites'
device-local `satVisibleBuf` there's no staging-buffer copy at all, just a loop over 6 entries using
the existing `projectSkyDirToScreen()`. `selectedPlanetIndex` (mutually exclusive with
`selectedSatIndex` — selecting one clears the other) is tried first (planet priority on an exact
overlap) at both click sites (`SatelliteSimUI.cpp`'s mouse-click handler and `KB_SELECT_SAT`'s
center-screen equivalent). `formatSelectedPlanetInfo()` fills `planetInfoLine[]` (name/magnitude/
distance/phase) — unlike `formatSelectedSatInfo()` (called only when the selection changes, since a
satellite's orbital elements are static), this is re-called **every frame** the selection is active
(right after `updatePlanets()` in `recordCompute()`), since a planet's distance/phase/magnitude
changes continuously. `buildSelectedSatPanel()` branches on `isPlanet` to pick its data source
(`planetBuf`'s own mapped memory directly for planets — never stale, vs. satellites' one-frame-stale
`lastPickedSkyDir` GPU round-trip) and info-line array, then shares the rest of the panel/reticule
rendering unchanged.

**Settings**: `showPlanets` (global) + `planetEnabled[kPlanetCount]` (per-planet), UI in
`buildSettingsConstellationsTab()` reusing the exact ON/OFF row pattern the constellation list
already uses. Persisted as `j["planets"]` (`{show_planets, list:[{name,enabled}]}`) in
`saveSettings()`/`loadSettings()`, same shape and same ungated (non-schema-versioned) treatment as
`j["constellations"]` — not a graphics-tuning value a schema mismatch needs to guard against.

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
- Compute→compute SSBO barriers use `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` on both sides
- `onResize` must recreate graphics pipelines (viewport baked in); compute pipelines are viewport-independent
- `glowBuf` is `HOST_COHERENT` so CPU can read back peak flare for magnitude UI without an explicit flush; previous frame's data is safe to read at the start of `recordCompute` (single frame in flight means queue is idle)

---

## Subsystem: Persistent Settings

`settings.json` is written next to the exe on settings-window close and on `cleanup()`, loaded in `init()` after `initConstellation()`.

Persisted fields: photometry params, `ui_scale`, settings window position, audio volumes, camera orientation (`az_deg`, `el_deg`, `fov_y_deg`), observer lat/lon, time scale index, keybindings (action → GLFW key code), constellation `enabled` + `highlight` state per name.

If the file is missing (first run) all defaults are used silently.

---

## Subsystem: GPU Performance Profiling

Built session 29 to replace guesswork ("N_VIEW is probably the bottleneck") with real
measurement — used to find and fix the terrain step-count bug and the aurora resolution/noise-bake
wins documented under "Active Development" above. Four pieces:

**In-app GPU timestamp queries** (`VulkanContext`): a `VK_QUERY_TYPE_TIMESTAMP` pool whose slot
count has changed several times as passes came and went — `VulkanContext::kTimestampCount` carries
the authoritative slot table and is the only place that mapping is documented. `updateGpuTimingStats`,
`kPerfLabels[]` and `savePerfSnapshot()`'s JSON keys all mirror it and must be updated together.
Single frame in flight, so results are resolved in `App::drawFrame` right after the fence wait —
no stall. Slot layout is a shared contract: App.cpp writes 0 (frame start), 5 (satellite+star draw
done), 6 (UI overlay done); `SatelliteSim` writes 1-3 in `recordCompute` (cloud march / orbit
compute / flare compute done) and 4 in `recordDraw` (sky background draw done — this is what
isolates the fullscreen atmosphere/terrain/ocean shader's own cost from the satellite/star point
draws that follow it in the same render pass; they used to be one fused bucket).
`SatelliteSim::updateGpuTimingStats()` EMA-smooths the six deltas into `gpuMsSmoothed[6]`,
displayed in Settings → Display → "GPU FRAME BREAKDOWN" (one-frame-stale, same pattern as
`peakMagnitude`).

**CPU frame timing** (`CpuBucket` / `cpuMsRaw[]` / `cpuMsSmoothed[]` / `beginCpuFrameTiming()`,
2026-08-10): the counterpart to the GPU timestamp buckets, displayed in Settings → Display →
"CPU FRAME BREAKDOWN" and logged as `cpu_timing_ms` (snapshots) / `knockout_sweep.baseline_cpu`
(sweeps). Buckets: `build_ui`, `update_positions`, `beam_readback`, `update_stars`,
`light_pollution_dome`, `update_planets`, plus a derived `other` = wall clock − GPU total −
everything measured (present/vsync wait, driver submit, App-side work, and any CPU block without a
bucket yet). **A large `other` is a finding, not a gap to hide** — it says the cost is somewhere
this table doesn't look.

Built because the GPU work above succeeded: once Medium's Anchorage GPU frame reached 15.5 ms, the
Release wall clock was 18.3 ms, and the ~2.8 ms remainder was *near-identical at Planetarium*
(2.73 ms against a 6.0 ms GPU frame — 31% of that frame). Fixed per-frame cost that doesn't scale
with rendering load is exactly what the GPU buckets exist to expose, and nothing equivalent existed
on the CPU side, so that number was opaque and unshrinkable without guessing.

Timers are scoped (`SatelliteSim::CpuTimer`, an RAII adder — it ACCUMULATES rather than assigns, so
one bucket can be timed at several sites or inside a loop). `beginCpuFrameTiming()` must stay the
**first statement in `buildUI()`**, the first sim entry point of a frame: it publishes the previous
*complete* frame and clears the accumulator, so it can never publish a half-filled one. The
resulting one-frame staleness deliberately matches `gpuMsRaw[]`'s, which is what lets a sweep step
sample a CPU frame and a GPU frame from the same moment instead of one lagging the other.

**Perf knockout toggles**: `debugDisableMask` (uint32) is a profiling-only bitmask. It rides in the
CloudParams UBO as `cloud.dbgDisableMask` (read by `sat_sky.frag` and `cloud_march.comp`), plus a
copy in `PointDrawPC` for `sat_point.frag`'s bit 4096 — all pushed from the single
`SatelliteSim::debugDisableMask` member each frame. Checkboxes in Settings → Display →
"KNOCKOUT PROFILING" (18 as of 2026-08-10, driven by the single `kDebugToggles` table at the top of
`SatelliteSimUI.cpp` — bit, display label, and stable JSON key per row; adding a row there adds a
checkbox AND a sweep step for free) each disable one shader block or dispatch — each with a mathematically-safe
zero/no-op fallback (e.g. terrain-skip leaves `tHit=-1`, the same value the "no hit" path already
produces) or, for the two producer-side bits, a "reproduce pre-feature behaviour" fallback. Default
mask 0 is bit-identical to normal rendering. Use this to isolate one block's real GPU cost via
before/after `gpuMsSmoothed` deltas, without a GPU capture tool — bit assignments: 1=terrain,
2=atmosphere, 4=sunOD (`optDepth`, called from 4 sites — zeroing it there is a single early-return
in the function itself, not 4 separate call-site edits), 8=oceanRefl, 16=airglowRed, 32=aurora
curtain, 64=cloud self-shadow cone, 128=Reflect-Orbital beams (both `cloud_march.comp`'s
volumetric term and `sat_sky.frag`'s ground-spot term), 256=cloud shadow (per-pixel, in
`cloud_march.comp`), 512=`beam_self_march.comp` DISPATCH itself (producer-side; repurposed
2026-08-09 from the now-retired `beam_cloud_block.comp`'s identical bit), 1024=scene depth pass
DISPATCH itself (producer-side — the big one: skipping it reverts the entire shared-depth
architecture to pre-unification occlusion behaviour), 2048=fog layer (C11), 4096=satellite point
cloud occlusion (`sat_point.frag` — added 2026-08-09 to isolate a reported perf question, see
`BEAM_CLOUD_PLAN.md`; ruled out as the cause, kept as a real diagnostic),
8192=Reflect-Orbital beam POINTING-RAY loop (`cloud_march.comp`'s per-pixel loop in `main()`),
16384=`cirrusMarchCS`, 32768=`cloudMarchCS` (the volumetric low/mid march itself),
65536=`sat_sky.frag`'s 64-bin satellite sky-glow loop, 131072=**beam tile cull OFF** (not a feature
knockout — an optimization A/B; see "Beam pointing-ray tile culling" below),
262144=**Potato sky** (swap `skyBgPipeline` → `skyBgMinimalPipeline`), 524288=**SKY_LITE sky**
(swap → `skyBgLitePipeline`) — see "Subsystem: Weak-Hardware Sky Tiers". These last two are
pipeline swaps, not in-shader branches; they're set by the Potato / Planetarium presets and are
NOT part of the `kDebugToggles` sweep.

**Bits 8192-65536 were added 2026-08-10** for the Anchorage worst-case profiling session, and all
four cover blocks that previously had NO knockout, so their cost was permanently invisible inside a
lumped bucket. Two of them (8192, 65536) additionally had no quality slider and **no preset reach**
— `applyGraphicsPreset` could not turn them off at any tier, so Planetarium paid for them in full.
Bit 8192 in particular closes the gap this document already flagged under "Subsystem:
Reflect-Orbital Beam Cloud Occlusion" ("knockout bit 128 gates … but NOT `cloud_march.comp`'s
per-pixel ray loop … flagged here so a future profiling session doesn't assume bit 128 isolates the
full beam rendering cost").

**Automated knockout sweep** (Settings → Display → "Run knockout sweep", `startKnockoutSweep`/
`updateKnockoutSweep` in `SatelliteSimUI.cpp`): walks the whole `kDebugToggles` table on its own —
baseline first, then one step per bit — holding each mask for `kSweepSettleFrames` (6) discarded
frames then averaging `gpuMsRaw[]` over `kSweepSampleFrames` (24), and appends ONE
`"record_kind": "knockout_sweep"` record carrying every step's full bucket breakdown plus its
`cost_ms` delta against the baseline. ~15 s for the whole table.

Two things make it trustworthy where a hand capture isn't. It reads `gpuMsRaw[]`, **not**
`gpuMsSmoothed[]` — the HUD's EMA (α=0.1) takes ~40 frames to settle, so a hand capture taken soon
after flipping a checkbox silently reads a blend of two configurations. And it **forces
`timePaused` for its duration** (restoring the user's value after), so all 18 steps measure the same
frame; without that, satellites move, beams re-target and clouds drift across a multi-second sweep
and the per-bit deltas mix real cost with scene change. The record's top-level `gpu_timing_ms` is
overwritten with the sweep's own baseline window rather than the EMA, which at write time is still
decaying out of the last knockout step. `cost_ms` is deliberately **not** clamped at zero — a
knockout can legitimately come out negative (noise, or a skip that makes a later pass do more work
because nothing occludes it any more, bit 1024 being the standing example), and that is information.

`analyze_profile.py`'s `print_sweep_report()` prints each sweep as a ranked cost table and, per row,
which bucket actually moved most — measured rather than assumed, since a bit can sit in a different
pass than expected (128 spans two shaders; 1024 is a producer whose skip shows up downstream).

**`perf_profiles/profile_log.jsonl`**: the "Save Snapshot" button (same panel) appends one JSON
record per press — GPU timing breakdown, resolution, observer lat/lon/altitude, sim time, active
knockout mask, GPU device name, quality settings, graphics preset name, and (2026-08-10) a `beams`
block: `active_count`/`ground_spot_count` and their capacities, plus `show_beam_rays`. Beam count is
a first-order driver of `cloud_march`'s cost (its pointing-ray loop iterates
`min(beamCount, 2048)` times per half-res pixel) and varies enormously by observer location, so
without it the Anchorage captures were not interpretable. JSON Lines (not a JSON array) so the log
grows by simple appending across sessions/restarts. Every record now carries `record_kind`
(`"snapshot"` or `"knockout_sweep"`). `SatelliteSim::savePerfSnapshot()` and
`buildPerfSnapshotJson()`/`appendPerfRecord()`, which the sweep shares so the two record kinds
can't drift apart.

**`tools/perf_analysis/`**: a small Python toolkit (gitignored `.venv`, `requirements.txt`: pandas
+ matplotlib) — `analyze_profile.py` reads the JSONL log and reports GPU cost by resolution bucket,
per-megapixel cost (flat across resolutions = purely resolution-bound), a matched-altitude
resolution ratio (isolates the resolution effect from confounding scene/altitude changes in the
same dataset — see the script for why raw correlation isn't enough), a knockout-toggle cost
summary, and Pearson correlations against scene variables, plus two PNG plots. Re-run this any
time a new round of snapshots is captured: `tools/perf_analysis/.venv/Scripts/python.exe
tools/perf_analysis/analyze_profile.py`.

See `TERRAIN_PLAN.md` session 29 log for the full narrative — what was measured, what was
concluded, and which prior assumptions (the session-24 transmittance-LUT guess) it overturned.

---

## Subsystem: Cloud Shadows

Marched **per pixel** inside `cloud_march.comp` (`cloudGroundShadow`), sunward from the terrain hit
point `scene_depth.comp` supplies, and delivered in `cloudTargetB.a` — the channel freed by
deleting `tEnterCombined`. `sat_sky.frag` consumes it as a single `directSun *= cloudB.a`.
Gated on `tScene < kNoSurfaceT`, so sky pixels pay nothing. Knockout bit 256.

This replaced `cloud_shadow.comp`, a fixed 128x128 observer-centred tangent-plane grid, which is
worth understanding because the failure modes were structural rather than tuning:

| | 128² grid | per-pixel |
|---|---|---|
| Resolution | 1250 m/texel uniformly at the 80 km default | screen-space; metres near camera, coarsens with distance |
| From altitude | observer-centred at fixed world extent, so it covered less and less of the visible ground | follows the view ray |
| Range | hard cutoff at `cloudShadowRangeM` (which had already caused a real beam bug, worked around via `blockOpacity`) | none |
| Swimming | needed `computeCloudShadowSnap()` + a residual subtracted by every consumer | nothing to snap — the value is a function of the world point being shaded, not the camera |

Deleted with it: the image/view/sampler/descriptor set/pipeline, `CloudShadowPC`, the snapping
block, two `SatDrawPC` fields, the "Cloud shadow range (m)" slider and its settings key, and the
pass's timestamp bucket. Same 12 steps and the same `3e-3` density→optical-depth constant the grid
used, so brightness is comparable. The march phase is jittered off `noiseTex` because adjacent
half-res texels can map to ground points kilometres apart at grazing angles.

---

## Subsystem: Resolution Scaling

Settings → Display → "Render scale" (50%-100%, default 100%, top of the tab — a user-facing perf
option, not a debug tool). Only the sky/terrain/ocean/cloud-composite background scales;
satellites, stars, and UI always render at native resolution, no exceptions — the explicit design
goal, given the earlier-session concern about losing tiny satellite point fidelity to a whole-
frame downscale.

**At the default (`renderScale==1.0`) this is a no-op — the code path is identical to before the
feature existed.** Below 100%, `SatelliteSim::recordPrePass` (a new `Simulation` interface hook,
default no-op, so the other simulations needed zero changes) renders the background into a low-res
offscreen target and blits it (`vkCmdBlitImage`, linear filter) directly into the swapchain image
*before* the main render pass opens. The main render pass then uses `ctx.renderPassLoad` instead of
`ctx.renderPass` — a second render pass object (`Simulation::activeRenderPass`, another new hook,
default `ctx.renderPass`) with the SAME attachment formats (so it stays compatible with the same
`ctx.framebuffers` — render-pass compatibility only requires matching format/sample-count, not
matching load/store ops) but LOAD instead of CLEAR for color, so the pre-pass's blit survives into
the frame instead of being cleared away.

**Depth is deliberately not blitted** — depth-format blit support isn't spec-guaranteed the way
color-format blit support effectively always is, a real portability risk specifically on the
lower-end hardware this feature targets. Consequence, accepted: satellites/stars are not occluded
by terrain while `renderScale<1.0` (a satellite that should hide behind a mountain may show
through). Only applies below 100%.

**`gl_FragCoord` gotcha — read this before adding any new `gl_FragCoord`-based lookup to
`sat_sky.frag`.** `gl_FragCoord.xy` is relative to whatever framebuffer the CURRENT draw call
targets, not always the full swapchain — a real bug shipped and was fixed same-session: the cloud
composite sample divided `gl_FragCoord.xy` by `textureSize(cloudTargetA,0)*2.0` (an assumed full-
res constant, since `cloud_march.comp`'s own dispatch is unaffected by `renderScale`), which
silently broke the moment the background could render into a smaller offscreen target — clouds
drifted off-center, fully distorted at 50%. The sky render-target size is now `cloud.skyScreenW`/
`skyScreenH` in the CloudParams UBO (`skyLowResExtent` when the scaled prepass runs — recordPrePass
and recordDraw's Pass 1 are mutually exclusive, so one per-frame value suffices — else
`ctx.swapExtent`); any `[0,1]`-normalized UV in `sat_sky.frag` derived from `gl_FragCoord` divides
by `vec2(cloud.skyScreenW, cloud.skyScreenH)`, never an assumed constant. (A fixed-frequency noise
seed like the terrain-march jitter lookup, `gl_FragCoord.xy * (1.0/128.0)`, is fine as-is — no
total-resolution assumption baked in, not a normalized UV.) The point shaders (`sat_point.frag`/
`star_point.frag`) never render scaled, so they keep a plain `screenSizePx` (= `ctx.swapExtent`) in
`PointDrawPC`.

`screenSizePx` originally lived in `SatDrawPC` (which grew 132→144 for it); it moved to the
CloudParams UBO / `PointDrawPC` in the 128-byte push-constant split. `buildSkyDrawPC(ctx)` /
`buildPointDrawPC(ctx)` fill the two draw push constants; the CloudParams UBO fill in
`recordCompute()` sets `skyScreenW`/`H` from `renderScale`.

See `TERRAIN_PLAN.md` session 29 log for the full design writeup and the bug's root-cause
narrative.

**Its value has shrunk since the pipeline unification.** `scene_depth.comp` and the cloud targets
are fixed at half the SWAP extent and do not scale, so at 1920x1009 dropping to 50% removes only
~1.46 Mpx of sky-pass work while ~0.96 Mpx of compute stays. With the sky pass now much cheaper
(beam occlusion gone, layers clamped at march time), 100% and 50% measure comparably in practice —
and 100% additionally gets exact hardware-depth occlusion for satellites/stars. Prefer 100%. If
render scale needs to matter again, the fix is making those two compute passes scale with it.

---

## Subsystem: Weak-Hardware Sky Tiers (Potato / SKY_LITE)

`sat_sky.frag` is ~2900 lines. On a **2015 MacBook Pro (AMD Radeon R9 M370X / GCN 1.0, macOS 12,
MoltenVK)** it compiles to one Metal fragment function whose register pressure collapses wavefront
occupancy — measured **~490 ms/frame, the entire frame**. This is not tunable by any quality slider
or `debugDisableMask` bit: those skip *execution*, not compiled *size*, and the constraint is peak
VGPR count + total texture-fetch latency that must be hidden, not instruction count. See
`SKY_OPTIMIZATION_PLAN.md` for the full investigation and `.gputrace` capture workflow
(`tools/make_capture_bundle.sh` — needs full Xcode to read).

Two stand-in fragment shaders, both bound through **`skyBgPipeLayout` / `skyDescSet` unchanged**
(each declares only the bindings it reads) and selected in `recordDraw()` Pass 1 by
`debugDisableMask` bit:

| Tier | Bit | Pipeline | Shader | Notes |
|---|---|---|---|---|
| **Potato** | `262144` | `skyBgMinimalPipeline` | `shaders/sat_sky_minimal.frag` (own file, ~370 lines) | closed-form analytic atmosphere (Kasten-Young airmass, one 32-tap arithmetic loop — no raymarch), day/night + city-detail textures, one flat drifting cloud shell w/ terminator lighting, cheap ocean (1 noise-tap slope + Fresnel + Blinn glint), textured moon, verbatim `lensFlare()`. **~60 FPS.** No Milky Way / volumetric clouds / aurora / airglow / real ocean waves — those don't fit the GCN1 occupancy ceiling (~31–32 KB SPV; the Milky Way's `atan2`/`asin` + panorama fetch is the specific thing that broke it). |
| **SKY_LITE** | `524288` | `skyBgLitePipeline` | `sat_sky.frag` **recompiled with `-DSKY_LITE`** → `sat_sky_lite.frag.spv` (2nd `add_custom_command` in CMakeLists) | `#ifdef SKY_LITE` cuts inside the real shader: Milky Way block, 64-bin satellite sky-glow loop, cloud layer loop `3→0` becomes `1→0`, the 3×3 `cloudTargetA/B` rgb blur → single tap, aurora surface glow (`auroraGlowAt` on terrain+ocean), zenith-ambient `N_ZT` 4→2, and the per-atmosphere-step **green/sodium airglow** (two `warpPerlin3` masks/step — the dominant cost) — city-glow upwelling KEPT but only on the first 3 march steps. **2 FPS → ~55 FPS** on the target. |

262144 wins if both bits are set. The always-on `Log::line` breadcrumb in `recordDraw` reports
`MINIMAL` / `LITE (SKY_LITE)` / `FULL sat_sky.frag` on any change (into `satlight_log.txt`) — this
is also the instrument for `potato-mode-intermittent-slow-start` (see memory).

**Preset wiring** (`applyGraphicsPreset`, `SatelliteSimUI.cpp`): **Potato** sets `kBitMinimalSky`
plus every compute-side knockout; **Planetarium** sets `kBitLiteSky` + `kBitCloudMarch |
kBitCirrusMarch` (coverage 0 at that tier) + `viewSamplesMin/Max 4/10`. Medium and up keep the full
shader unchanged. `GraphicsPreset::Potato` is enum value 6 (appended after `Custom` to preserve
persisted int indices); `kGraphicsPresetNames` and the preset-row UI list it first.

`skyLowResPipeline` (renderScale < 1.0 prepass) is **not** given a lite/minimal variant — both
those presets force `renderScale = 1.0`, so the prepass never runs for them.

### All-platform changes that came out of this pass (land on `main`, not tier-gated)

- **Sun corona bridge** (`sat_sky.frag` "Sun disc + atmospheric corona"): three stacked
  `pow(cosA, 1800/320/55)` lobes peaking at ~disc brightness, added between the hard disc and the
  `×0.12` wide `corona` Gaussian. Without it the disc dropped straight to the dim corona at its
  edge — a hard brightness step that read post-tonemap as a dark ring / "cutout." Same shape as
  `sat_sky_minimal.frag`'s Potato corona so the tiers match.
- **Moon `squish` dead code removed**: the disabled (`squish = 0`) atmospheric-refraction block
  still ran an `asin` + (below 15° elevation) two `tan` + `radians` to feed the identity
  `dir.z * (1.0 + 0.0)`. Gone; ray/disc intersection uses `dir` directly.
- **Cloud-shadow blur 5×5 → 3×3** (`kShadowBlurSpread = 1.7` keeps the ~radius-2 footprint;
  centre tap reuses the already-sampled `cloudBCenter.a`): −17 texture samples on every ground-hit
  pixel, the largest sample-count cut in the file. If ocean graininess returns, strengthen
  `cloudGroundShadow`'s dither in `cloud_march.comp` rather than widening this back out.
- **Push constants trimmed to the 128-byte `maxPushConstantsSize` floor** (2026-09-07). The same
  old AMD integrated parts that need Potato/Planetarium also report exactly the Vulkan-guaranteed
  minimum `maxPushConstantsSize` of 128, and `SatDrawPC` (176) / `CloudMarchPC` (148) both blew
  past it — the app could not create those pipeline layouts at all. `SatDrawPC` split into a
  128-byte sky core + a new 128-byte `PointDrawPC` for the point pipelines; `CloudMarchPC` trimmed
  to 128. Every per-frame-uniform tail field (`debugDisableMask`, `screenSizePx`,
  `skyGlareVisibility`, the `beam*` scalars, `mwSuppressEased`, `showBeamDebugRays`,
  `cloudShadowRangeM`) moved into the CloudParams frame UBO — see the **SatDrawPC** / **PointDrawPC**
  entries under "Subsystem: GPU Orbital Pipeline → Push constants" and the "Push-constant relief"
  block in `GpuCloudParams`. **Consequence for future work:** `SatOrbitPC` and `SatFlarePC` are
  already at exactly 128 — any new push-constant field on any of these structs must go in a UBO,
  not the push constant. Prefer the CloudParams UBO where the consumer already binds it.

---

## Fixed Simulation State

**Start epoch**: UTC 2036-06-21 00:00:00 → J2000 seconds = 1,150,891,200 (stored split: day 13,320 + 43,200 s)
**Observer**: 67°S 67°W → ECEF `obsDir = {0.1527, -0.3596, -0.9205}`, facing north — this is only the
compiled-in fallback used before `loadSettings()`/the intro cinematic override it; see below for
where the observer actually starts in practice.
**Moon phase offset**: `kMoonPhaseOffsetRad = 3.916 rad` → originally calibrated for 2026-03-30; moon phase at new epoch will differ

---

## Subsystem: Intro Cinematic (UC3)

`showIntro` drives `updateIntroCinematic()` (`SatelliteSim.cpp`, called from `recordCompute()` in
place of normal WASD/zoom input) through a fixed beat sheet, `kIntroKeyframes[]`
(`SatelliteSim.h`). `buildIntroOverlay()` (`SatelliteSimUI.cpp`) only draws the caption/skip-hint
text on top — the camera path itself *is* the cinematic. Whether it plays on launch is a real
persisted toggle, `playIntroOnStartup` (Display tab "Play intro on startup") — see the persistence
note near the bottom of this section; it is no longer forced on every launch.

**Window boots maximized** (`App::initWindow`, `glfwWindowHint(GLFW_MAXIMIZED, ...)` before
`glfwCreateWindow`) — windowed, not exclusive fullscreen, but full monitor work-area size by
default rather than the fixed `WIN_W`x`WIN_H` (1280x720). A small window the player has to
manually enlarge undercut the cinematic's impact and invited resizing instead of watching it.
`WIN_W`/`WIN_H` are still passed as the restore size for whenever the player un-maximizes later.

**Fixed vantage, locked every playback.** The intro always opens from `kIntroObserverLatDeg/LonDeg`
+ `kIntroStartAzDeg/ElDeg/FovDeg` (`SatelliteSim.h`) — the California coast at twilight facing the
SpaceX AI-datacenter satellites and the Reflect Orbital mirrors aimed at the nearby Topaz solar
farm — not whatever the player's current/saved observer position happens to be.
`updateIntroCinematic`'s one-time init block (`!introBasisValid`) force-sets
`obsDir`/`obsLatDeg`/`obsLonDeg`/`obsHeightOffset`/`camera.azDeg/elDeg/fovYDeg` from these
constants before computing the East/North tangent basis the rest of playback rides on. The
Display tab's "Replay Intro" button resets `introBasisValid = false`, so a replay re-locks to the
same spot regardless of where the player has since wandered off to. Also forces
`timeScaleIdx = 0` / `timePaused = false` / `timeDir = 1.0f` here — the beat sheet is tuned at 1x
and a replay runs on live state, not a fresh boot, so it can't rely on `loadSettings()` alone.

**`camera.azDeg` vs `obsFacing` — read this before touching keyframe azimuth.** `camera.azDeg` is
what `SkyCamera::viewMatrix()` actually renders with; `obsFacing` is only the ground-movement
tangent, and outside the intro `camera.azDeg` is *derived from* `obsFacing` every frame in
`buildUI` (search "Derive camera.azDeg from obsFacing"). That derivation is skipped entirely while
`showIntro` is true (same gate as the RMB-look block), so `updateIntroCinematic` must set BOTH
every frame — it used to set only `obsFacing`, which meant the rendered view never actually panned
in azimuth during playback and then snapped to match `obsFacing` on the first post-intro frame.
Fixed by computing the mixed azimuth once and assigning it to both.

**Beat sheet** (`kIntroKeyframes[]`, timings synced to `songbeat` = 7.61s, a music-tempo unit —
hand-tuned by the user, not derived): `kIntroYearIndex` (0) is a "2036" title card, bottom-anchored
like every other caption but at a larger font size (not centered — that was tried and reverted, it
read as inconsistent with the rest of the captions); `kIntroHintRevealIndex` (1) is the first
narrative line and also where the bottom-right skip hint first appears (not from frame 0 — see
below); the middle beats hold at ground level then pull to LEO; `kIntroTitleIndex` (6) is the
arrival "SAT LIGHT SIM" reveal (`kIntroBenchEndT` marks this as the benchmark cutoff);
`kIntroControlsIndex` (7) is the WASD/Q-E controls hint, Q/E text generated at render time from
live keybindings (`introControlsTextBuf`).

**Camera path is a Catmull-Rom/cubic-Hermite spline, not per-segment smoothstep.** The original
implementation eased in/out (smoothstep) independently within each keyframe segment, which gives
zero velocity at *every* waypoint, not just the first and last — the camera visibly decelerated to
a stop and re-accelerated at every beat boundary, reading as a stutter rather than one continuous
move. `updateIntroCinematic`'s local `hermite(field)` lambda instead estimates a time-weighted
tangent at each interior key from its two neighbors and blends with cubic Hermite basis functions,
so velocity carries through a waypoint instead of resetting there. Endpoint tangents fall back to
the one-sided neighbor difference, which happens to already be ~0 for this specific beat sheet
(the first two beats and the final hold beats each share identical values), so the start and end
still ease naturally with no special-cased boundary velocity.

**Controls go live mid-cinematic, at `kIntroControlsIndex`.** It read as broken to show "WASD to
move" / "Q / E to raise/lower height" on screen while those keys visibly did nothing — control now
unlocks the moment that caption is showing, not at the very end. Mechanism: `updateIntroCinematic`
checks `introCaptionIndex >= kIntroControlsIndex` and stops forcing
`obsHeightOffset`/`camera.azDeg/elDeg/fovYDeg`/`obsFacing` from that point on (safe with no
discontinuity, since the camera has already arrived at its final framing by then); `recordCompute`
separately runs the real WASD/Q-E/zoom movement block whenever `!showIntro || introCaptionIndex >=
kIntroControlsIndex`, replacing the old plain `else` so both can be true at once. `buildUI`'s
mouse-look block (RMB drag, cinematic pan, gamepad look, and its own `camera.azDeg`-from-
`obsFacing` derivation) uses the identical condition, so free-look and WASD are both live from the
same beat — the rest of the HUD (settings/view-controls windows, satellite picking) still waits
for the intro to actually end (`finishIntro`), since only WASD/Q-E/look are what the on-screen text
promises. Caution: any values kIntroKeyframes gives the final hold beat(s) *after*
`kIntroControlsIndex` for alt/az/el/fov are dead data for camera purposes once this fires — only
that beat's own `.t` still matters, as the auto-handoff time.

**Dismissal is a single defined key, not "any key."** `onKey()` only calls `finishIntro(true)` for
literal `GLFW_KEY_SPACE` (independent of whatever `Pause/Resume` is currently rebound to);
`pollGamepad()` only responds to `GLFW_GAMEPAD_BUTTON_START`. Mouse clicks do NOT skip —
`buildIntroOverlay` still covers the screen with a mouse-capture rect so a click doesn't leak
through to satellite picking, but it no longer calls `finishIntro`. This was a real usability
issue in the previous "any key/click" version: almost no one saw the cinematic play out, because
touching the keyboard or clicking into the window to focus it immediately skipped it.

**The intro hides the entire normal HUD**, not just its own overlay — `buildUI()` checks
`showIntro` before icon loading/scroll-zoom/satellite-picking and before the `uiVisible` check, so
none of the left/right HUD panels, settings/view-controls windows, or scene-interaction input run
while the cinematic owns the camera, regardless of whether the player has the rest of the UI
toggled on or off.

**`timeScaleIdx`'s compiled-in default was `1` ("10x"), not `0` ("1x")** — a real bug, now fixed.
`loadSettings()` overrides it from `settings.json`'s `time.scale_idx` when present, which is why it
looked "sometimes" wrong: a genuine first run (or any state where that key was never written)
booted at 10x. The intro's one-time init above also force-sets it defensively, since a replay runs
on live state that loadSettings() never touches again.

**Persistence: `playIntroOnStartup`, not a raw `showIntro` save.** `showIntro` is runtime
play-state (flips false the moment the cinematic ends), so persisting it directly conflated "did
today's playthrough finish" with "should it play again next launch." `playIntroOnStartup` (Display
tab checkbox, next to "Replay Intro") is the actual user preference, saved every close and applied
to `showIntro` at load time: `playIntroOnStartup = d.value("play_intro_on_startup", false)` inside
the `"display"` block. Absent-key default is `false` there (not the compiled-in `true`) so a
player upgrading from a build that predates this key doesn't suddenly get a cinematic that didn't
exist in their version — a genuine first run (no settings.json at all) never reaches that line, so
it still keeps the compiled-in `true` default from the "no file" early-return branch. Disabling it
does not need to separately handle "resume at last known location" — `obsDir`/`camera` are always
restored from settings.json's own `"observer"`/`"camera"` blocks regardless of `showIntro`, so
turning the intro off just resumes wherever the player last was.

---

## Active Development: Earth / Terrain Rendering

See `TERRAIN_PLAN.md` in the project root for the full step checklist and session log.
Read it at the start of any terrain-related session before making changes.

**Current state (as of 2026-07-17, session 29):**
- Steps 1, 2, 3, 4, 5, 5b, 6, 8 complete; C1–C8, C13, C15, C16 complete
- Phase E in progress (C13–C16: Cirrus rework, Anvil, Airglow, Aurora), sequenced ahead of
  C9/C11/C12. Full spec in `TERRAIN_PLAN.md`.
- **Cirrus (C13):** own standalone `cirrusMarch()` in `sat_sky.frag`, NOT a second `cloudMarch`
  call — `cloudMarch` already merges `layers[0]`/`[1]` (2-11km) into one low/mid shell, so there
  was no separate volumetric band to extend. Thin shell (700m) at `layers[1].shellAltM`,
  anisotropic streaks via a fixed global wind-axis compression (`cloud.cirrusWindAngle`/
  `cirrusStretch`, repurposed from the UBO's former `pad1`/`pad2`) — NOT a per-sample tangent
  decomposition (that's a no-op: the noise argument is purely radial from its own tangent frame).
  Sun-only lighting matching `evalCloudLayer`'s formula so it colour-matches the flat paste it
  crossfades against. See `TERRAIN_PLAN.md` session 21 log for the full writeup.
- **Airglow (C15):** three emissive bands (green 96km, sodium 90km, red 275km) gated by per-sample
  geographic day/night, not observer's. Green+sodium accumulate inside the existing `N_VIEW`
  atmosphere loop for free (their peaks fall inside its ~100km ceiling) and stay in `sat_sky.frag`.
  Red originally needed its own small 16-step supplemental march out to `R_EARTH+500km` since its
  peak/half-width sit well past that ceiling — extending the primary loop's far bound for one band
  would have coarsened the near-surface Rayleigh/Mie sampling everything else depends on; that red
  march itself moved to `cloud_march.comp` in session 29 (half-res, alongside aurora — see below),
  though the underlying peak/width/color constants and the day/night gating logic are unchanged.
  `CloudParams` UBO grew 176→192 bytes (all 3 pad slots were already consumed by C13) for 4 new gain
  fields (`airglowGain` master + per-band); peak altitude/width/color are hardcoded physical
  constants, not UBO fields. Reuses the analytic `warpPerlin3` noise (same one `cloudWarpOffset`
  uses) for horizontal patchiness — no new texture/binding. See `TERRAIN_PLAN.md` session 22 log for
  the original writeup, session 29 log for the move.
- **Raymarch-from-inside-a-volume fix (session 22):** `raySphere` reformulates `c = dot(ro,ro)-r*r`
  as `c = (|ro|-r)*(|ro|+r)` — the naive form catastrophically cancels at R_EARTH scale (~1e13
  float32 magnitude) exactly at grazing/near-tangent rays, i.e. every horizon, across all 29 call
  sites. Also: any shell march must classify the observer as below/inside/above the shell (keyed on
  `obsEffH`) rather than assuming a fixed forward root — `cloudMarch`/`cirrusMarch` already did this;
  the new airglow red-band march didn't, and broke (bright zenith band + horizon seams) once the
  observer flew (via the uncapped "Raise Elevation"/Q control) into or above the shell and looked
  outward, where the "always below" forward root goes negative. Fixed to match the established
  pattern. Any future shell march (Aurora/C16) needs this from the start — see
  `TERRAIN_PLAN.md` session 22 log.
- **Cloud march perf (session 22 follow-ups):** (1) `cloudMarch`'s C8 altitude-stratified stepping
  uncapped the real 3D step length for oblique rays (up to 50× the vertical step) — this is what
  made "clouds viewed from the side" undersample and band; now capped at a fixed `kCloudMaxStepM =
  250` (meters, not a multiple of the vertical step — see comment on why that matters at high
  `marchSteps`). (2) `cloud.lightSteps` (the "Light steps" slider) was declared but never read
  anywhere — the sun self-shadow cone hardcoded `N_CONE = 6` regardless; now wired up, and it's the
  dominant per-inCloud-sample cost. (3) That shadow cone is now also distance-gated
  (`cloud.shadowMaxDistM`, camera-relative) so far/orbital clouds skip it almost entirely, which
  paid for raising the render-distance cap (`cloud.maxRenderDistM`, replaces a hardcoded 80km) to
  reduce horizon pop-in. `CloudParams` grew again, 192→208 bytes. See `TERRAIN_PLAN.md` session 22
  log (multiple entries) for the full history.
- **Half-resolution cloud compute pass (session 23) — `cloudMarch()`/`cirrusMarch()` no longer live
  in `sat_sky.frag`.** They moved to `shaders/cloud_march.comp`, a new compute shader dispatched
  once per frame in `recordCompute()` at half `ctx.swapExtent` (1/4 the pixels). `sat_sky.frag`
  samples two `RGBA16_SFLOAT` targets (bindings 10/11) instead of marching per full-res pixel.
  Restructured (not just relocated): the compute-shader copies return an `(A, B)` affine-composite
  pair instead of mutating `color` in place, so cirrus-then-cloud combine into one exact
  `(A_total, B_total)` algebraically. No terrain data in the compute shader — `sat_sky.frag` does a
  post-hoc terrain-occlusion suppression using its own accurate `tSurface` against the sampled
  occlusion distance (exact for full occlusion, not for mid-shell partial truncation — accepted
  approximation). New `CloudMarchPC` push constant, new `cloudMarchDescSet` (7 bindings), 2 new
  `skyDescSet` bindings. See `TERRAIN_PLAN.md` session 23 log for the full design (why two targets,
  the barrier sequencing, the `init()` ordering constraints — several real mistakes were caught and
  fixed during design review before any code was written, don't repeat them).
- **Terrain-bleed bug fix + `cloudShadowFactor()` removed (session 23 follow-ups):** the terrain-
  suppression gate above initially used the opacity-gated `tCloudOcclude` (≥90% opaque only, meant
  for satellite depth), so most non-solid cloud rendered through terrain regardless of depth — a
  real bug, not the documented approximation. Fixed with a second, always-valid entry distance
  (`tEnterOut` from both march functions, combined via `min()`) stored in Target B's alpha;
  `cloudBlock` (displaced from that slot) is now derived from Target B's RGB instead. Separately:
  Release-build FPS testing showed the half-res compute move hadn't changed SURFACE performance at
  all (unchanged across the whole session, through every cloud-march fix) — `coverage=0` testing
  confirmed clouds were still the dominant surface cost anyway, pointing at `cloudShadowFactor()`
  (full-res cloud-shadow-on-terrain/ocean, untouched all session) as the real bottleneck. Removed
  outright per user decision (cloud shadowing on terrain isn't in use) rather than optimized — its
  `CloudParams` UBO slot reverted to `pad0`. See `TERRAIN_PLAN.md` session 23 log for both fixes.
- **Terrain/ocean/atmosphere perf follow-up (session 24):** fixed a real regression — the terrain
  march's altitude-scaled step count was `mix(320.0, 320.0, ...)` (a no-op, always paid the
  LEO-tuned 320-step budget at ground level too), restored to `mix(196.0, 320.0, ...)` matching its
  own comment. Also made 5 previously-hardcoded quality constants UBO-tunable (new sliders, all
  defaulting to prior fixed behavior): `N_VIEW`/`N_LIGHT` (main atmosphere loop, `cloud.viewSamples`/
  `lightSamples`) and ocean's `seaMap`/`seaMapDetail` octave counts + reflection sample count
  (`cloud.oceanSeaOctaves`/`oceanDetailOctaves`/`oceanReflSamples`). `CloudParams` grew 208→224
  bytes. **Session 29 update:** real GPU-timestamp profiling superseded the guess that `N_VIEW`/
  `N_LIGHT` (and a transmittance LUT) were the lead cost suspects — `optDepth`'s isolated cost was
  consistently near-zero; terrain's step-count formula (see below) and aurora (see its own entry)
  were the real dominant costs. See `TERRAIN_PLAN.md` session 24 log for the original follow-up,
  session 29 log for the profiling toolkit and the corrected picture, and "Subsystem: GPU
  Performance Profiling" below for how to re-run this kind of investigation.
- `SatDrawPC`, `PointDrawPC` and `CloudMarchPC` are all exactly **128 bytes** — the
  `maxPushConstantsSize` floor. `debugDisableMask` and the other per-frame tail fields ride in the
  CloudParams UBO now (`cloud.dbgDisableMask` etc.); see the **SatDrawPC** / **PointDrawPC** entries
  under "Subsystem: GPU Orbital Pipeline → Push constants" and the "Push-constant relief" block in
  `GpuCloudParams`. Both point pipeline layouts (`drawPipeLayout`, `starPipeLayout`) use
  `sizeof(PointDrawPC)`; `skyBgPipeLayout` uses `sizeof(SatDrawPC)`.
- Sky descriptor set has 22 bindings (0-21): GlowBuf, noise, moon, earthDay, earthNight, earthElev, earthSpec, earthClouds, cloudNoiseTex (sampler3D), CloudParams UBO, half-res cloud march targets A/B, lightDomeBuf, milkyWayTex, cityDayDetail, cityNightDetail, auroraNoiseTex (sampler3D), reflectBeamsBuf, beamGlowDomeBuf, sceneDepthTex, oceanGlintBuf, groundBeamsBuf. Binding 18 was `cloudShadowTex` until that pass was deleted; 19/20 were compacted down into 18/19 rather than leaving a hole, since the C++ side fills its binding array contiguously. groundBeamsBuf (21, perf follow-up) is the CPU-compacted, observer-range-culled subset of reflectBeamsBuf that sat_sky.frag's ground-spot loop reads instead of the raw (up to 2048-entry) buffer — see GpuGroundBeams in SatelliteSim.h. **As of 2026-08-10 its entries are `GpuGroundBeam` (32 bytes), not raw `GpuReflectBeam`** — a pre-solved record, see "Beam ground-spot CPU hoist" below
- GPU-side observer ground height lookup added; CPU observer height also corrected (see elevation encoding below)
- `sat_sky.frag` ground path: terrain march step count is path-length-adaptive as of session 29
  (`kN` scales with this ray's own `tExit`, clamped to a user-tuned [64,164] range — the old
  altitude-only formula gave a grazing/horizon ray and a steep ray from the same observer altitude
  identical step budgets regardless of how much further the grazing ray actually travels, a real
  bug contributing to reported terrain jitter, not just under-tuned; see `TERRAIN_PLAN.md` session
  29 log) + 12-step binary search; terrain hits use gradient-computed normals; sea-level sphere
  fallback; satellites/stars depth-tested against terrain (gl_FragDepth: close terrain → [0, 0.5),
  sky → 1.0)
- Ocean wave material: specular map (binding 6) gates UBO-tunable-octave noise wave normals +
  Blinn-Phong sun glint (exp=300) + Schlick Fresnel on sea-level sphere hits
- **Volumetric clouds (C7+C8):** shell march with full C8 lighting:
  - `cloudDensity` takes two UVW args — `uvwPresence` (Z=posZ) for Perlin R threshold,
    `uvwDetail` (Z=hNorm×kVertTiles) for Worley erosion. Keeps cloud-existence horizontal only.
  - **Altitude-stratified stepping:** `stepLen = (shellThick/N) / max(abs(dir.z), 0.02)`.
    Equal altitude per step regardless of ray angle — no oblique-angle slab artifacts.
  - **Spectral sun color:** `sunColorCloud` from `optDepth` at shell entry → orange/red at sunset.
    Night-side gated by Earth shadow test. Replaces old gray `vec3(1.0)` lighting.
  - **Night darkening:** ambient transitions from blue day dome to near-zero at night using
    per-sample `dot(normalize(pECEF), sunDirECEF)` geographic terminator check.
  - **City upwelling:** `earthNightTex` at mip 3 contributes warm orange into cloud bases at night.
- **Aurora (C16):** visual design centered on the geomagnetic pole (`kGeomagPoleECEF`, antipodal
  dipole model covers both hemispheres with one constant), colatitude oval band (`auroraOvalMask`)
  with ripple-warped centerline, anisotropic curtain-fold noise (`auroraCurtainNoise` — high freq
  along azimuth for many separate folds, low freq along colatitude so each fold reads as a long
  streak, not a blob). Day-gated PER-SAMPLE on that sample's own geographic day/night (mirrors
  airglowRed's `rDayness`/`rNight`). `CloudParams` grew 288→304 bytes for `stormStrength` +
  `auroraGain` (mirrored in `cloud_march.comp`, which hand-duplicates this UBO layout, and in
  `GpuCloudParams`). `kAuroraScale = 0.000001`, same order as `kAirglowScale`. Visual
  design/tuning **DONE, closed 2026-07-16 (session 28 follow-up #22) after 22 follow-up rounds**
  — brightness/exposure, per-sample day/night gating, fold axis/unit calibration, cloud occlusion
  depth-ordering, terrain/ocean/cloud ambient lighting + ocean reflection glint, atmospheric
  extinction, light-pollution/moonlight suppression, a sigmoid-based airglow blend at both shell
  edges, per-column elevation variation, and organic domain-warped shimmer evolution. See
  `TERRAIN_PLAN.md` session 28 log (all 22 follow-ups) for the full design and bug history.
  **Architecture changed significantly in session 29 for performance** (a ~40fps-swing cost down
  to a minor one) — read this before touching aurora code:
  - The sky curtain march itself (whole-ray bounding pre-check → adaptive-step march →
    light-pollution/moonlight suppression → extinction) moved OUT of `sat_sky.frag` into
    `cloud_march.comp`'s `auroraMarchCS`, running at half resolution alongside clouds/cirrus.
    Its result folds additively into the same `B_total` channel clouds already write — no new
    sampling code needed in `sat_sky.frag`.
  - `sat_sky.frag` keeps its own copies of `auroraFrame`/`auroraCoverage`/`auroraOvalMask`/
    `auroraCurtainNoise`/`auroraSampleAt` — still used by `auroraGlowAt` (terrain/ocean ambient
    lighting) and the ocean sky-reflection's own aurora sample, both legitimately full-resolution.
    `cloud_march.comp` has near-verbatim duplicates of the same functions for its own march — keep
    both copies in sync, same standing rule as the cloud/cirrus code this file already duplicates.
  - Most of the curtain-fold and column-window noise is now baked into a texture
    (`aurora_noise.comp`, 1024×16×256 RGBA8, `createAuroraNoisePipeline` — same one-shot-bake-at-
    init pattern as `cloud_noise.comp`) instead of computed live via `warpPerlin3` every sample —
    the direct fix for "aurora is much more expensive than clouds despite looking simpler," since
    clouds' noise was already baked in a prior session and aurora's never was. New descriptor
    binding 16 (`auroraNoiseTex`, sampler3D) on `sat_sky.frag`'s set, binding 8 on
    `cloud_march.comp`'s own set (separate descriptor sets, same underlying image/sampler).
  - Real behavior change, not just perf: aurora now respects terrain occlusion the same way clouds
    do (folded into the same terrain-gated composite branch) — previously it ignored terrain
    entirely. Judged more physically correct and accepted without further tuning.
  - `kAuroraStepsMin`/`kAuroraStepsMax` are 4/64 (was 24/160) — user-tuned in-app; the min rarely
    binds (a straight-down path through the ~200km shell is already ~14 steps at the target
    resolution). See `TERRAIN_PLAN.md` session 29 log for the full four-round history (step cap →
    pre-filters → noise bake → resolution move) and the specific approximations each step accepted.
- **Next:** C14 (Anvil) remains not started — deferred repeatedly in favor of C15/C16 per the
  2026-07-12 session — and can be picked up whenever; it has no dependency on C15/C16. Otherwise
  Phase E is complete (C13, C15, C16 done); C9/C11/C12 and noise-repetition cleanup are next in
  line per the 2026-07-12 planning session's priority order. Resolution scaling shipped in session
  29 (background-only, satellites/stars/UI always native res) — see "Subsystem: Resolution
  Scaling" below.

### Elevation texture encoding — READ THIS BEFORE TOUCHING TERRAIN CODE

**File:** `assets/textures/earth_elevation.png` (R8_UNORM, 21600×10800, land-only DEM)

**This is NOT ETOPO1 and has NO bathymetry / below-sea-level data.** Do not assume
pixel=0 means sea level — it does not. The actual encoding is:

| Pixel value | Meaning |
|-------------|---------|
| 0–14/255    | Compression noise in ocean regions — treat as sea level |
| **15/255**  | **Ocean / sea level baseline** |
| 16–255/255  | Land elevation, linearly scaled above sea level |
| 255/255     | ≈ 8848 m (Everest) |

**Correct formula used in `sat_sky.frag` and `SatelliteSim.cpp`:**
```
kElevOffset = 15.0/255.0 * kElevRange          // ≈ 529 m baseline
terrainH    = max(0, pixel * kElevRange − kElevOffset)
```
Failing to subtract `kElevOffset` makes every coastline on Earth appear as a ~530 m vertical
cliff because sea-level land reads as 529 m above the ocean sphere. This bug has been
introduced and re-introduced across multiple sessions. The ocean sphere sits at exactly
`R_EARTH`; the terrain height formula must produce 0 m for ocean-baseline pixels.
