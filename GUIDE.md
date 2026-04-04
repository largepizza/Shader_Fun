# SatelliteSim Developer Guide

A subsystem-by-subsystem reference for working on the satellite constellation visualizer.
Each section explains **what exists**, **why it works that way**, and **what to do** when extending it.

---

## Table of Contents

1. [Project orientation](#1-project-orientation)
2. [Adding controls](#2-adding-controls)
3. [Adding satellite types](#3-adding-satellite-types)
4. [Adding constellations](#4-adding-constellations)
5. [Adding UI elements](#5-adding-ui-elements)
6. [Orbital mechanics reference](#6-orbital-mechanics-reference)
7. [Photometry and shader calibration](#7-photometry-and-shader-calibration)
8. [TargetedReflector / mirror system](#8-targetedreflector--mirror-system)
9. [Vulkan pipeline reference](#9-vulkan-pipeline-reference)

---

## 1. Project orientation

The simulation runs in a fixed frame loop owned by `App`. Each frame:

```
buildUI(dt, ui)       ← Clay layout; camera look; mouse capture
recordCompute(cmd)    ← WASD nav; simTime advance; updatePositions(); GPU compute dispatch
recordDraw(cmd)       ← sky pass → satellite points → star catalog
```

Everything interesting lives in two files:

- `src/simulations/SatelliteSim.h` — data structures, enums, class declaration
- `src/simulations/SatelliteSim.cpp` — all simulation logic

The GPU does the per-satellite photometry (`sat_flare.comp`) and rendering (`sat_point.vert/frag`).
The CPU does orbital propagation (`updatePositions`), satellite type lookup, and surface normal computation.

**Key invariant**: `updatePositions(simTime)` must run before `initConstellation()` at init time (it populates `sunDirECI` which `alignTerminator` depends on).

---

## 2. Adding controls

Controls are the most tightly specified subsystem. There is a mandatory pipeline — deviating from it means the control won't appear in the Settings window and won't be rebindable.

### The mandatory pipeline

**Step 1**: Add an entry to the `KB` enum in `SatelliteSim.h`, before `KB_COUNT`:
```cpp
enum KB {
    KB_TOGGLE_UI  = 0,
    KB_PAUSE      = 1,
    KB_SLOWER     = 2,
    KB_FASTER     = 3,
    KB_REVERSE    = 4,
    KB_MOVE_BOOST = 5,
    KB_MOVE_FINE  = 6,
    KB_YOUR_KEY   = 7,  // ← add here
    KB_COUNT      = 8,  // ← bump this
};
```

**Step 2**: Add one line to `keybindings` in `SatelliteSim::init()`, in the same position as the enum:
```cpp
keybindings = {
    {"Toggle UI",    GLFW_KEY_TAB,         false, false},  // KB_TOGGLE_UI
    // ... existing entries ...
    {"Your Action",  GLFW_KEY_X,           false, false},  // KB_YOUR_KEY
};
```
Format: `{display name, GLFW_KEY_*, held, false}`. The last `false` is `listening` — always initialize to false.

**Step 3**: Update the assert:
```cpp
static_assert(KB_COUNT == 8, "KB enum and keybindings initializer are out of sync");
```

**Step 4**: Wire the action. Two patterns based on `held`:

**Event key** (`held=false`) — fires once when pressed, in `onKey()`:
```cpp
if (pressed(KB_YOUR_KEY))
    doSomething();
```

**Held key** (`held=true`) — polled every frame in `recordCompute()`:
```cpp
bool yourKey = win && glfwGetKey(win, keybindings[KB_YOUR_KEY].key) == GLFW_PRESS;
if (yourKey) doSomethingContinuous();
```

### What you get for free
- Settings window row with action name + current key + Rebind button
- `keyDisplayName()` display — covers all letters, digits, modifiers (LShift/RShift/LCtrl/RCtrl/LAlt/RAlt), F-keys, arrows, nav cluster, punctuation
- `(hold)` suffix in the settings display for held=true bindings
- Key capture and reassignment via the listening/rebind state machine

### What held vs event means
Movement modifiers (Shift boost, Ctrl fine) are `held=true` because their effect is continuous and they're polled in the movement loop. Toggle actions (pause, reverse, UI toggle) are `held=false` because they should fire once per keypress, not every frame.

---

## 3. Adding satellite types

Satellite types define the physical and optical properties of a class of satellite. They are independent of which constellation they appear in.

### SatelliteType fields
```cpp
struct SatelliteType {
    const char *name;
    glm::vec3 baseColor;    // visual tint (RGB)
    float crossSectionM2;   // reflective area in m²; brightness ∝ sqrt(area/10)
    SurfaceSpec primary;    // dominant reflective surface
    SurfaceSpec secondary;  // optional secondary; set weight=0 to disable
    float diffuse;          // Lambertian floor [0,1]; always-visible scatter
    float mirrorFrac;       // fraction of primary that is near-perfect mirror [0,1]
                            // 0 = no mirror peak; 0.97 = near-perfect (Reflect mirror)
};

struct SurfaceSpec {
    AttitudeMode attitude;  // how the surface normal is oriented each frame
    float specExp;          // Phong exponent (0 = Lambertian diffuse)
    float weight;           // contribution weight relative to primary
};
```

### AttitudeMode quick reference
| Mode | Surface normal | Best for |
|------|----------------|----------|
| `NadirPointing` | toward Earth center | Flat antenna/bus face (Starlink) |
| `SunTracking` | toward sun | Solar panels (most LEO/GEO) |
| `AntiNadir` | away from Earth | Space-facing radiators |
| `Perpendicular` | cross(surfN0, satNadir) | Secondary only — edge-on panels |
| `Tumbling` | spinning random axis | Debris, uncontrolled objects |
| `FlatMirror45` | normalize(sunDir + satNadir) | Flat mirror → reflects straight down |
| `TargetedReflector` | normalize(sunDir + toTarget) | Flat mirror → reflects toward ground target |

### Adding a new type
1. Append to `satTypes` in `initConstellation()`. The new type's index = position in the array.
2. Reference the index via `typeIdx` in any constellation config.
3. No GPU struct changes needed — all fields map to existing `GpuSatInput` members.

### Mirror peak calibration
`mirrorFrac` controls how much of the primary surface contributes the ultra-narrow spike:
```
mirrorPeak = irr0 × pow(dot(refl, satToObs), max(specExp×300, 8000)) × MIRROR_BOOST × mirrorFrac
```
At perfect alignment, approximate peak magnitude:
- `mirrorFrac=0.05, spec=18` (Starlink): effectFlare ≈ 27 → mag ≈ −2.7
- `mirrorFrac=0.97, spec=200` (Reflect mirror): effectFlare >> 1000 → mag ≈ −11

---

## 4. Adding constellations

### ConstellationConfig field order
```cpp
// Walker:
{ name, altM, incl, numPlanes, perPlane, typeIdx, enabled, OrbitDistribution::Walker }
// total = numPlanes × perPlane

// Disk with extra trailing fields:
{ name, altM, incl, numPlanes, perPlane, typeIdx, enabled, OrbitDistribution::Disk,
  altJitterM, raan, alignTerminator, numRings, ringSpacingM }
// total = numPlanes × perPlane, split across numRings rings
// alignTerminator=true: overrides incl+raan from J2000 sunDirECI, precesses at SSO rate
```

### Rules
- `perPlane` is never ignored — total satellites = `numPlanes × perPlane` for all distribution types
- `incl` is ignored when `alignTerminator=true`; inclination comes from `computeSSOInclination(altM)`
- `updatePositions(simTime)` must have run before `initConstellation()` for `alignTerminator` to work
- Total satellite count across all **enabled** constellations must not exceed `MAX_SATELLITES` (200,000)
- `hovConst[]` has room for 10 constellations and `buildUI` loops `ci < 10` — expand both if adding an 11th

### Capacity math
Current enabled total: ~57k sats (Starlink Gen1+Gen2 disabled for performance). With all enabled: ~103k.
`updatePositions()` is O(N) CPU: ~10 ms at 100k sats. At 200k, expect ~20 ms per frame.

---

## 5. Adding UI elements

### Clay layout model
All UI is declared between `Clay_BeginLayout()` and `Clay_EndLayout()` (called by the framework around `buildUI`). You declare what you want; Clay computes positions; UIRenderer renders them.

### Must-dos for every new panel or widget

**Mouse capture**: Call `ui.addMouseCaptureRect(x, y, w, h)` for every visible panel. Without this, clicks on the panel also fire in the 3D scene.

**Hover state**: Clay hover is only valid inside the element body. Store it in a member bool and use that bool for background colors the next frame:
```cpp
// In buildUI (inside CLAY() body):
bool nowHovered = Clay_Hovered();
myHoverBool = nowHovered;  // member — used next frame

// Elsewhere in CLAY() config:
.backgroundColor = myHoverBool ? Pal::btnHover : Pal::btnIdle,
```

**Runtime strings**: `CLAY_STRING(x)` requires a string literal. For dynamic text use a member char array:
```cpp
// In class: char myBuf[32];
snprintf(myBuf, sizeof(myBuf), "%.1f km", someValue);
Clay_String s{false, (int32_t)strlen(myBuf), myBuf};
CLAY_TEXT(s, CLAY_TEXT_CONFIG({...}));
```
Clay stores a raw pointer; the buffer must outlive the layout pass (member variables do, local stack variables don't).

### Color palette
All colors live in `namespace Pal` defined just before `buildUI()`. Add new named colors there rather than using inline `Clay_Color` literals. The theme is near-black backgrounds, mid-grey text, red accents.

### MSVC designator ordering
MSVC C++20 requires struct designators in declaration order. Key orderings:
- `Clay_LayoutConfig`: `sizing` → `padding` → `childGap` → `childAlignment` → `layoutDirection`
- `Clay_ElementDeclaration`: `layout` → `backgroundColor` → `cornerRadius` → `aspectRatio` → `image` → `floating` → `custom` → `clip` → `border` → `userData`
- `Clay_FloatingElementConfig`: `offset` → `zIndex` → `pointerCaptureMode` → `attachTo`

### Floating panels
Floating elements use `CLAY_ATTACH_TO_ROOT` and a zIndex to layer above the scene. Example pattern:
```cpp
CLAY(CLAY_ID("MyPanel"), {
    .floating = {
        .offset = {x, y},
        .zIndex = 10,
        .attachTo = CLAY_ATTACH_TO_ROOT
    }
}) { ... }
```
Never put `.clip` on a floating container that also has `backgroundColor` — scissor fires before rectangle, hiding the background.

---

## 6. Orbital mechanics reference

### Coordinate frames
- **ECI** (Earth-Centered Inertial): J2000 equatorial frame, doesn't rotate with Earth. Satellite positions computed here.
- **ECEF** (Earth-Centered Earth-Fixed): rotates with Earth. Observer ground positions stored here, rotated to ECI each frame by GMST.
- **ENU** (East-North-Up): local observer frame. Derived from `obsDir` (up), `obsFacing` (north), and east = cross(up, north).

### GMST rotation (ECEF → ECI)
```cpp
float gmst = (float)fmod(kOmegaEarth * t, two_pi);
// Point in ECI from ECEF unit vector ef:
vec3 eci = { cosG*ef.x - sinG*ef.y, sinG*ef.x + cosG*ef.y, ef.z };
```
`kOmegaEarth = 7.2921150e-5 rad/s` — Earth's sidereal rotation rate.

### Satellite position (Keplerian)
```cpp
// ECI position = Rz(RAAN) · Rx(incl) · position-in-orbital-plane
ex = cosR*cosU - sinR*sinU*cosI;
ey = sinR*cosU + cosR*sinU*cosI;
ez = sinU * sinI;
satECI = { ex, ey, ez } * R_sat;  // R_sat = kEarthRadius + altM
```
Phase `u = fmod(u0 + meanMotion * t, 2π)`. Use double precision for `fmod` then cast to float — at t≈8×10⁸ s, float precision is only ~96 s.

### SSO precession
Physically correct inclination: `computeSSOInclination(altM)` solves J2 precession formula.
Live RAAN: `raan_j2000 + kSSOPrecRate * t` where `kSSOPrecRate = 2π / (365.25 × 86400)`.
Reference RAAN anchored at J2000 from `sunDirECIAtJ2000()`.

### Observer movement (surface navigation)
`obsDir` and `obsFacing` are 3D unit vectors — no lat/lon arithmetic, no gimbal lock.
Movement: translate `obsDir` along the surface sphere, then parallel-transport `obsFacing`:
```cpp
glm::vec3 newPos = normalize(obsDir + speed*dt*(fwd*obsFacing + right*rightDir));
obsFacing = normalize(obsFacing - dot(obsFacing, newPos) * newPos);
obsDir = newPos;
```

---

## 7. Photometry and shader calibration

### Brightness pipeline (GPU, sat_flare.comp)
```
spec0   = irr0 × pow(dot(refl0, satToObs), specExp0)   ← primary Phong
spec1   = irr1 × pow(dot(refl1, satToObs), specExp1)   ← secondary Phong
mirror  = irr0 × pow(dot(refl0, satToObs), mirrorExp) × MIRROR_BOOST × mirrorFrac
specular = spec0 + mirror + spec1×w1 + diffuse
flare    = specular × litFactor × distFactor × crossSection × BRIGHTNESS_SCALE
effectFlare = flare / (1 + dayBright × DAY_SUPPRESSION)
```

`distFactor = (REF_RANGE / max(range, REF_RANGE))²` — 1/r² falloff, normalized to 500 km.
`litFactor` = soft penumbra via smoothstep over ±1% of Earth radius.
`dayBright = clamp((sunElev + 0.05) / 0.39, 0, 1)²`

### Calibration anchors
| effectFlare | Approx magnitude | Notes |
|-------------|-----------------|-------|
| 0.008 | 6.0 | Naked-eye limit, steady-state LEO sat |
| 0.25 | 3.5 | Notable flare |
| 8.0 | −0.3 | Iridium-class extreme |

### CPU–GPU constant mirror pairs
These must stay in sync between `SatelliteSim.cpp` and `sat_flare.comp`:
- `kBrightnessScale` ↔ `BRIGHTNESS_SCALE` (intentionally different: CPU used for magnitude UI only)
- `kDaySuppression` ↔ `DAY_SUPPRESSION`

The magnitude formula on CPU uses `kBrightnessScale=6.0`; the GPU renders with `BRIGHTNESS_SCALE=4.0`. This is intentional — the magnitude readout is calibrated separately from the visual render.

### Adding a new surface property
If you need a new per-satellite scalar (e.g. albedo, thermal emission):
1. Add a `float` field to `GpuSatInput` at the end (replacing `mirrorFrac`'s padding if possible, or adding a new struct)
2. Update `static_assert(sizeof(GpuSatInput) == 80)` — or accept 96 bytes and update the assert
3. Add the matching field to `SatInput` struct in `sat_flare.comp`
4. Set the field in the `satInputData[i].xxx = ...` block in `updatePositions()`
5. Use it in the shader

---

## 8. TargetedReflector / mirror system

This subsystem lets mirrors aim reflected sunlight at specific ground locations that rotate with Earth.

### Ground targets
`kNumReflectorTargets = 201` unit ECEF vectors, generated once in `initConstellation()`:
- Indices 0–199: random uniform-sphere samples
- Index 200: fixed at observer spawn (67°S 67°W) — guaranteed aim point when that site is dark

### Per-frame update
In `updatePositions()`, before the satellite loop:
1. Rotate all ECEF targets to ECI via GMST
2. Mark each valid: `dot(normalize(targetECI), sunDirECI) < 0` (night side only)

### Per-satellite normal
In `computeNormal(TargetedReflector, ...)`:
1. Find target index where `dot(-satNadir, normalize(targetECI))` is maximum (nearest to satellite's ground track)
2. `toTarget = normalize(targetECI - satECI_abs)`
3. Mirror normal = `normalize(sunDirECI + toTarget)`

**Half-vector proof**: `reflect(-sunDir, normalize(sunDir + T)) = T` for any unit T. So the reflected ray travels exactly toward the target.

Falls back to FlatMirror45 (straight down) when all targets are in daylight.

### Mirror slew rate
`satMirrorNormals[i]` stores the current physical orientation per satellite (persistent between frames).
Each frame: advance toward goal by at most `kMirrorRotRateDegPerSec × simDt` degrees.
`simDt = |dt × kTimeScales[timeScaleIdx]|` — scales with time warp so behavior is consistent at all speeds.
Zero-vector = uninitialized; snaps to target on first frame.

### Switching to FlatMirror45
Change `AttitudeMode::TargetedReflector` to `AttitudeMode::FlatMirror45` in the Reflect Mirror type definition in `initConstellation()`. No other changes needed — the slew system is gated on the attitude mode.

---

## 9. Vulkan pipeline reference

### Render passes per frame
1. **Sky pass** — fullscreen triangle, `skyBgPipeline`, draws atmosphere + sun + moon + sky glow. Reads glow SSBO (binding 0) and noise texture (binding 1).
2. **Satellite points** — `drawPipeline`, additive blending, reads `satVisibleBuf` via `gl_VertexIndex`. One vertex per satellite; `activeSatCount` vertices drawn.
3. **Star catalog** — `starPipeline`, additive blending, reads `starBuf` via `gl_VertexIndex`.

All three passes share `SatDrawPC` push constants (112 bytes): `skyView` mat4, `fovYRad`, `aspect`, `sunDirENU`, `moonDirENU`.

### Compute pass (before render pass)
`sat_flare.comp` — one invocation per satellite, group size 64.
Reads `satInputBuf` (CPU-written every frame, host-visible+coherent).
Writes `satVisibleBuf` (device-local; barrier to vertex stage before draw).

Barrier after dispatch:
```cpp
srcAccess = VK_ACCESS_SHADER_WRITE_BIT,  srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
dstAccess = VK_ACCESS_SHADER_READ_BIT,   dstStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
```
Use `VK_ACCESS_SHADER_READ_BIT`, **not** `VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT` — the vertex shader reads the SSBO, not a vertex buffer.

### onResize
Must recreate: `skyBgPipeline`, `drawPipeline`, `starPipeline` (viewport baked in).
Must NOT recreate: `compPipeline` (compute is viewport-independent).

### Descriptor sets summary
| Set | Bindings | Used by |
|-----|----------|---------|
| `descSet` | 0=satInputBuf, 1=satVisibleBuf | compute + draw |
| `skyDescSet` | 0=glowBuf SSBO, 1=noise texture | sky fragment |
| `starDescSet` | 0=starBuf | star vertex |
