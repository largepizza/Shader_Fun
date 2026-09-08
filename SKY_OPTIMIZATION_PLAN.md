# SKY_OPTIMIZATION_PLAN.md

Optimising `shaders/sat_sky.frag` for weak hardware — starting from what the macOS/MoltenVK
Potato work (`macOS-potato` branch) established, then carving a Planetarium-tier variant
**subtractively** from the full shader instead of building one up.

Read this before touching `sat_sky.frag` for performance. Companion to `TERRAIN_PLAN.md`.
Architecture summary lives in `CLAUDE.md` → "Subsystem: Weak-Hardware Sky Tiers".

---

## STATUS: complete (2026-09-07). Planetarium **2 → ~55 FPS** on the 2015 AMD Mac.

**Shipped:**
- `sat_sky_lite.frag.spv` (`sat_sky.frag -DSKY_LITE`), `skyBgLitePipeline`, bit `524288`, wired to
  the Planetarium preset. Cuts A/B/F + cloud rgb blur→1tap + aurora surface glow + zenith `N_ZT`
  4→2 + per-step green/sodium airglow (the dominant cost — two `warpPerlin3`/atmosphere-step).
  City-glow upwelling kept, limited to the first 3 march steps.
- Planetarium preset: `+kBitLiteSky +kBitCloudMarch +kBitCirrusMarch`, `viewSamplesMin/Max 4/10`.
- **All-platform (main):** sun corona bridge, moon `squish` dead-code removal, cloud-shadow blur
  5×5 → 3×3.

**Deliberately NOT done** (diminishing returns at ~55 FPS): cut E (ocean → Potato model — already
`dist`/`altFade`-gated so only a low+close observer pays), the `optDepth`-inside-the-atmosphere-
march replacement, a Low-tier `kBitLiteSky` path for mid-range GPUs (open follow-up if a
mid-range GPU ever needs it — the machinery is all there, just add the bit to the `Low` preset).

**Key lesson:** the cost was **texture-fetch latency + transcendental-heavy per-step work in loops**
(the airglow `warpPerlin3` masks, the per-step `earthNightTex` fetch, the Milky Way `atan2`/`asin`
+ panorama) — *not* raymarch step counts or SPV byte size. SPV size is a poor proxy: builds failed
at 31.6 KB and held at 41 KB.

---

## Context — what Potato proved

Target hardware: 2015 MacBook Pro, **AMD Radeon R9 M370X (GCN 1.0)**, macOS 12.7.6, MoltenVK 1.2.11.

- Full `sat_sky.frag` = **~490 ms/frame** there — the whole frame. Established by bisection
  (`SATLIGHTSIM_SKIP_SKYBG=1` → 9 ms). See `memory/satsky-frag-too-big-for-moltenvk.md`.
- The binding constraint is **fragment-shader register pressure → wavefront occupancy**, not SPV
  byte size. SPV size only loosely tracks it: the Milky Way panorama knocked Potato to 2 FPS at
  31.6 KB, yet later builds hold at 41.7 KB. `atan`/`asin` and **texture fetches** cost far more
  than their byte count (long latency that needs more concurrent wavefronts to hide); plain
  sequential `pow`/`dot`/`mix` cost almost nothing in register pressure.
- **What DOES run at 60 FPS on this GPU** (`shaders/sat_sky_minimal.frag`, Potato preset): closed-form
  analytic atmosphere (Kasten-Young airmass, one 32-tap arithmetic loop), day/night + city-detail
  textures, one flat cloud shell with terminator lighting, cheap ocean (1 noise tap → slope) +
  Fresnel + sun glint, textured moon, **the verbatim `lensFlare()`** (ghosts included — it ports
  fine, ~13 `pow` + 1 `atan` + 1 texture).
- We could not read exact occupancy numbers: `.gputrace` viewing and `xcrun metal` both need full
  Xcode (~7 GB, 14.2 last for Monterey); only Command Line Tools are installed. `tools/make_capture_bundle.sh`
  wraps the exe so a capture *can* be taken if Xcode is ever installed. We proceed empirically:
  edit → build → user tests FPS → revert on any cliff (every cliff so far has been an obvious
  drop to ~2 FPS, every revert clean).

## The MSL evidence (from the one capture we got — full `sat_sky.frag`, Planetarium)

Fragment `main()` after inlining: **1071 lines of MSL, 26 texture `.sample()` calls, 15 `for` loops.**
Loops, by MSL line and cost:

| # | Loop | Subsystem | glsl ~line | Cost |
|---|---|---|---|---|
| 1 | `i < kN` (kN 64–164) | terrain raymarch | 1445 | **HIGH** |
| 2 | `j < 12` | terrain binary-search refine | ~1470 | MED |
| 3 | `sy/sx −1..1` (9) | terrain normal / small blur | 1532 | LOW |
| 4 | `i_1 < N_VIEW` | atmosphere single-scatter march | ~1587 | **HIGH** |
| 5 | `gi < 64` | **satellite sky-glow histogram** | ~1620 | **MED-HIGH** |
| 6 | `sy_1/sx_1 −2..2` (**25**) | **cloud-shadow box blur — 25× `cloudTargetB.sample`** | 1969 | **HIGH** |
| 7 | `zi < 4` | sky ambient zenith integration (terrain hit) | ~2145 | LOW |
| 8 | `i_2 < 8` | ocean heightmap trace (`seaMap` secant) | ~2260 | MED |
| 9 | `ri < N_REFL` (6) | ocean sky-reflection march | ~2333 | MED |
| 10 | `ai < 6` | (aurora glow / flare accum) | ~2360 | LOW-MED |
| 11 | `fi < fCount` | ocean mirror-flare glints | ~2449 | LOW (usually 0) |
| 12 | `bi < activeBeamCount` | beam ground-spot loop | ~2470 | VAR |
| 13 | `li 3..0` (4) | `evalCloudLayer` ×4 flat cloud layers | ~2570 | **MED-HIGH** |

26 samples concentrate in: terrain elevation (4, in the normal calc), **cloud targets (5 + the 25
in loop 6)**, ground/city textures (7 around glsl 2458–2498), moon (1), Milky Way (1), sun/flare
occlusion checks (4).

**The single worst offender is loop 6: 25 `cloudTargetB` samples on every ground-hit pixel** for a
cloud-shadow denoise blur.

---

## Phase 0 — pure optimisation, no behaviour change, helps EVERY preset

Do this first. Ship it on `main`, not just a lite variant. Measure each on the Mac (FPS) and, if
possible, on a Windows GPU (should be neutral-to-positive everywhere).

1. **Moon `squish` dead code (glsl ~1804–1822).** `squish` is hardcoded `0.0` (the Bennett-formula
   line is commented out) but the code above still runs two `tan`, a `radians`, and divides to
   compute `Rlo`/`Rhi` that feed it, then `dirR = normalize(vec3(dir.xy, dir.z * (1.0 + squish)))`
   — an identity. Delete the whole `if (elDeg < 15.0)` block and use `dir` directly. Frees a
   branch + 2 transcendentals on every moon-region pixel, all platforms.

2. **Cloud-shadow blur 5×5 → 3×3 or separable (loop 6).** 25 taps → 9 (or 5+5 separable = 10 with a
   second pass, not worth it here). The comment already notes it started at 3×3 and was widened for
   ocean graininess — revisit whether a 3×3 plus a slightly stronger noise dither in
   `cloud_march.comp`'s `cloudGroundShadow` gets equivalent quality. Even 3×3 is −16 texture
   samples per ground pixel. **Biggest single win in the file.**

3. **Hoist the 4 terrain-elevation samples out of per-iteration paths** where possible — check the
   normal calc (glsl ~1500s) isn't re-fetching heights the march already had.

4. **`optDepth` is NOT on this list** — its isolated cost measured near-zero (session 29 profiling).
   Leave it. Do not "optimise" the 4 call sites into a shared function (already tried, reverted,
   see `CLAUDE.md` — it defeats the compile-time unroll).

5. Audit the MSL (`benchmarks/` capture, or regenerate) for other redundant `.sample()` of the same
   texture+UV, and for `for` loops with compile-time-constant bounds that didn't unroll.

**Expected:** measurable on the Mac (loop 6 alone is ~16 fewer texture fetches on ground pixels),
free elsewhere. This is the part that "grants better insights for all targets."

---

## Phase 1 — `sat_sky_lite.frag` variant, wired to Planetarium/Low

Same source file, second SPV compiled with `-DSKY_LITE` (CMake: add a second
`add_custom_command` for `sat_sky.frag` → `sat_sky_lite.frag.spv`, exactly like
`sat_sky_minimal.frag` already is a separate pipeline). New `skyBgLitePipeline`, selected by
preset the same way `skyBgMinimalPipeline` (bit 262144) already is — see
`createSkyBgPipeline()` / `recordDraw()` Pass 1.

`#ifdef SKY_LITE` gates, roughly in cut-value order:

| Cut | What | Reasoning |
|---|---|---|
| A | **Milky Way panorama** (loop-adjacent, glsl ~2700, L997 sample) | Known Potato-killer (`atan2`×2 + texture). Lowest tier can lose it; stars remain. |
| B | **64-bin sky-glow loop** (loop 5) | 64 iterations, no quality slider, no current preset reach (bit 65536). Satellite bloom still comes from the separate `flare_composite` pass. |
| C | **Beam ground-spot loop + beam-glow dome** (loop 12, glsl ~2470) | Reflect-Orbital is niche; drop at lowest tier (bit 128 already gates the compute side). |
| D | **Aurora glow on terrain/ocean** + ocean aurora reflection sample | Aurora is an event; the curtain already moved to `cloud_march.comp`. Cut the frag-side ambient. |
| E | **Ocean → Potato model**: replace `seaMap` secant trace (loop 8) + 3× `seaMapDetail` central-diff normals + N_REFL reflection march (loop 9) with 1–2 noise-tap slope + flat-sky Fresnel + Blinn glint. Removes 2–3 loops and most octave evals. |
| F | **`evalCloudLayer` 4 → 2 layers** (loop 13) | Keep the low/mid deck, drop the two thin extra layers. |
| G | **Cloud-shadow blur → single tap** (loop 6) | Beyond Phase 0's 3×3 — at lite tier accept the dither noise. |

Keep at lite tier (this is the "Planetarium look" we're protecting): the real atmosphere march
(just fewer samples — see Phase 2), terrain raymarch + binary search, day/night + city textures,
the flat cloud deck, sun disc + `lensFlare()`, moon disc.

**Open question the capture would have answered:** which of A–G is actually enough. Cut A+B+C
first, build, test. Add D→G only as needed. Don't cut all seven blind.

---

## Phase 2 — preset tuning, zero shader change

`applyGraphicsPreset(GraphicsPreset::Planetarium)` in `SatelliteSimUI.cpp` can already push the
UBO-driven sample counts down with no recompile:

- `cloud.viewSamplesMin` / `viewSamplesMax` (atmosphere march, loop 4) — Planetarium should be ~6/10.
- `cloud.lightSamples` (`optDepth` inner) — ~4.
- `kTerrainStepsMax` equivalent (`cloud`-driven? check — terrain kN clamp, loop 1) — cap ~96.
- `cloud.oceanSeaOctaves` / `oceanDetailOctaves` / `oceanReflSamples` — floor them.

Verify what Planetarium currently sets vs Medium — this may be low-hanging fruit already half-done.

---

## Benchmarking — keep it in-repo

- Traces / large scratch → `benchmarks/` (gitignored). Never `~` or `/`.
- `tools/make_capture_bundle.sh` builds `build/SatLightSimCapture.app` for GPU capture (needs full
  Xcode to *read* the result).
- Primary signal remains: in-app FPS badge + `perf_profiles/profile_log.jsonl` snapshots + the
  knockout sweep. The Mac has no working GPU timer attribution (`memory/` notes), so trust the
  wall-clock FPS and the `App.cpp` `SATLIGHTSIM_FRAME_TRACE=1` phase breakdown.
- Regenerate the MSL any time: it's the MoltenVK translation of the `.spv`; a capture dumps it as
  hashed blobs (`file` reports "C++ source text"). Useful for spotting redundant samples / unrolled
  loop bloat without Xcode.

---

## Session log

### 2026-09-07 — plan created
Forked from `macOS-potato`. Potato mode (additive minimal shader) committed there as the
proven-hardware benchmark. This plan is the subtractive counterpart on the real `sat_sky.frag`,
Phase 0 intended to land on `main`.

### 2026-09-07 — Phase 0 done (untested)
`sat_sky.frag`:
1. **Moon `squish` block removed** (was glsl ~1804–1822). It computed `elDeg` (an `asin`), and
   under `elDeg < 15°` two `tan` + `radians` + divides, all feeding `squish` which was hardcoded
   `0` — then `dirR = normalize(vec3(dir.xy, dir.z*(1.0+squish)))`, an identity. Now uses `dir`
   directly for the ray/disc intersection. `dirR` fully gone.
2. **Cloud-shadow blur 5×5 → 3×3** (glsl ~1951). 25 `cloudTargetB.a` taps → 8 (centre tap reuses
   the already-sampled `cloudBCenter.a`). `kShadowBlurSpread = 1.7` keeps the footprint near the
   old radius-2. **−17 texture samples on every ground-hit pixel** — the largest sample-count cut
   available in the file.

Not done (behaviour-change risk, deferred): terrain normal 4-tap central diff → forward diff;
binary-search 12 → 8 iters; the 3×3 cloudTargetA/B rgb blur (18 taps, glsl ~1531) — that one is
beam-glow-edge-specific, so it belongs in Phase 1's SKY_LITE (which cuts beams anyway).

Test: Planetarium FPS delta; no regression on moon disc near the horizon, or cloud-shadow
graininess on close ocean.

### 2026-09-07 — Phase 0 tested + Phase 1 infra done (untested)
Phase 0 shipped: **no Planetarium FPS change (still 2)** — expected, shaving 17 texture samples
doesn't cross the occupancy threshold. Kept as a real all-targets optimisation.

Phase 1 built:
- **CMake:** second compile `sat_sky.frag -DSKY_LITE` → `sat_sky_lite.frag.spv`.
- **C++:** `skyBgLitePipeline` (mirrors `skyBgMinimalPipeline` — created/resized/destroyed
  alongside it), selected by **debugDisableMask bit 524288** in `recordDraw` (262144 Potato wins
  if both set). Breadcrumb logs `LITE (SKY_LITE)`.
- **Preset:** Planetarium now sets `kBitLiteSky` **and** `viewSamplesMax 48 → 16` (Phase 2 — a
  flat-Earth tier doesn't need a 48-step atmosphere integral).
- **`#ifdef SKY_LITE` cuts in `sat_sky.frag`:** A Milky Way block, B 64-bin sky-glow loop, C beam
  ground-spot loop, F cloud layer loop `3→0` becomes `1→0`, plus the **3×3 cloudTargetA/B rgb blur
  → single tap** (−16 dynamic texture samples/pixel, and it was NOT terrain-gated so it was the
  biggest unconditional cost left at Planetarium once terrain is off — which bit 1 already does).
- SPV 158 KB → 147 KB (−7%). Static `OpImageSample` 32 → 29 (the blur is one op in an un-unrolled
  loop, so the real per-pixel saving is ~16, not 3).

**Full shader unchanged** (still 32 samples) — Medium+ and any capable GPU keep it exactly.

Test: Planetarium FPS. Visual: Milky Way gone (stars stay), no beam ground spots, otherwise
identical to old Planetarium.

### 2026-09-07 — Phase 1 tested: **2 FPS → 15 FPS** on the 2015 Mac
Huge jump — cutting the MW (`atan2`/`asin` + panorama fetch) + the 18-tap cloud rgb blur +
`viewSamplesMax` 48→16 crossed the occupancy cliff. Confirms register pressure / texture-latency
was the wall, not any one feature's math.

Follow-up same session:
- **Beam ground-spot loop RESTORED in SKY_LITE** (cut C reverted). User specifically values the
  Reflect-Orbital beams and they measured cheap (loop is bounded by `groundBeamCount`, ~1.6 ms
  worst case at Medium/full-res/256 beams). Beam pointing rays were never cut (compute pass).
- **Planetarium atmosphere march `viewSamplesMin/Max` 6/16 → 4/10.**

SKY_LITE cuts now: A Milky Way, B 64-bin sky-glow loop, F cloud layers 4→2, cloud rgb blur→1 tap.
(C beams kept, D aurora-glow not yet cut.)

### 2026-09-07 — round 3: sun corona bridge + more SKY_LITE cuts (15 → ~22 FPS so far)
- **Sun corona bridge — FULL shader, all platforms.** The disc (`sunCol`≈1.5) dropped straight to
  the `×0.12` wide corona at its edge → a dark ring / "cutout" post-tonemap (present in the base
  app, not just Potato). Added three stacked `pow(cosA, 1800/320/55)` lobes peaking at ~disc
  brightness, same shape as `sat_sky_minimal.frag`'s Potato corona. User: "the glow is fantastic."
- **SKY_LITE cut D:** aurora surface glow (terrain + ocean `auroraGlowAt`). Ocean aurora
  *reflection* march was already dead here via `dbgSkipOceanRefl`.
- **SKY_LITE zenith ambient loop** `N_ZT` 4 → 2.
- **SKY_LITE atmosphere-loop city-upwelling + green/sodium airglow CUT** — that block was a
  `textureLod(earthNightTex)` on EVERY atmosphere step (~10 texture fetches/pixel) + asin/atan +
  two `warpPerlin3` airglow masks. Kept only `sampleSunDotGeo` (one dot) for the terminator gate.
  Likely the biggest single remaining runtime cut.
- **Preset:** Planetarium also knocks out `kBitCloudMarch | kBitCirrusMarch` (coverage 0 here).

SKY_LITE SPV 149 → 141 KB. Full shader +0.5 KB (corona only).

### 2026-09-07 — round 4: **22 → 60 FPS** (airglow was the killer), city glow restored optimised
The atmosphere-loop cut jumped Planetarium to 60 FPS in some regions. The **green/sodium airglow
coverage masks** (two `warpPerlin3` calls per atmosphere step) were the real cost, not the
`earthNightTex` fetch alone.

City-upwelling put back for SKY_LITE (sim read "hollow" at night without it) but optimised:
- **airglow bands stay cut** (this tier has no nightglow anyway)
- city `earthNightTex` lookup only on **`i < 3`** atmosphere steps — `densR = exp(-h/8km)` makes
  everything past ~3 steps negligible, so ~3 fetches/pixel not ~10
- `sampleSunDotGeo` (terminator gate, needs every step) cheapened to `dot(normalize(sp), sunDir)`
  — frame-invariant, skips the ECEF matrix transform

SKY_LITE SPV 141 → 142 KB.

### 2026-09-07 — push constants trimmed to the 128-byte floor (all-platform, `main`)
Not a perf change — a hardware-compatibility fix, same 2015 AMD Mac / weak-hardware theme. That
GPU's `maxPushConstantsSize` is exactly 128 (the Vulkan-guaranteed minimum), and `SatDrawPC` (176)
+ `CloudMarchPC` (148) both exceeded it, so their pipeline layouts couldn't be created.

- `SatDrawPC` → 128-byte sky core (skyView / fov / aspect / gmst / waveTime / sun / moon / obsECEF),
  used only by `skyBgPipeLayout`.
- New `PointDrawPC` (128) for `drawPipeLayout` + `starPipeLayout` — carries the two genuinely
  per-draw flags (`noTwinkle` on the planet draw, `manualTerrainTest` on the trail draws) plus a
  full-res `screenSizePx` and a `debugDisableMask` copy for `sat_point.frag`'s bit 4096.
- `CloudMarchPC` → 128 (mat4 + 4 float + 3 vec4).
- Everything else that was trailing past offset 128 was per-frame-uniform and moved into the
  **CloudParams UBO** (`GpuCloudParams` / `cloud_params.glsl` grew 496 → 544, "Push-constant
  relief" block; `check_cloud_params.py` green): `debugDisableMask`, `skyGlareVisibility`,
  `beamMaxRangeM`, `beamSkyGlowGain`, `beamGlowBleedGain`, `beamProximityGlow`, `mwSuppressEased`,
  `showBeamDebugRays`, `cloudShadowRangeM`, and the sky render-target size as `skyScreenW`/`skyScreenH`.
- `buildSatDrawPC()` → `buildSkyDrawPC()` / `buildPointDrawPC()`.

No new descriptor bindings, no new pipelines, no spec constants. Builds clean (incl. `-DSKY_LITE`),
all `static_assert`s pass, user confirmed the app still runs on the Mac. `CLAUDE.md` updated
(struct entries + "Weak-Hardware Sky Tiers → All-platform changes"). **Future:** `SatOrbitPC` /
`SatFlarePC` sit at exactly 128 too — new push-constant fields on any of these go in a UBO.

### Next (options, in rough FPS/effort order)
1. **Aurora surface glow (cut D)** — `auroraGlowAt` on terrain/ocean + the ocean aurora-reflection
   sample. `kBitAurora` already gates the curtain; this is the frag-side ambient. Low risk.
2. **Sky-ambient-at-terrain-hit zenith integration** (loop L554, `zi<4`, each with an `optDepth`)
   — replace with a 1-sample analytic term at SKY_LITE.
3. **Ocean detail (cut E)** — already `dist`/`altFade`-gated so only a low+close observer pays;
   lower value than first thought. If cut: `seaMap` secant octaves 3→2, skip `seaMapDetail`
   central-diff (perturb sphere normal from 1–2 noise taps).
4. **Broaden reach** — Phase 0 opts (moon squish, 3×3 shadow blur) should land on `main` for all
   platforms. Consider a `kBitLiteSky`-style path for the **Low** tier too (mid-range GPUs), not
   just Planetarium.
