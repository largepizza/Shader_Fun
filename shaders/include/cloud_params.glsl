// ── cloud_params.glsl ─────────────────────────────────────────────────────────────────────────
// The CloudParams uniform block, shared by every stage that reads it.
//
// This block used to be hand-copied into each consumer. It has grown from 176 to 336 bytes across
// sessions, and every growth meant editing each copy plus the C++ GpuCloudParams in lockstep — a
// silent, whole-buffer corruption if any one of them was missed, with no compiler to catch it.
// One definition removes that failure mode entirely.
//
// Consumers bind it at different indices, so define CLOUD_PARAMS_BINDING before including:
//     #define CLOUD_PARAMS_BINDING 4
//     #include "cloud_params.glsl"
//
// The C++ side (GpuCloudParams in src/simulations/SatelliteSim.h) still has to mirror this by
// hand — GLSL and C++ cannot share a declaration. THAT pairing is the one remaining place a
// mismatch can hide, so any field added here must be added there in the same change.
//
// Not every consumer reads every field. Fields are kept in full regardless, because the layout is
// what matters: dropping one a given shader does not use would shift every field after it.
#ifndef CLOUD_PARAMS_BINDING
#error "define CLOUD_PARAMS_BINDING before including cloud_params.glsl"
#endif

struct CloudLayer {
    float shellAltM;
    float driftMult;
    float alphaMax;
    float mipLod;
    float coverageMult;
    float densityMult;
    float enabled;
    float pad;
};

layout(set = 0, binding = CLOUD_PARAMS_BINDING) uniform CloudParams {
    float coverage;
    float density;
    float driftRate;
    float sunGain;
    float ambientGain;
    float hgG;
    float marchSteps;
    float lightSteps;
    float cloudPhase;
    float extinctionCoeff;
    float cirrusWindAngle;
    float cirrusStretch;
    float airglowGain;
    float airglowGreenGain;
    float airglowRedGain;
    float airglowSodiumGain;
    float shadowMaxDistM;
    float maxRenderDistM;
    float viewSamplesMin;
    float lightSamples;
    float oceanSeaOctaves;
    float oceanDetailOctaves;
    float oceanReflSamples;
    float viewSamplesMax;
    float sunGainZenith;
    float moonGain;           // shared moonlight brightness master — replaces this file's old
    float pad1;               // reserved
    float pad2;               // reserved
    vec4 mwBasisRow0;
    vec4 mwBasisRow1;
    vec4 mwBasisRow2;
    CloudLayer layers[4];
    float stormStrength;
    float auroraGain;
    float auroraCloudGain;
    float auroraGroundGain;
    float auroraCoverageFreq;
    float auroraCoverageAzFreq;
    float auroraCoverageDriftRate;
    float auroraShimmerRate;
    // Struct grew 320->336 (session 30, see GpuCloudParams in SatelliteSim.h for why this was
    // appended here instead of reusing pad1/pad2 above).
    float cloudTwilightAmbientGain;
    float cloudBaseVariance;
    float cloudErosionEdge;
    float cloudErosionCore;
    // Grew 336->352. sunGainElevBand: sin(sun elevation) at which sunGainZenith fully replaces
    // sunGain. Was a hardcoded smoothstep(0,1,...), i.e. only half-way to the zenith value at 30
    // degrees elevation, so a gain tuned for sunset stayed dominant most of the morning.
    float sunGainElevBand;
    // Edges of the twilight ambient bell, in sin(sun elevation) at the cloud sample.
    // Hi: above this the term is zero (raise to start it EARLIER, before the sun has set).
    // Lo: below this the term is zero (lower to carry it deeper into night).
    float twilightBandHi;
    float twilightBandLo;
    // Mip level the VOLUMETRIC march reads the 2D coverage map at. Was hardcoded 4.0, which on
    // the 8K source is ~78 km per texel — every small feature the flat 2D layer shows was already
    // filtered away before the density function saw it, so no coverage value could bring it back.
    float coverageMipLod;
    // Flat-2D-layer calibration. The volumetric and flat paths need DIFFERENT numbers to look the
    // same, because they are structurally different: the volumetric thresholds the coverage map
    // and then erodes it, applies a height profile and accumulates through transmittance, while
    // the flat layer thresholds the same map once and multiplies. Driving both from one set of
    // sliders is what made the 3D->2D crossfade impossible to match. These map the shared slider
    // values onto the flat layer's equivalent, measured against the volumetric at MIP 0.
    float flatCoverageScale;   // volumetric coverage 1.00 matched flat coverage 0.69
    float flatSunGainScale;    // volumetric sun gain 0.46 matched flat sun gain 1.84 (~4x)
    // fogTopAltM/fogDensity (C11, repurposed from pad10/pad11 — zero size change): the ground fog
    // shell (sea level to fogTopAltM) and its density scale. See fogMarchCS in cloud_march.comp.
    float fogTopAltM;
    float fogDensity;
    // Distance-based 3D->2D crossfade, replacing the altitude-only one as the effective control.
    // Keyed on the distance to the cloud shell along THIS ray, so it bounds the march directly —
    // unlike maxRenderDistM, which caps march LENGTH from the shell entry and therefore does
    // nothing from orbit (there the span is just the ~9 km shell crossing, already far under any
    // cap). Also gives a natural horizon transition from the ground, which altitude cannot.
    float cloudDistFadeStartM; // fully volumetric nearer than this
    float cloudDistFadeEndM;   // fully flat-2D beyond this
    // fogCoverage/fogSunGain (C11, repurposed from pad12/pad13 — zero size change): global
    // coverage gate for the ground fog shell's patchiness, and its own sun-lit brightness gain
    // (separate from cloud.sunGain per [[feedback_shared_gain_sliders]]). See fogMarchCS.
    float fogCoverage;
    float fogSunGain;
    // Terrain march distance fade (S4, RELEASE_v1_1_PLAN.md session 31): mirrors
    // cloudDistFadeStartM/EndM above but gates the per-pixel terrain-relief raymarch in
    // sat_sky.frag rather than a volumetric/flat crossfade. Keyed on the ray's own march reach
    // (tExit, which already grows with observer altitude via tCap) — grazing/high-altitude rays
    // fade OUT the march's step budget and, beyond terrainDistFadeEndM, skip it outright, falling
    // back to the already-computed tSeaLvl (a smooth textured Earth, not "nothing" — the same
    // result "terrain off" debug knockout already produces). Below terrainDistFadeStartM, full
    // detail — chosen so the old always-250km ground-level reach is entirely unaffected.
    float terrainDistFadeStartM;
    float terrainDistFadeEndM;
    // Cloud opacity tuning knob (repurposed from pad14): multiplies the volumetric cloud march's
    // per-step extinction coefficient directly. `density` alone cannot push a cloud past full
    // opacity because the per-sample density `d` is clamped to [0,1] before extinction is derived
    // from it (cloudDensity() in cloud_march.comp) — once a column is saturated at d=1, raising
    // `density` further does nothing, so a persistently "leaky" thick cloud (city lights or the
    // sun disc visibly showing through what should read as solid overcast) had no slider that
    // could ever fix it. This scales the extinction-per-metre constant itself, independent of
    // cloud shape/coverage. Default 1.0 reproduces the original hardcoded behavior exactly.
    float cloudOpacityScale;
    // City-lights blur-through-cloud (repurposed from pad15): sat_sky.frag's terrain branch
    // blends earthNightTex/cityNightDetailTex toward a blurrier mip as this pixel's local cloud
    // opacity rises, so light diffused through haze/overcast reads as a soft glow instead of a
    // sharp copy of the raw city-lights texture cutting through partially-transmissive cloud —
    // even a fully correct opacity value still looks wrong if what leaks through is pixel-sharp.
    // This is the mip LOD used at full (opacity=1) blur strength; 0 disables the effect.
    float cityLightBlurLod;
    // Atmospheric scattering strength gains: scale the physical Rayleigh/Mie coefficients
    // (BETA_R_BASE/BETA_M_BASE, common.glsl) uniformly across every consumer that shadows them
    // (sat_sky.frag's evalCloudLayer/main — atmosphere, terrain ambient, ocean reflection, moon/
    // sun attenuation — and cloud_march.comp's cloudMarchCS/cirrusMarchCS/fogMarchCS). Default 1.0
    // reproduces the original hardcoded constants exactly. Rayleigh gain scales the whole vec3
    // uniformly, so it preserves the R:G:B ratio that gives sunsets their color — it only changes
    // how much path length it takes for that color to show up (deepens/shallows the effect, not
    // the hue). Mie gain scales the wavelength-neutral haze term that dilutes Rayleigh's
    // saturation with white/grey — raise it to soften an overly saturated sunset, lower it for a
    // purer, more intensely colored one.
    float atmosRayleighGain;
    float atmosMieGain;
    // Domain-warp shear controls (repurposed from pad16/pad17 - zero size change; the block
    // stays 416). Together these set how much the warp DISTORTS the cloud noise as opposed to
    // merely translating it.
    //
    // The warp maps x -> x*base + A*W(f*x). That stays a well-behaved (non-folding) deformation
    // only while A*f*|grad W| / base < 1. Past 1 the Jacobian determinant passes through zero and
    // the field FOLDS - mirrored back on itself and infinitely compressed along one direction at
    // the fold line. That reads as pinched/squished noise, horizontal banding, and cloud tops
    // undulating like "wavy chips".
    //
    // With |grad W| ~ 2 in the bake's cell space (the FBM's 2x octave contributes nearly as much
    // gradient as the base octave despite only 0.3 amplitude weight), the ratio for the detail
    // lookup is cloudWarpStrength * 2 * cloudWarpFreq / kCloudHorizFreq. Defaults 32 / 3.0 give
    // 0.4 - comfortably clear of folding. It was 32 / 6.0 = 0.8, i.e. right at the edge.
    //
    // Raise STRENGTH for bigger cloud-structure displacement, lower FREQUENCY to keep that
    // displacement smooth. Shear is the PRODUCT, so trading one against the other is how you get
    // large-scale movement without pinching.
    float cloudWarpStrength;
    float cloudWarpFreq;
    // ── Erosion redesign (416 -> 432; APPENDED, not folded into pad1/pad2, which look free but
    // are already repurposed as the city-detail world offsets — see GpuCloudParams) ────────────
    //
    // cloudSurfaceCarve: how the vertical height profile enters cloudDensity().
    //   0 = the original MULTIPLICATIVE form, `eroded * heightProfile`.
    //   1 = fully SUBTRACTIVE, `eroded - (1 - heightProfile)`.
    // This is the single most important knob of the four. Multiplying by a smooth analytic ramp
    // scales the erosion noise toward zero exactly at the cloud's top and bottom surfaces, so
    // those surfaces are defined by the ramp (smooth, per-column, fixed 0.05 width) and the noise
    // only ever appears as speckle superimposed on the interior. Subtracting instead DISPLACES
    // the surface by the noise, which is what makes a boundary bumpy. Containment is preserved at
    // any blend: heightProfile -> 0 drives both forms to zero.
    //
    // cloudErosionBillow: polarity of the erosion field, 0 = original, 1 = flipped.
    // The bake's G channel is an inverted-Worley FBM, which PEAKS at Worley feature points. Used
    // as the erosion threshold that means density is removed in round blobs centred on those
    // points, and what survives is the connected Voronoi web between them — a sponge with holes
    // punched through it. Flipping to (1 - G) removes the web instead and leaves the blobs, i.e.
    // rounded lumps: cauliflower. The three octave weights sum to exactly 1.0, so (1 - G) is
    // exactly the non-inverted Worley FBM and the flip is free.
    //
    // cloudErosionBillowH: normalised column height over which erosion ramps from web-polarity
    // (wispy, streaky — correct for cloud BASES) up to full cloudErosionBillow (billowy lumps —
    // correct for cloud TOPS). This is Schneider/Nubis's height-varying erosion character. 0
    // disables the ramp and applies cloudErosionBillow everywhere.
    //
    // cloudErosionFreq: multiplier on the erosion lookup's coordinate (was a hardcoded 1.5).
    // Raise for smaller-scale surface detail; it re-tiles the same 192^3 bake, so far above ~4
    // expect visible repetition.
    float cloudSurfaceCarve;
    float cloudErosionBillow;
    float cloudErosionBillowH;
    float cloudErosionFreq;
    // ── Directional-shading contrast (432 -> 448) ────────────────────────────────────────────
    // A cloud's only source of lit-side/dark-side variation is sunOptDepth from the self-shadow
    // cone: the view-angle phase function is constant across the whole cloud from any one
    // viewpoint, and every ambient term is additive and non-directional. So everything that
    // compresses sunOptDepth's dynamic range flattens the cloud, and four separate things did.
    //
    // cloudMultiScatter: 0 = pure single-scatter exp(-OD), 1 = the original 3-octave sum. The
    //   slow-decaying octaves fill shadows in on purpose; at typical sunset optical depths they
    //   supplied more than half the remaining light, so most of the "shadow" carried no shape.
    // cloudShadowFloorT: hard floor on sun transmittance (was a fixed 0.05) — directly sets how
    //   dark a fully shadowed cloud side may get, so it caps contrast on its own.
    // cloudGrazeShadow: sunOptDepth multiplier at zero sun elevation (was a fixed
    //   kGrazeShadowFloor = 0.35, i.e. self-shadowing was cut to a third exactly at sunset).
    // cloudConeLenScale: multiplier on the cone's shellThick*2 length cap, which bounds the
    //   maximum optical depth the cone can accumulate and therefore the darkest back-lit shadow
    //   available at any sun angle.
    float cloudMultiScatter;
    float cloudShadowFloorT;
    float cloudGrazeShadow;
    float cloudConeLenScale;
    // ── Shape-aware shading + flat-layer decoupling (448 -> 464) ─────────────────────────────
    // cloudVertShadeGain: strength of the analytic top-bright/bottom-dark ramp, which is a pure
    //   function of height and therefore paints horizontal bands ("lasagna"). It is additionally
    //   weighted by sun elevation now, since a vertical light gradient only makes sense when the
    //   light comes from above. 0 hands all shading to the self-shadow cone.
    // cloudDensityAO / cloudAOPower: occlusion driven by the sample's own local density, the one
    //   cheap shading input that carries real 3D cloud shape rather than height. Power sets where
    //   1-exp(-d*power) bites, and needs tuning against whatever `density` is set to.
    // flatDensityScale: the flat 2D layer's opacity, decoupled from the volumetric `density`.
    //   The two reach opacity by different routes — volumetric accumulates transmittance across
    //   many samples, flat multiplies once — so a `density` low enough to give the volumetric
    //   soft shading left the flat layer visibly translucent. Raise this to compensate.
    float cloudVertShadeGain;
    float cloudDensityAO;
    float cloudAOPower;
    float flatDensityScale;
    // ── Flat-2D Rayleigh decoupling (464 -> 480) ─────────────────────────────────────────────
    // flatRayleighGain: an EXTRA multiplier on the Rayleigh coefficient used by sat_sky.frag's
    //   evalCloudLayer ONLY — i.e. the flat 2D paste, not the volumetric march, not the sky.
    //   It stacks on top of atmosRayleighGain (which stays global), so 1.0 is an exact no-op.
    //   Exists because the two cloud paths reach their colour by structurally different routes:
    //   the flat layer applies BETA_R twice over (once for sunColorFlat, the sun transmittance
    //   AT the cloud, once for attn/airlight, the camera->cloud path) as a single closed-form
    //   multiply, while the volumetric accumulates through transmittance per step. So a Rayleigh
    //   gain tuned to give the volumetric clouds the right sunset depth does NOT land the flat
    //   layer in the same place, and the 3D->2D crossfade shows a hue/depth step across
    //   cloudDistFadeStartM..EndM. This is the knob that closes that step, in the same spirit as
    //   flatCoverageScale/flatSunGainScale/flatDensityScale above.
    // flatTwilightAmbientGain (claimed pad18): the flat layer's share of the twilight sky-ambient
    //   term, stacked on the shared cloudTwilightAmbientGain the volumetric march uses — so the
    //   shared slider still moves both together and this one only sets their RATIO. Before it
    //   existed the flat path had no ambient term at all and went black through twilight while
    //   the volumetric clouds beside it stayed sky-lit blue; see evalCloudLayer in sat_sky.frag.
    //   1.0 = the same strength the volumetric gets.
    // The remaining pads keep the block a multiple of 16 bytes (std140 rounds the block size up
    // to 16 regardless, so the C++ mirror must too or the descriptor range undershoots the block).
    float flatRayleighGain;
    float flatTwilightAmbientGain;
    // ── Orbital terminator gate (claimed pad19/pad20 — zero size change, block stays 480) ─────
    // Read only by sat_sky.frag's N_VIEW atmosphere loop. An ARTISTIC suppression of scattered
    // sunlight past the terminator, not a correction to the scattering: the integral was measured
    // to fall ~1 decade per 6 degrees across the terminator, which is real twilight's rate. What
    // it compensates for is that this renderer composites an eyeballed HDR scene instead of
    // shooting at a fixed exposure, so that physically-correct tail reads far brighter than the
    // orbital photography it gets compared against — leaving hard-cutoff clouds looking like dark
    // patches against a still-lit atmosphere.
    //
    // atmosTermStrength: 0 = exact previous behaviour (the multiply becomes 1.0 — keep this as the
    //   A/B), 1 = full suppression past the cutoff. Additionally scaled by observer altitude over
    //   40-100 km, so it is inert at ground level. That gating is load-bearing, not caution:
    //   applied globally it takes the western afterglow down 4-8x with the sun 2 degrees below the
    //   horizon and removes the Belt of Venus, because the samples it suppresses are the ones over
    //   night-side ground — the unwanted wash seen from orbit, but the actual post-sunset sky seen
    //   from below.
    // atmosTermWidth: half-width of the rolloff in sin(sun elevation), centred on the geometric
    //   terminator — smaller is a harder cliff. Centring on 0 rather than exposing both edges is
    //   deliberate: it anchors the cut to the same place the clouds cut, which is the entire point
    //   of the feature. 0.08 puts SZA 92 about 23x down, 0.035 effectively removes it outright.
    float atmosTermStrength;
    float atmosTermWidth;
    // ── Airglow coverage + polar boost (480 -> 496) ──────────────────────────────────────────
    // airglowCoverageGain: blends green/sodium/red airglow from a uniform, gently-shimmering
    // sky (0) to a genuinely patchy coverage field with real dark gaps between brighter patches
    // (1) — the same threshold-remap idiom auroraCoverage/cloud coverage already use, applied
    // to airglow's existing but much milder airPatch/rPatch shimmer (+-40%, which reads as
    // nearly flat on screen).
    // airglowPolarGain: RED-band-only extra multiplier that ramps up toward the geomagnetic
    // pole (see kGeomagPoleECEF), patchily rather than as a uniform ring. Real 630nm redline
    // nightglow brightens toward the auroral zone from diffuse particle precipitation, well
    // beyond the discrete aurora curtain that auroraGain/auroraGroundGain/auroraCloudGain
    // already control — green/sodium deliberately do NOT get this term, only broader coverage
    // patchiness (see airglowCoverageGain above).
    float airglowCoverageGain;
    float airglowPolarGain;
    // ── Zodiacal light (496 -> 512) ───────────────────────────────────────────────────────────
    // pad21/pad22 repurposed: zodiacalWidthDeg (ecliptic-latitude Gaussian sigma near the sun),
    // zodiacalOuterFadeDeg (elongation at which the cone has faded out). eclipticPoleENU.xyz is a
    // CPU-computed (updatePositions(), mirrors the Milky Way's mwBasisRow0-2 idiom) ENU-frame unit
    // vector toward the ecliptic north pole — NOT a full lon/lat basis like the Milky Way's, since
    // zodiacal light is a pure analytic falloff (elongation from the sun + ecliptic latitude), not
    // texture-sampled, so it only ever needs the latitude term: asin(dot(dir, eclipticPoleENU.xyz)).
    // eclipticPoleENU.w = zodiacalGain, the master brightness multiplier.
    float zodiacalWidthDeg;      // was pad21
    float zodiacalOuterFadeDeg;  // was pad22
    vec4 eclipticPoleENU;
} cloud;
