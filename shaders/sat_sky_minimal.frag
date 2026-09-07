#version 450

// ── Minimal sky/ground shader — Potato preset (knockout bit 262144) ───────────────────────────
// Cheap stand-in for sat_sky.frag on GPUs where that ~2800-line shader is unusably slow
// (measured ~490 ms/frame — the ENTIRE frame — on an AMD Radeon R9 M370X via MoltenVK on
// macOS 12; the cost is the shader's sheer size, not any single feature).
//
// The atmosphere is a CLOSED-FORM single-scatter approximation — no raymarch:
//   * optical depth along a ray  ~ scaleHeight * air-density-at-the-ray's-lowest-point
//                                  * Kasten-Young airmass (same airmass the extinction
//                                    subsystem uses; finite at the horizon)
//   * evaluated at the view ray's closest approach to the surface, so it works from the
//     GROUND (dense, blue overhead, bright horizon) AND from SPACE (thin wash over the
//     sunlit disc, bright blue band at the limb, pure black looking away from Earth)
//   * Rayleigh + Mie phase against the sun; wavelength dependence carried by (1 - e^-od)
//   * sun-path airmass explodes as the sun crosses the horizon, so day -> twilight -> night
//     falls out with no special case
// Ground = day/night Earth texture across the terminator + aerial perspective.
// No volumetric clouds / ocean waves / Milky Way / aurora / lens flare.
//
// Reuses skyBgPipeLayout / skyDescSet unchanged (declares only the 4 bindings it reads) and the
// same gl_FragDepth convention as sat_sky.frag.

#include "common.glsl"   // R_EARTH, R_ATMOS, PI, raySphere, BETA_R_BASE, BETA_M_BASE, H_R, H_M,
                         // SUN_INTENSITY, phaseR, phaseM
#include "terrain.glsl"  // observerEffHeight, observerPos, enuBasis, posToUV

// NOTE: the Milky Way was tried here (Sept 2026) and dropped for good. Textured panorama → 2 FPS
// (atan/asin equirect projection + a texture fetch are the occupancy killers on this GCN1/MoltenVK
// combo). A no-trig procedural galactic band fit performance-wise but looked poor — the discrete
// star catalogue (star_point.*) alone covers the night sky here.

// ── TUNE ─────────────────────────────────────────────────────────────────────────────────────
const float kSkyRayleighGain = 2.0;   // overall blue-sky brightness
const float kSkyMieGain      = 0.2;   // haze / white horizon / around-sun brightness
const float kNightSkyFloor   = 0.001; // faint airglow so the ground-level night zenith isn't #000

const float kCloudShellAltM  = 3000.0; // flat cloud deck altitude (single thin shell)
const float kCloudCoverage   = 0.55;   // 0..1 — higher = more sky covered
const float kCloudDensity    = 3.0;    // coverage-threshold hardness (edge sharpness)
const float kCloudAlphaMax   = 0.95;   // deck never fully opaque
const float kCloudDriftRate  = 0.03;   // longitude drift per radian of Earth rotation (0 = locked)
const float kCloudBrightness = 1.0;    // global multiplier on the lit cloud colour
const vec3  kCloudDayColor   = vec3(1.00, 0.98, 0.95);  // lit cloud, sun well up
const vec3  kCloudDuskColor  = vec3(0.78, 0.36, 0.15);  // warm band straddling the terminator
const vec3  kCloudNightColor = vec3(0.008, 0.011, 0.020); // unlit cloud, past the terminator

const float kCityDetailTileM = 20000.0;  // metres per detail-texture tile repeat
const float kCityFadeNearM   = 30000.0;  // full city detail strength inside this range
const float kCityFadeFarM    = 300000.0; // city detail fully gone beyond this range

const float kOceanWaveScale  = 0.0016; // wave-noise frequency (1/metres)
const float kOceanWaveStr    = 0.20;   // normal-perturbation strength (0 = mirror flat)
const float kOceanGlintExp   = 180.0;  // sun-glint tightness (higher = smaller, sharper)
const float kOceanGlintGain  = 1.5;    // sun-glint brightness
const float kOceanFresnel    = 0.45;   // max sky-reflection strength at grazing angles

const float kMoonAngR        = 0.004578 * 3.0; // lunar disc angular radius, drawn 3x for visibility
const float kMoonRadiance    = 1.0;    // lunar surface brightness (vs the sky it competes with)

const float kFlareCoronaGain = 0.25;   // procedural sun corona/halo brightness
const float kFlareRayReach   = 14.0;   // corona exponent along ray directions (lower = longer rays)
const float kFlareGhostGain  = 0.55;   // reflected ghost / bokeh artifact brightness
// ─────────────────────────────────────────────────────────────────────────────────────────────

layout(set = 0, binding = 3)  uniform sampler2D earthDayTex;
layout(set = 0, binding = 4)  uniform sampler2D earthNightTex;
layout(set = 0, binding = 5)  uniform sampler2D earthElevTex;
layout(set = 0, binding = 6)  uniform sampler2D earthSpecTex;layout(set = 0, binding = 7)  uniform sampler2D earthCloudsTex;
layout(set = 0, binding = 1)  uniform sampler2D noiseTex;            // RGBA white noise, REPEAT
layout(set = 0, binding = 2)  uniform sampler2D moonTex;            // lunar albedo, equirect-ish
layout(set = 0, binding = 14) uniform sampler2D cityDayDetailTex;
layout(set = 0, binding = 15) uniform sampler2D cityNightDetailTex;

layout(push_constant) uniform PC {
    mat4  skyView;
    float fovYRad, aspect, gmst, pad;
    vec4  sunDirENU;   // xyz dir in ENU, w = sin(elevation)
    vec4  moonDirENU;  // xyz dir in ENU, w = illuminated fraction
    vec4  obsECEFDir;  // xyz = observer ECEF up-direction, w = height offset
} pc;

layout(location = 0) in vec3 enuDir;
layout(location = 1) in flat vec4 sunDirENU;
layout(location = 2) in flat vec4 moonDirENU;
layout(location = 0) out vec4 outColor;

// Perlin's quintic smootherstep — C2 continuous (zero 1st AND 2nd derivative at both ends),
// unlike smoothstep (a cubic, whose 2nd derivative jumps at the endpoints, leaving a visible
// kink exactly where a value settles to 0 or 1 — e.g. where the sky finishes fading to black).
float smootherstep(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Kasten & Young 1989 relative airmass. 1.0 at the zenith, ~38 near the horizon. The published
// formula is only defined for elevation >= 0; below the horizon we return a large, growing value
// (the sun is being extinguished by an ever-thicker slab), which is what makes night dark.
float airmassKY(float sinEl) {
    if (sinEl <= 0.0) return 40.0 - sinEl * 260.0;
    float elDeg = degrees(asin(min(sinEl, 1.0)));
    return 1.0 / (sinEl + 0.50572 * pow(elDeg + 6.07995, -1.6364));
}

// Closed-form atmosphere. `dir` is the view direction (ENU). Returns in-scattered sky radiance;
// `viewTrans` is the scalar transmittance along the view ray (for aerial perspective on ground).
// enu[XYZ]/sunDirECEF let it evaluate the sun's elevation AT THE SCATTERING POINT rather than at
// the observer — without that, an orbital observer's whole visible atmosphere shares one sun
// angle (no terminator, and the whole disc's brightness sliding as the observer moves lat/lon).
vec3 analyticSky(vec3 obsPos, vec3 dir, float tGround, vec3 sunDir3,
                 vec3 enuX, vec3 enuY, vec3 enuZ, vec3 sunDirECEF,
                 out float viewTrans) {
    // ── Density geometry: the view ray's LOW point (closest approach to Earth's centre, capped
    //    at a surface hit). Dense air here — this sets the optical-depth MAGNITUDE.
    float b     = dot(obsPos, dir);
    float tLow  = clamp(-b, 0.0, (tGround > 0.0) ? tGround : 1.0e12);
    vec3  pLow  = obsPos + dir * tLow;
    float hLow  = clamp(length(pLow) - R_EARTH, 0.0, 1.0e6);
    float rayEl = abs(dot(dir, normalize(pLow)));   // |sin(elevation)| of the ray at that point

    // ── Lighting geometry. Scattered sunlight originates ALONG the view ray, at a range of
    //    altitudes — and near the terminator the TOP of the atmosphere is still lit while the
    //    BOTTOM is already in Earth's shadow. A single sample makes that transition a hard line;
    //    too few samples make it a few discrete bands. This walks the view ray's atmosphere
    //    segment and integrates the sun visibility. (The only "raymarch" in the shader — a
    //    handful of iterations of pure arithmetic, nothing like the old shader's 160 with
    //    texture fetches per step.)
    const int kSunTaps = 32;
    vec2  tShell  = raySphere(obsPos, dir, R_ATMOS);
    float hitsAir = step(0.0, tShell.y);           // 0 -> ray never reaches air -> deep space
    float segA = max(tShell.x, 0.0);
    float segB = (tShell.y > 0.0) ? tShell.y : 0.0;
    if (tGround > 0.0) segB = min(segB, tGround);

    float sunVis   = 0.0;
    float sunSinEl = 0.0;
    for (int i = 0; i < kSunTaps; ++i) {
        float t  = mix(segA, segB, (float(i) + 0.5) / float(kSunTaps));
        vec3  sp = obsPos + dir * t;
        // Soft geometric Earth shadow at this sample: bSun >= 0 -> the sun ray heads away from
        // Earth (unoccluded); else how far its closest approach clears the solid Earth, soft over
        // ~200 km so neighbouring taps' transitions overlap and the terminator reads as a smooth
        // gradient rather than stepped bands.
        float bs = dot(sp, sunDir3);
        float sv = 1.0;
        if (bs < 0.0) {
            float mr = length(sp + sunDir3 * (-bs));
            sv = smootherstep(R_EARTH - 60000.0, R_EARTH + 160000.0, mr);
        }
        sunVis   += sv;
        sunSinEl += dot(normalize(sp.x * enuX + sp.y * enuY + sp.z * enuZ), sunDirECEF);
    }
    sunVis   /= float(kSunTaps);
    sunSinEl /= float(kSunTaps);   // mean sun elevation over the lit part of the ray
    // One more quintic pass on the integrated result — softens the shoulders where sunVis
    // settles to exactly 0 (full night) or 1 (full day), which is where a residual kink shows.
    sunVis = smootherstep(0.0, 1.0, sunVis);

    float densR = exp(-hLow / H_R);
    float densM = exp(-hLow / H_M);
    float amV   = airmassKY(max(rayEl, 0.02));
    float amS   = airmassKY(sunSinEl);

    vec3  odR = BETA_R_BASE       * (H_R * densR * amV) * hitsAir;
    float odM = BETA_M_BASE * 1.1 * (H_M * densM * amV) * hitsAir;
    vec3  odRsun = BETA_R_BASE       * (H_R * densR * amS) * hitsAir;
    float odMsun = BETA_M_BASE * 1.1 * (H_M * densM * amS) * hitsAir;

    viewTrans = exp(-dot(odR + vec3(odM), vec3(1.0 / 3.0)));

    float mu = dot(dir, sunDir3);
    float pR = phaseR(mu);
    float pM = phaseM(mu);

    vec3 sunAttenR = exp(-odRsun);
    vec3 sunAttenM = exp(-vec3(odMsun));

    vec3 inR = pR * (vec3(1.0) - exp(-odR)) * sunAttenR;   // blue-dominant, reddens toward low sun
    vec3 inM = pM * (1.0 - exp(-odM)) * sunAttenM;         // wavelength-neutral haze / around-sun

    // Azimuthal (mu-weighted) bias: near sunset the SUNWARD sky is brighter and warmer, the
    // ANTI-SUN sky dimmer and cooler (Earth's shadow / the Belt of Venus). The midday sky is
    // close to azimuth-independent, so this only ramps in at a low sun.
    float lowSun = 1.0 - smoothstep(0.03, 0.38, sunSinEl);
    float azBias = mix(1.0, mix(0.42, 1.18, smoothstep(-0.55, 0.55, mu)), lowSun);

    vec3 sky = SUN_INTENSITY * sunVis * azBias * (inR * kSkyRayleighGain + inM * kSkyMieGain);

    // Faint night airglow, only where there's actually air (fades out from space).
    sky += vec3(0.38, 0.52, 1.0) * kNightSkyFloor * densR * hitsAir;

    return max(sky, vec3(0.0));
}

// ── Flat cloud deck ──────────────────────────────────────────────────────────────────────────
// One thin shell at R_EARTH + kCloudShellAltM. Samples earthCloudsTex at the hit point's
// geographic lat/lon (same equirectangular convention as posToUV), thresholds to an alpha, lights
// it by the sun angle at the CLOUD's own location (so night-side cloud stays dark regardless of
// where the observer is), and alpha-composites over `color`. No volumetric march, no drift, no
// aerial re-scatter — the Potato stand-in for evalCloudLayer / cloud_march.comp.
vec3 flatClouds(vec3 color, vec3 obsPos, vec3 dir, float tGround,
                vec3 enuX, vec3 enuY, vec3 enuZ, vec3 sunDirECEF) {
    vec2  tc = raySphere(obsPos, dir, R_EARTH + kCloudShellAltM);
    float t  = (tc.x > 1.0) ? tc.x : tc.y;
    if (t <= 1.0) return color;                        // shell not in front of us
    if (tGround > 0.0 && t >= tGround) return color;   // deck is behind ground we already hit

    vec3  hitENU = obsPos + dir * t;
    vec3  cECEF  = hitENU.x * enuX + hitENU.y * enuY + hitENU.z * enuZ;
    // Slow longitude drift so the deck isn't frozen to the ground. pc.gmst is Earth's rotation
    // angle (advances with sim time), so this is a genuine cloud-vs-terrain relative motion.
    vec2  uv = posToUV(cECEF);
    uv.x = fract(uv.x + pc.gmst * kCloudDriftRate);
    float raw   = texture(earthCloudsTex, uv).r;
    float alpha = clamp((raw - (1.0 - kCloudCoverage)) * kCloudDensity, 0.0, kCloudAlphaMax);
    if (alpha <= 0.0) return color;

    // Terminator lighting — deliberately NOT a scattering calc. Three-stop gradient keyed on the
    // sun angle at the cloud's own location: day white → a warm dusk band centred just past the
    // terminator → night. `night` and the dusk gaussian are hand-shaped, not physical.
    float sunDot  = dot(normalize(cECEF), sunDirECEF);
    float night   = smoothstep(0.12, 0.0, sunDot);                          // fully night AT the terminator
    float dusk    = exp(-pow((sunDot - 0.06) / 0.075, 2.0)) * (1.0 - night); // narrow warm bump, day side only
    vec3  cloudLit = mix(kCloudDayColor, kCloudNightColor, night);
    cloudLit = mix(cloudLit, kCloudDuskColor, dusk);
    return mix(color, cloudLit * kCloudBrightness, alpha);
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// VERBATIM copy of sat_sky.frag's lensFlare() (lines ~785-931) — 1:1 port test. If this holds
// performance on the target hardware, the procedural corona/ray/ghost approximation above it in
// git history is unnecessary. Uses noiseTex (binding 1). Coordinate space: ShaderToy-style flare
// UV, x in [-0.5*aspect, +0.5*aspect], y in [-0.5, +0.5].
// ════════════════════════════════════════════════════════════════════════════════════════════════
vec3 lensFlare(vec2 uv, vec2 pos, float intens, float bokehMult) {
    vec2  uvd  = uv * length(uv);
    vec2  d    = uv - pos;
    float dist = pow(length(d), 0.1);
    float ang  = atan(d.y, d.x);

    float noiseSeed = sin(ang * 4.0 + pos.x) * 4.0 - cos(ang * 3.0 + pos.y);
    float noiseU    = fract(noiseSeed * 0.125 + 0.5);
    float angNoise  = texture(noiseTex, vec2(noiseU, 0.25)).r;

    float scale = 1200.0;
    float f0 = 1.0 / (length(d) * scale + 1.0);
    f0 = f0 + f0 * (sin(angNoise * 16.0) * 20.8 + dist);
    f0 *= 0.1;

    float f1 = max(0.01 - pow(length(uv + 1.2 * pos), 1.9), 0.0) * 4.0;
    f1 *= 0.6;

    float ghostFade = smoothstep(0.03, 0.12, length(pos));

    float f2  = max(1.0 / (1.0 + 32.0 * pow(length(uvd + 0.80 * pos), 2.0)), 0.0) * 0.25 * bokehMult;
    float f22 = max(1.0 / (1.0 + 32.0 * pow(length(uvd + 0.85 * pos), 2.0)), 0.0) * 0.23 * bokehMult;
    float f23 = max(1.0 / (1.0 + 32.0 * pow(length(uvd + 0.90 * pos), 2.0)), 0.0) * 0.21 * bokehMult;

    vec2  uvx = mix(uv, uvd, -0.5);
    float f4  = max(0.01 - pow(length(uvx + 0.40 * pos), 2.4), 0.0) * 6.0;
    float f42 = max(0.01 - pow(length(uvx + 0.45 * pos), 2.4), 0.0) * 5.0;
    float f43 = max(0.01 - pow(length(uvx + 0.50 * pos), 2.4), 0.0) * 3.0;

    uvx = mix(uv, uvd, -0.4);
    float f5  = max(0.01 - pow(length(uvx + 0.20 * pos), 5.5), 0.0) * 2.0;
    float f52 = max(0.01 - pow(length(uvx + 0.40 * pos), 5.5), 0.0) * 2.0;
    float f53 = max(0.01 - pow(length(uvx + 0.60 * pos), 5.5), 0.0) * 2.0;

    uvx = mix(uv, uvd, -0.5);
    float f6  = max(0.01 - pow(length(uvx - 0.300 * pos), 1.6), 0.0) * 6.0;
    float f62 = max(0.01 - pow(length(uvx - 0.325 * pos), 1.6), 0.0) * 3.0;
    float f63 = max(0.01 - pow(length(uvx - 0.350 * pos), 1.6), 0.0) * 5.0;

    vec3 c = vec3(0.0);
    c += vec3(f0);
    c += vec3(f1 * 0.5);
    c.r += (f2  + f4  + f5  + f6)  * 0.4 * ghostFade * bokehMult;
    c.g += (f22 + f42 + f52 + f62) * 0.4 * ghostFade * bokehMult;
    c.b += (f23 + f43 + f53 + f63) * 0.4 * ghostFade * bokehMult;
    c = c * 1.3 - vec3(length(uvd) * 0.05);

    return max(c, vec3(0.0));
}

void main() {
    vec3 dir = normalize(enuDir);

    vec3 enuX, enuY, enuZ;
    enuBasis(pc.obsECEFDir.xyz, enuX, enuY, enuZ);
    vec3 dirECEF    = dir.x * enuX + dir.y * enuY + dir.z * enuZ;
    vec3 sunDirECEF = sunDirENU.x * enuX + sunDirENU.y * enuY + sunDirENU.z * enuZ;

    float obsEffH = observerEffHeight(earthElevTex, earthSpecTex, pc.obsECEFDir);
    vec3  obsPos  = observerPos(obsEffH);                                  // ENU
    vec3  obsECEF = obsPos.x * enuX + obsPos.y * enuY + obsPos.z * enuZ;   // ECEF

    vec2  tBase   = raySphere(obsPos, dir, R_EARTH);
    float tGround = tBase.x;
    float sunSinEl = clamp(sunDirENU.w, -1.0, 1.0);

    // sin() of the geometric horizon depression — 0 at sea level, negative in space (you can see
    // the sun below the local horizontal because the limb has dropped away). Sun/flare visibility
    // is gated on sunSinEl > limbZ, not > 0, so the flare doesn't blink out at orbital altitude.
    float obsR  = length(obsPos);
    float limbZ = (obsR > R_EARTH) ? -sqrt(max(0.0, 1.0 - (R_EARTH * R_EARTH) / (obsR * obsR))) : 0.0;

    float viewTrans;
    vec3  sky = analyticSky(obsPos, dir, tGround, normalize(sunDirENU.xyz),
                            enuX, enuY, enuZ, sunDirECEF, viewTrans);

    vec3  color;
    float tOcclude = -1.0;

    if (tGround > 0.0) {
        // ── Ground ───────────────────────────────────────────────────────────
        vec3 pECEF  = obsECEF + dirECEF * tGround;
        vec2 uv     = posToUV(pECEF);
        vec3 dayC   = texture(earthDayTex, uv).rgb;
        vec3 nightC = texture(earthNightTex, uv).rgb;

        // ── City-detail texture blend ────────────────────────────────────────
        // Tile a hi-freq detail texture over bright night-texture pixels (cities) within range of
        // the observer. Over dark terrain, or past kCityFadeFarM, this is a no-op. Simplified from
        // sat_sky.frag (no edge-noise disguise, no cloud blur), but the UV is now Earth-fixed:
        // cosine-corrected equirectangular metres, a pure function of the ECEF hit point, so the
        // pattern is glued to the terrain with zero swim. One seam at longitude ±180° (the fract
        // wrap in posToUV) — 113° from the 67°W spawn, never in frame at ground level.
        {
            float cityFade = 1.0 - smoothstep(kCityFadeNearM, kCityFadeFarM, tGround);
            if (cityFade > 0.001) {
                float cityLum  = dot(nightC, vec3(0.2126, 0.7152, 0.0722));
                float cityMask = smoothstep(0.01, 0.3, cityLum) * cityFade;
                if (cityMask > 0.001) {
                    float latC   = (0.5 - uv.y) * PI;
                    vec2  worldM = vec2(uv.x * (2.0 * PI * R_EARTH) * cos(latC),
                                        uv.y * (PI * R_EARTH));
                    vec2  duv = worldM / kCityDetailTileM;
                    dayC   = mix(dayC,   texture(cityDayDetailTex,   duv).rgb, cityMask);
                    nightC = mix(nightC, texture(cityNightDetailTex, duv).rgb, cityMask);
                }
            }
        }

        float localSun = dot(normalize(pECEF), sunDirECEF);       // local terminator, [-1, 1]
        float lit      = smoothstep(-0.10, 0.18, localSun);
        vec3  groundDay   = dayC * (0.05 + 0.95 * lit);
        vec3  groundNight = nightC * 1.3;                          // city lights only
        vec3  ground = mix(groundNight, groundDay, lit);

        // ── Ocean sun-glint (lean) ──────────────────────────────────────────
        // Perturb the local up with one scrolling noiseTex tap, Blinn-Phong against the sun. No
        // Fresnel, no sky reflection, no heightfield trace — the single striking water cue at
        // near-zero cost. Scalar sun-elevation reject comes first so land/night pixels pay nothing.
        if (sunSinEl > -0.05 && texture(earthSpecTex, uv).r > 0.5) {
            vec3  hitENU = obsPos + dir * tGround;
            vec2  wc     = hitENU.xy * kOceanWaveScale + pc.gmst * 0.15;
            vec2  slope  = (texture(noiseTex, wc).xy - 0.5) * kOceanWaveStr;
            vec3  N      = normalize(normalize(hitENU) + vec3(slope, 0.0));

            // Fresnel sky-reflection (no march — reuses the analytic sky colour). Distance-faded
            // so far ocean doesn't turn into a mirror sheet.
            float fres = pow(1.0 - max(dot(N, -dir), 0.0), 4.0) * kOceanFresnel
                         * exp(-tGround / 40000.0);
            ground = mix(ground, sky * 0.9 + vec3(0.02, 0.04, 0.07), fres);

            // Sun glint (Blinn-Phong against the perturbed normal).
            vec3  H    = normalize(normalize(sunDirENU.xyz) - dir);
            float spec = pow(max(dot(N, H), 0.0), kOceanGlintExp);
            ground += vec3(1.0, 0.96, 0.88) * spec * kOceanGlintGain
                      * smoothstep(-0.04, 0.06, sunSinEl) * exp(-tGround / 60000.0);
        }

        // Aerial perspective: attenuate the surface, add sky in-scatter in front of it.
        color    = ground * viewTrans + sky * (1.0 - viewTrans);
        tOcclude = tGround;
    } else {
        // ── Sky ──────────────────────────────────────────────────────────────
        color = sky;

        // Sun disc + a bright atmospheric corona right around it. The ghosts/streaks are the
        // verbatim lensFlare() call applied post-merge below, but its f0 term is faint and patchy
        // close to the disc — without this bridge the hard disc reads as "punched out" with a dark
        // ring before the flare glow starts. This part IS atmospheric scatter (cloud/earth
        // occludable), so it stays in the sky branch; lensFlare (a lens artifact) overlays all.
        float cosSun   = dot(dir, normalize(sunDirENU.xyz));
        float sunAbove = smoothstep(limbZ - 0.05, limbZ + 0.02, sunSinEl);
        float g = max(cosSun, 0.0);
        color += vec3(1.0, 0.96, 0.88) * smoothstep(0.99988, 0.99997, cosSun);          // disc
        color += vec3(1.0, 0.94, 0.82) * (pow(g, 1800.0) * 0.85 + pow(g, 320.0) * 0.13
                                          + pow(g, 55.0) * 0.025) * sunAbove;            // corona bridge

        // ── Textured moon disc ───────────────────────────────────────────────
        // Ray/sphere against the enlarged lunar disc; orthographic UV on the near face; sun-angle
        // phase terminator + sqrt limb darkening + a faint earthshine floor. Stripped from
        // sat_sky.frag: no refraction squish, no parallactic rotation beyond the fixed 180° flip
        // (kMoonTexRotDeg was 180 → a UV negate, no trig). Face-frame "up" = ECEF north in ENU.
        //
        // Daytime disappearance is CONTRAST, not opacity: the disc's own radiance is dimmed by the
        // atmospheric column (viewTrans) and ADDED on top of the sky's in-scattered airlight
        // (already in `color`). A bright gibbous/full moon still stands out against a blue sky; a
        // thin crescent or new moon washes into it. No day/night hack.
        {
            vec3  m3   = normalize(moonDirENU.xyz);
            float cosM = dot(dir, m3);
            float cov  = smoothstep(cos(kMoonAngR * 1.15), cos(kMoonAngR * 0.98), cosM);
            if (cov > 0.0) {
                float dm = max(0.0, cosM * cosM - (1.0 - kMoonAngR * kMoonAngR));
                vec3  n  = normalize((cosM - sqrt(dm)) * dir - m3);   // normal on near hemisphere
                vec3  mz = -m3;
                vec3  mx = normalize(cross(vec3(enuX.z, enuY.z, enuZ.z), mz));
                vec3  my = cross(mz, mx);
                vec2  mUV = 0.5 - vec2(dot(n, mx), dot(n, my)) * 0.5;  // *0.5+0.5 then 180° flip
                float lit  = max(0.0, dot(n, normalize(sunDirENU.xyz)));
                float limb = 0.35 + 0.65 * sqrt(max(0.0, dot(n, mz)));
                vec3  moonSurf = texture(moonTex, mUV).rgb * (lit * limb + 0.0015) * kMoonRadiance;
                color += moonSurf * viewTrans * cov;
            }
        }
    }

    // Flat cloud deck — composited last so it occludes both the ground and the sky/discs.
    color = flatClouds(color, obsPos, dir, tGround, enuX, enuY, enuZ, sunDirECEF);

    // ── Sun lens flare — VERBATIM sat_sky.frag lensFlare() (1:1 port test) ────────────
    // Applied post-merge so it overlays the Earth + clouds as a pure camera-optics artifact,
    // exactly as sat_sky.frag's "Camera lens flares (post-tonemap)" block does. Projection math
    // copied from that block; cloud/terrain occlusion of the source omitted (no screen-space
    // depth/cloud textures in this shader).
    if (sunDirENU.w > limbZ - 0.05) {
        float tanHF     = tan(pc.fovYRad * 0.5);
        float invTanHF2 = 1.0 / (tanHF * 2.0);
        vec3  fragCamDir = mat3(pc.skyView) * enuDir;
        vec2  fragUV     = vec2(fragCamDir.x, -fragCamDir.y) * invTanHF2;
        vec3  sunCam     = mat3(pc.skyView) * normalize(sunDirENU.xyz);
        if (sunCam.z < -0.01) {
            float above   = sunDirENU.w - limbZ;
            vec2  sunUV    = vec2(sunCam.x, -sunCam.y) / (-sunCam.z * tanHF * 2.0);
            float sunFade  = clamp(above * 8.0, 0.0, 1.0);
            color += lensFlare(fragUV, sunUV, 10.0 * clamp(above / 0.5, 0.0, 1.0), 2.0)
                     * vec3(1.4, 1.2, 0.9) * sunFade * 0.45;
        }
    }

    outColor = vec4(color, 1.0);

    // Match sat_sky.frag: surface hits within 150 km write [0, 0.5) so the point passes are
    // occluded; everything else writes 1.0 so they pass.
    const float kOcclusionCap = 150000.0;
    gl_FragDepth = (tOcclude >= 0.0 && tOcclude < kOcclusionCap)
                   ? tOcclude / (kOcclusionCap * 2.0)
                   : 1.0;
}
