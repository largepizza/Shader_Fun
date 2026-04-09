#version 450

// ── Camera + sun push constants (same layout as C++ SatDrawPC, 112 bytes) ─────
// The pipeline layout declares VK_SHADER_STAGE_VERTEX_BIT|FRAGMENT_BIT so both
// stages share one push constant range.  The fragment uses skyView/fovYRad to
// project glowBuf ENU directions into screen UV for the lens flare pass.
layout(push_constant) uniform PC {
    mat4  skyView;    // ENU -> camera space (rotation, no translation)
    float fovYRad;    // vertical field of view in radians
    float aspect;     // viewport width / height
    float pad[2];
    vec4  sunDirENU;  // xyz = sun dir in ENU, w = sin(sun elevation)
    vec4  moonDirENU; // xyz = moon dir in ENU, w = illuminated fraction
} pc;

layout(location = 0) in  vec3 enuDir;           // interpolated ENU view ray (not normalised)
layout(location = 1) in flat vec4 sunDirENU;    // passed through from vertex (same as pc.sunDirENU)
layout(location = 2) in flat vec4 moonDirENU;   // moon dir + phase pass-through

// Top-N bright satellite flares -- written by CPU each frame, no smoothing.
// kMaxGlows must match the constant in SatelliteSim.h.
layout(std430, set = 0, binding = 0) readonly buffer GlowBuf {
    int  count;
    vec4 entries[128]; // xyz = ENU unit dir, w = effectFlare intensity
} glowBuf;

// RGBA noise texture (binding 1): tiled REPEAT sampler, used for angular corona
// variation in lensFlare().  Replaces the original ShaderToy's iChannel0 lookup.
layout(set = 0, binding = 1) uniform sampler2D noiseTex;

// Moon surface texture (binding 2): near-side face disc image.
// Sampled with an orthographic projection of the surface normal onto the moon's
// local face frame — maps the near hemisphere to the full [0,1] UV range.
layout(set = 0, binding = 2) uniform sampler2D moonTex;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// ── Atmosphere geometry (meters) ───────────────────────────────────────────────
const float R_EARTH = 6371000.0;
const float R_ATMOS = 6471000.0;   // 100 km above surface

// ── Rayleigh scattering (wavelength-dependent: R=650nm, G=510nm, B=440nm) ─────
const vec3  BETA_R = vec3(5.8e-6, 13.5e-6, 33.1e-6);  // 1/m, sea level
const float H_R    = 7994.0;   // Rayleigh scale height (m)

// ── Mie scattering (aerosols, wavelength-independent) ─────────────────────────
const float BETA_M = 2.1e-5;   // 1/m, sea level
const float H_M    = 1200.0;   // Mie scale height (m)
const float G_MIE  = 0.76;     // forward-scatter asymmetry (higher = sharper corona)

// ── Lighting / tone mapping ────────────────────────────────────────────────────
const float SUN_INTENSITY  = 1.0;
const float EXPOSURE_DAY   =  1.8;   // sun at zenith -- prevents white washout
const float EXPOSURE_NIGHT = 10.0;   // below horizon -- amplifies dim twilight glow

// ── Ray march quality ──────────────────────────────────────────────────────────
const int N_VIEW  = 12;   // view ray samples
const int N_LIGHT = 4;    // sun-direction samples per view sample

float phaseR(float cosA) {
    return 0.75 * (1.0 + cosA * cosA);
}
float phaseM(float cosA) {
    float g2  = G_MIE * G_MIE;
    float den = pow(max(1e-4, 1.0 + g2 - 2.0 * G_MIE * cosA), 1.5);
    return 1.5 * ((1.0 - g2) / (2.0 + g2)) * (1.0 + cosA * cosA) / den;
}
vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b  = dot(ro, rd);
    float c  = dot(ro, ro) - r * r;
    float d  = b * b - c;
    if (d < 0.0) return vec2(-1.0);
    float sq = sqrt(d);
    return vec2(-b - sq, -b + sq);
}
vec2 optDepth(vec3 p, vec3 d, float segTotal) {
    float sLen = segTotal / float(N_LIGHT);
    float odR = 0.0, odM = 0.0;
    for (int i = 0; i < N_LIGHT; ++i) {
        float h = max(0.0, length(p + d * (float(i) + 0.5) * sLen) - R_EARTH);
        odR += exp(-h / H_R);
        odM += exp(-h / H_M);
    }
    return vec2(odR, odM) * sLen;
}

// ── Lens flare (adapted from "Lens Flare Example" by peterekepeter, public domain)
// ─────────────────────────────────────────────────────────────────────────────
// Produces the visible corona/bloom around the source AND the reflected ghost
// artifacts that appear along the flare axis (source -> screen centre -> beyond).
// Diffraction spikes are intentionally omitted; instead the irregular corona
// shape (human-eye / dirty-lens airy-disk pattern) is produced entirely by the
// noise texture lookup on f0.
//
// Coordinate space: ShaderToy-style UV.
//   x in [-0.5*aspect, +0.5*aspect],  y in [-0.5, +0.5].
//
// Parameters:
//   uv     -- current fragment position in flare UV space
//   pos    -- source (satellite / sun) position in flare UV space
//   intens -- normalised brightness [0,1]; controls f0 scale and ghost strength
//
// Returns an HDR additive RGB contribution.
// The call site multiplies by a tint and an overall scale factor.
// ─────────────────────────────────────────────────────────────────────────────
// bokehMult: independent brightness scalar for the ghost/bokeh elements (f2–f6).
// Use a small value (e.g. 0.3) for satellites, larger (e.g. 2.0) for the sun.
// Separates corona brightness (intens) from artifact brightness (bokehMult).
vec3 lensFlare(vec2 uv, vec2 pos, float intens, float bokehMult) {

    // uvd: radially distorted UV -- uv * |uv|.
    // Near screen centre uvd ~= 0; toward edges it bends outward.
    // Ghost artifacts use uvd so their positions follow the curved optical path
    // of real multi-element lens reflections.
    vec2 uvd = uv * length(uv);

    // d: displacement from current fragment to source.
    vec2 d = uv - pos;

    // dist: radius^0.1 -- nearly 1.0 everywhere, dips to 0 right at the source.
    // Used as a small radial term in f0's shimmer modulation.
    float dist = pow(length(d), 0.1);

    // ang: polar angle [-pi, +pi] around the source.
    // Used to sample the noise texture angularly so the corona has irregular lobes.
    float ang = atan(d.y, d.x);

    // ── Angular corona noise via texture lookup ────────────────────────────────
    // Replicates the original ShaderToy formula:
    //   noise(sin(ang*4 + pos.x)*4 - cos(ang*3 + pos.y))
    //
    // The argument is a smoothly-varying scalar that changes both with the angle
    // around the source (ang) and with the source's screen position (pos.x, pos.y).
    // This means each satellite at a different screen position has a unique corona
    // shape -- the lobes don't align between adjacent satellites.
    //
    // Mapping the scalar to a UV coordinate for noiseTex:
    //   We use a 1D slice along the texture's x-axis (v = 0.5, middle row).
    //   The u coordinate wraps via the REPEAT sampler so any float value is valid.
    //   The noise value (red channel) is then passed into sin(...*16)*0.1 which
    //   creates fine angular variation (+/-10%) around the corona rim.
    // noiseSeed is a smoothly-varying float that changes with angle and source position.
    // We map it into [0,1] UV space by dividing by the expected range (~8) and adding
    // 0.5 to centre it, then rely on REPEAT wrapping for values outside [0,1].
    // Using fract() explicitly makes the wrapping behaviour unambiguous.
    // The v coordinate is fixed at 0.25 (upper quarter of texture, away from the
    // edge to avoid any border artifacts on some hardware).
    float noiseSeed = sin(ang * 4.0 + pos.x) * 4.0 - cos(ang * 3.0 + pos.y);
    float noiseU    = fract(noiseSeed * 0.125 + 0.5); // map [-8,+8] -> [0,1], wrapping
    float angNoise  = texture(noiseTex, vec2(noiseU, 0.25)).r;

    // ── Source glow: Lorentzian corona centered on the source ─────────────────
    // The Lorentzian  1/(r * scale + 1)  is wider and softer than a Gaussian,
    // matching real lens-coating scatter on a bright point source.
    //
    float scale = 1200.0; // corona radius: higher = tighter. 60 = wide (visible at 200px), 1200 = tight (visible at ~15px)
    //   r = 0.005 (~5px at 1080p):  f0 = 1/(0.005*60+1) = 0.77
    //   r = 0.02  (~22px):          f0 = 1/(0.02 *60+1) = 0.45
    //   r = 0.05  (~54px):          f0 = 1/(0.05 *60+1) = 0.25
    //   r = 0.10  (~108px):         f0 = 1/(0.10 *60+1) = 0.14
    //   r = 0.20  (~216px):         f0 = 1/(0.20 *60+1) = 0.077
    // This gives a wide, visible corona that extends well past the satellite dot
    // and fades naturally without a hard edge.  The old scale of 200 fell to
    // <0.05 at only 50px, making the corona invisible at our additive blend scale.
    //
    // The modulation line applies the noise-driven angular shimmer:
    //   sin(angNoise * 16) * 0.1  -- fine ripple from texture (+/- 10% per lobe)
    //   dist * 0.1                -- barely-there radial taper (~constant ~1)
    //   + 0.8                     -- base boost so the corona is always bright
    // sin(noise*16) oscillates rapidly around the corona, creating 8-16 irregular
    // bright lobes -- the airy-disk / human-eye diffraction pattern.
    float f0 = 1.0 / (length(d) * scale + 1.0);
    f0 = f0 + f0 * (sin(angNoise * 16.0) * 20.8 + dist);
    // Scale by intensity so dimmer satellites have a proportionally smaller corona.
    f0 *= 0.1;// + intens * 0.5);

    // ── Large near-source bloom: soft blob mirrored through screen centre ──────
    // Placed at -1.2*pos (reflected slightly beyond centre).
    // Represents light that bounced backward through the lens and re-emerged near
    // the entrance pupil.  Multiplier 4.0 (reduced from original 7.0) and
    // contribution capped below to prevent peripheral over-saturation.
    float f1 = max(0.01 - pow(length(uv + 1.2 * pos), 1.9), 0.0) * 4.0;
    f1 *= 0.6;

    // ── Ghost artifacts: fade when source is near screen centre ───────────────
    // When pos ~= (0,0) (looking directly at the source), uvd + k*pos ~= uvd,
    // which is nearly zero everywhere near centre.  The Lorentzian denominator
    // (1 + 32*r^2) then approaches 1 everywhere, lighting up the entire screen.
    //
    // ghostFade = smoothstep(0.03, 0.12, |pos|):
    //   source within ~3% screen height of centre: ghosts = 0
    //   source more than 12% screen height off-centre: ghosts full
    // This also makes physical sense: looking directly at the source means ghost
    // reflection paths don't form visible off-axis elements.
    float ghostFade = smoothstep(0.03, 0.12, length(pos));

    // ── Bokeh halos: large circular rings reflected through screen centre ──────
    // Classic rainbow-ringed bokeh circles opposite the source.
    // Lorentzian  1/(1 + 32*r^2)  matches wide, soft real ghost disc profiles.
    // Three slightly offset RGB positions produce chromatic aberration fringing.

    float f2  = max(1.0/(1.0 + 32.0*pow(length(uvd + 0.80*pos), 2.0)), 0.0) * 0.25 * bokehMult;
    float f22 = max(1.0/(1.0 + 32.0*pow(length(uvd + 0.85*pos), 2.0)), 0.0) * 0.23 * bokehMult;
    float f23 = max(1.0/(1.0 + 32.0*pow(length(uvd + 0.90*pos), 2.0)), 0.0) * 0.21 * bokehMult;

    // ── Star-shaped secondary bokeh (between source and centre) ───────────────
    // uvx = 1.5*uv - 0.5*uvd.  The 2.4 exponent gives a slightly star-shaped
    // profile (intermediate between circle and square).
    // RGB variants at 0.40/0.45/0.50*pos create a second tier of chromatic split.
    vec2 uvx = mix(uv, uvd, -0.5);
    float f4  = max(0.01 - pow(length(uvx + 0.40*pos), 2.4), 0.0) * 6.0;
    float f42 = max(0.01 - pow(length(uvx + 0.45*pos), 2.4), 0.0) * 5.0;
    float f43 = max(0.01 - pow(length(uvx + 0.50*pos), 2.4), 0.0) * 3.0;

    // ── Compact sparkle dots along the flare axis ─────────────────────────────
    // High exponent (5.5) = sharp dropoff = tight bright pinpoints at 0.2/0.4/0.6*pos.
    uvx = mix(uv, uvd, -0.4);
    float f5  = max(0.01 - pow(length(uvx + 0.20*pos), 5.5), 0.0) * 2.0;
    float f52 = max(0.01 - pow(length(uvx + 0.40*pos), 5.5), 0.0) * 2.0;
    float f53 = max(0.01 - pow(length(uvx + 0.60*pos), 5.5), 0.0) * 2.0;

    // ── Broad streaks on the camera-side of centre ────────────────────────────
    // Negative multiplier places these between centre and the source.
    // Low exponent (1.6) = broad, diffuse -- reads as a smear on the front element.
    uvx = mix(uv, uvd, -0.5);
    float f6  = max(0.01 - pow(length(uvx - 0.300*pos), 1.6), 0.0) * 6.0;
    float f62 = max(0.01 - pow(length(uvx - 0.325*pos), 1.6), 0.0) * 3.0;
    float f63 = max(0.01 - pow(length(uvx - 0.350*pos), 1.6), 0.0) * 5.0;

    // ── Assemble ──────────────────────────────────────────────────────────────
    vec3 c = vec3(0.0);

    // Source corona -- achromatic (warm white set by call-site tint).
    c += vec3(f0);
    c += vec3(f1 * 0.5);  // bloom at -1.2*pos

    // Ghost terms: chromatic, gated by ghostFade to prevent centre blowout.
    // bokehMult independently scales all ghost/reflection artifacts from the corona (f0).
    c.r += (f2  + f4  + f5  + f6)  * 0.4 * ghostFade * bokehMult;
    c.g += (f22 + f42 + f52 + f62) * 0.4 * ghostFade * bokehMult;
    c.b += (f23 + f43 + f53 + f63) * 0.4 * ghostFade * bokehMult;

    // Slight vignette: outer screen positions have more lens distortion.
    c = c * 1.3 - vec3(length(uvd) * 0.05);

    return max(c, vec3(0.0));
}

void main() {
    vec3 dir    = normalize(enuDir);
    vec3 sunDir = normalize(sunDirENU.xyz);

    vec3 obsPos = vec3(0.0, 0.0, R_EARTH + 1.0);

    vec2  tAtmos = raySphere(obsPos, dir, R_ATMOS);
    float tEnd   = tAtmos.y;

    float segLen = tEnd / float(N_VIEW);
    float cosA   = dot(dir, sunDir);
    float pR     = phaseR(cosA);
    float pM     = phaseM(cosA);

    vec3  accumR  = vec3(0.0);
    float accumM  = 0.0;
    float odR_cam = 0.0;
    float odM_cam = 0.0;

    for (int i = 0; i < N_VIEW; ++i) {
        vec3  sp  = obsPos + dir * ((float(i) + 0.5) * segLen);
        float len = length(sp);
        if (len < R_EARTH) sp *= R_EARTH / len;
        float h = max(0.0, length(sp) - R_EARTH);

        float densR = exp(-h / H_R) * segLen;
        float densM = exp(-h / H_M) * segLen;
        odR_cam += densR;
        odM_cam += densM;

        vec2 tSunEarth = raySphere(sp, sunDir, R_EARTH);
        if (tSunEarth.x > 0.0 && tSunEarth.y > 0.0) continue;

        vec2 tSun  = raySphere(sp, sunDir, R_ATMOS);
        vec2 sunOD = (tSun.y > 0.0) ? optDepth(sp, sunDir, tSun.y) : vec2(0.0);

        vec3 tau  = BETA_R       * (odR_cam + sunOD.x)
                  + BETA_M * 1.1 * (odM_cam + sunOD.y);
        vec3 attn = exp(-tau);

        accumR += attn * densR;
        accumM += dot(attn, vec3(1.0 / 3.0)) * densM;
    }

    vec3 color = SUN_INTENSITY * (pR * BETA_R * accumR + vec3(pM * BETA_M * accumM));

    // ── Moon disc ─────────────────────────────────────────────────────────────
    // kMoonTexRotDeg: rotates the texture CW in the UV plane to align the image's
    // north pole with the physical lunar north pole as seen from the observer.
    // Tune this until the terminator's shadow boundary matches the image poles.
    const float kMoonTexRotDeg = 180.0;
    const float kMoonAngR      = 0.004578 * 3.0;
    const float kMoonBright    = 0.54;
    if (moonDirENU.z > -kMoonAngR * 2.0) {
        vec3  moonDir3 = normalize(moonDirENU.xyz);

        // ── Atmospheric refraction squish ─────────────────────────────────────
        // Near the horizon, differential refraction lifts the bottom limb more
        // than the top, compressing the apparent disc height.  The Bennett formula
        // gives refraction R(el) in arcminutes; the squish fraction is the
        // difference in R across the disc diameter, divided by the disc diameter.
        float squish = 0.0;
        float elDeg  = degrees(asin(clamp(moonDirENU.z, -1.0, 1.0)));
        if (elDeg < 15.0) {
            float r   = degrees(kMoonAngR);             // disc angular radius, degrees
            float elo = max(elDeg - r, 0.2);            // lower limb elevation (clamped off ground)
            float ehi = elDeg + r;                      // upper limb elevation
            float Rlo = 1.02 / tan(radians(elo + 10.3 / (elo + 5.11))); // arcmin
            float Rhi = 1.02 / tan(radians(ehi + 10.3 / (ehi + 5.11)));
            squish = clamp((Rlo - Rhi) / (2.0 * r * 60.0), 0.0, 0.5);
        }
        // Stretching dir.z before intersection maps screen pixels into a
        // vertically compressed disc-space — the silhouette becomes a physical
        // ellipse (shorter in elevation) matching the naked-eye refraction effect.
        vec3  dirR  = normalize(vec3(dir.xy, dir.z * (1.0 + squish)));

        vec3  oc    = -moonDir3;
        float bm    = dot(oc, dirR);
        float cm    = 1.0 - kMoonAngR * kMoonAngR;
        float discm = bm * bm - cm;
        if (discm >= 0.0) {
            vec3  hp = (-bm - sqrt(discm)) * dirR;
            vec3  n  = normalize(hp - moonDir3);
            float diffuse  = max(0.0, dot(n, sunDir)) * moonDirENU.w;
            float mu       = max(0.0, dot(n, -moonDir3));
            float limbDark = 0.35 + 0.65 * sqrt(mu);
            const float kEarth = 0.018 * 0.2;

            // Build the moon's local face frame: moonZ points toward the observer
            // (tidally locked near side), moonX/moonY span the visible face plane.
            vec3 moonZ = -moonDir3;
            vec3 refUp = abs(moonZ.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 moonX = normalize(cross(refUp, moonZ));
            vec3 moonY = cross(moonZ, moonX);

            // Orthographic projection of the surface normal onto the face plane.
            // At the disc centre n == moonZ → UV (0.5, 0.5); at the limb UV spans [0,1].
            vec2 moonUV = vec2(dot(n, moonX), dot(n, moonY)) * 0.5 + 0.5;

            // Rotate UV around disc centre by kMoonTexRotDeg to align image north pole
            // with the physical lunar north pole. Positive = CCW rotation of the texture.
            float rotRad = radians(kMoonTexRotDeg);
            float cosR = cos(rotRad), sinR = sin(rotRad);
            vec2  uvc  = moonUV - 0.5;
            moonUV = vec2(cosR * uvc.x - sinR * uvc.y,
                          sinR * uvc.x + cosR * uvc.y) + 0.5;

            vec3 texColor = texture(moonTex, moonUV).rgb;

            vec3 moonColor = texColor * (diffuse + kEarth) * limbDark * kMoonBright;
            vec3 moonAttn  = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));
            color += moonColor * moonAttn;
        }
    }

    // ── Satellite constellation sky glow (pre-tonemap) ────────────────────────
    // Wide Gaussian (kSig = 0.90 rad ~= 51 deg) summed over all glowBuf entries.
    // This is aggregate light-pollution glow from whole constellations, NOT
    // per-satellite bloom.  Many satellites together produce a diffuse brightening
    // of the sky dome above them -- analogous to urban skyglow.
    // Runs pre-tonemap so the exposure system scales it: invisible at noon,
    // visible at dusk, prominent at night.
    if (glowBuf.count > 0) {
        vec3  flareAttn = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));
        float hClip     = smoothstep(-0.02, 0.03, dir.z);
        const float kSig = 0.90; // wide sigma -- aggregate constellation glow, not per-sat
        for (int gi = 0; gi < glowBuf.count; ++gi) {
            vec4  e      = glowBuf.entries[gi];
            if (e.z < -0.08) continue;
            vec3  fd     = normalize(e.xyz);
            float angle  = acos(clamp(dot(dir, fd), -1.0, 1.0));
            float glow   = exp(-angle * angle / (2.0 * kSig * kSig)) * 0.01;
            float gElev  = smoothstep(-0.08, 0.02, e.z);
            float intens = clamp(log2(max(e.w, 1.0)) / 4.0, 0.0, 1.5);
            color += hClip * gElev * glow * intens * 0.06 * vec3(1.0, 0.96, 0.88) * flareAttn;
        }
    }

    // ── Ground blend ──────────────────────────────────────────────────────────
    float skyBlend = smoothstep(-0.05, 0.03, dir.z);
    if (skyBlend < 1.0) {
        float daylight = clamp(sunDirENU.w * 4.0 + 0.3, 0.0, 1.0);
        vec3  ground   = vec3(0.035, 0.028, 0.022) * daylight;
        color = mix(ground, color, skyBlend);
    }

    // ── Auto-exposure tone mapping ─────────────────────────────────────────────
    float dayness  = clamp((sunDirENU.w + 0.2) / 1.2, 0.0, 1.0);
    float exposure = mix(EXPOSURE_NIGHT, EXPOSURE_DAY, pow(dayness, 0.4));
    color = vec3(1.0) - exp(-exposure * color);

    // ── Night ambient floor ────────────────────────────────────────────────────
    float nightAmt = 1.0 - clamp(dayness * 5.0, 0.0, 1.0);
    color += vec3(0.0008, 0.001, 0.002) * nightAmt;

    // ── Moonlight ambient ──────────────────────────────────────────────────────
    float moonEl    = clamp(moonDirENU.z, 0.0, 1.0);
    float moonIllum = moonDirENU.w;
    color += vec3(0.0025, 0.003, 0.004) * moonIllum * moonEl * nightAmt;

    // ── Moon glow: tight corona + wide diffuse halo ───────────────────────────
    if (moonDirENU.z > -0.05) {
        vec3  moonDir3  = normalize(moonDirENU.xyz);
        float moonAngle = acos(clamp(dot(dir, moonDir3), -1.0, 1.0));
        float moonFade  = smoothstep(-0.05, 0.02, moonDirENU.z);
        float hClip     = smoothstep(-0.02, 0.03, dir.z);

        // Tight inner corona — peaks at the disc edge (~0.014 rad), falls to ~8% at 3× disc radius.
        // sigma = 0.012 rad ≈ 0.7°; gives a crisp bloom ring without polluting the wider sky.
        float corona = exp(-moonAngle * moonAngle / (2.0 * 0.012 * 0.012)) * nightAmt;
        color += hClip * moonFade * corona * vec3(0.92, 0.94, 1.00) * moonIllum * 0.04;

        // Wide diffuse halo — very broad Gaussian (sigma ≈ 1.8 rad) that lifts the whole
        // night sky slightly around the moon, matching the real scattered moonlight glow.
        float scale = 100.0;
        float halo  = exp(-moonAngle * moonAngle / (2.0 * 0.018 * 0.018 * scale * scale));
        color += hClip * moonFade * halo * vec3(0.88, 0.90, 1.00) * moonIllum * 0.012;
    }

    // ── Sun disc + atmospheric corona ─────────────────────────────────────────
    if (sunDirENU.w > -0.1) {
        float angle     = acos(clamp(cosA, -1.0, 1.0));
        float disc      = 1.0 - smoothstep(0.007, 0.010, angle);
        float corona    = exp(-angle * angle / (2.0 * 0.035 * 0.035));
        float fade      = smoothstep(-0.12, 0.02, sunDirENU.w);
        float horizClip = smoothstep(-0.02, 0.03, dir.z);
        vec3  sunCol    = vec3(1.5, 1.3, 1.0);
        color += horizClip * fade * (disc * sunCol + corona * sunCol * 0.12);
    }

    // ── Camera lens flares (post-tonemap) ─────────────────────────────────────
    // Applied after all physics-based rendering so they read as pure camera
    // optical artifacts on top of the scene.
    //
    // UV space: x in [-0.5*aspect, +0.5*aspect], y in [-0.5, +0.5].
    //
    // Fragment projection:
    //   fragCamDir = mat3(skyView) * enuDir  (camera-space ray, z ~= -1)
    //   fragUV = vec2(camDir.x, -camDir.y) * invTanHF2
    //   No perspective divide since z ~= -1 throughout the fullscreen tri.
    //
    // Source projection (satellite or sun):
    //   satCam = mat3(skyView) * normalize(enu)
    //   satUV  = vec2(satCam.x, -satCam.y) / (-satCam.z * tanHF * 2)
    //   Perspective divide by -satCam.z is required here.
    {
        float tanHF     = tan(pc.fovYRad * 0.5);
        float invTanHF2 = 1.0 / (tanHF * 2.0);

        vec3 fragCamDir = mat3(pc.skyView) * enuDir;
        vec2 fragUV     = vec2(fragCamDir.x, -fragCamDir.y) * invTanHF2;

        vec3 flareAccum = vec3(0.0);

        // ── Satellite flares ────────────────────────────────────────────────────
        // Threshold kFlareThreshold: below this effectFlare value the satellite is
        // too dim to generate visible lens artifacts.  The Gaussian sky-glow loop
        // above handles dim-to-medium satellites as aggregate light pollution.
        //
        // Intensity curve: log2(e.w) / log2(16) maps [1..16] -> [0..1], compressed
        // so a cluster of moderate-bright sats each contribute modestly while a
        // single very bright sat (ISS-class) gets close to full intensity.
        //
        // kFlareThreshold is intentionally LOW (1.0) so that satellites approaching
        // flare brightness fade in gradually rather than popping on all at once.
        // The per-entry entryScale (intens^1.5) means a satellite just above threshold
        // contributes almost nothing; only truly bright ones dominate.
        //
        // entryScale curve (intens = log2(e.w) / log2(16)):
        //   e.w =  1.0 (at threshold):     intens = 0.00,  entryScale = 0.000
        //   e.w =  1.5:                     intens = 0.14,  entryScale = 0.007
        //   e.w =  2.0:                     intens = 0.25,  entryScale = 0.031
        //   e.w =  4.0:                     intens = 0.50,  entryScale = 0.177
        //   e.w = 16.0 (ISS-class):         intens = 1.00,  entryScale = 1.000
        // This matches the gradual visibility increase of the old point-sprite spikes
        // while avoiding any hard threshold pop.
        const float kFlareThreshold = 1.0;
        for (int gi = 0; gi < glowBuf.count; ++gi) {
            vec4 e = glowBuf.entries[gi];
            if (e.z < -0.05) continue;
            if (e.w < kFlareThreshold) continue;

            vec3 satCam = mat3(pc.skyView) * normalize(e.xyz);
            if (satCam.z >= -0.01) continue;
            vec2 satUV = vec2(satCam.x, -satCam.y) / (-satCam.z * tanHF * 2.0);

            float intens = clamp(log2(max(e.w, 1.0)) / log2(16.0), 0.0, 1.0);
            // Smooth entry: intens^1.5 gives near-zero contribution just above threshold,
            // growing naturally to 1.0 at ISS-class brightness.
            float entryScale = intens * sqrt(intens); // = intens^1.5

            // Warm white tint; lensFlare() adds its own chromatic aberration on ghosts.
            vec3 tint = vec3(1.3, 1.15, 1.0);
            flareAccum += lensFlare(fragUV, satUV, intens, 0.3) * tint * entryScale * 0.35;
        }

        // ── Sun lens flare ──────────────────────────────────────────────────────
        // The sun is always at intens=1.0 when above the horizon.
        // A separate horizon fade ensures the flare vanishes as the sun sets.
        // The sun disc and atmospheric corona are handled above (physics-based);
        // this adds only the camera optical artifact layer on top.
        if (pc.sunDirENU.w > -0.01) {
            float sunIntensity = 10 * clamp(pc.sunDirENU.w, 0, 1.0);
            vec3 sunCam = mat3(pc.skyView) * normalize(pc.sunDirENU.xyz);
            if (sunCam.z < -0.01) {
                vec2 sunUV    = vec2(sunCam.x, -sunCam.y) / (-sunCam.z * tanHF * 2.0);
                float sunFade = clamp(pc.sunDirENU.w * 5.0 + 0.5, 0.0, 1.0);
                vec3  sunTint = vec3(1.4, 1.2, 0.9);
                flareAccum += lensFlare(fragUV, sunUV, sunIntensity, 2.0) * sunTint * sunFade * 0.45;
            }
        }

        color += flareAccum;
    }

    outColor = vec4(color, 1.0);
}
