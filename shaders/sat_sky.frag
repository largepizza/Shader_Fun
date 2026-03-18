#version 450

layout(location = 0) in  vec3 enuDir;
layout(location = 1) in flat vec4 sunDirENU;  // xyz = sun dir in ENU, w = sin(elevation)
layout(location = 2) in flat vec4 moonDirENU; // xyz = moon dir in ENU, w = illuminated fraction

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
// Auto-exposure: noon sky is orders of magnitude brighter than twilight.
// A fixed exposure either blows out the day or loses the night.
// Exposure is derived analytically from sun elevation each frame.
const float EXPOSURE_DAY   =  1.8;   // sun at zenith — prevents white washout
const float EXPOSURE_NIGHT = 10.0;   // below horizon — amplifies dim twilight glow

// ── Ray march quality ──────────────────────────────────────────────────────────
const int N_VIEW  = 12;   // view ray samples
const int N_LIGHT = 4;    // sun-direction samples per view sample

// Rayleigh phase function (symmetric, peaks at 0° and 180°)
float phaseR(float cosA) {
    return 0.75 * (1.0 + cosA * cosA);
}

// Modified Henyey-Greenstein Mie phase (forward-scattering peak at sun)
float phaseM(float cosA) {
    float g2  = G_MIE * G_MIE;
    float den = pow(max(1e-4, 1.0 + g2 - 2.0 * G_MIE * cosA), 1.5);
    return 1.5 * ((1.0 - g2) / (2.0 + g2)) * (1.0 + cosA * cosA) / den;
}

// Ray-sphere intersection (sphere centred at origin, radius r).
// Returns (t_near, t_far). vec2(-1) if no intersection.
vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b  = dot(ro, rd);
    float c  = dot(ro, ro) - r * r;
    float d  = b * b - c;
    if (d < 0.0) return vec2(-1.0);
    float sq = sqrt(d);
    return vec2(-b - sq, -b + sq);
}

// Optical depth (Rayleigh .x, Mie .y) from p along d over segTotal metres.
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

void main() {
    vec3 dir    = normalize(enuDir);
    vec3 sunDir = normalize(sunDirENU.xyz);

    // Observer 1 m above Earth's surface (avoids self-intersection in sphere math).
    // In ENU, (0,0,1) is the local zenith; Earth centre is (0,0,-R_EARTH).
    vec3 obsPos = vec3(0.0, 0.0, R_EARTH + 1.0);

    // ── Ray extent ─────────────────────────────────────────────────────────────
    // Always march to atmosphere exit — avoids a hard discontinuity at the horizon.
    // Ground rays get the same atmospheric path length; ground colour is blended
    // in afterwards using a smooth elevation ramp.
    vec2  tAtmos = raySphere(obsPos, dir, R_ATMOS);
    float tEnd   = tAtmos.y;

    // ── Atmospheric in-scatter ray march ───────────────────────────────────────
    float segLen = tEnd / float(N_VIEW);
    float cosA   = dot(dir, sunDir);
    float pR     = phaseR(cosA);
    float pM     = phaseM(cosA);

    vec3  accumR  = vec3(0.0);
    float accumM  = 0.0;
    float odR_cam = 0.0;
    float odM_cam = 0.0;

    for (int i = 0; i < N_VIEW; ++i) {
        // Clamp sample altitude — below-horizon samples are pushed to the surface
        // so the march stays physically inside the atmosphere.
        vec3  sp  = obsPos + dir * ((float(i) + 0.5) * segLen);
        float len = length(sp);
        if (len < R_EARTH) sp *= R_EARTH / len;   // project onto surface
        float h = max(0.0, length(sp) - R_EARTH);

        float densR = exp(-h / H_R) * segLen;
        float densM = exp(-h / H_M) * segLen;
        odR_cam += densR;
        odM_cam += densM;

        // Skip if Earth blocks the sun from this sample.
        vec2 tSunEarth = raySphere(sp, sunDir, R_EARTH);
        if (tSunEarth.x > 0.0 && tSunEarth.y > 0.0) continue;

        // Optical depth from sample toward the sun through remaining atmosphere.
        vec2 tSun  = raySphere(sp, sunDir, R_ATMOS);
        vec2 sunOD = (tSun.y > 0.0) ? optDepth(sp, sunDir, tSun.y) : vec2(0.0);

        // Extinction along view path + sun path (Rayleigh + Mie).
        // Mie extinction ≈ 1.1 × scattering coefficient (aerosol absorption).
        vec3 tau  = BETA_R       * (odR_cam + sunOD.x)
                  + BETA_M * 1.1 * (odM_cam + sunOD.y);
        vec3 attn = exp(-tau);

        accumR += attn * densR;
        accumM += dot(attn, vec3(1.0 / 3.0)) * densM;
    }

    // ── Assemble in-scatter colour ─────────────────────────────────────────────
    vec3 color = SUN_INTENSITY * (pR * BETA_R * accumR + vec3(pM * BETA_M * accumM));

    // ── Moon disc (background emitter, attenuated by atmosphere) ─────────────
    // The moon lives behind the atmosphere just like a star, so it gets the same
    // Rayleigh/Mie extinction.  By adding its contribution to the HDR color here
    // (before tone-map) the exposure system automatically makes it invisible
    // during daytime (bright in-scatter dominates) and visible at night.
    //
    // Angular radius of Moon ≈ 0.2622° = 0.004578 rad.
    // kMoonBright calibrated so full moon at zenith tone-maps to ~0.7 at night.
    const float kMoonAngR   = 0.004578 * 3.0; // 3× to make the moon disc more visible at low resolution
    const float kMoonBright = 0.12;

    if (moonDirENU.z > -kMoonAngR * 2.0) {  // moon at or above horizon
        vec3  moonDir3 = normalize(moonDirENU.xyz);
        // Ray-sphere intersection in ENU unit space.
        // Sphere: centre = moonDir3, radius = kMoonAngR.
        vec3  oc     = -moonDir3;           // origin (observer) to sphere centre
        float bm     = dot(oc, dir);
        float cm     = 1.0 - kMoonAngR * kMoonAngR;  // dot(oc,oc) = 1
        float discm  = bm * bm - cm;
        if (discm >= 0.0) {
            vec3  hp = (-bm - sqrt(discm)) * dir;
            vec3  n  = normalize(hp - moonDir3);   // outward surface normal

            // Lambertian diffuse — moonDirENU.w = illuminated fraction.
            float diffuse = max(0.0, dot(n, sunDir)) * moonDirENU.w;

            // Earthshine: faint fill on the dark limb.
            const float kEarth = 0.018 * 0.2;

            // Limb darkening: mu = cosine between normal and toward-camera direction.
            float mu       = max(0.0, dot(n, -moonDir3));
            float limbDark = 0.35 + 0.65 * sqrt(mu);

            const vec3 kAlbedo = vec3(0.88, 0.85, 0.82);
            vec3 moonColor = kAlbedo * (diffuse + kEarth) * limbDark * kMoonBright;

            // Atmospheric extinction: identical treatment to background starlight.
            vec3 moonAttn = exp(-(BETA_R * odR_cam + BETA_M * 1.1 * odM_cam));
            color += moonColor * moonAttn;
        }
    }

    // ── Ground: blend in surface colour below horizon ─────────────────────────
    // dir.z = sin(elevation). Smooth ramp over ±3° straddles the mathematical
    // horizon so the blend is seamless — no hard switch, no seam.
    float skyBlend  = smoothstep(-0.05, 0.03, dir.z);  // 0 = ground, 1 = sky
    if (skyBlend < 1.0) {
        float daylight = clamp(sunDirENU.w * 4.0 + 0.3, 0.0, 1.0);
        vec3  ground   = vec3(0.035, 0.028, 0.022) * daylight;
        color = mix(ground, color, skyBlend);
    }

    // ── Analytical auto-exposure tone mapping ──────────────────────────────────
    // Map sun elevation [-0.1, 1.0] → dayness [0, 1], then interpolate exposure
    // on a power curve so the transition compresses near sunset, not midday.
    // Extend twilight range: sun must be 11.5° below horizon (w=-0.2) before full night.
    float dayness  = clamp((sunDirENU.w + 0.2) / 1.2, 0.0, 1.0);
    float exposure = mix(EXPOSURE_NIGHT, EXPOSURE_DAY, pow(dayness, 0.4));
    color = vec3(1.0) - exp(-exposure * color);

    // ── Night ambient floor ─────────────────────────────────────────────────────
    // Base airglow keeps the sky from going pitch-black after astronomical twilight.
    float nightAmt = 1.0 - clamp(dayness * 5.0, 0.0, 1.0);
    color += vec3(0.0008, 0.001, 0.002) * nightAmt;

    // ── Moonlight: brighter night sky scaled by moon phase and altitude ─────────
    // moonDirENU.z = sin(moon elevation); moonDirENU.w = illuminated fraction.
    float moonEl    = clamp(moonDirENU.z, 0.0, 1.0);
    float moonIllum = moonDirENU.w;
    color += vec3(0.0025, 0.003, 0.004) * moonIllum * moonEl * nightAmt;

    // ── Soft atmospheric halo around the moon (visible through thin cloud/haze) ─
    if (moonDirENU.z > -0.05) {
        vec3  moonDir3  = normalize(moonDirENU.xyz);
        float moonCosA  = dot(dir, moonDir3);
        float moonAngle = acos(clamp(moonCosA, -1.0, 1.0));
        // Tight inner glow — shows as a pale ring around the moon disc.
        float scale     = 100.0;
        float halo      = exp(-moonAngle * moonAngle / (2.0 * 0.018 * 0.018 * scale * scale)); //
        float moonFade  = smoothstep(-0.05, 0.02, moonDirENU.z);
        float hClip     = smoothstep(-0.02, 0.03, dir.z);
        vec3  haloCol   = vec3(0.88, 0.90, 1.00) * moonIllum * 0.012;
        color += hClip * moonFade * halo * haloCol;
    }

    // ── Sun disc + corona (added post-tonemap so it stays visibly bright) ──────
    if (sunDirENU.w > -0.1) {
        float angle     = acos(clamp(cosA, -1.0, 1.0));
        float disc      = 1.0 - smoothstep(0.007, 0.010, angle);
        float corona    = exp(-angle * angle / (2.0 * 0.035 * 0.035));
        float fade      = smoothstep(-0.12, 0.02, sunDirENU.w);
        // Smooth horizon clip: fades out as the sun disc dips below the ground plane.
        float horizClip = smoothstep(-0.02, 0.03, dir.z);
        vec3  sunCol    = vec3(1.5, 1.3, 1.0);
        color += horizClip * fade * (disc * sunCol + corona * sunCol * 0.12);
    }

    outColor = vec4(color, 1.0);
}
