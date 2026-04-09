#version 450

layout(push_constant) uniform PC {
    vec3  moonCamCenter;
    float moonRadius;
    float fovYRad;
    float aspect;
    float moonSinEl;
    float sunSinEl;       // sin(sun elevation) — drives daytime fade
    vec3  sunDirCam;
    float moonPhase;
    vec3  enuUpInCam;     // ENU "up" in camera space — for horizon clipping
    float pad2;
} pc;

layout(location = 0) in  vec2 fragNDC;
layout(location = 0) out vec4 outColor;

void main() {
    // ── Daytime fade ──────────────────────────────────────────────────────────
    // When the sun is above the horizon the sky is too bright for the moon to
    // show.  Fade smoothly: fully visible at nautical twilight, gone by ~15°.
    float dayFade = smoothstep(0.25, -0.05, pc.sunSinEl);
    if (dayFade <= 0.0) discard;

    // ── Reconstruct camera-space ray direction from NDC ───────────────────────
    float tanHalfFov = tan(pc.fovYRad * 0.5);
    vec3 rayDir = normalize(vec3(
         fragNDC.x * tanHalfFov * pc.aspect,
        -fragNDC.y * tanHalfFov,            // Vulkan NDC y-down → camera y-up
        -1.0
    ));

    // ── Per-fragment horizon clipping ─────────────────────────────────────────
    // dot(rayDir, enuUpInCam) = sin(elevation) for this fragment's sky direction.
    float sinEl     = dot(rayDir, pc.enuUpInCam);
    float horizFade = smoothstep(-0.015, 0.025, sinEl);
    if (horizFade <= 0.0) discard;

    // ── Ray–sphere intersection ───────────────────────────────────────────────
    vec3  oc   = -pc.moonCamCenter;
    float b    = dot(oc, rayDir);
    float c    = dot(oc, oc) - pc.moonRadius * pc.moonRadius;
    float disc = b * b - c;
    if (disc < 0.0) discard;

    float t  = -b - sqrt(disc);
    vec3  hp = t * rayDir;
    vec3  n  = normalize(hp - pc.moonCamCenter);   // outward surface normal

    // ── Lambertian diffuse from the Sun ───────────────────────────────────────
    float diffuse = max(0.0, dot(n, pc.sunDirCam)) * pc.moonPhase;

    // ── Earthshine: faint illumination on the unlit limb ─────────────────────
    const float kEarthshine = 0.018;

    // ── Lunar albedo (gray regolith, slightly warm) ───────────────────────────
    const vec3 kAlbedo = vec3(0.88, 0.85, 0.82);

    // ── Limb darkening ────────────────────────────────────────────────────────
    vec3  camDir   = normalize(-pc.moonCamCenter);
    float mu       = max(0.0, dot(n, camDir));
    float limbDark = 0.35 + 0.65 * pow(mu, 0.5);

    vec3 color = kAlbedo * (diffuse + kEarthshine) * limbDark * 2.2;

    // ── Atmospheric extinction and horizon reddening ──────────────────────────
    float moonEl   = clamp(pc.moonSinEl, 0.01, 1.0);
    float airmass  = clamp(1.0 / moonEl, 1.0, 20.0);
    const vec3 kExtR = vec3(0.014, 0.009, 0.004);
    vec3 extinct     = exp(-kExtR * (airmass - 1.0));
    float horizGlow  = clamp(1.0 - moonEl * 4.0, 0.0, 1.0);
    vec3  warmTint   = vec3(1.25, 0.85, 0.50) * horizGlow * 0.4;
    color = color * extinct + warmTint * dot(color, vec3(0.33));

    // ── Final alpha: horizon fade × daytime fade ──────────────────────────────
    outColor = vec4(color, horizFade * dayFade);
}
