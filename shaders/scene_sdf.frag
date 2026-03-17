#version 450

// ── Input from fullscreen.vert ────────────────────────────────────────────────
layout(location = 0) in vec2 uv; // unused; ray reconstructed from gl_FragCoord

// ── Camera UBO ────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invViewProj;
    vec4 camPos;        // xyz = world pos
    vec4 screenParams;  // xy = resolution, z = near, w = far
} cam;

// ── SDF object SSBO ───────────────────────────────────────────────────────────
struct SDFObject {
    mat4 invModel;
    vec4 params;     // shape-specific
    vec4 albedo;     // xyz = albedo, w = emissive
    vec4 rm;         // x = roughness, y = metallic
    int  shapeType;
    int  pad0, pad1, pad2;
};
layout(set = 0, binding = 1) readonly buffer SDFData {
    int       numObjects;
    int       pad0, pad1, pad2;
    SDFObject objects[];
} sdf;

// ── Light UBO ─────────────────────────────────────────────────────────────────
struct GPULight {
    vec4 direction; // xyz = toward-light (L), w = intensity
    vec4 color;     // xyz = color
};
layout(set = 0, binding = 2) uniform LightUBO {
    int     numLights;
    int     pad0, pad1, pad2;
    GPULight lights[4];
} lightData;

// ── Output ────────────────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

// gl_FragDepth written explicitly for proper compositing with mesh objects.

// ── SDF primitives (evaluated in object space) ────────────────────────────────
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sdTorus(vec3 p, float R, float r) {
    vec2 q = vec2(length(p.xz) - R, p.y);
    return length(q) - r;
}
float sdCapsule(vec3 p, float halfH, float r) {
    p.y -= clamp(p.y, -halfH, halfH);
    return length(p) - r;
}

float evalSDF(vec3 lp, int type, vec4 params) {
    if (type == 0) return sdSphere(lp, params.x);
    if (type == 1) return sdBox(lp, params.xyz);
    if (type == 2) return sdTorus(lp, params.x, params.y);
    if (type == 3) return sdCapsule(lp, params.x, params.y);
    return 1e10;
}

// ── Scene evaluation ──────────────────────────────────────────────────────────
float sceneDist(vec3 p, out int hitIdx) {
    float d = 1e10;
    hitIdx = -1;
    for (int i = 0; i < sdf.numObjects; i++) {
        vec3  lp = (sdf.objects[i].invModel * vec4(p, 1.0)).xyz;
        float di = evalSDF(lp, sdf.objects[i].shapeType, sdf.objects[i].params);
        if (di < d) { d = di; hitIdx = i; }
    }
    return d;
}
float sceneDist(vec3 p) { int dummy; return sceneDist(p, dummy); }

// ── Normal via central differences ────────────────────────────────────────────
vec3 calcNormal(vec3 p) {
    const float e = 0.0005;
    return normalize(vec3(
        sceneDist(p + vec3(e,0,0)) - sceneDist(p - vec3(e,0,0)),
        sceneDist(p + vec3(0,e,0)) - sceneDist(p - vec3(0,e,0)),
        sceneDist(p + vec3(0,0,e)) - sceneDist(p - vec3(0,0,e))
    ));
}

// ── Lighting (matches scene_mesh.frag) ───────────────────────────────────────
const vec3  AMB_COLOR = vec3(0.10, 0.12, 0.16);
const float SPEC_EXP  = 64.0;

vec3 shade(vec3 albedo, float roughness, float metallic, float emissive, vec3 N, vec3 V) {
    vec3 color = albedo * AMB_COLOR;

    float specF0  = mix(0.04, 0.9, metallic);
    float specPow = SPEC_EXP * (1.0 - roughness * 0.95);

    for (int i = 0; i < lightData.numLights; i++) {
        vec3  L         = normalize(lightData.lights[i].direction.xyz);
        float intensity = lightData.lights[i].direction.w;
        vec3  Lcolor    = lightData.lights[i].color.xyz * intensity;

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        color += albedo * Lcolor * NdotL * (1.0 - metallic * 0.9);

        vec3  H     = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        color += Lcolor * specF0 * pow(NdotH, specPow) * NdotL;
    }

    color += albedo * emissive;
    return color;
}

// ── Main ──────────────────────────────────────────────────────────────────────
void main() {
    if (sdf.numObjects == 0) discard;

    // Reconstruct world-space ray
    vec2 res = cam.screenParams.xy;
    vec2 ndc = (gl_FragCoord.xy / res) * 2.0 - 1.0;

    vec4 nearH = cam.invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 farH  = cam.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 rO = nearH.xyz / nearH.w;
    vec3 rD = normalize(farH.xyz / farH.w - rO);

    // Sphere-trace
    const int   MAX_STEPS = 128;
    const float MAX_DIST  = 200.0;
    const float HIT_EPS   = 0.0005;

    float t = 0.0;
    int hitIdx = -1;
    for (int i = 0; i < MAX_STEPS; i++) {
        float d = sceneDist(rO + rD * t, hitIdx);
        if (d < HIT_EPS) break;
        t += d;
        if (t >= MAX_DIST) { hitIdx = -1; break; }
    }

    if (hitIdx < 0) discard;

    vec3 hitPos = rO + rD * t;
    vec3 N      = calcNormal(hitPos);
    vec3 V      = -rD;

    SDFObject obj = sdf.objects[hitIdx];
    outColor = vec4(shade(obj.albedo.xyz, obj.rm.x, obj.rm.y, obj.albedo.w, N, V), 1.0);

    // Write depth for correct compositing with mesh geometry
    vec4 clip = cam.viewProj * vec4(hitPos, 1.0);
    gl_FragDepth = clip.z / clip.w;
}
