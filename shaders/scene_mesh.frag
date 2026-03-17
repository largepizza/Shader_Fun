#version 450

// ── Inputs ────────────────────────────────────────────────────────────────────
layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

// ── Camera UBO ────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invViewProj;
    vec4 camPos;
    vec4 screenParams;
} cam;

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

// ── Push constants ────────────────────────────────────────────────────────────
layout(push_constant) uniform MeshPC {
    mat4  model;
    vec4  albedo;
    float roughness;
    float metallic;
    float emissive;
    float _pad;
} pc;

// ── Output ────────────────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

// ── Blinn-Phong with per-light loop ──────────────────────────────────────────
const vec3  AMB_COLOR = vec3(0.10, 0.12, 0.16);
const float SPEC_EXP  = 64.0;

vec3 shade(vec3 albedo, float roughness, float metallic, vec3 N, vec3 V) {
    vec3 color = albedo * AMB_COLOR;

    float specF0   = mix(0.04, 0.9, metallic);
    float specPow  = SPEC_EXP * (1.0 - roughness * 0.95);

    for (int i = 0; i < lightData.numLights; i++) {
        vec3  L         = normalize(lightData.lights[i].direction.xyz);
        float intensity = lightData.lights[i].direction.w;
        vec3  Lcolor    = lightData.lights[i].color.xyz * intensity;

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        // Diffuse (attenuated by metallic — metals have no diffuse)
        color += albedo * Lcolor * NdotL * (1.0 - metallic * 0.9);

        // Specular
        vec3  H     = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        color += Lcolor * specF0 * pow(NdotH, specPow) * NdotL;
    }

    return color;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(cam.camPos.xyz - fragWorldPos);

    vec3 color = shade(pc.albedo.rgb, pc.roughness, pc.metallic, N, V);

    // Emissive (adds flat luminance regardless of lights, e.g. for sun disc)
    color += pc.albedo.rgb * pc.emissive;

    outColor = vec4(color, 1.0);
}
