#version 450

// ── Inputs ────────────────────────────────────────────────────────────────────
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// ── Camera UBO ────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invViewProj;
    vec4 camPos;
    vec4 screenParams;
} cam;

// ── Push constants (model matrix + material) ──────────────────────────────────
layout(push_constant) uniform MeshPC {
    mat4  model;
    vec4  albedo;
    float roughness;
    float metallic;
    float emissive;
    float _pad;
} pc;

// ── Outputs ───────────────────────────────────────────────────────────────────
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

void main() {
    vec4 worldPos = pc.model * vec4(inPos, 1.0);
    fragWorldPos  = worldPos.xyz;
    fragNormal    = normalize(mat3(transpose(inverse(pc.model))) * inNormal);
    fragUV        = inUV;
    gl_Position   = cam.viewProj * worldPos;
}
