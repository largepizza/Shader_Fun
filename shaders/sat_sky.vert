#version 450

// ── Camera + sun push constants (matches C++ SatDrawPC, 112 bytes) ────────────
layout(push_constant) uniform PC {
    mat4  skyView;    // ENU → camera space             — offset 0
    float fovYRad;    //                                 — offset 64
    float aspect;     //                                 — offset 68
    float pad[2];     //                                 — offsets 72, 76
    vec4  sunDirENU;  // xyz = sun direction in ENU, w = sin(elevation) — offset 80
    vec4  moonDirENU; // xyz = moon direction in ENU, w = illuminated fraction — offset 96
} pc;

layout(location = 0) out vec3 enuDir;           // interpolated ENU ray direction (not normalized)
layout(location = 1) out flat vec4 sunDirENU;   // pass-through to fragment (flat = no interp)
layout(location = 2) out flat vec4 moonDirENU;  // moon dir + phase pass-through

void main() {
    // Fullscreen triangle: 3 vertices cover the entire clip space.
    vec2 uv  = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = uv * 2.0 - 1.0;

    // Reconstruct camera-space ray direction for this triangle corner.
    // Vulkan NDC y is down; camera convention is y-up, so negate ndc.y.
    float tanHalfFov = tan(pc.fovYRad * 0.5);
    vec3 camDir = vec3(ndc.x * tanHalfFov * pc.aspect, -ndc.y * tanHalfFov, -1.0);

    // skyView maps ENU → camera space.
    // Its transpose (= inverse for rotation matrices) maps camera → ENU.
    enuDir     = mat3(transpose(pc.skyView)) * camDir;
    sunDirENU  = pc.sunDirENU;
    moonDirENU = pc.moonDirENU;

    gl_Position = vec4(ndc, 0.5, 1.0);
}
