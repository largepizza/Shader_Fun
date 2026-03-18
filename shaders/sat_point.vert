#version 450

// ── SSBO: written by sat_flare.comp, read here via gl_VertexIndex ─────────────
struct SatVisible {
    vec3  skyDir;         // unit vector in local ENU (x=East, y=North, z=Up)
    float flareIntensity; // [0, 1+]
    vec3  baseColor;      // satellite tint
    float angularSize;    // base point size (pixels)
};
layout(set = 0, binding = 1) readonly buffer SatVisibleBuf {
    SatVisible satellites[];
};

// ── Camera push constants ─────────────────────────────────────────────────────
// skyView: transforms ENU direction vectors into camera space.
//   Camera convention: +X=right, +Y=up, -Z=forward.
// fovYRad: vertical field of view in radians.
// aspect:  viewport width / height.
layout(push_constant) uniform PC {
    mat4  skyView;
    float fovYRad;
    float aspect;
    float pad[2];
} pc;

layout(location = 0) out vec3  fragColor;
layout(location = 1) out float fragIntensity;

void main() {
    SatVisible sat = satellites[gl_VertexIndex];

    // ── Project ENU sky direction through the camera ──────────────────────────
    // skyView transforms a direction (w=0) from ENU to camera space.
    // In camera space: +X=right, +Y=up, -Z=forward (satellite in front → cam.z < 0).
    vec3 cam = (pc.skyView * vec4(sat.skyDir, 0.0)).xyz;

    // Invisible (below horizon / shadow / below threshold): clip before rasterization.
    if (sat.flareIntensity <= 0.0) {
        gl_Position  = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.001;
        fragColor     = vec3(0.0);
        fragIntensity = 0.0;
        return;
    }

    // Satellite behind camera: push outside clip volume so hardware discards it.
    if (cam.z >= -0.001) {
        gl_Position  = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.001;
        fragColor     = vec3(0.0);
        fragIntensity = 0.0;
        return;
    }

    // Perspective projection.
    // tanHalfFov = tan(fovY/2). NDC x range [-1,1] corresponds to fovX,
    // NDC y range [-1,1] corresponds to fovY.
    // Vulkan Y is down, so we negate cam.y.
    float tanHalfFov = tan(pc.fovYRad * 0.5);
    float ndcX =  cam.x / (-cam.z) / (tanHalfFov * pc.aspect);
    float ndcY = -cam.y / (-cam.z) /  tanHalfFov;

    // z=0.5 is mid-depth (no depth test, value doesn't matter but must be in [0,1]).
    gl_Position  = vec4(ndcX, ndcY, 0.5, 1.0);
    gl_PointSize = sat.angularSize;  // sized by compute shader already

    fragColor     = sat.baseColor;
    fragIntensity = sat.flareIntensity;
}
