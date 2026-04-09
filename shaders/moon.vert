#version 450

// Push constants matching C++ MoonPC (64 bytes)
layout(push_constant) uniform PC {
    vec3  moonCamCenter;  // moon centre in camera space (+X right, +Y up, -Z forward)
    float moonRadius;     // moon radius in same units as moonCamCenter
    float fovYRad;        // vertical FoV (radians)
    float aspect;         // viewport width / height
    float moonSinEl;      // sin(elevation) of the moon centre in ENU
    float sunSinEl;       // sin(elevation) of the sun — used for daytime fade
    vec3  sunDirCam;      // unit vector toward sun in camera space
    float moonPhase;      // illuminated fraction [0, 1]
    vec3  enuUpInCam;     // ENU "up" (0,0,1) expressed in camera space
    float pad2;
} pc;

layout(location = 0) out vec2 fragNDC;

void main() {
    // Coarse culls — skip rasterisation entirely:
    //   (a) moon centre more than one disc-radius below the local horizon
    //   (b) moon behind the camera
    //   (c) sun high enough that the sky blows out the moon (handled in frag, but
    //       if the sun is way above we can skip the quad entirely to save work)
    if (pc.moonSinEl < -0.01 || pc.moonCamCenter.z >= -0.001) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        fragNDC     = vec2(0.0);
        return;
    }

    float dist       = -pc.moonCamCenter.z;   // positive depth
    float tanHalfFov = tan(pc.fovYRad * 0.5);

    // NDC position of the moon centre.
    float cx = pc.moonCamCenter.x / dist / (tanHalfFov * pc.aspect);
    float cy = -pc.moonCamCenter.y / dist / tanHalfFov;   // Vulkan Y-down

    // NDC half-extents with 10 % margin so the full disc fits inside the quad.
    float halfW = (pc.moonRadius / dist / (tanHalfFov * pc.aspect)) * 1.1;
    float halfH = (pc.moonRadius / dist / tanHalfFov) * 1.1;

    // Two-triangle quad (6 vertices, no VBO).
    const vec2 corners[6] = vec2[](
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
        vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
    );

    vec2 c   = corners[gl_VertexIndex];
    vec2 ndc = vec2(cx + c.x * halfW, cy + c.y * halfH);

    gl_Position = vec4(ndc, 0.5, 1.0);
    fragNDC     = ndc;
}
