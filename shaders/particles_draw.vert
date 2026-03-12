#version 450

struct Particle {
    vec2 pos;
    vec2 vel;
    vec4 color;
};

// Same SSBO — read-only in the vertex shader
layout(set = 0, binding = 0) readonly buffer ParticleBuffer {
    Particle particles[];
};

layout(location = 0) out vec4 fragColor;
layout(location = 1) out float fragSpeed;

void main() {
    Particle p = particles[gl_VertexIndex];

    gl_Position  = vec4(p.pos, 0.0, 1.0);
    gl_PointSize = 2.0;

    fragColor = p.color;
    fragSpeed = length(p.vel);
}
