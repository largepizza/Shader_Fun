#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// The compute shader outputs R channel as alive (1.0) or dead (0.0)
layout(set = 0, binding = 0) uniform sampler2D golGrid;

void main() {
    float alive = texture(golGrid, uv).r;

    // Color scheme: dark blue background, bright green/cyan alive cells
    vec3 deadColor  = vec3(0.02, 0.02, 0.08);
    vec3 aliveColor = vec3(0.1, 0.9, 0.4);

    // Add a slight glow by checking surrounding pixels (cheap bloom-like effect)
    vec2 texelSize = 1.0 / textureSize(golGrid, 0);
    float glow = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            glow += texture(golGrid, uv + vec2(dx, dy) * texelSize).r;
        }
    }
    glow /= 9.0;

    vec3 color = mix(deadColor, aliveColor, alive);
    // Add ambient glow from nearby alive cells
    color += vec3(0.0, 0.15, 0.05) * glow * (1.0 - alive);

    outColor = vec4(color, 1.0);
}
