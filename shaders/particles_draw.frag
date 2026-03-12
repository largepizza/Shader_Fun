#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in float fragSpeed;

layout(location = 0) out vec4 outColor;

void main() {
    // gl_PointCoord is [0,1] across the point sprite
    vec2 centered = gl_PointCoord - 0.5;
    float dist    = length(centered);

    // Discard corners to make circular point sprites
    if (dist > 0.5) discard;

    // Soft falloff: bright centre, fade at edges
    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

    // Brighter for faster particles
    float brightness = 1.0 + fragSpeed * 3.0;
    outColor = vec4(fragColor.rgb * brightness * alpha, alpha);
}
