#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inMode;

layout(push_constant) uniform PC {
    vec2 screenSize;
} pc;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;
layout(location = 2) out float fragMode;

void main() {
    // Convert screen-space pixels (top-left origin) to Vulkan NDC (top-left = -1,-1)
    vec2 ndc = (inPos / pc.screenSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    fragUV    = inUV;
    fragColor = inColor;
    fragMode  = inMode;
}
