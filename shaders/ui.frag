#version 450

layout(location = 0) in vec2  fragUV;
layout(location = 1) in vec4  fragColor;
layout(location = 2) in float fragMode;

layout(location = 0) out vec4 outColor;

// Font atlas — single-channel R8 texture
layout(set = 0, binding = 0) uniform sampler2D fontAtlas;

void main() {
    if (fragMode > 0.5) {
        // Text glyph: use font atlas red channel as alpha mask
        float alpha = texture(fontAtlas, fragUV).r;
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
    } else {
        // Solid rectangle
        outColor = fragColor;
    }
}
