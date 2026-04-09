#version 450

layout(location = 0) in vec2  fragUV;
layout(location = 1) in vec4  fragColor;
layout(location = 2) in float fragMode;

layout(location = 0) out vec4 outColor;

// binding 0 = font atlas (R8_UNORM, alpha mask for text glyphs)
layout(set = 0, binding = 0) uniform sampler2D fontAtlas;

// binding 1 = icon atlas (R8G8B8A8_UNORM, RGBA sprite sheet)
layout(set = 0, binding = 1) uniform sampler2D iconAtlas;

void main() {
    if (fragMode < 0.5) {
        // Solid rectangle
        outColor = fragColor;
    } else if (fragMode < 1.5) {
        // Text glyph: red channel of font atlas is the alpha mask
        float alpha = texture(fontAtlas, fragUV).r;
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
    } else {
        // Icon sprite: RGBA sample, tinted by fragColor
        vec4 tex = texture(iconAtlas, fragUV);
        outColor = tex * fragColor;
    }
}
