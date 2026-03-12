#version 450

// Generates a fullscreen triangle without any vertex buffers.
// Trick: 3 vertices cover the entire NDC space using gl_VertexIndex.
layout(location = 0) out vec2 uv;

void main() {
    // Vertices at (-1,-1), (3,-1), (-1,3) cover the full screen
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 uvs[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 2.0)
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    uv = uvs[gl_VertexIndex];
}
