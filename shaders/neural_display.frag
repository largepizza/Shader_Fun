#version 450
layout(location = 0) in  vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D grid;

layout(set = 0, binding = 1, std430) readonly buffer PongBuf {
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float ai_paddle_y;
    float opp_paddle_y;
    float reward;
    int   ai_score;
    int   opp_score;
    float rewardDisplay;
} pong;

layout(push_constant) uniform PC {
    int gridW;
    int gridH;
    int viewMode; // 0=activity, 1=trace, 2=weight
} pc;

vec3 hsv2rgb(float h, float s, float v) {
    vec3 c = abs(mod(vec3(h * 6.0) + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0;
    return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}

const float SPLIT = 0.60;   // neural grid occupies [0, SPLIT]
const float GAP   = 0.015;  // separator gap

const float PADDLE_H     = 0.10;
const float PADDLE_X_AI  = 0.05;
const float PADDLE_X_OPP = 0.95;
const float PADDLE_W     = 0.018;
const float BALL_R       = 0.018;

void main() {
    float x = uv.x;

    // ── Neural grid ───────────────────────────────────────────────────────────
    if (x < SPLIT) {
        vec2  nuv  = vec2(x / SPLIT, uv.y);
        vec4  cell = texture(grid, nuv);
        float act  = cell.r;
        float wgt  = cell.g;
        float tr   = cell.b;  // eligibility trace

        vec3 col;
        if (pc.viewMode == 1) {
            // Trace field: green-tinted glow
            col = hsv2rgb(0.33, 0.7, tr * 0.95 + 0.03);
        } else if (pc.viewMode == 2) {
            // Weight field: blue=weak → gold=strong
            col = hsv2rgb(mix(0.62, 0.11, wgt), 0.85, 0.85);
        } else {
            // Default: hue=weight, brightness=activation
            float hue = mix(0.62, 0.11, wgt);
            col = hsv2rgb(hue, 0.75, act * 0.9 + 0.05);
            // Reward tint on grid
            col += vec3(max(pong.reward, 0.0) * 0.12, 0.0, max(-pong.reward, 0.0) * 0.12);
        }

        // Input column: cyan highlight
        float npx = nuv.x * float(pc.gridW);
        if (npx < 1.5)
            col = mix(col, vec3(0.0, 0.9, 0.75) * (act * 0.6 + 0.4), 0.75);
        // Output column: magenta highlight
        if (npx > float(pc.gridW) - 1.5)
            col = mix(col, vec3(0.85, 0.2, 0.75) * (act * 0.6 + 0.4), 0.75);

        outColor = vec4(col, 1.0);

    // ── Separator ─────────────────────────────────────────────────────────────
    } else if (x < SPLIT + GAP) {
        outColor = vec4(0.03, 0.03, 0.06, 1.0);

    // ── Pong field ────────────────────────────────────────────────────────────
    } else {
        float px = (x - SPLIT - GAP) / (1.0 - SPLIT - GAP);
        float py = uv.y;

        vec3 col = vec3(0.04, 0.04, 0.09);

        // Center dashed line
        float dash = mod(py * 22.0, 1.0);
        if (abs(px - 0.5) < 0.006 && dash < 0.5)
            col = vec3(0.12, 0.12, 0.20);

        // Ball: glow + solid white core
        float bd = length(vec2(px - pong.ball_x, py - pong.ball_y));
        col += vec3(0.25, 0.35, 0.9) * max(0.0, 1.0 - bd / (BALL_R * 5.0));
        if (bd < BALL_R) col = vec3(1.0);

        // AI paddle (cyan, left)
        vec2 aiRel = abs(vec2(px - PADDLE_X_AI, py - pong.ai_paddle_y));
        if (aiRel.x < PADDLE_W && aiRel.y < PADDLE_H)
            col = mix(col, vec3(0.15, 0.95, 0.75), 0.9);

        // Opponent paddle (orange, right)
        vec2 oppRel = abs(vec2(px - PADDLE_X_OPP, py - pong.opp_paddle_y));
        if (oppRel.x < PADDLE_W && oppRel.y < PADDLE_H)
            col = mix(col, vec3(0.95, 0.55, 0.15), 0.9);

        // Reward flash: green border on hit, red on miss
        float rd = abs(pong.rewardDisplay);
        if (rd > 0.01) {
            float edge  = min(min(px, 1.0 - px), min(py, 1.0 - py));
            float flash = rd * max(0.0, 0.05 - edge) / 0.05;
            vec3  fcol  = pong.rewardDisplay > 0.0
                        ? vec3(0.0, 0.6, 0.1)
                        : vec3(0.7, 0.0, 0.0);
            col = mix(col, fcol, flash * 0.7);
        }

        outColor = vec4(col, 1.0);
    }
}
