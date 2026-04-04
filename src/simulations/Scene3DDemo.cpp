#include "Scene3DDemo.h"
#include "../UIRenderer.h"
#include "clay.h"

#include <cstdio>
#include <cstring>

void Scene3DDemo::init(VulkanContext &ctx)
{
    // ── Ground grid (mesh) ────────────────────────────────────────────────
    auto [gv, gi] = Scene3D::makeGrid(8.0f, 20);
    auto ground = scene.addMesh("Ground", std::move(gv), std::move(gi));
    ground->material.albedo = {0.30f, 0.35f, 0.28f};
    ground->material.roughness = 0.9f;

    // ── Box (mesh) ────────────────────────────────────────────────────────
    auto [bv, bi] = Scene3D::makeBox({0.6f, 0.6f, 0.6f});
    auto box = scene.addMesh("Box", std::move(bv), std::move(bi));
    box->transform.position = {-2.0f, 0.6f, 0.0f};
    box->material.albedo = {0.8f, 0.4f, 0.15f};
    box->material.roughness = 0.4f;

    // ── Sphere (mesh) ─────────────────────────────────────────────────────
    auto [sv, si] = Scene3D::makeSphere(0.7f, 24, 32);
    auto meshSphere = scene.addMesh("MeshSphere", std::move(sv), std::move(si));
    meshSphere->transform.position = {2.0f, 0.7f, 0.0f};
    meshSphere->material.albedo = {0.2f, 0.5f, 0.9f};
    meshSphere->material.roughness = 0.15f;
    meshSphere->material.metallic = 0.8f;

    // ── SDF sphere (animates vertically) ──────────────────────────────────
    sdfSphere = scene.addSDF("SDFSphere", SDFShape::Sphere, {0.8f, 0, 0});
    sdfSphere->transform.position = {0.0f, 1.5f, -2.5f};
    sdfSphere->material.albedo = {0.9f, 0.2f, 0.3f};
    sdfSphere->material.roughness = 0.3f;

    // ── SDF torus (rotates) ───────────────────────────────────────────────
    sdfTorus = scene.addSDF("SDFTorus", SDFShape::Torus, {0.9f, 0.3f, 0});
    sdfTorus->transform.position = {0.0f, 1.0f, 2.5f};
    sdfTorus->material.albedo = {0.3f, 0.9f, 0.5f};
    sdfTorus->material.roughness = 0.5f;

    // ── SDF capsule ───────────────────────────────────────────────────────
    auto capsule = scene.addSDF("SDFCapsule", SDFShape::Capsule, {0.7f, 0.25f, 0});
    capsule->transform.position = {2.8f, 1.2f, -2.0f};
    capsule->material.albedo = {0.9f, 0.8f, 0.2f};
    capsule->material.roughness = 0.6f;

    // ── Directional light (sun) ───────────────────────────────────────────
    // direction = toward the light (L vector), so (0.6, 1.0, 0.4) = upper-right-front sun
    auto &sun = scene.addLight(glm::normalize(glm::vec3(0.6f, 1.0f, 0.4f)),
                               {1.0f, 0.92f, 0.78f}, 1.1f);
    sun.discDist = 30.0f;
    sun.discRadius = 1.0f;

    // ── Point light (lamp) ───────────────────────────────────────────────
    auto &lamp = scene.addLight(glm::normalize(glm::vec3(-0.5f, 1.0f, -0.3f)),
                                {0.8f, 0.2f, 0.2f}, 0.5f);
    lamp.discDist = 10.0f;
    lamp.discRadius = 0.4f;

    scene.init(ctx);
    scene.uploadMeshes(ctx);
}

void Scene3DDemo::onResize(VulkanContext &ctx)
{
    scene.onResize(ctx);
}

void Scene3DDemo::recordDraw(VkCommandBuffer cmd, VulkanContext &ctx, float dt)
{
    totalTime += dt;

    // Animate SDF sphere (bob up/down)
    sdfSphere->transform.position.y = 1.5f + 0.6f * sinf(totalTime * 1.2f);

    // Animate SDF torus (rotate around Y)
    sdfTorus->transform.rotation.y = totalTime * 40.0f;
    sdfTorus->transform.rotation.x = 20.0f; // tilt so ring is visible

    scene.render(cmd, ctx, dt);

    // Reset mouse delta after each frame
    dmx = dmy = 0.0f;
}

void Scene3DDemo::buildUI(float dt, UIRenderer &ui)
{
    // Update camera from input
    if (win)
    {
        scene.camera.update(win, dt, dmx, dmy);
    }

    UIInput inp = ui.input();
    bool captured = scene.camera.captured;

    // Info panel (top-left, floating)
    CLAY(CLAY_ID("InfoPanel"), {.layout = {
                                    .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                                    .padding = {12, 12, 10, 10},
                                    .childGap = 6,
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM},
                                .backgroundColor = {10, 12, 18, 200},
                                .cornerRadius = CLAY_CORNER_RADIUS(6),
                                .floating = {.offset = {12, 12}, .zIndex = 5, .attachTo = CLAY_ATTACH_TO_ROOT}})
    {
        CLAY_TEXT(CLAY_STRING("Scene3D Demo"),
                  CLAY_TEXT_CONFIG({.textColor = {220, 220, 255, 255}, .fontSize = 20}));

        // Camera mode indicator
        Clay_Color modeCol = captured ? Clay_Color{100, 255, 120, 255} : Clay_Color{200, 200, 200, 200};
        CLAY_TEXT(captured ? CLAY_STRING("Camera: CAPTURED  (RMB to release)")
                           : CLAY_STRING("Camera: free  (Right-click to capture)"),
                  CLAY_TEXT_CONFIG({.textColor = modeCol, .fontSize = 16}));

        CLAY_TEXT(CLAY_STRING("WASD = move   Q/E = down/up   Mouse = look"),
                  CLAY_TEXT_CONFIG({.textColor = {160, 160, 180, 200}, .fontSize = 14}));

        // FPS
        static char fpsBuf[32];
        float fps = dt > 0.0f ? 1.0f / dt : 0.0f;
        snprintf(fpsBuf, sizeof(fpsBuf), "%.0f fps", fps);
        Clay_String fpsStr{false, (int32_t)strlen(fpsBuf), fpsBuf};
        CLAY_TEXT(fpsStr, CLAY_TEXT_CONFIG({.textColor = {160, 200, 160, 220}, .fontSize = 14}));
    }

    ui.addMouseCaptureRect(12, 12, 320, 130);
}

void Scene3DDemo::cleanup(VkDevice device)
{
    scene.cleanup(device);
    if (win)
    {
        glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void Scene3DDemo::onKey(GLFWwindow *w, int key, int action)
{
    win = w; // cache window pointer for camera polling
}

void Scene3DDemo::onCursorPos(GLFWwindow *w, double x, double y)
{
    win = w;
    if (firstMouse)
    {
        prevX = x;
        prevY = y;
        firstMouse = false;
    }
    dmx += (float)(x - prevX);
    dmy += (float)(y - prevY);
    prevX = x;
    prevY = y;
}
