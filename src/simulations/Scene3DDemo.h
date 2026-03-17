#pragma once
#include "../Simulation.h"
#include "../Scene3D.h"

// Demo showcasing the Scene3D framework:
//   - A ground grid (mesh)
//   - A box (mesh)
//   - A sphere SDF (animates up/down)
//   - A torus SDF (rotates)
//   - Free-look camera: right-click to capture, WASD to move, mouse to look
class Scene3DDemo : public Simulation {
public:
    void init(VulkanContext& ctx) override;
    void onResize(VulkanContext& ctx) override;
    void recordDraw(VkCommandBuffer cmd, VulkanContext& ctx, float dt) override;
    void buildUI(float dt, UIRenderer& ui) override;
    void cleanup(VkDevice device) override;

    void onKey(GLFWwindow* win, int key, int action) override;
    void onCursorPos(GLFWwindow* win, double x, double y) override;

    VkClearValue clearColor() const override { return {{{0.05f, 0.06f, 0.10f, 1.0f}}}; }
    const char*  name()       const override { return "Scene3D Demo"; }

private:
    Scene3D scene;

    // SDF objects we want to animate
    std::shared_ptr<SDFObject3D> sdfSphere;
    std::shared_ptr<SDFObject3D> sdfTorus;

    float totalTime = 0.0f;

    // Mouse tracking for camera delta
    GLFWwindow* win = nullptr;
    double prevX = 0, prevY = 0;
    bool firstMouse = true;
    float dmx = 0, dmy = 0; // mouse delta for current frame
};
