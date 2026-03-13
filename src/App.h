#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <memory>
#include "VulkanContext.h"
#include "Simulation.h"
#include "UIRenderer.h"

class App {
public:
    // Pass ownership of the simulation to run.
    explicit App(std::unique_ptr<Simulation> sim);
    void run();

private:
    GLFWwindow*              window   = nullptr;
    bool                     resized  = false;
    double                   lastTime = 0.0;
    VulkanContext            ctx;
    std::unique_ptr<Simulation> sim;
    UIRenderer               ui;
    float                    scrollX  = 0.0f; // accumulated scroll for Clay
    float                    scrollY  = 0.0f;

    void initWindow();
    void mainLoop();
    void drawFrame();

    static void cbResize(GLFWwindow* w, int, int);
    static void cbKey(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void cbCursorPos(GLFWwindow* w, double x, double y);
    static void cbScroll(GLFWwindow* w, double dx, double dy);
};
