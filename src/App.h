#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <memory>
#include "VulkanContext.h"
#include "Simulation.h"

class App {
public:
    // Pass ownership of the simulation to run.
    explicit App(std::unique_ptr<Simulation> sim);
    void run();

private:
    GLFWwindow*              window  = nullptr;
    bool                     resized = false;
    double                   lastTime = 0.0;
    VulkanContext            ctx;
    std::unique_ptr<Simulation> sim;

    void initWindow();
    void mainLoop();
    void drawFrame();

    static void cbResize(GLFWwindow* w, int, int);
    static void cbKey(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void cbCursorPos(GLFWwindow* w, double x, double y);
};
