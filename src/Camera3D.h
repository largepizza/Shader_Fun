#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Free-look camera. Call update() every frame.
// Right-click to capture/release mouse. WASD to move, mouse to look.
struct Camera3D {
    glm::vec3 pos       = {0.0f, 1.5f, 6.0f};
    float     yaw       = -90.0f; // degrees, -90 faces -Z
    float     pitch     = -10.0f; // degrees
    float     fovY      = 60.0f;
    float     nearZ     = 0.05f;
    float     farZ      = 200.0f;
    float     speed     = 5.0f;
    float     mouseSens = 0.12f;
    bool      captured  = false;

    glm::vec3 forward() const {
        float yr = glm::radians(yaw), pr = glm::radians(pitch);
        return glm::normalize(glm::vec3(
            cos(pr) * cos(yr),
            sin(pr),
            cos(pr) * sin(yr)));
    }

    glm::vec3 right() const {
        return glm::normalize(glm::cross(forward(), glm::vec3(0, 1, 0)));
    }

    // View matrix (world → camera space)
    glm::mat4 viewMatrix() const {
        return glm::lookAt(pos, pos + forward(), glm::vec3(0, 1, 0));
    }

    // Projection matrix for Vulkan (Y-flipped, depth in [0,1])
    glm::mat4 projMatrix(float aspect) const {
        glm::mat4 p = glm::perspective(glm::radians(fovY), aspect, nearZ, farZ);
        p[1][1] *= -1.0f; // Vulkan Y-down
        return p;
    }

    // Call once per frame before recording draw commands.
    // dmx/dmy: raw mouse delta (pixels).
    void update(GLFWwindow* win, float dt, float dmx, float dmy) {
        // Toggle capture on right-click press
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!captured) {
                glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                captured = true;
            }
        } else {
            if (captured) {
                glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                captured = false;
            }
        }

        // Mouse look (only when captured)
        if (captured) {
            yaw   += dmx * mouseSens;
            pitch -= dmy * mouseSens;
            if (pitch >  89.0f) pitch =  89.0f;
            if (pitch < -89.0f) pitch = -89.0f;
        }

        // Keyboard movement (only when captured)
        if (captured) {
            float s = speed * dt;
            glm::vec3 fwd = forward();
            glm::vec3 rgt = right();
            if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) pos += fwd * s;
            if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) pos -= fwd * s;
            if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) pos -= rgt * s;
            if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) pos += rgt * s;
            if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) pos.y -= s;
            if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) pos.y += s;
        }
    }
};
