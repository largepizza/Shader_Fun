cmake_minimum_required(VERSION 3.20)
project(ShaderFun VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static-link the MSVC runtime (/MT in Release, /MTd in Debug) so the exe
# runs on machines without the Visual C++ Redistributable installed.
# Must be set before FetchContent so GLFW builds with the same runtime.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Emit compile_commands.json so VS Code IntelliSense knows all include paths and flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Vulkan SDK must be installed: https://vulkan.lunarg.com/
find_package(Vulkan REQUIRED)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glm)

# Clay UI layout library (single header)
# Disable all example targets so Clay doesn't pull in SDL2, raylib, etc.
set(CLAY_INCLUDE_ALL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLAY_INCLUDE_DEMOS        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    clay
    GIT_REPOSITORY https://github.com/nicbarker/clay.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(clay)

# stb single-header libs (for stb_truetype)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_executable(ShaderFun ${SOURCES})
target_include_directories(ShaderFun PRIVATE src ${clay_SOURCE_DIR} ${stb_SOURCE_DIR})
target_link_libraries(ShaderFun PRIVATE Vulkan::Vulkan glfw glm::glm)

# Shader compilation
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK and ensure VULKAN_SDK env var is set.")
endif()

set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

file(GLOB SHADERS
    "${SHADER_SRC_DIR}/*.vert"
    "${SHADER_SRC_DIR}/*.frag"
    "${SHADER_SRC_DIR}/*.comp"
)

set(SPV_FILES "")
foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV "${SHADER_OUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPV_FILES ${SPV})
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${SPV_FILES})
add_dependencies(ShaderFun CompileShaders)

# Copy compiled shaders next to the executable
add_custom_command(TARGET ShaderFun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_OUT_DIR}
        $<TARGET_FILE_DIR:ShaderFun>/shaders
    COMMENT "Copying shaders to output directory"
)

# Copy assets (icons, textures, etc.) next to the executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
    add_custom_command(TARGET ShaderFun POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:ShaderFun>/assets"
        COMMENT "Copying assets to output directory"
    )
endif()
