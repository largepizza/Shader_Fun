cmake_minimum_required(VERSION 3.20)

# CMake 4.0 REMOVED compatibility with projects declaring cmake_minimum_required(VERSION <3.5),
# turning what used to be a deprecation warning into a hard configure error. Two of this
# project's FetchContent dependencies still declare one — glfw 3.4 ("3.4...3.28") and
# nlohmann/json v3.11.3 ("3.1...3.14") — so a stock CMake 4 (Homebrew's default on macOS since
# early 2025, and increasingly the Windows/Linux default too) cannot configure this project at
# all. This variable is CMake's own documented escape hatch: it seeds each dependency's
# cmake_minimum_required floor without patching vendored sources or pinning older dep tags.
#
# This is why the macOS setup originally hardcoded an absolute path to a portable CMake 3.31 in
# .vscode/settings.json. That worked around the symptom on exactly one machine, at the cost of
# committing a developer's home directory to the repo. Fixing it here fixes it for every
# machine and every generator, with nothing user-specific to commit.
#
# Unknown to CMake < 3.31, where it is silently ignored — harmless, since those versions still
# accept the old floors natively. Overridable from the command line for anyone who wants to see
# the raw deprecation warnings.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# macOS deployment target — must be set BEFORE project(), which is what bakes LC_BUILD_VERSION
# into every object file. Without it, clang defaults to the BUILD machine's OS version, so a
# binary produced on a macOS 14 CI runner refuses to launch on anything older ("not supported on
# this version of macOS"). 11.0 is the oldest release with both Apple Silicon and MoltenVK
# support, and covers the Monterey (12.x) target hardware. See CMAKE_OSX_ARCHITECTURES in
# CMakePresets.json for the separate arm64/x86_64 question.
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version" FORCE)
endif()

# Single source of truth for the version — edit VERSION to cut a release.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" APP_VERSION)
string(STRIP "${APP_VERSION}" APP_VERSION)           # remove trailing newline
string(REPLACE "." "_" VERSION_UNDERSCORED "${APP_VERSION}")
set(EXE_BASENAME "SAT_LIGHT_SIM_V_${VERSION_UNDERSCORED}")  # e.g. SAT_LIGHT_SIM_V_1_0_0

project(SatLightSim VERSION ${APP_VERSION} LANGUAGES CXX)
message(STATUS "Building ${EXE_BASENAME}")

# Git commit + build date for diagnostics/bug reports (NEW-1). "unknown" for
# source-zip builds with no .git directory so this never fails a from-scratch
# build. Only updates when CMake re-configures, not on every incremental build.
find_package(Git QUIET)
set(GIT_COMMIT "unknown")
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_RESULT
    )
    if(GIT_RESULT EQUAL 0 AND NOT GIT_COMMIT_OUT STREQUAL "")
        set(GIT_COMMIT "${GIT_COMMIT_OUT}")
    endif()
endif()
string(TIMESTAMP BUILD_DATE "%Y-%m-%d" UTC)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
    @ONLY
)

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

# miniaudio — single-header audio engine (MIT). Provides MP3/WAV/FLAC decoding
# and OS audio output with no external dependencies.
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(miniaudio)

# nlohmann/json — header-only JSON parser (MIT). Used to load constellations.json at runtime.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_executable(SatLightSim ${SOURCES})
set_target_properties(SatLightSim PROPERTIES OUTPUT_NAME "${EXE_BASENAME}")
target_include_directories(SatLightSim PRIVATE src ${clay_SOURCE_DIR} ${stb_SOURCE_DIR} ${miniaudio_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/generated")

# UC6 (RELEASE_v1_1_PLAN.md): screenshot PNG encoding runs on a std::thread. A no-op on MSVC
# (thread support is built into the runtime there), but GCC/Clang on Linux/macOS need an explicit
# pthread link or std::thread silently fails to link.
find_package(Threads REQUIRED)
target_link_libraries(SatLightSim PRIVATE Vulkan::Vulkan glfw glm::glm nlohmann_json::nlohmann_json Threads::Threads)

# miniaudio's CoreAudio backend needs these frameworks explicitly. GLFW's target already
# brings Cocoa/IOKit/CoreFoundation transitively, but not CoreAudio/AudioToolbox — without
# this the link fails with undefined _AudioObjectGetPropertyData etc.
if(APPLE)
    target_link_libraries(SatLightSim PRIVATE
        "-framework CoreFoundation" "-framework CoreAudio" "-framework AudioToolbox")
endif()

# Shader compilation
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK and ensure VULKAN_SDK env var is set.")
endif()

set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_INC_DIR "${SHADER_SRC_DIR}/include")
set(SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

file(GLOB SHADERS CONFIGURE_DEPENDS
    "${SHADER_SRC_DIR}/*.vert"
    "${SHADER_SRC_DIR}/*.frag"
    "${SHADER_SRC_DIR}/*.comp"
)

# Shared shader headers live in shaders/include/ with a .glsl extension — a SEPARATE directory
# AND a separate extension, so the glob above never tries to compile one as a standalone
# translation unit. Naming a header .comp, or leaving it in shaders/, breaks the build.
file(GLOB SHADER_HEADERS CONFIGURE_DEPENDS "${SHADER_INC_DIR}/*.glsl")

set(SPV_FILES "")
foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV "${SHADER_OUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -I "${SHADER_INC_DIR}" ${SHADER} -o ${SPV}
        # Every shader depends on ALL headers, not just the ones it includes. Coarse — touching
        # one header recompiles all ~28 shaders — but glslc takes seconds even on the two big
        # ones, and it is correct on every generator. The precise alternative (glslc -MD +
        # add_custom_command DEPFILE) needs CMake 3.27 for Visual Studio generators; this project
        # requires 3.20 and builds with MSVC, so DEPFILE is not available here. If the build ever
        # moves to Ninja, switching is a small change.
        DEPENDS ${SHADER} ${SHADER_HEADERS}
        COMMENT "Compiling shader: ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPV_FILES ${SPV})
endforeach()

# Second compile of sat_sky.frag with -DSKY_LITE — a Planetarium-tier variant that #ifdef's out
# the heaviest subsystems (Milky Way, 64-bin sky-glow loop, beam ground-spot, cirrus/high cloud
# layers) for GPUs where the full shader collapses fragment-shader occupancy. Selected at runtime
# by a debugDisableMask bit, exactly like sat_sky_minimal.frag. See SKY_OPTIMIZATION_PLAN.md.
set(SKY_LITE_SPV "${SHADER_OUT_DIR}/sat_sky_lite.frag.spv")
add_custom_command(
    OUTPUT ${SKY_LITE_SPV}
    COMMAND ${GLSLC} -I "${SHADER_INC_DIR}" -DSKY_LITE "${SHADER_SRC_DIR}/sat_sky.frag" -o ${SKY_LITE_SPV}
    DEPENDS "${SHADER_SRC_DIR}/sat_sky.frag" ${SHADER_HEADERS}
    COMMENT "Compiling shader: sat_sky.frag (SKY_LITE variant)"
    VERBATIM
)
list(APPEND SPV_FILES ${SKY_LITE_SPV})

add_custom_target(CompileShaders ALL DEPENDS ${SPV_FILES})
add_dependencies(SatLightSim CompileShaders)

# Copy compiled shaders next to the executable
add_custom_command(TARGET SatLightSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_OUT_DIR}
        $<TARGET_FILE_DIR:SatLightSim>/shaders
    COMMENT "Copying shaders to output directory"
)

# Copy data files next to the executable on every build when the source changes.
# Both files are tracked in source control; copy_if_different keeps the output
# in sync without unnecessary writes when nothing has changed.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_custom_command(TARGET SatLightSim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/constellations.schema.json"
            "$<TARGET_FILE_DIR:SatLightSim>/constellations.schema.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/constellations.json"
            "$<TARGET_FILE_DIR:SatLightSim>/constellations.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/reflector_targets.json"
            "$<TARGET_FILE_DIR:SatLightSim>/reflector_targets.json"
        COMMENT "Copying data files to output directory"
    )
endif()

# Copy assets (icons, textures, etc.) next to the executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
    add_custom_command(TARGET SatLightSim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:SatLightSim>/assets"
        COMMENT "Copying assets to output directory"
    )
endif()

# THIRD_PARTY_NOTICES.txt must ship next to the executable in every distributed
# copy — MIT/zlib library licenses require the license text to travel with the
# binary, not just live in an in-app tab (see UC2 in RELEASE_v1_1_PLAN.md).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.txt")
    add_custom_command(TARGET SatLightSim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.txt"
            "$<TARGET_FILE_DIR:SatLightSim>/THIRD_PARTY_NOTICES.txt"
        COMMENT "Copying THIRD_PARTY_NOTICES.txt to output directory"
    )
endif()

# ── Release packaging ─────────────────────────────────────────────────────────
#   cmake --build <build-dir> --config Release --target package-release
#
# Stages the shippable tree and archives it to dist/. Not part of ALL, so a normal build is
# unaffected. cmake/PackageRelease.cmake holds the actual copy list — CI (release.yml) and
# release.bat both call this target instead of maintaining their own copies of it.
#
# PLATFORM_TAG names the archive. On macOS it distinguishes a universal binary from a
# single-architecture one, because those are NOT interchangeable on a player's machine: an
# arm64-only build on an Intel Mac dies with "Bad CPU type in executable" before main() runs.
if(WIN32)
    set(SATLIGHT_PLATFORM_TAG "Windows")
elseif(APPLE)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _osx_arch_count)
    if(_osx_arch_count GREATER 1)
        set(SATLIGHT_PLATFORM_TAG "macOS_universal")
    elseif(CMAKE_OSX_ARCHITECTURES)
        set(SATLIGHT_PLATFORM_TAG "macOS_${CMAKE_OSX_ARCHITECTURES}")
    else()
        set(SATLIGHT_PLATFORM_TAG "macOS_${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
else()
    set(SATLIGHT_PLATFORM_TAG "Linux")
endif()

# Bundling the Vulkan loader + MoltenVK only makes sense on macOS, where there is no system
# Vulkan at all. Windows and Linux both get a loader from the GPU driver / distro package.
set(SATLIGHT_PACKAGE_VULKAN_SDK "")
if(APPLE)
    set(SATLIGHT_PACKAGE_VULKAN_SDK "$ENV{VULKAN_SDK}")
endif()

add_custom_target(package-release
    COMMAND ${CMAKE_COMMAND}
        -DEXE_FILE=$<TARGET_FILE:SatLightSim>
        -DRUNTIME_DIR=$<TARGET_FILE_DIR:SatLightSim>
        -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DOUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/dist
        -DAPP_VERSION=${APP_VERSION}
        -DEXE_BASENAME=${EXE_BASENAME}
        -DPLATFORM_TAG=${SATLIGHT_PLATFORM_TAG}
        -DVULKAN_SDK_DIR=${SATLIGHT_PACKAGE_VULKAN_SDK}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageRelease.cmake
    COMMENT "Packaging release archive into dist/"
    VERBATIM
)
# add_custom_target's own DEPENDS takes FILES, not targets — the target ordering has to come
# from add_dependencies, or `--target package-release` would happily archive a stale exe.
add_dependencies(package-release SatLightSim)
