# PackageRelease.cmake — run in script mode (cmake -P) by the `package-release` target.
#
# Stages a distributable tree next to the source root and compresses it. This is the SINGLE
# source of truth for "what ships": .github/workflows/release.yml and release.bat both drive
# the `package-release` target rather than repeating the copy list. Before this existed the
# same list appeared five times (3 CI jobs + 2 release.bat branches) and had already drifted.
#
# Required -D arguments (all absolute paths):
#   EXE_FILE      full path to the built executable
#   RUNTIME_DIR   directory the build copied shaders/ and assets/ into ($<TARGET_FILE_DIR:...>)
#   SRC_DIR       project source root
#   OUT_DIR       where the staged tree and the archive go
#   APP_VERSION   e.g. 1.1.0
#   EXE_BASENAME  e.g. SAT_LIGHT_SIM_V_1_1_0
#   PLATFORM_TAG  Windows | Linux | macOS | macOS_universal
# Optional:
#   VULKAN_SDK_DIR   macOS only — when set, the Vulkan loader + MoltenVK are bundled and a
#                    launcher .command is written (see the block at the bottom for why).

cmake_minimum_required(VERSION 3.20)

foreach(arg EXE_FILE RUNTIME_DIR SRC_DIR OUT_DIR APP_VERSION EXE_BASENAME PLATFORM_TAG)
    if(NOT DEFINED ${arg})
        message(FATAL_ERROR "PackageRelease.cmake: -D${arg} is required")
    endif()
endforeach()

if(NOT EXISTS "${EXE_FILE}")
    message(FATAL_ERROR "PackageRelease.cmake: executable not found: ${EXE_FILE}\n"
                        "Build the SatLightSim target before packaging.")
endif()

set(ARCHIVE_BASE "SAT_LIGHT_SIM_v${APP_VERSION}_${PLATFORM_TAG}")
set(STAGE "${OUT_DIR}/${ARCHIVE_BASE}")

message(STATUS "Packaging ${ARCHIVE_BASE}")
file(REMOVE_RECURSE "${STAGE}")
file(MAKE_DIRECTORY "${STAGE}")

# ── Payload ───────────────────────────────────────────────────────────────────
# shaders/ and assets/ come from RUNTIME_DIR, not SRC_DIR: shaders there are the compiled
# .spv files (including the SKY_LITE variant, which has no source-tree counterpart).
file(COPY "${EXE_FILE}" DESTINATION "${STAGE}"
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

foreach(dir shaders assets)
    if(NOT EXISTS "${RUNTIME_DIR}/${dir}")
        message(FATAL_ERROR "PackageRelease.cmake: ${RUNTIME_DIR}/${dir} is missing — "
                            "the build's POST_BUILD copy steps did not run.")
    endif()
    file(COPY "${RUNTIME_DIR}/${dir}" DESTINATION "${STAGE}")
endforeach()

foreach(f data/constellations.json
          data/constellations.schema.json
          data/reflector_targets.json
          THIRD_PARTY_NOTICES.txt)
    if(NOT EXISTS "${SRC_DIR}/${f}")
        message(FATAL_ERROR "PackageRelease.cmake: required file missing: ${SRC_DIR}/${f}")
    endif()
    file(COPY "${SRC_DIR}/${f}" DESTINATION "${STAGE}")
endforeach()

# ── macOS: Vulkan loader + MoltenVK + launcher ────────────────────────────────
# Vulkan on macOS only exists via MoltenVK, loaded at runtime through the Vulkan loader
# (libvulkan.1.dylib) as an ICD — not linked in directly. The exe is built against
# $VULKAN_SDK, a path that exists only on the build machine, so on a player's Mac (no SDK
# installed) it fails to launch with a dyld "Library not loaded" error unless the loader,
# MoltenVK, and an ICD manifest travel with it and the process is pointed at them.
if(DEFINED VULKAN_SDK_DIR AND NOT VULKAN_SDK_DIR STREQUAL "")
    if(NOT EXISTS "${VULKAN_SDK_DIR}/lib")
        message(FATAL_ERROR "PackageRelease.cmake: VULKAN_SDK_DIR=${VULKAN_SDK_DIR} has no lib/ — "
                            "cannot bundle the Vulkan runtime.")
    endif()
    file(MAKE_DIRECTORY "${STAGE}/lib")

    # FOLLOW_SYMLINK_CHAIN dereferences the SDK's libvulkan.dylib -> libvulkan.1.x.y.dylib
    # symlink chain, so real file content is bundled rather than a dangling link back to the
    # build machine's own $VULKAN_SDK.
    file(GLOB VK_DYLIBS "${VULKAN_SDK_DIR}/lib/libvulkan*.dylib"
                        "${VULKAN_SDK_DIR}/lib/libMoltenVK.dylib")
    if(VK_DYLIBS STREQUAL "")
        message(FATAL_ERROR "PackageRelease.cmake: no libvulkan*/libMoltenVK dylibs under "
                            "${VULKAN_SDK_DIR}/lib")
    endif()
    file(COPY ${VK_DYLIBS} DESTINATION "${STAGE}/lib" FOLLOW_SYMLINK_CHAIN)

    # The launcher is what players double-click. It regenerates the MoltenVK ICD manifest with
    # an absolute path on every run (correct no matter where the archive is unzipped to) and
    # sets DYLD_FALLBACK_LIBRARY_PATH so dyld recovers from the exe's baked-in, build-machine-
    # only library path by falling back to the bundled lib/ beside it.
    # Bracket-quoted so nothing here is expanded by CMake; @EXE_BASENAME@ is substituted below.
    set(LAUNCHER [=[#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/vulkan/icd.d"
cat > "$DIR/vulkan/icd.d/MoltenVK_icd.json" <<ICD
{
    "file_format_version" : "1.0.0",
    "ICD": { "library_path": "$DIR/lib/libMoltenVK.dylib", "api_version" : "1.2.0" }
}
ICD
export VK_ICD_FILENAMES="$DIR/vulkan/icd.d/MoltenVK_icd.json"
export VK_DRIVER_FILES="$DIR/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_FALLBACK_LIBRARY_PATH="$DIR/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
exec "$DIR/@EXE_BASENAME@" "$@"
]=])
    string(REPLACE "@EXE_BASENAME@" "${EXE_BASENAME}" LAUNCHER "${LAUNCHER}")
    file(WRITE "${STAGE}/${EXE_BASENAME}.command" "${LAUNCHER}")
    file(CHMOD "${STAGE}/${EXE_BASENAME}.command"
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

# ── Archive ───────────────────────────────────────────────────────────────────
# zip on Windows (Explorer opens it natively), tar.gz elsewhere (preserves the exec bit,
# which zip does not — the macOS launcher and the Linux binary both need it).
if(PLATFORM_TAG STREQUAL "Windows")
    set(ARCHIVE "${OUT_DIR}/${ARCHIVE_BASE}.zip")
    set(TAR_ARGS cfv "${ARCHIVE}" --format=zip .)
else()
    set(ARCHIVE "${OUT_DIR}/${ARCHIVE_BASE}.tar.gz")
    set(TAR_ARGS czf "${ARCHIVE}" .)
endif()

file(REMOVE "${ARCHIVE}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar ${TAR_ARGS}
    WORKING_DIRECTORY "${STAGE}"
    RESULT_VARIABLE tar_result
    OUTPUT_QUIET
)
if(NOT tar_result EQUAL 0)
    message(FATAL_ERROR "PackageRelease.cmake: archiving failed (${tar_result})")
endif()

file(SIZE "${ARCHIVE}" ARCHIVE_BYTES)
math(EXPR ARCHIVE_MB "${ARCHIVE_BYTES} / 1048576")
message(STATUS "Packaged ${ARCHIVE} (${ARCHIVE_MB} MB)")
