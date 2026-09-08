@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
::  SAT LIGHT SIM Release Builder (Windows convenience wrapper)
::
::  Thin driver over the CMake presets + the `package-release` target. It does NOT
::  contain its own copy of the staging file list any more — that lives in
::  cmake/PackageRelease.cmake, which CI (.github/workflows/release.yml) also uses,
::  so a local archive and a tagged one have identical layouts by construction.
::
::  Output:
::    dist\SAT_LIGHT_SIM_v<version>_Windows.zip
::    dist\SAT_LIGHT_SIM_v<version>_Linux.tar.gz     (via WSL)
::
::  macOS: built by GitHub Actions as a universal (arm64 + x86_64) binary — push a
::  vX.Y.Z tag to trigger it. Cannot be cross-built from Windows.
::
::  Usage:
::    release.bat             — build both platforms
::    release.bat windows     — Windows only
::    release.bat linux       — Linux (WSL) only
:: ============================================================================

set PROJ=%~dp0
if "%PROJ:~-1%"=="\" set PROJ=%PROJ:~0,-1%

:: Read version from file (display only — CMake reads VERSION itself)
set /p APP_VERSION=<"%PROJ%\VERSION"
for /f "tokens=* delims= " %%v in ("%APP_VERSION%") do set APP_VERSION=%%v
set ARCHIVE_BASE=SAT_LIGHT_SIM_v%APP_VERSION%

set DO_WIN=1
set DO_LIN=1
if /i "%1"=="windows" ( set DO_LIN=0 )
if /i "%1"=="linux"   ( set DO_WIN=0 )

echo.
echo  Release Builder  —  v%APP_VERSION%
echo  Output: %PROJ%\dist
echo.

:: ── Windows Release ──────────────────────────────────────────────────────────
if %DO_WIN%==1 (
    echo [Windows] Clearing build cache...
    if exist "%PROJ%\build-win-release" rmdir /s /q "%PROJ%\build-win-release"

    echo [Windows] Configuring...
    cmake --preset windows-release
    if errorlevel 1 ( echo [Windows] Configure FAILED & exit /b 1 )

    echo [Windows] Building...
    cmake --build --preset windows-release --parallel
    if errorlevel 1 ( echo [Windows] Build FAILED & exit /b 1 )

    echo [Windows] Packaging...
    cmake --build --preset windows-package
    if errorlevel 1 ( echo [Windows] Packaging FAILED & exit /b 1 )

    echo [Windows] Done. ^> dist\%ARCHIVE_BASE%_Windows.zip
    echo.
)

:: ── Linux Release via WSL ────────────────────────────────────────────────────
if %DO_LIN%==1 (
    where wsl >nul 2>&1
    if errorlevel 1 (
        echo [Linux] WSL not found — skipping Linux build.
        echo         Install WSL + Vulkan SDK inside it to enable this step.
        goto :mac_note
    )

    for /f "delims=" %%i in ('wsl wslpath -u "%PROJ%"') do set WSL_PROJ=%%i

    echo [Linux] Clearing build cache...
    wsl bash -lc "rm -rf !WSL_PROJ!/build-linux-release"

    echo [Linux] Configuring...
    wsl bash -lc "cd !WSL_PROJ! && cmake --preset linux-release"
    if errorlevel 1 ( echo [Linux] Configure FAILED & exit /b 1 )

    echo [Linux] Building...
    wsl bash -lc "cd !WSL_PROJ! && cmake --build --preset linux-release --parallel"
    if errorlevel 1 ( echo [Linux] Build FAILED & exit /b 1 )

    echo [Linux] Packaging...
    wsl bash -lc "cd !WSL_PROJ! && cmake --build --preset linux-package"
    if errorlevel 1 ( echo [Linux] Packaging FAILED & exit /b 1 )

    echo [Linux] Done. ^> dist\%ARCHIVE_BASE%_Linux.tar.gz
    echo.
)

:mac_note
echo  macOS: push a version tag to trigger GitHub Actions:
echo    git tag v%APP_VERSION% ^&^& git push origin v%APP_VERSION%
echo  The workflow builds Windows, Linux, and a universal (arm64 + x86_64) macOS
echo  binary, and attaches all three as release artifacts.
echo.
echo  Done.
endlocal
