@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
::  ShaderFun Release Builder
::  Builds Windows (native MSVC) and Linux (via WSL) release packages.
::  Mac: handled by GitHub Actions — see .github/workflows/release.yml
::
::  Usage:
::    release.bat             — build both platforms
::    release.bat windows     — Windows only
::    release.bat linux       — Linux (WSL) only
:: ============================================================================

set PROJ=%~dp0
if "%PROJ:~-1%"=="\" set PROJ=%PROJ:~0,-1%
set DIST=%PROJ%\dist
set BUILD_WIN=%PROJ%\build-win-release
set BUILD_LIN=%PROJ%\build-linux-release

set DO_WIN=1
set DO_LIN=1
if /i "%1"=="windows" ( set DO_LIN=0 )
if /i "%1"=="linux"   ( set DO_WIN=0 )

echo.
echo  ShaderFun Release Builder
echo  Output: %DIST%
echo.

:: ── Windows Release ──────────────────────────────────────────────────────────
if %DO_WIN%==1 (
    echo [Windows] Configuring...
    cmake -B "%BUILD_WIN%" -S "%PROJ%" -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_INSTALL_PREFIX="%DIST%\windows"
    if errorlevel 1 ( echo [Windows] Configure FAILED & exit /b 1 )

    echo [Windows] Building...
    cmake --build "%BUILD_WIN%" --config Release
    if errorlevel 1 ( echo [Windows] Build FAILED & exit /b 1 )

    echo [Windows] Packaging to dist\windows\ ...
    if exist "%DIST%\windows" rmdir /s /q "%DIST%\windows"
    mkdir "%DIST%\windows"
    :: Copy only the distributable files (exe, shaders, assets, data)
    copy /Y "%BUILD_WIN%\Release\ShaderFun.exe"              "%DIST%\windows\" >nul
    xcopy /E /I /Y "%BUILD_WIN%\Release\shaders"             "%DIST%\windows\shaders" >nul
    xcopy /E /I /Y "%BUILD_WIN%\Release\assets"              "%DIST%\windows\assets"  >nul
    copy /Y "%BUILD_WIN%\Release\constellations.json"        "%DIST%\windows\" >nul 2>&1
    copy /Y "%BUILD_WIN%\Release\constellations.schema.json" "%DIST%\windows\" >nul 2>&1

    echo [Windows] Done. ^> dist\windows\
    echo.
)

:: ── Linux Release via WSL ────────────────────────────────────────────────────
if %DO_LIN%==1 (
    where wsl >nul 2>&1
    if errorlevel 1 (
        echo [Linux] WSL not found — skipping Linux build.
        echo         Install WSL and set up the Vulkan SDK inside it to enable this step.
        goto :mac_note
    )

    :: Convert the project path to a WSL-style /mnt/... path
    for /f "delims=" %%i in ('wsl wslpath -u "%PROJ%"') do set WSL_PROJ=%%i

    echo [Linux] Configuring...
    wsl bash -lc "cmake -B !WSL_PROJ!/build-linux-release -S !WSL_PROJ! -DCMAKE_BUILD_TYPE=Release"
    if errorlevel 1 ( echo [Linux] Configure FAILED & exit /b 1 )

    echo [Linux] Building...
    wsl bash -lc "cmake --build !WSL_PROJ!/build-linux-release --parallel"
    if errorlevel 1 ( echo [Linux] Build FAILED & exit /b 1 )

    echo [Linux] Packaging to dist\linux\ ...
    if exist "%DIST%\linux" rmdir /s /q "%DIST%\linux"
    mkdir "%DIST%\linux"
    :: Copy distributable files from the Linux build directory (on Windows FS, accessible directly)
    copy /Y "%BUILD_LIN%\ShaderFun"                          "%DIST%\linux\" >nul
    xcopy /E /I /Y "%BUILD_LIN%\shaders"                     "%DIST%\linux\shaders" >nul
    xcopy /E /I /Y "%BUILD_LIN%\assets"                      "%DIST%\linux\assets"  >nul
    copy /Y "%BUILD_LIN%\constellations.json"                "%DIST%\linux\" >nul 2>&1
    copy /Y "%BUILD_LIN%\constellations.schema.json"         "%DIST%\linux\" >nul 2>&1

    echo [Linux] Done. ^> dist\linux\
    echo.
)

:mac_note
echo ============================================================================
echo  macOS builds require a Mac runner — use GitHub Actions:
echo    Push a tag v*.*.* to trigger .github/workflows/release.yml
echo    The workflow builds Windows, Linux, and macOS and uploads all three
echo    as release artifacts automatically.
echo ============================================================================
echo.
echo Done.
endlocal
