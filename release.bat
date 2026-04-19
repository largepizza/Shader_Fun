@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
::  ShaderFun Release Builder
::  Reads version from VERSION file, builds release packages, and compresses
::  them into distributable archives ready for Git/itch.io upload.
::
::  Output:
::    dist\SAT_LIGHT_SIM_v<version>_Windows.zip
::    dist\SAT_LIGHT_SIM_v<version>_Linux.tar.gz
::
::  Mac: handled by GitHub Actions — push a vX.Y.Z tag to trigger it.
::
::  Usage:
::    release.bat             — build both platforms
::    release.bat windows     — Windows only
::    release.bat linux       — Linux (WSL) only
:: ============================================================================

set PROJ=%~dp0
if "%PROJ:~-1%"=="\" set PROJ=%PROJ:~0,-1%

:: Read version from file
set /p APP_VERSION=<"%PROJ%\VERSION"
:: Trim any trailing whitespace/CR
for /f "tokens=* delims= " %%v in ("%APP_VERSION%") do set APP_VERSION=%%v

:: Build derived names
set VERSION_UNDERSCORED=%APP_VERSION:.=_%
set EXE_NAME=SAT_LIGHT_SIM_V_%VERSION_UNDERSCORED%
set ARCHIVE_BASE=SAT_LIGHT_SIM_v%APP_VERSION%

set DIST=%PROJ%\dist
set BUILD_WIN=%PROJ%\build-win-release
set BUILD_LIN=%PROJ%\build-linux-release

set DO_WIN=1
set DO_LIN=1
if /i "%1"=="windows" ( set DO_LIN=0 )
if /i "%1"=="linux"   ( set DO_WIN=0 )

echo.
echo  Release Builder  —  v%APP_VERSION%  ^(%EXE_NAME%^)
echo  Output: %DIST%
echo.

if exist "%DIST%" rmdir /s /q "%DIST%"
mkdir "%DIST%"

:: ── Windows Release ──────────────────────────────────────────────────────────
if %DO_WIN%==1 (
    echo [Windows] Clearing build cache...
    if exist "%BUILD_WIN%" rmdir /s /q "%BUILD_WIN%"

    echo [Windows] Configuring...
    cmake -B "%BUILD_WIN%" -S "%PROJ%" -DCMAKE_BUILD_TYPE=Release
    if errorlevel 1 ( echo [Windows] Configure FAILED & exit /b 1 )

    echo [Windows] Building...
    cmake --build "%BUILD_WIN%" --config Release --parallel
    if errorlevel 1 ( echo [Windows] Build FAILED & exit /b 1 )

    echo [Windows] Staging...
    mkdir "%DIST%\windows"
    copy /Y "%BUILD_WIN%\Release\%EXE_NAME%.exe"             "%DIST%\windows\" >nul
    xcopy /E /I /Y "%BUILD_WIN%\Release\shaders"             "%DIST%\windows\shaders\" >nul
    xcopy /E /I /Y "%BUILD_WIN%\Release\assets"              "%DIST%\windows\assets\"  >nul
    copy /Y "%PROJ%\data\constellations.json"                 "%DIST%\windows\" >nul
    copy /Y "%PROJ%\data\constellations.schema.json"          "%DIST%\windows\" >nul

    echo [Windows] Compressing to %ARCHIVE_BASE%_Windows.zip ...
    powershell -NoProfile -Command ^
        "Compress-Archive -Path '%DIST%\windows\*' -DestinationPath '%DIST%\%ARCHIVE_BASE%_Windows.zip' -Force"
    if errorlevel 1 ( echo [Windows] Compression FAILED & exit /b 1 )

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
    wsl bash -lc "cmake -B !WSL_PROJ!/build-linux-release -S !WSL_PROJ! -DCMAKE_BUILD_TYPE=Release"
    if errorlevel 1 ( echo [Linux] Configure FAILED & exit /b 1 )

    echo [Linux] Building...
    wsl bash -lc "cmake --build !WSL_PROJ!/build-linux-release --parallel"
    if errorlevel 1 ( echo [Linux] Build FAILED & exit /b 1 )

    echo [Linux] Staging...
    mkdir "%DIST%\linux"
    copy /Y "%BUILD_LIN%\%EXE_NAME%"                        "%DIST%\linux\" >nul
    xcopy /E /I /Y "%BUILD_LIN%\shaders"                    "%DIST%\linux\shaders\" >nul
    xcopy /E /I /Y "%BUILD_LIN%\assets"                     "%DIST%\linux\assets\"  >nul
    copy /Y "%PROJ%\data\constellations.json"                 "%DIST%\linux\" >nul
    copy /Y "%PROJ%\data\constellations.schema.json"          "%DIST%\linux\" >nul

    echo [Linux] Compressing to %ARCHIVE_BASE%_Linux.tar.gz ...
    tar -czf "%DIST%\%ARCHIVE_BASE%_Linux.tar.gz" -C "%DIST%\linux" .
    if errorlevel 1 ( echo [Linux] Compression FAILED & exit /b 1 )

    echo [Linux] Done. ^> dist\%ARCHIVE_BASE%_Linux.tar.gz
    echo.
)

:mac_note
echo  macOS: push a version tag to trigger GitHub Actions:
echo    git tag v%APP_VERSION% ^&^& git push origin v%APP_VERSION%
echo  The workflow builds Windows, Linux, and macOS and attaches all three
echo  as release artifacts at github.com/YOUR_REPO/releases
echo.
echo  Done.
endlocal
