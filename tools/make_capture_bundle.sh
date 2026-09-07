#!/bin/bash
# Wraps the built executable in a minimal .app bundle that declares MetalCaptureEnabled,
# so `MTL_CAPTURE_ENABLED=1` + MoltenVK's auto-capture (or Xcode) can grab a GPU frame.
#
# Run from the project root AFTER building. Then, from build/ (so assets resolve):
#   cd build
#   MTL_CAPTURE_ENABLED=1 \
#   MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=2 \
#   MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=$HOME/satlight.gputrace \
#   ./SatLightSimCapture.app/Contents/MacOS/satlightsim
# then open ~/satlight.gputrace in Xcode.
set -e

EXE=$(ls build/SAT_LIGHT_SIM_V_* 2>/dev/null | head -1)
[ -z "$EXE" ] && { echo "No built executable in build/ — build first."; exit 1; }

APP=build/SatLightSimCapture.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cp "$EXE" "$APP/Contents/MacOS/satlightsim"

# The exe resolves shaders/ and *.json relative to its OWN directory, not the CWD.
# Symlink the runtime assets from build/ next to the bundled binary.
MACOS_DIR="$APP/Contents/MacOS"
for item in shaders assets constellations.json constellations.schema.json reflector_targets.json settings.json; do
    [ -e "build/$item" ] && ln -sfn "$(cd build && pwd)/$item" "$MACOS_DIR/$item"
done

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>       <string>satlightsim</string>
    <key>CFBundleIdentifier</key>       <string>com.local.satlightsim.capture</string>
    <key>CFBundleName</key>             <string>SatLightSimCapture</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key>   <string>10.15</string>
    <key>MetalCaptureEnabled</key>      <true/>
</dict>
</plist>
PLIST

echo "Built $APP"
echo "Run it from build/ so shaders/ and *.json resolve next to the CWD."
