#pragma once
// Include miniaudio declarations (NOT the implementation — MINIAUDIO_IMPLEMENTATION
// is defined only in AudioSystem.cpp). This brings in the type definitions needed
// for the pointer members below without triggering any conflicting forward declarations.
#include "miniaudio.h"

#include <string>
#include <vector>

// ── AudioSystem ───────────────────────────────────────────────────────────────
// Thin wrapper around miniaudio that provides:
//   • A sequential music playlist (streaming, no gap between tracks).
//   • Fire-and-forget UI sound effects.
//   • Three independent volume tiers: master > music, master > sfx.
//
// Lifecycle (managed by App):
//   init()      — start the audio device (WASAPI on Windows)
//   update(dt)  — call once per frame to advance the playlist on track end
//   cleanup()   — stop all audio and release the device
//
// Simulation integration:
//   App calls sim->setAudio(&audio) after both are initialised so each
//   simulation can set up its own playlist and trigger UI sounds from buildUI.
// ─────────────────────────────────────────────────────────────────────────────
class AudioSystem {
public:
    AudioSystem()  = default;
    ~AudioSystem() { cleanup(); }

    // Non-copyable — owns native audio resources.
    AudioSystem(const AudioSystem&)            = delete;
    AudioSystem& operator=(const AudioSystem&) = delete;

    void init();
    void cleanup();
    void update(float dt);   // advance playlist when current track ends

    // ── Music playlist ───────────────────────────────────────────────────────
    // Tracks play in order, then loop back to the first track.
    void addTrack(const std::string& path);
    void clearTracks();
    void startMusic();       // begin from track 0
    void stopMusic();

    // ── UI sound effects (fire-and-forget, very short) ───────────────────────
    void playSfx(const std::string& path);

    // ── Volume controls (0.0 – 1.0) ──────────────────────────────────────────
    // master  : overall output gain applied to both music and sfx
    // music   : sub-mix gain for streaming background tracks only
    // sfx     : sub-mix gain for UI sound effects only
    void  setMasterVolume(float v);
    void  setMusicVolume(float v);
    void  setSfxVolume(float v);
    float getMasterVolume() const { return masterVol_; }
    float getMusicVolume()  const { return musicVol_;  }
    float getSfxVolume()    const { return sfxVol_;    }

    bool isInitialized() const { return initialized_; }

private:
    void loadTrack(int idx);    // load + start tracks_[idx]
    void applyMusicVolume();    // push musicVol_ into the music group

    ma_engine*      engine_     = nullptr;
    ma_sound_group* musicGroup_ = nullptr;  // music sub-mix node
    ma_sound_group* sfxGroup_   = nullptr;  // SFX  sub-mix node
    ma_sound*       music_      = nullptr;  // currently streaming track

    std::vector<std::string> tracks_;
    int  trackIdx_   = 0;
    bool initialized_= false;

    float masterVol_ = 0.8f;
    float musicVol_  = 0.6f;
    float sfxVol_    = 1.0f;
};
