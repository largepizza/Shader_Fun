// AudioSystem.cpp — miniaudio implementation.
// MINIAUDIO_IMPLEMENTATION must appear in exactly one translation unit.
// miniaudio pulls in minimp3 and minivorbis automatically; no extra libs needed.
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "AudioSystem.h"
#include <algorithm>
#include <cstdio>

// ── init ─────────────────────────────────────────────────────────────────────
void AudioSystem::init()
{
    engine_ = new ma_engine;
    if (ma_engine_init(nullptr, engine_) != MA_SUCCESS) {
        fprintf(stderr, "[AudioSystem] ma_engine_init failed — audio disabled.\n");
        delete engine_;
        engine_ = nullptr;
        return;
    }
    ma_engine_set_volume(engine_, masterVol_);

    // Music sub-mix: all streaming background tracks attach here.
    musicGroup_ = new ma_sound_group;
    if (ma_sound_group_init(engine_, 0, nullptr, musicGroup_) != MA_SUCCESS) {
        fprintf(stderr, "[AudioSystem] Failed to create music group.\n");
        delete musicGroup_;
        musicGroup_ = nullptr;
    } else {
        ma_sound_group_set_volume(musicGroup_, musicVol_);
    }

    // SFX sub-mix: all fire-and-forget UI sounds attach here.
    sfxGroup_ = new ma_sound_group;
    if (ma_sound_group_init(engine_, 0, nullptr, sfxGroup_) != MA_SUCCESS) {
        fprintf(stderr, "[AudioSystem] Failed to create SFX group.\n");
        delete sfxGroup_;
        sfxGroup_ = nullptr;
    } else {
        ma_sound_group_set_volume(sfxGroup_, sfxVol_);
    }

    initialized_ = true;
}

// ── cleanup ───────────────────────────────────────────────────────────────────
void AudioSystem::cleanup()
{
    if (!initialized_) return;

    stopMusic();

    if (sfxGroup_) {
        ma_sound_group_uninit(sfxGroup_);
        delete sfxGroup_;
        sfxGroup_ = nullptr;
    }
    if (musicGroup_) {
        ma_sound_group_uninit(musicGroup_);
        delete musicGroup_;
        musicGroup_ = nullptr;
    }
    if (engine_) {
        ma_engine_uninit(engine_);
        delete engine_;
        engine_ = nullptr;
    }

    initialized_ = false;
}

// ── update ────────────────────────────────────────────────────────────────────
// Poll for track completion each frame and advance the playlist.
// miniaudio's end-of-stream callback runs on the audio thread, so polling
// ma_sound_at_end() from the main thread is the safe, simple approach here.
void AudioSystem::update(float /*dt*/)
{
    if (!initialized_ || !music_) return;

    if (ma_sound_at_end(music_)) {
        ma_sound_uninit(music_);
        delete music_;
        music_ = nullptr;

        if (!tracks_.empty()) {
            trackIdx_ = (trackIdx_ + 1) % (int)tracks_.size();
            loadTrack(trackIdx_);
        }
    }
}

// ── playlist management ───────────────────────────────────────────────────────
void AudioSystem::addTrack(const std::string& path)
{
    tracks_.push_back(path);
}

void AudioSystem::clearTracks()
{
    stopMusic();
    tracks_.clear();
    trackIdx_ = 0;
}

void AudioSystem::startMusic()
{
    if (tracks_.empty() || !initialized_) return;
    trackIdx_ = 0;
    loadTrack(0);
}

void AudioSystem::stopMusic()
{
    if (music_) {
        ma_sound_uninit(music_);
        delete music_;
        music_ = nullptr;
    }
}

// ── loadTrack (private) ───────────────────────────────────────────────────────
// Stream from disk (MA_SOUND_FLAG_STREAM) so large music files don't sit in RAM.
// The file is read relative to the working directory, which is the exe directory
// at runtime (matching the same convention used by shader and icon loading).
void AudioSystem::loadTrack(int idx)
{
    if (!initialized_ || idx < 0 || idx >= (int)tracks_.size()) return;

    stopMusic();
    music_ = new ma_sound;

    ma_uint32 flags = MA_SOUND_FLAG_STREAM;  // stream; don't decode entire file
    if (ma_sound_init_from_file(engine_,
                                tracks_[idx].c_str(),
                                flags,
                                musicGroup_,   // attach to music sub-mix
                                nullptr,
                                music_) != MA_SUCCESS)
    {
        fprintf(stderr, "[AudioSystem] Failed to load track: %s\n", tracks_[idx].c_str());
        delete music_;
        music_ = nullptr;
        return;
    }

    ma_sound_start(music_);
}

// ── SFX ───────────────────────────────────────────────────────────────────────
// ma_engine_play_sound creates a self-managed, fire-and-forget node attached to
// sfxGroup_ so the SFX sub-mix volume applies automatically.
void AudioSystem::playSfx(const std::string& path)
{
    if (!initialized_) return;
    ma_engine_play_sound(engine_, path.c_str(), sfxGroup_);
}

// ── Volume setters ────────────────────────────────────────────────────────────
void AudioSystem::setMasterVolume(float v)
{
    masterVol_ = std::clamp(v, 0.0f, 1.0f);
    if (engine_) ma_engine_set_volume(engine_, masterVol_);
}

void AudioSystem::setMusicVolume(float v)
{
    musicVol_ = std::clamp(v, 0.0f, 1.0f);
    applyMusicVolume();
}

void AudioSystem::setSfxVolume(float v)
{
    sfxVol_ = std::clamp(v, 0.0f, 1.0f);
    if (sfxGroup_) ma_sound_group_set_volume(sfxGroup_, sfxVol_);
}

void AudioSystem::applyMusicVolume()
{
    if (musicGroup_) ma_sound_group_set_volume(musicGroup_, musicVol_);
}
