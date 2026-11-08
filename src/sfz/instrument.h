#pragma once

#include "../io/audio.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace clay {

enum class Trigger { attack, release, releaseKey, first, legato };
enum class Loop { automatic, none, oneShot, continuous, sustain };

struct Condition {
    uint8_t cc;
    uint8_t low;
    uint8_t high;
};

struct Region {
    int sample = -1;
    int lowKey = 0, highKey = 127, rootKey = 60;
    int lowVelocity = 0, highVelocity = 127;
    int offset = 0, end = -1, loopStart = -1, loopEnd = -1;
    int sequenceLength = 1, sequencePosition = 1;
    int sustainCC = 64, sustainLow = 1;
    bool sustainEnabled = true, releaseDead = false;
    Trigger trigger = Trigger::attack;
    Loop loop = Loop::automatic;
    float volume = 0, amplitude = 1, velocityTrack = 1, pan = 0;
    float keyTrack = 100, tune = 0, bendUp = 200, bendDown = -200;
    float delay = 0, envelopeDelay = 0, attack = 0, hold = 0, decay = 0, sustain = 1;
    float start = 0, release = 0.001f, releaseDecay = 0;
    float randomLow = 0, randomHigh = 1;
    std::vector<Condition> conditions;
};

struct Instrument {
    std::vector<Region> regions;
    std::array<std::vector<int>, 128> byKey;
    std::array<float, 128> initialCC = [] {
        std::array<float, 128> values{};
        values[7] = 100;
        values[10] = 63.5f;
        values[11] = 127;
        return values;
    }();
    std::array<bool, 128> usedCC{};
};

Instrument loadInstrument(const std::filesystem::path& path, AudioBank& bank);

} // namespace clay
