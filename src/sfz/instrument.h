#pragma once

#include "../io/audio.h"
#include "../cuda/render_types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <utility>
#include <vector>

namespace clay {

enum class Trigger { attack, release, releaseKey, first, legato };
enum class Loop { automatic, none, oneShot, continuous, sustain };
enum class OffMode { fast, normal, time };

struct Condition {
    uint8_t cc;
    uint8_t low;
    uint8_t high;
};

struct Crossfade {
    int inLow = 0, inHigh = 0, outLow = 127, outHigh = 127;
    bool power = true;
};

struct Modulation {
    ModTarget target;
    int cc;
    float depth = 0;
    int curve = 0, smooth = 0;
};

using Curve = std::array<float, 128>;

struct Filter {
    FilterType type = FilterType::none;
    float cutoff = 0, resonance = 0, keyTrack = 0, velocityTrack = 0, random = 0;
    int keyCenter = 60;
};

struct Equalizer {
    float frequency = 0, bandwidth = 1, gain = 0;
    bool enabled = false;
};

struct Lfo {
    int wave = 0;
    float frequency = 0, phase = 0, delay = 0, fade = 0;
    std::map<ModTarget, float> depths;
    std::vector<Modulation> modulation;
};

struct SecondaryEnvelope {
    std::array<float, 7> values{0, 0, 0, 0, 1, 0, 0};
    std::array<float, 7> velocity{};
    float depth = 0, velocityDepth = 0;
    bool enabled = false, dynamic = false;
};

struct FlexPoint {
    float seconds = 0, level = 0;
    std::map<int, float> timeCC;
};

struct PitchEnvelope {
    std::map<int, FlexPoint> points;
    int sustain = -1;
    float depth = 0;
    std::map<int, float> depthCC;
};

struct Region {
    int sample = -1;
    int lowKey = 0, highKey = 127, rootKey = 60;
    int lowVelocity = 0, highVelocity = 127;
    int offset = 0, end = -1, loopStart = -1, loopEnd = -1;
    int transpose = 0;
    int sequenceLength = 1, sequencePosition = 1;
    int switchLow = -1, switchHigh = -1, switchDefault = -1;
    int switchDown = -1, switchUp = -1, switchPrevious = -1;
    int notePolyphony = 0;
    int sustainCC = 64, sustainLow = 1;
    int64_t group = 0, offBy = 0;
    bool sustainEnabled = true, releaseDead = false, switchRange = false, triggerOnNote = true;
    bool chokedByGroup = false, envelopeDynamic = false, selfMask = true;
    bool explicitSequencePosition = false;
    Trigger trigger = Trigger::attack;
    Loop loop = Loop::automatic;
    OffMode offMode = OffMode::fast;
    Generator generator = Generator::none;
    float oscillatorPhase = 0;
    float volume = 0, amplitude = 1, velocityTrack = 1, pan = 0, width = 1, position = 0;
    float keyTrack = 100, tune = 0, bendUp = 200, bendDown = -200;
    int bendSmooth = 0;
    float delay = 0, envelopeDelay = 0, attack = 0, hold = 0, decay = 0, sustain = 1;
    float start = 0, release = 0.001f, releaseDecay = 0;
    float randomLow = 0, randomHigh = 1;
    float ampRandom = 0, pitchRandom = 0, offTime = .006f, loopCrossfade = .001f;
    Crossfade keyFade, velocityFade;
    std::map<int, Crossfade> controllerFades;
    bool powerControllerFade = true;
    std::vector<std::pair<int, float>> velocityCurve;
    std::vector<Condition> conditions;
    std::vector<Condition> ccTriggers;
    std::vector<Modulation> modulation;
    std::array<float, 7> velocityEnvelope{};
    std::array<Filter, 2> filters;
    std::array<Equalizer, 3> eq;
    std::map<int, Lfo> lfos;
    std::array<SecondaryEnvelope, 2> secondary;
    std::map<int, PitchEnvelope> pitchEnvelopes;
    int offsetRandom = 0;
};

struct Instrument {
    std::vector<Region> regions;
    std::array<std::vector<int>, 128> byKey;
    std::array<std::vector<int>, 128> byCC;
    std::array<bool, 128> switchKeys{};
    int initialSwitch = -1;
    bool hasOffBy = false;
    std::array<float, 128> initialCC = [] {
        std::array<float, 128> values{};
        values[7] = 100;
        values[10] = 63.5f;
        values[11] = 127;
        return values;
    }();
    std::array<bool, 128> usedCC{};
    std::map<int, float> extendedCCDefaults;
    std::map<int, Curve> curves;
};

Instrument loadInstrument(const std::filesystem::path& path, AudioBank& bank);
float curveValue(const Instrument& instrument, int curve, float value);

} // namespace clay
