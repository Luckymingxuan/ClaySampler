#pragma once

enum class ModTarget {
    amplitude, volume, pan, width, position, pitch,
    envDelay, envAttack, envHold, envDecay, envSustain, envStart, envRelease,
    offset, delay, cutoff, resonance, cutoff2, resonance2, filterVelocity, filterVelocity2,
    lfoFrequency, lfoPhase, lfoDelay, lfoFade,
    filterEnvDelay, filterEnvAttack, filterEnvHold, filterEnvDecay, filterEnvSustain, filterEnvStart, filterEnvRelease,
    pitchEnvDelay, pitchEnvAttack, pitchEnvHold, pitchEnvDecay, pitchEnvSustain, pitchEnvStart, pitchEnvRelease,
    filterEnvDepth, pitchEnvDepth,
    eq1Frequency, eq1Bandwidth, eq1Gain, eq2Frequency, eq2Bandwidth, eq2Gain, eq3Frequency, eq3Bandwidth, eq3Gain
};

enum class FilterType { none, low1, high1, low2, high2, band2, notch2, low4, high4, band4, low6, high6, band6, peak };
enum class Generator { none, sine, silence, noise };
constexpr int voiceFilterSlots = 5;

struct VoiceFilter {
    FilterType type;
    float cutoff, resonance, bandwidth, gain;
};

struct EnvelopeShape {
    int attackStart, holdStart, decayStart, releaseStart, fadeStart;
    float start, attackStep, decayRate, sustain, releaseValue, releaseRate, fadeValue, fadeStep;
    float depth, releaseSeconds;
};

struct VoiceSpan {
    unsigned long long sample;
    double position;
    double step;
    double pitchSlope;
    int channels, begin, end, sourceEnd, loopFirst, loopEnd, loopCrossfade;
    int stereoMatrix;
    Generator generator;
    unsigned noiseSeed;
    int noiseStart;
    int firstModulation, modulationCount;
    int firstLfo, lfoCount;
    int filterTrack, nextFilteredSpan;
    unsigned long long filteredOffset;
    unsigned long long phaseOffset;
    int variablePitch;
    float minimumPitchRatio;
    int filterControls[4];
    int sampleRate;
    VoiceFilter filters[voiceFilterSlots];
    EnvelopeShape extraEnvelope[2];
    int activeEnvelope[2];
    int attackStart, holdStart, decayStart, releaseStart, fadeStart, bendEnd;
    float gain, pan, left, right, start, attackStep, decayRate, sustain, releaseValue, releaseRate;
    float fadeValue, fadeStep;
    float widthDirect, widthCross, positionLeft, positionRight;
    float width, stereoPosition;
};

struct Tile {
    unsigned long long output;
    int frame, frames, first, count;
    int controls[4];
};

struct ControlPoint {
    int frame;
    float value, step;
};

struct VoiceModulation {
    int first, last;
    ModTarget target;
    float depth;
    int shapeFirst = -1, shapeLast = 0;
};

struct FilterTile {
    int span, frame;
};
