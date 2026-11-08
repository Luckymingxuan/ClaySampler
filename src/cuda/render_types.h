#pragma once

struct VoiceSpan {
    unsigned long long sample;
    double position;
    double step;
    double pitchSlope;
    int channels, begin, end, sourceEnd, loopFirst, loopEnd, loopCrossfade;
    int stereoMatrix;
    int attackStart, holdStart, decayStart, releaseStart, fadeStart, bendEnd;
    float gain, pan, left, right, start, attackStep, decayRate, sustain, releaseValue, releaseRate;
    float fadeValue, fadeStep;
    float widthDirect, widthCross, positionLeft, positionRight;
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
