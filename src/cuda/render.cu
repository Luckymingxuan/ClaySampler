#include "render_types.h"

constexpr int taps = 48;
constexpr int phases = 2048;

__device__ float envelope(const VoiceSpan& voice, int frame)
{
    if (voice.releaseStart >= 0 && frame >= voice.releaseStart) {
        if (frame >= voice.fadeStart)
            return fmaxf(0, voice.fadeValue - (frame - voice.fadeStart + 1) * voice.fadeStep);
        return voice.releaseValue * expf((frame - voice.releaseStart + 1) * voice.releaseRate);
    }
    if (frame < voice.attackStart) return voice.start;
    if (frame < voice.holdStart) return fminf(1, voice.start + (frame - voice.attackStart + 1) * voice.attackStep);
    if (frame < voice.decayStart) return 1;
    if (voice.decayRate == 0) return voice.sustain;
    return fmaxf(voice.sustain, expf((frame - voice.decayStart + 1) * voice.decayRate));
}

__device__ float2 fetch(const float* samples, const VoiceSpan& voice, int frame)
{
    if (frame < 0 || frame >= voice.sourceEnd)
        return make_float2(0, 0);
    auto index = voice.sample + static_cast<unsigned long long>(frame) * voice.channels;
    auto left = samples[index];
    return make_float2(left, voice.channels == 1 ? left : samples[index + 1]);
}

__device__ float2 interpolateAt(const float* samples, const float* sinc, const VoiceSpan& voice, double pos)
{
    const auto center = __double2int_rd(pos);
    if (pos == center)
        return fetch(samples, voice, center);
    const int phase = __double2int_rn((pos - center) * phases);
    float2 result = make_float2(0, 0);
    if (center >= 23 && center < voice.sourceEnd - 24) {
        auto index = voice.sample + static_cast<unsigned long long>(center - 23) * voice.channels;
        if (voice.channels == 1) {
            #pragma unroll 8
            for (int tap = 0; tap < taps; ++tap)
                result.x = fmaf(sinc[tap * (phases + 1) + phase], samples[index + tap], result.x);
            result.y = result.x;
        } else {
            #pragma unroll 8
            for (int tap = 0; tap < taps; ++tap) {
                auto weight = sinc[tap * (phases + 1) + phase];
                result.x = fmaf(weight, samples[index + 2 * tap], result.x);
                result.y = fmaf(weight, samples[index + 2 * tap + 1], result.y);
            }
        }
    } else {
        for (int tap = 0; tap < taps; ++tap) {
            auto sample = fetch(samples, voice, center - 23 + tap);
            auto weight = sinc[tap * (phases + 1) + phase];
            result.x = fmaf(weight, sample.x, result.x);
            result.y = fmaf(weight, sample.y, result.y);
        }
    }
    return result;
}

__device__ float2 interpolate(const float* samples, const float* sinc, const VoiceSpan& voice, int frame)
{
    const auto elapsed = frame - voice.begin;
    double distance = elapsed;
    if (voice.pitchSlope != 0)
        distance = expm1(voice.pitchSlope * elapsed) / expm1(voice.pitchSlope);
    double pos = voice.position + distance * voice.step;
    const int period = voice.loopEnd - voice.loopFirst;
    if (voice.loopEnd > 0 && pos >= voice.loopEnd)
        pos = voice.loopFirst + fmod(pos - voice.loopFirst, double(period));
    auto sample = interpolateAt(samples, sinc, voice, pos);
    const int fadeStart = voice.loopEnd - voice.loopCrossfade;
    if (voice.loopEnd > 0 && voice.loopCrossfade > 0 && pos >= fadeStart) {
        auto entry = interpolateAt(samples, sinc, voice, pos - period);
        auto phase = float((pos - fadeStart) / voice.loopCrossfade);
        auto blend = .5f - .5f * cosf(3.14159265358979323846f * phase);
        sample.x = fmaf(blend, entry.x - sample.x, sample.x);
        sample.y = fmaf(blend, entry.y - sample.y, sample.y);
    }
    return sample;
}

__device__ short pcm16(float value)
{
    return static_cast<short>(__float2int_rn(fminf(32767, fmaxf(-32768, value * 32768))));
}

__device__ float controlAt(const ControlPoint* controls, int low, int high, int frame)
{
    while (low + 1 < high) {
        int mid = low + (high - low) / 2;
        if (controls[mid].frame <= frame)
            low = mid;
        else
            high = mid;
    }
    const auto point = controls[low];
    return fmaf(point.step, frame - point.frame, point.value);
}

extern "C" __global__ void renderTiles(const float* samples, const float* sinc, const VoiceSpan* voices,
                                      const Tile* tiles, const int* indices, const ControlPoint* controls,
                                      short2* output, int firstTile)
{
    const Tile tile = tiles[firstTile + blockIdx.x];
    const auto lane = int(threadIdx.x);
    if (lane >= tile.frames)
        return;
    const auto frame = tile.frame + lane;
    const auto volume = controlAt(controls, tile.controls[0], tile.controls[1], frame);
    const auto expression = controlAt(controls, tile.controls[1], tile.controls[2], frame);
    const auto pan = controlAt(controls, tile.controls[2], tile.controls[3], frame);
    float left = 0, right = 0;
    for (int i = tile.first; i < tile.first + tile.count; ++i) {
        const VoiceSpan& voice = voices[indices[i]];
        if (frame < voice.begin || frame >= voice.end)
            continue;
        const auto level = envelope(voice, frame);
        if (level <= 0)
            continue;
        const auto sample = interpolate(samples, sinc, voice, frame);
        auto leftGain = voice.left;
        auto rightGain = voice.right;
        if (pan != 0) {
            constexpr float quarterPi = .7853981633974483f;
            auto p = fminf(1, fmaxf(-1, voice.pan + pan));
            leftGain = voice.gain * cosf((1 + p) * quarterPi);
            rightGain = voice.gain * cosf((1 - p) * quarterPi);
        }
        if (voice.stereoMatrix) {
            const auto l = sample.x * leftGain;
            const auto r = sample.y * rightGain;
            left = fmaf(fmaf(r, voice.widthCross, l * voice.widthDirect), level * voice.positionLeft, left);
            right = fmaf(fmaf(l, voice.widthCross, r * voice.widthDirect), level * voice.positionRight, right);
        } else {
            left = fmaf(sample.x, level * leftGain, left);
            right = fmaf(sample.y, level * rightGain, right);
        }
    }
    const auto amplitude = volume * expression;
    output[tile.output + lane] = make_short2(pcm16(left * amplitude), pcm16(right * amplitude));
}
