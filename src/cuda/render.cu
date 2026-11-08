#include "render_types.h"

constexpr int taps = 48;
constexpr int phases = 2048;

template<class Envelope>
__device__ float envelope(const Envelope& voice, int frame)
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

__device__ float2 interpolate(const float* samples, const float* sinc, const VoiceSpan& voice, int frame, double pos)
{
    if (pos < 0) {
        const auto elapsed = frame - voice.begin;
        double distance = elapsed;
        if (voice.pitchSlope != 0)
            distance = expm1(voice.pitchSlope * elapsed) / expm1(voice.pitchSlope);
        pos = voice.position + distance * voice.step;
    }
    if (voice.generator != Generator::none) {
        if (voice.generator == Generator::noise) {
            unsigned bits = voice.noiseSeed + static_cast<unsigned>(frame - voice.noiseStart);
            bits = (bits ^ (bits >> 16)) * 0x7feb352du;
            bits = (bits ^ (bits >> 15)) * 0x846ca68bu;
            bits ^= bits >> 16;
            const auto value = float(bits >> 8) * (2.f / 16777216.f) - 1;
            return make_float2(value, value);
        }
        const auto value = voice.generator == Generator::sine
            ? -sinf(6.28318530717958647692f * float(pos - floor(pos))) : 0.f;
        return make_float2(value, value);
    }
    const int period = voice.loopEnd - voice.loopFirst;
    if (voice.loopEnd > 0 && pos >= voice.loopEnd)
        pos = voice.loopFirst + fmod(pos - voice.loopFirst, double(period));
    if (pos >= voice.sourceEnd)
        return make_float2(0, 0);
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

__device__ float modulationAt(const VoiceModulation& mod, const ControlPoint* controls, int frame)
{
    auto value = mod.first < 0 ? mod.depth : mod.depth * controlAt(controls, mod.first, mod.last, frame);
    if (mod.shapeFirst >= 0)
        value *= controlAt(controls, mod.shapeFirst, mod.shapeLast, frame);
    return value;
}

constexpr float quarterPi = .7853981633974483f;

__device__ float2 panGains(const VoiceSpan& voice, float offset)
{
    if (offset == 0)
        return make_float2(voice.left, voice.right);
    const auto p = fminf(1, fmaxf(-1, voice.pan + offset));
    return make_float2(voice.gain * cosf((1 + p) * quarterPi), voice.gain * cosf((1 - p) * quarterPi));
}

__device__ float2 voiceAudio(const float* samples, const float* sinc, const VoiceSpan& voice,
                             const ControlPoint* controls, const VoiceModulation* modulation,
                             float pan, int frame, bool filterInput = false, double phase = -1)
{
    auto level = envelope(voice, frame);
    if (level <= 0)
        return make_float2(0, 0);
    float db = 0, widthOffset = 0, positionOffset = 0;
    for (int j = voice.firstModulation; j < voice.firstModulation + voice.modulationCount; ++j) {
        const auto mod = modulation[j];
        if (mod.target > ModTarget::position)
            continue;
        const auto value = modulationAt(mod, controls, frame);
        switch (mod.target) {
        case ModTarget::amplitude: level *= value; break;
        case ModTarget::volume: db += value; break;
        case ModTarget::pan: pan += value; break;
        case ModTarget::width: widthOffset += value; break;
        case ModTarget::position: positionOffset += value; break;
        default: break;
        }
    }
    if (db != 0)
        level *= exp2f(db * .166096404744368f);
    const auto sample = interpolate(samples, sinc, voice, frame, phase);
    if (filterInput && voice.channels == 1)
        return make_float2(sample.x * level, sample.x * level);
    const auto gains = panGains(voice, pan);
    const auto l = sample.x * gains.x;
    const auto r = sample.y * gains.y;
    if (voice.stereoMatrix) {
        auto direct = voice.widthDirect;
        auto cross = voice.widthCross;
        if (widthOffset != 0) {
            const auto width = fminf(1, fmaxf(-1, voice.width + widthOffset));
            direct = cosf((1 - width) * quarterPi);
            cross = cosf((1 + width) * quarterPi);
        }
        auto positionLeft = voice.positionLeft;
        auto positionRight = voice.positionRight;
        if (positionOffset != 0) {
            const auto position = fminf(1, fmaxf(-1, voice.stereoPosition + positionOffset));
            positionLeft = cosf((1 + position) * quarterPi);
            positionRight = cosf((1 - position) * quarterPi);
        }
        return make_float2(fmaf(r, cross, l * direct) * (level * positionLeft),
                           fmaf(l, cross, r * direct) * (level * positionRight));
    }
    return make_float2(l * level, r * level);
}

__device__ float jobGain(const ControlPoint* controls, const int* offsets, int frame)
{
    return controlAt(controls, offsets[0], offsets[1], frame) *
           controlAt(controls, offsets[1], offsets[2], frame);
}

extern "C" __global__ void renderFilterSources(const float* samples, const float* sinc, const VoiceSpan* voices,
    const FilterTile* tiles, const ControlPoint* controls, const VoiceModulation* modulation,
    const double* phase, float2* filtered, int firstTile)
{
    const auto tile = tiles[firstTile + blockIdx.x];
    const auto& voice = voices[tile.span];
    const auto frame = tile.frame + int(threadIdx.x);
    if (frame >= voice.end)
        return;
    const auto* offsets = voice.filterControls;
    const auto position = voice.variablePitch ? phase[voice.phaseOffset + frame - voice.begin] : -1;
    auto sample = voiceAudio(samples, sinc, voice, controls, modulation,
        controlAt(controls, offsets[2], offsets[3], frame), frame, true, position);
    const auto gain = jobGain(controls, offsets, frame);
    filtered[voice.filteredOffset + frame - voice.begin] = make_float2(sample.x * gain, sample.y * gain);
}

extern "C" __global__ void prepareVoicePitch(const VoiceSpan* voices, const int* heads,
    const ControlPoint* controls, const VoiceModulation* modulation, double* phase, int count)
{
    const auto track = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (track >= count)
        return;
    const auto head = heads[track];
    if (!voices[head].variablePitch)
        return;
    auto position = voices[head].position;
    int previousLoopEnd = 0, previousLoopStart = 0;
    for (int span = head; span >= 0; span = voices[span].nextFilteredSpan) {
        const auto& voice = voices[span];
        if (!voice.loopEnd && previousLoopEnd > 0 && position >= previousLoopEnd)
            position = previousLoopStart + fmod(position - previousLoopStart,
                double(previousLoopEnd - previousLoopStart));
        previousLoopEnd = voice.loopEnd;
        previousLoopStart = voice.loopFirst;
        const auto slope = exp(voice.pitchSlope);
        auto step = voice.step;
        for (int frame = voice.begin; frame < voice.end; ++frame) {
            phase[voice.phaseOffset + frame - voice.begin] = position;
            float cents = 0;
            auto depth = voice.extraEnvelope[1].depth;
            for (int j = voice.firstModulation; j < voice.firstModulation + voice.modulationCount; ++j) {
                const auto mod = modulation[j];
                if (mod.target == ModTarget::pitch)
                    cents += modulationAt(mod, controls, frame);
                if (mod.target == ModTarget::pitchEnvDepth)
                    depth += modulationAt(mod, controls, frame);
            }
            if (voice.activeEnvelope[1])
                cents += depth * envelope(voice.extraEnvelope[1], frame);
            position += step * exp2f(fminf(24000, fmaxf(-24000, cents)) / 1200);
            step *= slope;
        }
    }
}

struct Coefficients {
    float b0, b1, b2, a1, a2;
};

__device__ Coefficients coefficients(FilterType type, float cutoff, float resonance, int rate,
                                     float bandwidth, float gain)
{
    const auto w = 6.28318530717958647692f * fminf(fminf(20000.f, rate * .499f), fmaxf(1.f, cutoff)) / rate;
    if (type == FilterType::low1 || type == FilterType::high1) {
        const auto p = expf(-w);
        if (type == FilterType::low1)
            return {1 - p, 0, 0, -p, 0};
        const auto high = (1 + p) * .5f;
        return {high, -high, 0, -p, 0};
    }
    float s, c;
    sincosf(w, &s, &c);
    if (type == FilterType::peak) {
        const auto a = exp2f(fminf(60, fmaxf(-96, gain)) * .083048202372184f);
        const auto width = fminf(12, fmaxf(.01f, bandwidth));
        const auto alpha = s * sinhf(fminf(20, .346573590279973f * width * w / s));
        const auto inv = 1 / (1 + alpha / a);
        const auto mid = -2 * c * inv;
        return {(1 + alpha * a) * inv, mid, (1 - alpha * a) * inv, mid, (1 - alpha / a) * inv};
    }
    const auto alpha = .5f * s * exp2f(-.166096404744368f * fminf(60, fmaxf(-60, resonance)));
    const auto inv = 1.f / (1 + alpha);
    const auto a1 = -2 * c * inv;
    const auto a2 = (1 - alpha) * inv;
    if (type == FilterType::band2 || type == FilterType::band4 || type == FilterType::band6)
        return {alpha * inv, 0, -alpha * inv, a1, a2};
    if (type == FilterType::notch2)
        return {inv, a1, inv, a1, a2};
    const bool high = type == FilterType::high2 || type == FilterType::high4 || type == FilterType::high6;
    const auto b = (1 + (high ? c : -c)) * .5f * inv;
    return {b, (high ? -2 : 2) * b, b, a1, a2};
}

__device__ int sections(FilterType type)
{
    if (type == FilterType::none) return 0;
    if (type == FilterType::peak) return 1;
    if (type >= FilterType::low6) return 3;
    if (type >= FilterType::low4) return 2;
    return 1;
}

__device__ void smoothCoefficients(Coefficients& current, const Coefficients& target, float amount)
{
    current.b0 = fmaf(amount, target.b0 - current.b0, current.b0);
    current.b1 = fmaf(amount, target.b1 - current.b1, current.b1);
    current.b2 = fmaf(amount, target.b2 - current.b2, current.b2);
    current.a1 = fmaf(amount, target.a1 - current.a1, current.a1);
    current.a2 = fmaf(amount, target.a2 - current.a2, current.a2);
}

__device__ float filterSample(float x, const Coefficients& c, float& z1, float& z2)
{
    auto y = fmaf(c.b0, x, z1);
    z1 = fmaf(-c.a1, y, fmaf(c.b1, x, z2));
    z2 = fmaf(-c.a2, y, c.b2 * x);
    return y;
}

extern "C" __global__ void applyVoiceFilters(const VoiceSpan* voices, const int* heads,
    const ControlPoint* controls, const VoiceModulation* modulation, float2* filtered, int count)
{
    const auto track = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (track >= count)
        return;
    bool hasFilters = false;
    for (const auto& filter : voices[heads[track]].filters)
        hasFilters |= filter.type != FilterType::none;
    if (!hasFilters) return;
    float2 z1[voiceFilterSlots][3]{}, z2[voiceFilterSlots][3]{};
    Coefficients current[voiceFilterSlots]{};
    bool initialized = false;
    for (int span = heads[track]; span >= 0; span = voices[span].nextFilteredSpan) {
        const auto& voice = voices[span];
        const auto rate = voice.sampleRate;
        const auto amount = 1 - expf(-1000.f / rate);
        int count[voiceFilterSlots];
        for (int f = 0; f < voiceFilterSlots; ++f)
            count[f] = sections(voice.filters[f].type);
        bool moving = voice.activeEnvelope[0] != 0;
        for (int j = voice.firstModulation; j < voice.firstModulation + voice.modulationCount; ++j) {
            auto target = modulation[j].target;
            moving |= (target >= ModTarget::cutoff && target <= ModTarget::resonance2) ||
                target == ModTarget::filterEnvDepth || target >= ModTarget::eq1Frequency;
        }
        for (int frame = voice.begin; frame < voice.end; ++frame) {
            if (moving || !initialized) {
                float cents[2]{}, db[2]{};
                float eq[3][3]{};
                auto depth = voice.extraEnvelope[0].depth;
                for (int j = voice.firstModulation; j < voice.firstModulation + voice.modulationCount; ++j) {
                    const auto mod = modulation[j];
                    if (mod.target == ModTarget::cutoff) cents[0] += modulationAt(mod, controls, frame);
                    if (mod.target == ModTarget::cutoff2) cents[1] += modulationAt(mod, controls, frame);
                    if (mod.target == ModTarget::resonance) db[0] += modulationAt(mod, controls, frame);
                    if (mod.target == ModTarget::resonance2) db[1] += modulationAt(mod, controls, frame);
                    if (mod.target == ModTarget::filterEnvDepth) depth += modulationAt(mod, controls, frame);
                    if (mod.target >= ModTarget::eq1Frequency) {
                        auto id = int(mod.target) - int(ModTarget::eq1Frequency);
                        eq[id / 3][id % 3] += modulationAt(mod, controls, frame);
                    }
                }
                if (voice.activeEnvelope[0])
                    cents[0] += depth * envelope(voice.extraEnvelope[0], frame);
                for (int f = 0; f < voiceFilterSlots; ++f) {
                    if (!count[f])
                        continue;
                    const auto& filter = voice.filters[f];
                    const auto target = f < 2 ? coefficients(filter.type,
                        filter.cutoff * exp2f(cents[f] / 1200), filter.resonance + db[f], rate, 0, 0) :
                        coefficients(filter.type, filter.cutoff + eq[f - 2][0], 0, rate,
                            filter.bandwidth + eq[f - 2][1], filter.gain + eq[f - 2][2]);
                    if (initialized) smoothCoefficients(current[f], target, amount);
                    else current[f] = target;
                }
                initialized = true;
            }
            auto& sample = filtered[voice.filteredOffset + frame - voice.begin];
            auto value = sample;
            for (int f = 0; f < voiceFilterSlots; ++f) {
                for (int section = 0; section < count[f]; ++section) {
                    value.x = filterSample(value.x, current[f], z1[f][section].x, z2[f][section].x);
                    value.y = filterSample(value.y, current[f], z1[f][section].y, z2[f][section].y);
                }
            }
            sample = value;
        }
    }
}

__device__ float2 mixFrame(const float* samples, const float* sinc, const VoiceSpan* voices,
                          const Tile& tile, const int* indices, const ControlPoint* controls,
                          const VoiceModulation* modulation, const float2* filtered, int frame)
{
    const auto pan = controlAt(controls, tile.controls[2], tile.controls[3], frame);
    float left = 0, right = 0, wetLeft = 0, wetRight = 0;
    for (int i = tile.first; i < tile.first + tile.count; ++i) {
        const VoiceSpan& voice = voices[indices[i]];
        if (frame < voice.begin || frame >= voice.end)
            continue;
        if (voice.filterTrack >= 0) {
            auto sample = filtered[voice.filteredOffset + frame - voice.begin];
            if (voice.channels == 1) {
                auto offset = pan;
                for (int j = voice.firstModulation; j < voice.firstModulation + voice.modulationCount; ++j) {
                    const auto mod = modulation[j];
                    if (mod.target == ModTarget::pan)
                        offset += modulationAt(mod, controls, frame);
                }
                const auto gains = panGains(voice, offset);
                sample.x *= gains.x;
                sample.y *= gains.y;
            }
            wetLeft += sample.x;
            wetRight += sample.y;
        } else {
            auto sample = voiceAudio(samples, sinc, voice, controls, modulation, pan, frame);
            left += sample.x;
            right += sample.y;
        }
    }
    const auto amplitude = jobGain(controls, tile.controls, frame);
    return make_float2(fmaf(left, amplitude, wetLeft), fmaf(right, amplitude, wetRight));
}

#if CLAY_FLOAT_OUTPUT
using OutputFrame = float2;
#else
using OutputFrame = short2;
#endif

extern "C" __global__ void renderTiles(const float* samples, const float* sinc, const VoiceSpan* voices,
                                      const Tile* tiles, const int* indices, const ControlPoint* controls,
                                      const VoiceModulation* modulation, const float2* filtered,
                                      OutputFrame* output, int firstTile)
{
    const Tile tile = tiles[firstTile + blockIdx.x];
    const auto lane = int(threadIdx.x);
    if (lane >= tile.frames)
        return;
    const auto mixed = mixFrame(samples, sinc, voices, tile, indices, controls, modulation, filtered, tile.frame + lane);
#if CLAY_FLOAT_OUTPUT
    output[tile.output + lane] = mixed;
#else
    output[tile.output + lane] = make_short2(pcm16(mixed.x), pcm16(mixed.y));
#endif
}
