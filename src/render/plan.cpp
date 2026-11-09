#include "plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>

namespace clay {
namespace {

constexpr int forever = std::numeric_limits<int>::max();
constexpr float silence = 1e-4f;
constexpr int referenceBlock = 1024;
constexpr float masterGainDb = -7.35f;

enum class VoiceKind { attack, release, controller };

struct Voice {
    VoiceSpan span;
    int key;
    int velocity;
    int triggerFrame;
    const Region* region;
    double baseStep;
    float releaseSeconds;
    bool noteOff = false;
    VoiceKind kind;
    bool choked = false;
    bool sostenuto = false;
    std::vector<int> flexShapes;
};

struct Release {
    int key, velocity, duration;
    const Region* region;
    bool sostenuto;
};

struct Bend {
    int begin = 0, end = 0, from = 0, to = 0;
};

struct LfoTrack {
    const Lfo* source;
    int begin, end, delay, fade;
    std::vector<VoiceModulation> inputs;
    std::vector<std::pair<ModTarget, int>> outputs;
};

float trackAt(const std::vector<ControlPoint>& track, int frame)
{
    const auto last = std::upper_bound(track.begin(), track.end(), frame,
        [](int value, const ControlPoint& point) { return value < point.frame; });
    const auto& point = last == track.begin() ? track.front() : *std::prev(last);
    return point.value + (frame - point.frame) * point.step;
}

float lfoWave(int wave, double phase)
{
    const auto p = static_cast<float>(phase - std::floor(phase));
    switch (wave) {
    case 0: return p < .25f ? 4 * p : p > .75f ? 4 * p - 4 : 2 - 4 * p;
    case 1: {
        const auto x = 2 * p - 1;
        return -4 * x * (1 - std::abs(x));
    }
    case 2: return p < .75f;
    case 3: return p < .5f;
    case 4: return p < .25f;
    case 5: return p < .125f;
    case 6: return 2 * p - 1;
    case 7: return 1 - 2 * p;
    default: return 0;
    }
}

float velocityGain(const Region& r, int velocity)
{
    auto normalized = float(velocity) / 127;
    auto gain = normalized * normalized;
    if (!r.velocityCurve.empty()) {
        const auto last = std::lower_bound(r.velocityCurve.begin(), r.velocityCurve.end(), velocity,
            [](const auto& point, int value) { return point.first < value; });
        if (last->first == velocity) {
            gain = last->second;
        } else {
            const auto first = std::prev(last);
            const auto phase = float(velocity - first->first) / (last->first - first->first);
            gain = first->second + phase * (last->second - first->second);
        }
    }
    const auto track = std::abs(r.velocityTrack) * (1 - gain);
    return r.velocityTrack < 0 ? track : 1 - track;
}

float crossfadeGain(const Crossfade& fade, float value, bool controllerScale)
{
    const auto end = [controllerScale](int high) { return controllerScale && high == 127 ? 126 : high; };
    const auto inHigh = end(fade.inHigh);
    const auto outHigh = end(fade.outHigh);
    float up = 1, down = 1;
    if (value < fade.inLow)
        up = 0;
    else if (inHigh > fade.inLow)
        up = std::min(1.f, float(value - fade.inLow) / (inHigh - fade.inLow));
    if (outHigh > fade.outLow)
        down = std::clamp(float(outHigh - value) / (outHigh - fade.outLow), 0.f, 1.f);
    return fade.power ? std::sqrt(up * down) : up * down;
}

std::vector<Bend> bendTimeline(const Midi& midi)
{
    // sfizz_render interpolates within its 1024-frame processing block.
    std::vector<Bend> bends;
    int previousFrame = 0;
    int previousValue = 0;
    for (const auto& event : midi.events) {
        const bool reset = event.kind == EventKind::control && event.a == 121;
        if (event.kind != EventKind::bend && !reset)
            continue;
        const auto value = reset ? 0 : event.a;
        auto begin = std::max(previousFrame, event.frame / referenceBlock * referenceBlock);
        if (previousValue != value) {
            bends.push_back({begin, event.frame, previousValue, value});
            bends.push_back({event.frame, event.frame, value, value});
        }
        previousFrame = event.frame;
        previousValue = value;
    }
    return bends;
}

template<class Transform>
std::vector<ControlPoint> automation(const Midi& midi, int cc, float initial, int smoothFrames,
                                     Transform transform, bool settle = false)
{
    float previous = transform(initial);
    std::vector<ControlPoint> points{{0, previous, 0}};
    struct Target {
        int first, last;
        float from, to;
    };
    std::vector<Target> ramps;
    int last = 0;
    for (const auto& event : midi.events) {
        float value = 0;
        if (event.kind == EventKind::control && event.a == 121) value = initial;
        else if (cc == 128 && event.kind == EventKind::bend) value = float(event.a);
        else if (event.kind == EventKind::control && event.a == cc) value = float(event.b);
        else continue;
        const auto target = transform(value);
        if (settle && event.frame == 0) {
            points.front().value = target;
        } else if (target != previous) {
            int first = std::max(last, event.frame / referenceBlock * referenceBlock);
            ramps.push_back({first, event.frame, previous, target});
        }
        previous = target;
        last = event.frame;
    }
    const auto add = [&](int frame, float value, float step) {
        if (points.back().frame == frame) {
            points.back() = {frame, value, step};
        } else if (points.back().step != step ||
                   std::abs(points.back().value + (frame - points.back().frame) * step - value) > 1e-6f) {
            points.push_back({frame, value, step});
        }
    };
    if (smoothFrames < 2) {
        for (const auto& ramp : ramps) {
            if (ramp.first < ramp.last)
                add(ramp.first, ramp.from, (ramp.to - ramp.from) / (ramp.last - ramp.first));
            add(ramp.last, ramp.to, 0);
        }
        return points;
    }
    float current = points.front().value;
    float target = current;
    float step = 0;
    size_t next = 0;
    constexpr int group = 16;
    const int smooth = std::max(group, smoothFrames);
    for (int frame = 0; next < ramps.size() || current != target; frame += group) {
        if (current == target && next < ramps.size() && frame < ramps[next].first)
            frame = std::max(frame, ramps[next].first / group * group);
        auto nextTarget = target;
        const int end = frame + group - 1;
        while (next < ramps.size() && end >= ramps[next].last)
            nextTarget = ramps[next++].to;
        if (next < ramps.size() && end >= ramps[next].first) {
            const auto& ramp = ramps[next];
            nextTarget = ramp.from + (ramp.to - ramp.from) *
                (float(end - ramp.first) / (ramp.last - ramp.first));
        }
        if (nextTarget != target) {
            target = nextTarget;
            step = (target - current) / smooth;
        }
        if (current == target || step == 0) {
            add(frame, current, 0);
            continue;
        }
        add(frame, current + step, step);
        const auto limit = std::abs((target - current) / step);
        if (limit <= group) {
            add(frame + std::max(0, int(std::ceil(limit)) - 1), target, 0);
            current = target;
        } else {
            current += group * step;
        }
    }
    return points;
}

template<class Envelope>
float envelopeLevel(const Envelope& voice, int frame)
{
    if (voice.releaseStart >= 0 && frame >= voice.releaseStart) {
        if (frame >= voice.fadeStart)
            return std::max(0.f, voice.fadeValue - (frame - voice.fadeStart + 1) * voice.fadeStep);
        return voice.releaseValue * std::exp((frame - voice.releaseStart + 1) * voice.releaseRate);
    }
    if (frame < voice.attackStart)
        return voice.start;
    if (frame < voice.holdStart)
        return std::min(1.f, voice.start + (frame - voice.attackStart + 1) * voice.attackStep);
    if (frame < voice.decayStart)
        return 1;
    if (voice.decayRate == 0)
        return voice.sustain;
    return std::max(voice.sustain, std::exp((frame - voice.decayStart + 1) * voice.decayRate));
}

template<class Envelope>
void beginRelease(Envelope& envelope, int frame, int rate, float level, float seconds)
{
    envelope.releaseStart = frame;
    envelope.releaseValue = level;
    envelope.releaseRate = -9 / (std::clamp(seconds, .006f, 100.f) * rate);
    const auto duration = level > silence
        ? static_cast<int>(std::log(silence / level) / envelope.releaseRate) : 0;
    envelope.fadeStart = frame + duration;
    envelope.fadeValue = level * std::exp(envelope.releaseRate * duration);
    envelope.fadeStep = std::max(silence, envelope.fadeValue) / (rate * .05f);
}

double travel(const VoiceSpan& voice, int frames)
{
    if (voice.pitchSlope == 0)
        return voice.step * voice.minimumPitchRatio * frames;
    return voice.step * voice.minimumPitchRatio * std::expm1(voice.pitchSlope * frames) / std::expm1(voice.pitchSlope);
}

double wrap(const VoiceSpan& voice, double position)
{
    if (voice.loopEnd > 0 && position >= voice.loopEnd)
        return voice.loopFirst + std::fmod(position - voice.loopFirst, voice.loopEnd - voice.loopFirst);
    return position;
}

int voiceEnd(const VoiceSpan& voice, int envelopeEnd)
{
    if (voice.loopEnd > 0 || voice.generator != Generator::none)
        return envelopeEnd;
    const auto rampFrames = std::max(0, voice.bendEnd - voice.begin);
    const auto rampTravel = travel(voice, rampFrames);
    const auto remaining = voice.sourceEnd - voice.position;
    const auto step = voice.step * voice.minimumPitchRatio;
    const auto length = rampFrames && voice.pitchSlope != 0 && remaining <= rampTravel
        ? std::ceil(std::log1p(std::max(0., remaining) * std::expm1(voice.pitchSlope) / step) /
                    voice.pitchSlope)
        : rampFrames + std::ceil((remaining - rampTravel) /
                    (step * std::exp(voice.pitchSlope * rampFrames)));
    if (length > forever - voice.begin)
        return envelopeEnd;
    return std::min(envelopeEnd, voice.begin + std::max(0, static_cast<int>(length)));
}

class Planner {
public:
    Planner(const Instrument& instrument, AudioBank& bank, int rate, unsigned seed, int limit, TailPolicy tail)
        : instrument_(instrument), bank_(bank), rate_(rate), limit_(limit), tail_(tail), cc_(instrument.initialCC),
          sequence_(instrument.regions.size()), random_(seed), variation_(seed ^ 0x85ebca6b),
          currentSwitch_(instrument.initialSwitch)
    {
    }

    RenderPlan render(const Midi& midi)
    {
        midi_ = &midi;
        hasBend_ = std::any_of(midi.events.begin(), midi.events.end(),
            [](const Event& event) { return event.kind == EventKind::bend && event.a != 0; });
        constexpr int defaults[] = {7, 11, 10};
        for (int cc : defaults) {
            if (instrument_.usedCC[cc]) {
                plan_.controls.push_back({{0, cc == 10 ? 0.f : 1.f, 0}});
            } else {
                const auto curve = cc == 10 ? 1 : 4;
                plan_.controls.push_back(automation(midi, cc, instrument_.initialCC[cc], rate_ / 100,
                    [&](float value) { return curveValue(instrument_, curve, value); }, tail_ == TailPolicy::natural));
            }
        }
        const auto bends = bendTimeline(midi);
        size_t nextBend = 0;
        for (const auto& event : midi.events) {
            while (nextBend < bends.size() && bends[nextBend].begin <= event.frame)
                changeBend(bends[nextBend++]);
            prune(event.frame);
            switch (event.kind) {
            case EventKind::on: noteOn(event); break;
            case EventKind::off: noteOff(event); break;
            case EventKind::control: control(event); break;
            case EventKind::bend: break;
            }
        }
        prune(midi.end);
        for (auto& voice : active_) {
            // Natural mode ends notes through the prepared MIDI and its pedal releases.
            if (tail_ != TailPolicy::natural && voice.kind == VoiceKind::attack && voice.region->loop != Loop::oneShot)
                release(voice, midi.end);
            if (voice.span.end == forever && limit_ == forever)
                throw std::runtime_error("a looping one-shot or release sample has no finite end");
            emit(voice.span, voice.span.end);
        }
        if (tail_ != TailPolicy::natural)
            plan_.frames = std::max(plan_.frames, midi.end);
        if (plan_.spans.empty())
            throw std::runtime_error("no playable notes for this SFZ and MIDI");
        buildLfoTracks();
        return std::move(plan_);
    }

private:
    void emit(VoiceSpan span, int end)
    {
        span.end = std::min(span.end, end);
        plan_.tailClipped |= span.end > limit_ && span.begin < span.end;
        span.end = std::min(span.end, limit_);
        if (span.begin < span.end) {
            for (int i = span.firstLfo; i < span.firstLfo + span.lfoCount; ++i)
                lfos_[i].end = std::max(lfos_[i].end, span.end);
            plan_.frames = std::max(plan_.frames, span.end);
            plan_.spans.push_back(span);
        }
    }

    void prune(int frame)
    {
        auto tail = std::remove_if(active_.begin(), active_.end(), [&](const Voice& voice) {
            if (voice.span.end > frame)
                return false;
            emit(voice.span, voice.span.end);
            return true;
        });
        active_.erase(tail, active_.end());
    }

    void split(Voice& voice, int frame, double position)
    {
        emit(voice.span, frame);
        voice.span.begin = frame;
        voice.span.position = position;
    }

    bool pedal(const Region& region) const
    {
        return region.sustainEnabled && cc_[region.sustainCC] >= region.sustainLow;
    }

    bool matches(const Region& r, int velocity, float draw) const
    {
        return velocity >= r.lowVelocity && velocity <= r.highVelocity &&
            draw >= r.randomLow && draw < r.randomHigh &&
            switchesMatch(r) &&
            controllersMatch(r);
    }

    bool switchesMatch(const Region& r) const
    {
        return (r.switchLow < 0 || (currentSwitch_ >= r.switchLow && currentSwitch_ <= r.switchHigh)) &&
            (r.switchDown < 0 || down_[r.switchDown]) &&
            (r.switchUp < 0 || !down_[r.switchUp]) &&
            (r.switchPrevious < 0 || previousNote_ == r.switchPrevious);
    }

    bool controllersMatch(const Region& region) const
    {
        return std::all_of(region.conditions.begin(), region.conditions.end(), [&](const Condition& c) {
            return cc_[c.cc] >= c.low && cc_[c.cc] <= c.high;
        });
    }

    void applyBend(Voice& voice) const
    {
        auto& span = voice.span;
        if (voice.region->bendSmooth > 0) {
            span.bendEnd = span.begin;
            span.pitchSlope = 0;
            span.step = voice.baseStep;
            return;
        }
        const auto cents = [&](int value) {
            auto normalized = std::max(-8191, value) / 8191.0;
            return normalized >= 0 ? normalized * voice.region->bendUp : -normalized * voice.region->bendDown;
        };
        const auto to = cents(bend_.to);
        span.bendEnd = bend_.end;
        span.pitchSlope = 0;
        if (span.begin < bend_.end) {
            const auto from = cents(bend_.from);
            const auto slope = (to - from) / (bend_.end - bend_.begin);
            span.pitchSlope = slope * (std::log(2.) / 1200);
            span.step = voice.baseStep * std::exp2((from + (span.begin - bend_.begin) * slope) / 1200);
        } else {
            span.step = voice.baseStep * std::exp2(to / 1200);
        }
    }

    void changeBend(const Bend& bend)
    {
        prune(bend.begin);
        bend_ = bend;
        for (auto& voice : active_) {
            if (voice.region->bendSmooth > 0)
                continue;
            const int start = std::max(bend.begin, voice.span.begin);
            const auto phase = voice.span.position + travel(voice.span, start - voice.span.begin);
            split(voice, start, phase);
            applyBend(voice);
            voice.span.end = voiceEnd(voice.span, fadeEnd(voice.span));
        }
    }

    int fadeEnd(const VoiceSpan& span) const
    {
        return span.releaseStart < 0 ? forever : span.fadeStart + rate_ / 20 + 1;
    }

    float controllerValue(const Modulation& mod, int key, int velocity, VoiceKind kind) const
    {
        if (mod.cc == 140) return keyDelta_;
        if (mod.cc == 141) return std::abs(keyDelta_);
        float value = 0;
        if (mod.cc < 128) value = cc_[mod.cc];
        else if (mod.cc == 133) value = float(key);
        else if ((mod.cc == 131 && kind == VoiceKind::attack) ||
                 (mod.cc == 132 && kind == VoiceKind::release)) value = float(velocity);
        else if (mod.cc == 134) value = keysDown_ > 0 ? 127.f : 0.f;
        else if (mod.cc == 135) value = 127 * randomValue_;
        else if (mod.cc == 136) value = 127 * (2 * randomValue_ - 1);
        else if (mod.cc == 137) value = alternate_ ? 127.f : 0.f;
        else if (const auto it = instrument_.extendedCCDefaults.find(mod.cc);
                 it != instrument_.extendedCCDefaults.end()) value = it->second;
        return curveValue(instrument_, mod.curve, value);
    }

    int controllerTrack(const Modulation& mod)
    {
        const auto key = std::make_tuple(mod.cc, mod.curve, mod.smooth);
        auto result = tracks_.emplace(key, static_cast<int>(plan_.controls.size()));
        if (result.second) {
            plan_.controls.push_back(automation(*midi_, mod.cc, instrument_.initialCC[mod.cc],
                mod.smooth * rate_ / 1000, [&](float value) { return curveValue(instrument_, mod.curve, value); },
                tail_ == TailPolicy::natural));
        }
        return result.first->second;
    }

    int smoothBendTrack(const Region& region)
    {
        const auto key = std::make_tuple(region.bendUp, region.bendDown, region.bendSmooth);
        auto result = bendTracks_.emplace(key, static_cast<int>(plan_.controls.size()));
        if (result.second) {
            plan_.controls.push_back(automation(*midi_, 128, 0, region.bendSmooth * rate_ / 1000,
                [&](float value) {
                    auto bend = std::max(-8191.f, value) / 8191;
                    return bend >= 0 ? bend * region.bendUp : -bend * region.bendDown;
                }, tail_ == TailPolicy::natural));
        }
        return result.first->second;
    }

    int controllerFadeTrack(int cc, const Crossfade& fade)
    {
        auto key = std::make_tuple(cc, fade.inLow, fade.inHigh, fade.outLow, fade.outHigh, fade.power);
        auto result = fadeTracks_.emplace(key, static_cast<int>(plan_.controls.size()));
        if (result.second) {
            plan_.controls.push_back(automation(*midi_, cc, instrument_.initialCC[cc], rate_ / 200,
                [&](float value) { return crossfadeGain(fade, value, true); }, tail_ == TailPolicy::natural));
        }
        return result.first->second;
    }

    float param(const Region& r, ModTarget target, float base, int key, int velocity, VoiceKind kind) const
    {
        if (target >= ModTarget::envDelay && target <= ModTarget::envRelease)
            base += r.velocityEnvelope[int(target) - int(ModTarget::envDelay)] * (velocity / 127.f);
        for (const auto& m : r.modulation)
            if (m.target == target)
                base += m.depth * controllerValue(m, key, velocity, kind);
        return base;
    }

    void initSecondary(Voice& voice)
    {
        auto& span = voice.span;
        for (int slot = 0; slot < 2; ++slot) {
            const auto& source = voice.region->secondary[slot];
            if (!source.enabled)
                continue;
            const auto start = slot == 0 ? ModTarget::filterEnvDelay : ModTarget::pitchEnvDelay;
            std::array<float, 7> values;
            for (size_t p = 0; p < values.size(); ++p) {
                const auto base = source.values[p] + source.velocity[p] * (voice.velocity / 127.f);
                const auto target = static_cast<ModTarget>(int(start) + p);
                const auto value = param(*voice.region, target, base, voice.key, voice.velocity, voice.kind);
                values[p] = std::clamp(value, 0.f, p == 4 || p == 5 ? 1.f : 100.f);
            }
            auto& shape = span.extraEnvelope[slot];
            span.activeEnvelope[slot] = 1;
            shape.start = values[5];
            shape.attackStart = span.begin + int(values[0] * rate_);
            const auto attack = values[1] * rate_;
            shape.attackStep = attack > 0 ? 1 / attack : 1;
            shape.holdStart = shape.attackStart + std::max(0, int(std::ceil((1 - shape.start) * attack)) - 1);
            shape.decayStart = shape.holdStart + int(values[2] * rate_);
            shape.decayRate = values[3] > 0 ? -9 / (std::max(.006f, values[3]) * rate_) : 0;
            shape.sustain = values[4];
            shape.releaseStart = -1;
            shape.releaseSeconds = values[6];
            shape.depth = source.depth + source.velocityDepth * (voice.velocity / 127.f);
        }
    }

    float appendLfos(Voice& voice, int frame, bool& modulatedStereo)
    {
        auto& span = voice.span;
        span.firstLfo = static_cast<int>(lfos_.size());
        float pitchBound = 0;
        for (const auto& entry : voice.region->lfos) {
            const auto& lfo = entry.second;
            LfoTrack track{&lfo, frame, frame, 0, 0, {}, {}};
            auto delay = lfo.delay;
            auto fade = lfo.fade;
            auto bounds = lfo.depths;
            for (auto& depth : bounds)
                depth.second = std::abs(depth.second);
            for (const auto& m : lfo.modulation) {
                const auto value = controllerValue(m, voice.key, voice.velocity, voice.kind);
                if (m.target == ModTarget::lfoDelay) delay += m.depth * value;
                else if (m.target == ModTarget::lfoFade) fade += m.depth * value;
                else {
                    int source = m.cc < 128 ? controllerTrack(m) : -1;
                    track.inputs.push_back({source, 0, m.target, source < 0 ? m.depth * value : m.depth});
                    if (m.target < ModTarget::lfoFrequency || m.target >= ModTarget::eq1Frequency) {
                        auto maximum = std::abs(value);
                        if (source >= 0) {
                            for (const auto& point : plan_.controls[source])
                                maximum = std::max(maximum, std::abs(point.value));
                        }
                        bounds[m.target] += std::abs(m.depth) * maximum;
                    }
                }
            }
            track.delay = int(std::clamp(delay, 0.f, 100.f) * rate_);
            track.fade = int(std::clamp(fade, 0.f, 100.f) * rate_);
            for (const auto& entry : bounds) {
                const auto target = entry.first;
                if (entry.second == 0)
                    continue;
                modulatedStereo |= target == ModTarget::width || target == ModTarget::position;
                if (target == ModTarget::pitch)
                    pitchBound += entry.second;
                const auto index = static_cast<int>(plan_.controls.size());
                plan_.controls.emplace_back();
                track.outputs.emplace_back(target, index);
                plan_.modulation.push_back({index, 0, target, 1});
            }
            if (!track.outputs.empty())
                lfos_.push_back(std::move(track));
        }
        span.lfoCount = static_cast<int>(lfos_.size()) - span.firstLfo;
        return pitchBound;
    }

    void buildLfoTracks()
    {
        std::mt19937 sampleHold(1);
        for (const auto& lfo : lfos_) {
            const auto& source = *lfo.source;
            const int start = lfo.begin + lfo.delay;
            const int stride = source.wave <= 1 ? 16 : 1;
            double phase = 0;
            int cycle = -1;
            float randomSample = 0;
            std::array<float, int(ModTarget::eq3Gain) + 1> baseDepth{};
            for (const auto& entry : source.depths)
                baseDepth[int(entry.first)] = entry.second;
            for (const auto& output : lfo.outputs)
                plan_.controls[output.second].push_back({0, 0, 0});
            for (int frame = start; frame < lfo.end; frame += stride) {
                auto frequency = source.frequency;
                auto shift = source.phase;
                auto depth = baseDepth;
                for (const auto& m : lfo.inputs) {
                    const auto value = m.first < 0 ? m.depth : m.depth * trackAt(plan_.controls[m.first], frame);
                    if (m.target == ModTarget::lfoFrequency) frequency += value;
                    else if (m.target == ModTarget::lfoPhase) shift += value;
                    else depth[int(m.target)] += value;
                }
                auto wave = lfoWave(source.wave, phase + shift);
                if (source.wave == 12) {
                    const auto nextCycle = static_cast<int>(phase);
                    if (cycle != nextCycle) {
                        randomSample = 2 * std::generate_canonical<float, 24>(sampleHold) - 1;
                        cycle = nextCycle;
                    }
                    wave = randomSample;
                }
                const auto fade = lfo.fade == 0 ? 1.f : std::min(1.f, float(frame - start) / lfo.fade);
                for (const auto& output : lfo.outputs) {
                    auto& points = plan_.controls[output.second];
                    const auto value = depth[int(output.first)] * wave * fade;
                    if (points.back().frame == frame) {
                        points.back().value = value;
                    } else {
                        if (stride > 1 && frame - points.back().frame == stride)
                            points.back().step = (value - points.back().value) / stride;
                        if (stride > 1 || value != points.back().value)
                            points.push_back({frame, value, 0});
                    }
                }
                phase += double(std::clamp(frequency, 0.f, 1000.f)) * stride / rate_;
            }
        }
    }

    float appendPitchEnvelopes(Voice& voice)
    {
        float bound = 0;
        for (const auto& entry : voice.region->pitchEnvelopes) {
            const auto& source = entry.second;
            std::vector<VoiceModulation> bindings;
            if (source.depth != 0) {
                bindings.push_back({-1, 0, ModTarget::pitch, source.depth});
                bound += std::abs(source.depth);
            }
            for (const auto& cc : source.depthCC) {
                const Modulation m{ModTarget::pitch, cc.first, cc.second};
                const auto value = controllerValue(m, voice.key, voice.velocity, voice.kind);
                if (cc.first < 128) {
                    bindings.push_back({controllerTrack(m), 0, ModTarget::pitch, cc.second});
                    bound += std::abs(cc.second);
                } else if (value != 0) {
                    bindings.push_back({-1, 0, ModTarget::pitch, cc.second * value});
                    bound += std::abs(cc.second * value);
                }
            }
            if (bindings.empty())
                continue;
            const auto shape = static_cast<int>(plan_.controls.size());
            voice.flexShapes.push_back(shape);
            auto& track = plan_.controls.emplace_back();
            track.push_back({0, 0, 0});
            auto write = [&](int frame, float level, float step) {
                if (track.back().frame == frame) track.back() = {frame, level, step};
                else track.push_back({frame, level, step});
            };
            int frame = voice.triggerFrame;
            float level = 0;
            for (const auto& point : source.points) {
                auto time = point.second.seconds;
                for (const auto& cc : point.second.timeCC)
                    time += cc.second * controllerValue({ModTarget::pitch, cc.first}, voice.key, voice.velocity, voice.kind);
                auto frames = int(std::clamp(time, 0.f, 100.f) * rate_);
                if (frames > 0)
                    write(frame, level, (point.second.level - level) / frames);
                level = point.second.level;
                frame += frames;
                write(frame, level, 0);
            }
            for (auto binding : bindings) {
                binding.shapeFirst = shape;
                plan_.modulation.push_back(binding);
            }
        }
        return bound;
    }

    void trigger(const Region& r, int frame, int key, int velocity, VoiceKind kind, int duration = 0,
                 int controller = -1)
    {
        if (instrument_.hasOffBy)
            choke(r, frame, controller >= 0 ? controller : key, kind);
        if (r.notePolyphony)
            limitNoteVoices(r, frame, key, velocity);
        const Sample* sample = r.generator == Generator::none ? &bank_.decode(r.sample) : nullptr;
        const auto vary = [&](float extent) {
            return extent == 0 ? 0.f : extent * std::generate_canonical<float, 24>(variation_);
        };
        Voice voice{};
        voice.key = key;
        voice.velocity = velocity;
        voice.triggerFrame = frame;
        voice.region = &r;
        voice.kind = kind;
        const auto modulated = [&](ModTarget target, float base) { return param(r, target, base, key, velocity, kind); };
        const auto seconds = [&](ModTarget target, float base) { return std::clamp(modulated(target, base), 0.f, 100.f); };
        voice.releaseSeconds = seconds(ModTarget::envRelease, r.release);
        const auto baseRate = sample ? sample->rate : 440 * std::exp2((r.rootKey - 69) / 12.);
        voice.baseStep = (baseRate / rate_) * std::exp2(
            (r.keyTrack * (key - r.rootKey) + r.tune + 100 * r.transpose + vary(r.pitchRandom)) / 1200.0);
        auto& span = voice.span;
        span.sample = sample ? sample->offset : 0;
        span.channels = sample ? sample->channels : 1;
        span.generator = r.generator;
        if (r.generator == Generator::noise) {
            span.noiseSeed = variation_();
            span.noiseStart = frame;
        }
        span.begin = frame + int(seconds(ModTarget::delay, r.delay) * rate_);
        span.sourceEnd = r.end;
        const auto randomizedOffset = r.offsetRandom ?
            std::uniform_int_distribution<int>(0, r.offsetRandom)(variation_) : 0;
        if (sample) {
            span.position = std::trunc(std::clamp(double(modulated(ModTarget::offset, float(r.offset))) +
                randomizedOffset, 0., double(r.end)));
        } else {
            auto phase = r.oscillatorPhase >= 0 ? r.oscillatorPhase : randomValue_;
            span.position = phase - std::floor(phase);
        }
        applyBend(voice);
        if (sample && (r.loop == Loop::continuous || r.loop == Loop::sustain)) {
            span.loopFirst = r.loopStart;
            span.loopEnd = r.loopEnd;
            span.loopCrossfade = static_cast<int>(std::lround(
                std::min<double>(r.loopStart, double(r.loopCrossfade) * sample->rate)));
        }
        span.start = std::clamp(modulated(ModTarget::envStart, r.start), 0.f, 1.f);
        span.attackStart = span.begin + int(seconds(ModTarget::envDelay, r.envelopeDelay) * rate_);
        const auto attackFrames = seconds(ModTarget::envAttack, r.attack) * rate_;
        span.attackStep = attackFrames > 0 ? 1 / attackFrames : 1;
        span.holdStart = span.attackStart + std::max(0, int(std::ceil((1 - span.start) * attackFrames)) - 1);
        span.decayStart = span.holdStart + int(seconds(ModTarget::envHold, r.hold) * rate_);
        const auto decay = seconds(ModTarget::envDecay, r.decay);
        span.decayRate = decay > 0 ? -9 / (std::max(.006f, decay) * rate_) : 0;
        span.sustain = std::clamp(modulated(ModTarget::envSustain, r.sustain), 0.f, 1.f);
        span.releaseStart = -1;
        initSecondary(voice);
        span.filterTrack = -1;
        span.sampleRate = rate_;
        for (size_t i = 0; i < r.filters.size(); ++i) {
            const auto& filter = r.filters[i];
            if (filter.type == FilterType::none)
                continue;
            if (span.filterTrack < 0)
                span.filterTrack = plan_.filterTracks++;
            const auto velocityTarget = i == 0 ? ModTarget::filterVelocity : ModTarget::filterVelocity2;
            const auto cents = filter.keyTrack * (key - filter.keyCenter) + vary(filter.random) +
                modulated(velocityTarget, filter.velocityTrack) * velocity / 127.f;
            span.filters[i] = {filter.type,
                std::clamp(filter.cutoff * std::exp2(std::clamp(cents, -24000.f, 24000.f) / 1200), 0.f, 20000.f),
                filter.resonance, 0, 0};
        }
        constexpr float eqDefaults[] = {50, 500, 5000};
        for (size_t i = 0; i < r.eq.size(); ++i) {
            const auto& eq = r.eq[i];
            if (!eq.enabled)
                continue;
            if (span.filterTrack < 0)
                span.filterTrack = plan_.filterTracks++;
            span.filters[i + 2] = {FilterType::peak, eq.frequency == 0 ? eqDefaults[i] : eq.frequency,
                                   0, eq.bandwidth, eq.gain};
        }
        span.firstModulation = static_cast<int>(plan_.modulation.size());
        bool modulatedStereo = false;
        float pitchBound = std::abs(span.extraEnvelope[1].depth);
        for (const auto& m : r.modulation) {
            if ((m.target >= ModTarget::envDelay && m.target < ModTarget::cutoff) ||
                (m.target >= ModTarget::filterVelocity && m.target < ModTarget::filterEnvDepth))
                continue;
            modulatedStereo |= m.target == ModTarget::width || m.target == ModTarget::position;
            if (m.target == ModTarget::pitch || m.target == ModTarget::pitchEnvDepth)
                pitchBound += std::abs(m.cc < 128 ? m.depth : m.depth * controllerValue(m, key, velocity, kind));
            if (m.cc < 128)
                plan_.modulation.push_back({controllerTrack(m), 0, m.target, m.depth});
            else
                plan_.modulation.push_back({-1, 0, m.target, m.depth * controllerValue(m, key, velocity, kind)});
        }
        for (const auto& fade : r.controllerFades)
            plan_.modulation.push_back({controllerFadeTrack(fade.first, fade.second), 0, ModTarget::amplitude, 1});
        if (r.bendSmooth > 0 && hasBend_) {
            pitchBound += std::max(std::abs(r.bendUp), std::abs(r.bendDown));
            plan_.modulation.push_back({smoothBendTrack(r), 0, ModTarget::pitch, 1});
        }
        pitchBound += appendLfos(voice, frame, modulatedStereo);
        pitchBound += appendPitchEnvelopes(voice);
        span.minimumPitchRatio = std::exp2(-std::min(24000.f, pitchBound) / 1200);
        if (pitchBound > 0) {
            span.variablePitch = 1;
            if (span.filterTrack < 0)
                span.filterTrack = plan_.filterTracks++;
        }
        span.modulationCount = static_cast<int>(plan_.modulation.size()) - span.firstModulation;
        const auto noteGain = kind == VoiceKind::controller ? 1.f : velocityGain(r, velocity) *
            crossfadeGain(r.keyFade, key, false) * crossfadeGain(r.velocityFade, velocity, true);
        const auto gain = r.amplitude * noteGain *
            std::pow(10.f, ((tail_ == TailPolicy::natural ? 0 : masterGainDb) + r.volume +
                vary(r.ampRandom) - r.releaseDecay * duration / rate_ + 3) / 20);
        constexpr float quarterPi = .7853981633974483f;
        constexpr float panCompensation = 1.4125375446227544f;
        const bool stereoMatrix = span.channels == 2 && (modulatedStereo || r.width != 1 || r.position != 0);
        span.stereoMatrix = stereoMatrix;
        const auto positionGain = span.channels == 1 ? 1.f : .99881488f;
        span.gain = gain * (stereoMatrix ? panCompensation : positionGain);
        span.pan = r.pan;
        span.width = r.width;
        span.stereoPosition = r.position;
        span.left = span.gain * std::cos((1 + r.pan) * quarterPi);
        span.right = span.gain * std::cos((1 - r.pan) * quarterPi);
        span.widthDirect = stereoMatrix ? std::cos((1 - r.width) * quarterPi) : 1.f;
        span.widthCross = stereoMatrix ? std::cos((1 + r.width) * quarterPi) : 0.f;
        span.positionLeft = stereoMatrix ? std::cos((1 + r.position) * quarterPi) : 1.f;
        span.positionRight = stereoMatrix ? std::cos((1 - r.position) * quarterPi) : 1.f;
        span.end = voiceEnd(span, forever);
        active_.push_back(voice);
    }

    void release(Voice& voice, int frame)
    {
        if (voice.span.releaseStart >= 0 || voice.kind != VoiceKind::attack || voice.region->loop == Loop::oneShot)
            return;
        if (voice.region->envelopeDynamic)
            voice.releaseSeconds = param(*voice.region, ModTarget::envRelease, voice.region->release,
                voice.key, voice.velocity, voice.kind);
        startRelease(voice, frame, voice.releaseSeconds);
    }

    void startRelease(Voice& voice, int frame, float seconds)
    {
        auto& span = voice.span;
        if (frame >= span.end)
            return;
        if (frame < span.begin) {
            span.end = span.begin;
            return;
        }
        const auto level = envelopeLevel(span, frame - 1);
        const bool exitLoop = voice.region->loop == Loop::sustain && span.loopEnd > 0;
        if (exitLoop || span.releaseStart >= 0) {
            auto phase = span.position + travel(span, frame - span.begin);
            split(voice, frame, exitLoop ? wrap(span, phase) : phase);
            applyBend(voice);
            if (exitLoop)
                span.loopEnd = 0;
        }
        beginRelease(span, frame, rate_, level, seconds);
        for (int shape : voice.flexShapes) {
            auto& track = plan_.controls[shape];
            track.erase(std::lower_bound(track.begin(), track.end(), frame,
                [](const ControlPoint& point, int value) { return point.frame < value; }), track.end());
            track.push_back({frame, 0, 0});
        }
        for (int slot = 0; slot < 2; ++slot) {
            auto& shape = span.extraEnvelope[slot];
            if (span.activeEnvelope[slot] && shape.releaseStart < 0)
                beginRelease(shape, frame, rate_, envelopeLevel(shape, frame - 1), shape.releaseSeconds);
        }
        span.end = voiceEnd(span, fadeEnd(span));
    }

    void choke(const Region& incoming, int frame, int key, VoiceKind cause)
    {
        std::vector<std::pair<int, int>> released;
        for (auto& voice : active_) {
            const auto& r = *voice.region;
            if (voice.span.end <= frame || voice.kind == VoiceKind::release || voice.choked || !r.chokedByGroup ||
                r.offBy != incoming.group ||
                (r.group == incoming.group && voice.key == key && controllersMatch(r)))
                continue;
            forceOff(voice, frame);
            if (cause != VoiceKind::controller && voice.kind == VoiceKind::attack)
                released.emplace_back(voice.key, voice.velocity);
        }
        for (const auto& note : released)
            releaseRegions(frame, note.first, note.second, std::max(0, frame - noteStart_[note.first]));
    }

    void forceOff(Voice& voice, int frame)
    {
        const auto& r = *voice.region;
        voice.choked = true;
        const auto time = r.offMode == OffMode::normal ? voice.releaseSeconds :
            r.offMode == OffMode::fast ? .006f : r.offTime;
        startRelease(voice, frame, time);
    }

    void limitNoteVoices(const Region& region, int frame, int key, int velocity)
    {
        std::vector<Voice*> candidates;
        int activeCount = 0;
        for (auto& voice : active_) {
            if (voice.choked || voice.span.end <= frame || voice.key != key || voice.region->group != region.group)
                continue;
            ++activeCount;
            if (!region.selfMask || voice.velocity <= velocity)
                candidates.push_back(&voice);
        }
        if (activeCount < region.notePolyphony)
            return;
        std::stable_sort(candidates.begin(), candidates.end(), [&](const Voice* a, const Voice* b) {
            return region.selfMask ? a->velocity < b->velocity : a->triggerFrame < b->triggerFrame;
        });
        for (auto* candidate : candidates) {
            if (candidate->choked)
                continue;
            for (auto& sister : active_) {
                if (!sister.choked && sister.key == candidate->key && sister.kind == candidate->kind &&
                    sister.triggerFrame == candidate->triggerFrame)
                    forceOff(sister, frame);
            }
            if (--activeCount < region.notePolyphony)
                break;
        }
    }

    void noteOn(const Event& event)
    {
        const bool switching = instrument_.switchKeys[event.a];
        if (switching)
            currentSwitch_ = event.a;
        if (!down_[event.a] && !switching)
            ++keysDown_;
        if (!switching) {
            keyDelta_ = lastNote_ < 0 ? 0.f : float(event.a - lastNote_);
            lastNote_ = event.a;
            randomValue_ = std::generate_canonical<float, 24>(extendedRandom_);
            alternate_ = !alternate_;
        }
        down_[event.a] = true;
        noteVelocity_[event.a] = event.b;
        noteStart_[event.a] = event.frame;
        const float draw = std::generate_canonical<float, 24>(random_);
        for (const auto index : instrument_.byKey[event.a]) {
            const auto& region = instrument_.regions[index];
            const bool step = sequence_[index]++ % region.sequenceLength == unsigned(region.sequencePosition - 1);
            if (!step || !matches(region, event.b, draw))
                continue;
            if (region.trigger == Trigger::attack ||
                (region.trigger == Trigger::first && keysDown_ == 1) ||
                (region.trigger == Trigger::legato && keysDown_ > 1))
                trigger(region, event.frame, event.a, event.b, VoiceKind::attack);
        }
        previousNote_ = event.a;
    }

    bool hasAttackVoice(const Region& r) const
    {
        return std::any_of(active_.begin(), active_.end(), [&](const Voice& voice) {
            return voice.kind == VoiceKind::attack && voice.key >= r.lowKey && voice.key <= r.highKey &&
                voice.velocity >= r.lowVelocity && voice.velocity <= r.highVelocity;
        });
    }

    void noteOff(const Event& event)
    {
        if (!down_[event.a])
            return;
        down_[event.a] = false;
        if (!instrument_.switchKeys[event.a])
            --keysDown_;
        for (auto& voice : active_) {
            if (voice.key == event.a && voice.kind == VoiceKind::attack) {
                voice.noteOff = true;
                if (!pedal(*voice.region) && !voice.sostenuto)
                    release(voice, event.frame);
            }
        }
        releaseRegions(event.frame, event.a, noteVelocity_[event.a], event.frame - noteStart_[event.a]);
    }

    void releaseRegions(int frame, int key, int velocity, int duration)
    {
        const float draw = std::generate_canonical<float, 24>(random_);
        for (const auto index : instrument_.byKey[key]) {
            const auto& r = instrument_.regions[index];
            const auto counter = sequence_[index];
            const bool sequenceMatches = counter == 0 ? !r.explicitSequencePosition :
                (counter - 1) % r.sequenceLength == unsigned(r.sequencePosition - 1);
            if ((r.trigger != Trigger::release && r.trigger != Trigger::releaseKey) ||
                !sequenceMatches || !matches(r, velocity, draw))
                continue;
            if (r.trigger == Trigger::release && (pedal(r) || sostenutoKeys_[key])) {
                delayed_.push_back({key, velocity, duration, &r, sostenutoKeys_[key]});
            } else if (r.trigger == Trigger::releaseKey || r.releaseDead || hasAttackVoice(r)) {
                trigger(r, frame, key, velocity, VoiceKind::release, duration);
            }
        }
    }

    void updateDynamicEnvelopes(const std::array<float, 128>& before, int frame)
    {
        for (auto& voice : active_) {
            if (voice.choked)
                continue;
            for (const auto& mod : voice.region->modulation) {
                if (mod.cc >= 128 || before[mod.cc] == cc_[mod.cc] || mod.depth == 0)
                    continue;
                const bool filter = mod.target >= ModTarget::filterEnvDelay && mod.target <= ModTarget::filterEnvRelease;
                const bool pitch = mod.target >= ModTarget::pitchEnvDelay && mod.target <= ModTarget::pitchEnvRelease;
                if ((filter && voice.region->secondary[0].dynamic) || (pitch && voice.region->secondary[1].dynamic))
                    throw std::runtime_error("changing a dynamic pitch or filter envelope mid-voice is unsupported");
                if (!voice.region->envelopeDynamic || mod.target < ModTarget::envDelay || mod.target > ModTarget::envRelease)
                    continue;
                if (mod.target != ModTarget::envRelease)
                    throw std::runtime_error("changing a dynamic amplitude envelope mid-voice is unsupported for CC" +
                        std::to_string(mod.cc));
                if (voice.span.releaseStart >= 0 && frame < voice.span.fadeStart) {
                    const auto seconds = param(*voice.region, ModTarget::envRelease, voice.region->release,
                        voice.key, voice.velocity, voice.kind);
                    startRelease(voice, frame, seconds);
                    break;
                }
            }
        }
    }

    void control(const Event& event)
    {
        if (event.a == 121) {
            const auto before = cc_;
            cc_ = instrument_.initialCC;
            updateDynamicEnvelopes(before, event.frame);
            if (tail_ == TailPolicy::natural) {
                setSostenuto(false);
                releasePedals(event.frame, 66, before[66]);
            }
            for (auto& voice : active_) {
                if (voice.noteOff && !pedal(*voice.region))
                    release(voice, event.frame);
            }
            return;
        }
        const bool natural = tail_ == TailPolicy::natural;
        if (!natural && event.a == 66 && event.b != 0)
            throw std::runtime_error("unsupported MIDI sostenuto CC66");
        if (natural && (event.a == 120 || event.a == 123)) {
            for (auto& voice : active_)
                emit(voice.span, event.frame);
            active_.clear();
            delayed_.clear();
            down_.fill(false);
            sostenutoKeys_.fill(false);
            keysDown_ = 0;
            return;
        }
        if (event.a >= 120)
            throw std::runtime_error("unsupported MIDI channel mode CC" + std::to_string(event.a));
        const bool sostenuto = natural && event.a == 66;
        if (!instrument_.usedCC[event.a] && !sostenuto)
            return;
        const auto before = cc_;
        const auto previous = before[event.a];
        cc_[event.a] = static_cast<float>(event.b);
        if (previous != cc_[event.a])
            updateDynamicEnvelopes(before, event.frame);
        if (sostenuto)
            setSostenuto(event.b == 127);
        releasePedals(event.frame, event.a, previous);
        const auto& triggers = instrument_.byCC[event.a];
        if (triggers.empty())
            return;
        const auto draw = std::generate_canonical<float, 24>(random_);
        for (auto index : triggers) {
            const auto& region = instrument_.regions[index];
            auto condition = std::find_if(region.ccTriggers.begin(), region.ccTriggers.end(),
                [&](const Condition& range) { return range.cc == event.a; });
            if (event.b < condition->low || event.b > condition->high ||
                draw < region.randomLow || draw >= region.randomHigh)
                continue;
            const bool step = sequence_[index]++ % region.sequenceLength == unsigned(region.sequencePosition - 1);
            if (step && switchesMatch(region) && previous != event.b && controllersMatch(region))
                trigger(region, event.frame, region.rootKey, event.b, VoiceKind::controller, 0, event.a);
        }
    }

    void setSostenuto(bool down)
    {
        if (sostenutoDown_ == down)
            return;
        sostenutoDown_ = down;
        sostenutoKeys_ = down ? down_ : std::array<bool, 128>{};
        for (auto& voice : active_)
            voice.sostenuto = down && !voice.noteOff;
        if (!down) {
            for (auto& note : delayed_)
                note.sostenuto = false;
        }
    }

    void releasePedals(int frame, int cc, float previous)
    {
        const bool sostenutoUp = tail_ == TailPolicy::natural && cc == 66 && !sostenutoDown_;
        for (auto& voice : active_) {
            const auto& r = *voice.region;
            const bool sustainChanged = r.sustainEnabled && r.sustainCC == cc && previous >= r.sustainLow;
            if (voice.noteOff && (sustainChanged || sostenutoUp) && !pedal(r) && !voice.sostenuto)
                release(voice, frame);
        }
        std::vector<Release> due;
        auto end = std::remove_if(delayed_.begin(), delayed_.end(), [&](const Release& note) {
            const auto& r = *note.region;
            if ((r.sustainCC != cc && !sostenutoUp) || pedal(r) || note.sostenuto)
                return false;
            if (r.releaseDead || hasAttackVoice(r))
                due.push_back(note);
            return true;
        });
        delayed_.erase(end, delayed_.end());
        for (const auto& note : due)
            trigger(*note.region, frame, note.key, note.velocity, VoiceKind::release, note.duration);
    }

    const Instrument& instrument_;
    AudioBank& bank_;
    int rate_;
    int limit_;
    TailPolicy tail_;
    std::array<float, 128> cc_;
    std::array<bool, 128> down_{};
    std::array<bool, 128> sostenutoKeys_{};
    bool sostenutoDown_ = false;
    std::array<int, 128> noteStart_{}, noteVelocity_{};
    std::vector<unsigned> sequence_;
    std::mt19937 random_, variation_;
    std::mt19937 extendedRandom_{1};
    std::vector<Voice> active_;
    std::vector<Release> delayed_;
    int keysDown_ = 0;
    int currentSwitch_;
    Bend bend_;
    const Midi* midi_ = nullptr;
    bool hasBend_ = false;
    bool alternate_ = false;
    int lastNote_ = -1;
    int previousNote_ = -1;
    float keyDelta_ = 0, randomValue_ = 0;
    std::vector<LfoTrack> lfos_;
    std::map<std::tuple<int, int, int>, int> tracks_;
    std::map<std::tuple<float, float, int>, int> bendTracks_;
    std::map<std::tuple<int, int, int, int, int, bool>, int> fadeTracks_;
    RenderPlan plan_;
};

} // namespace

RenderPlan planRender(const Instrument& instrument, AudioBank& bank, const Midi& midi,
                      int rate, unsigned seed, int limit, TailPolicy tail)
{
    return Planner(instrument, bank, rate, seed, limit, tail).render(midi);
}

} // namespace clay
