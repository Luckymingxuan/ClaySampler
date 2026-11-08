#include "plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

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
    const Region* region;
    double baseStep;
    bool noteOff = false;
    VoiceKind kind;
    bool choked = false;
};

struct Release {
    int key, velocity, duration;
    const Region* region;
};

struct Bend {
    int begin = 0, end = 0, from = 0, to = 0;
};

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

float crossfadeGain(const Crossfade& fade, int value, bool velocity)
{
    const auto end = [velocity](int high) { return velocity && high == 127 ? 126 : high; };
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

std::vector<ControlPoint> automation(const Midi& midi, int cc, float initial, bool replaced, int rate)
{
    auto transform = [cc](float value) {
        auto normalized = value / 127;
        return cc == 10 ? 2 * normalized - 1 : normalized * normalized;
    };
    float previous = replaced ? (cc == 10 ? 0.f : 1.f) : transform(initial);
    std::vector<ControlPoint> points{{0, previous, 0}};
    if (replaced)
        return points;
    struct Target {
        int first, last;
        float from, to;
    };
    std::vector<Target> ramps;
    int last = 0;
    for (const auto& event : midi.events) {
        if (event.kind != EventKind::control || (event.a != cc && event.a != 121))
            continue;
        const auto target = transform(event.a == 121 ? initial : float(event.b));
        if (target != previous) {
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
    float current = points.front().value;
    float target = current;
    float step = 0;
    size_t next = 0;
    constexpr int group = 16;
    const int smooth = std::max(group, rate / 100);
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

float envelopeLevel(const VoiceSpan& voice, int frame)
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

double travel(const VoiceSpan& voice, int frames)
{
    if (voice.pitchSlope == 0)
        return voice.step * frames;
    return voice.step * std::expm1(voice.pitchSlope * frames) / std::expm1(voice.pitchSlope);
}

double wrap(const VoiceSpan& voice, double position)
{
    if (voice.loopEnd > 0 && position >= voice.loopEnd)
        return voice.loopFirst + std::fmod(position - voice.loopFirst, voice.loopEnd - voice.loopFirst);
    return position;
}

int voiceEnd(const VoiceSpan& voice, int envelopeEnd)
{
    if (voice.loopEnd > 0)
        return envelopeEnd;
    const auto rampFrames = std::max(0, voice.bendEnd - voice.begin);
    const auto rampTravel = travel(voice, rampFrames);
    const auto remaining = voice.sourceEnd - voice.position;
    const auto length = rampFrames && voice.pitchSlope != 0 && remaining <= rampTravel
        ? std::ceil(std::log1p(std::max(0., remaining) * std::expm1(voice.pitchSlope) / voice.step) /
                    voice.pitchSlope)
        : rampFrames + std::ceil((remaining - rampTravel) /
                    (voice.step * std::exp(voice.pitchSlope * rampFrames)));
    if (length > forever - voice.begin)
        return envelopeEnd;
    return std::min(envelopeEnd, voice.begin + std::max(0, static_cast<int>(length)));
}

class Planner {
public:
    Planner(const Instrument& instrument, AudioBank& bank, int rate, unsigned seed)
        : instrument_(instrument), bank_(bank), rate_(rate), cc_(instrument.initialCC),
          sequence_(instrument.regions.size()), random_(seed), variation_(seed ^ 0x85ebca6b),
          currentSwitch_(instrument.initialSwitch)
    {
    }

    RenderPlan render(const Midi& midi)
    {
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
        // Match the offline convention of releasing any still-held voices at EOF.
        for (auto& voice : active_) {
            if (voice.kind == VoiceKind::attack && voice.region->loop != Loop::oneShot)
                release(voice, midi.end);
            if (voice.span.end == forever)
                throw std::runtime_error("a looping one-shot or release sample has no finite end");
            emit(voice.span, voice.span.end);
        }
        plan_.frames = std::max(plan_.frames, midi.end);
        if (plan_.spans.empty())
            throw std::runtime_error("no playable notes for this SFZ and MIDI");
        constexpr int defaults[] = {7, 11, 10};
        for (size_t i = 0; i < plan_.controls.size(); ++i) {
            const int cc = defaults[i];
            plan_.controls[i] = automation(midi, cc, instrument_.initialCC[cc], instrument_.usedCC[cc], rate_);
        }
        return std::move(plan_);
    }

private:
    void emit(VoiceSpan span, int end)
    {
        span.end = std::min(span.end, end);
        if (span.begin < span.end) {
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
            (r.switchLow < 0 || (currentSwitch_ >= r.switchLow && currentSwitch_ <= r.switchHigh)) &&
            controllersMatch(r);
    }

    bool controllersMatch(const Region& region) const
    {
        return std::all_of(region.conditions.begin(), region.conditions.end(), [&](const Condition& c) {
            return cc_[c.cc] >= c.low && cc_[c.cc] <= c.high;
        });
    }

    void applyBend(Voice& voice) const
    {
        const auto cents = [&](int value) {
            auto normalized = std::max(-8191, value) / 8191.0;
            return normalized >= 0 ? normalized * voice.region->bendUp : -normalized * voice.region->bendDown;
        };
        auto& span = voice.span;
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

    void trigger(const Region& r, int frame, int key, int velocity, VoiceKind kind, int duration = 0,
                 int controller = -1)
    {
        if (instrument_.hasOffBy)
            choke(r, frame, controller >= 0 ? controller : key, kind);
        const auto& sample = bank_.decode(r.sample);
        const auto vary = [&](float extent) {
            return extent == 0 ? 0.f : extent * std::generate_canonical<float, 24>(variation_);
        };
        Voice voice{};
        voice.key = key;
        voice.velocity = velocity;
        voice.region = &r;
        voice.kind = kind;
        voice.baseStep = (double(sample.rate) / rate_) * std::exp2(
            (r.keyTrack * (key - r.rootKey) + r.tune + 100 * r.transpose + vary(r.pitchRandom)) / 1200.0);
        auto& span = voice.span;
        span.sample = sample.offset;
        span.channels = sample.channels;
        span.begin = frame + int(r.delay * rate_);
        span.sourceEnd = r.end;
        span.position = r.offset;
        applyBend(voice);
        if (r.loop == Loop::continuous || r.loop == Loop::sustain) {
            span.loopFirst = r.loopStart;
            span.loopEnd = r.loopEnd;
            span.loopCrossfade = static_cast<int>(std::lround(
                std::min<double>(r.loopStart, double(r.loopCrossfade) * sample.rate)));
        }
        span.start = r.start;
        span.attackStart = span.begin + int(r.envelopeDelay * rate_);
        const auto attackFrames = r.attack * rate_;
        span.attackStep = attackFrames > 0 ? 1 / attackFrames : 1;
        span.holdStart = span.attackStart + std::max(0, int(std::ceil((1 - r.start) * attackFrames)) - 1);
        span.decayStart = span.holdStart + int(r.hold * rate_);
        span.decayRate = r.decay > 0 ? -9 / (std::max(.006f, r.decay) * rate_) : 0;
        span.sustain = r.sustain;
        span.releaseStart = -1;
        const auto noteGain = kind == VoiceKind::controller ? 1.f : velocityGain(r, velocity) *
            crossfadeGain(r.keyFade, key, false) * crossfadeGain(r.velocityFade, velocity, true);
        const auto gain = r.amplitude * noteGain *
            std::pow(10.f, (masterGainDb + r.volume + vary(r.ampRandom) - r.releaseDecay * duration / rate_ + 3) / 20);
        constexpr float quarterPi = .7853981633974483f;
        constexpr float panCompensation = 1.4125375446227544f;
        const bool stereoMatrix = sample.channels == 2 && (r.width != 1 || r.position != 0);
        span.stereoMatrix = stereoMatrix;
        const auto positionGain = sample.channels == 1 ? 1.f : .99881488f;
        span.gain = gain * (stereoMatrix ? panCompensation : positionGain);
        span.pan = r.pan;
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
        startRelease(voice, frame, voice.region->release);
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
        span.releaseStart = frame;
        span.releaseValue = level;
        span.releaseRate = -9 / (std::max(.006f, seconds) * rate_);
        auto exponentialFrames = span.releaseValue > silence
            ? static_cast<int>(std::log(silence / span.releaseValue) / span.releaseRate) : 0;
        span.fadeStart = frame + exponentialFrames;
        span.fadeValue = span.releaseValue * std::exp(span.releaseRate * exponentialFrames);
        span.fadeStep = std::max(silence, span.fadeValue) / (rate_ * .05f);
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
            voice.choked = true;
            const auto time = r.offMode == OffMode::normal ? r.release :
                r.offMode == OffMode::fast ? .006f : r.offTime;
            startRelease(voice, frame, time);
            if (cause != VoiceKind::controller && voice.kind == VoiceKind::attack)
                released.emplace_back(voice.key, voice.velocity);
        }
        for (const auto& note : released)
            releaseRegions(frame, note.first, note.second, std::max(0, frame - noteStart_[note.first]));
    }

    void noteOn(const Event& event)
    {
        const bool switching = instrument_.switchKeys[event.a];
        if (switching)
            currentSwitch_ = event.a;
        if (!down_[event.a] && !switching)
            ++keysDown_;
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
                if (!pedal(*voice.region))
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
            if ((r.trigger != Trigger::release && r.trigger != Trigger::releaseKey) ||
                !matches(r, velocity, draw))
                continue;
            if (r.trigger == Trigger::release && pedal(r)) {
                delayed_.push_back({key, velocity, duration, &r});
            } else if (r.trigger == Trigger::releaseKey || r.releaseDead || hasAttackVoice(r)) {
                trigger(r, frame, key, velocity, VoiceKind::release, duration);
            }
        }
    }

    void control(const Event& event)
    {
        if (event.a == 121) {
            cc_ = instrument_.initialCC;
            for (auto& voice : active_) {
                if (voice.noteOff && !pedal(*voice.region))
                    release(voice, event.frame);
            }
            return;
        }
        if (event.a == 66 && event.b != 0)
            throw std::runtime_error("unsupported MIDI sostenuto CC66");
        if (event.a >= 120)
            throw std::runtime_error("unsupported MIDI channel mode CC" + std::to_string(event.a));
        if (!instrument_.usedCC[event.a])
            return;
        const auto previous = cc_[event.a];
        cc_[event.a] = static_cast<float>(event.b);
        for (auto& voice : active_) {
            const auto& r = *voice.region;
            if (voice.noteOff && r.sustainEnabled && r.sustainCC == event.a &&
                previous >= r.sustainLow && !pedal(r))
                release(voice, event.frame);
        }
        std::vector<Release> due;
        auto end = std::remove_if(delayed_.begin(), delayed_.end(), [&](const Release& releaseNote) {
            const auto& r = *releaseNote.region;
            if (r.sustainCC != event.a || pedal(r))
                return false;
            if (r.releaseDead || hasAttackVoice(r))
                due.push_back(releaseNote);
            return true;
        });
        delayed_.erase(end, delayed_.end());
        for (const auto& note : due)
            trigger(*note.region, event.frame, note.key, note.velocity, VoiceKind::release, note.duration);
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
            const bool switched = region.switchLow < 0 ||
                (currentSwitch_ >= region.switchLow && currentSwitch_ <= region.switchHigh);
            if (step && switched && previous != event.b && controllersMatch(region))
                trigger(region, event.frame, region.rootKey, event.b, VoiceKind::controller, 0, event.a);
        }
    }

    const Instrument& instrument_;
    AudioBank& bank_;
    int rate_;
    std::array<float, 128> cc_;
    std::array<bool, 128> down_{};
    std::array<int, 128> noteStart_{}, noteVelocity_{};
    std::vector<unsigned> sequence_;
    std::mt19937 random_, variation_;
    std::vector<Voice> active_;
    std::vector<Release> delayed_;
    int keysDown_ = 0;
    int currentSwitch_;
    Bend bend_;
    RenderPlan plan_;
};

} // namespace

RenderPlan planRender(const Instrument& instrument, AudioBank& bank, const Midi& midi,
                      int rate, unsigned seed)
{
    return Planner(instrument, bank, rate, seed).render(midi);
}

} // namespace clay
