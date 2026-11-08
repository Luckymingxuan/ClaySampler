#include "instrument.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string_view>

namespace clay {
namespace {

using Path = std::filesystem::path;
using Opcodes = std::map<std::string, std::string>;
constexpr std::string_view whitespace = " \t\r\n";

std::string trim(std::string_view value)
{
    auto first = value.find_first_not_of(whitespace);
    if (first == value.npos)
        return {};
    auto last = value.find_last_not_of(whitespace);
    return std::string(value.substr(first, last - first + 1));
}

bool nameChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string unquote(std::string value)
{
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        return value.substr(1, value.size() - 2);
    return value;
}

Path filePath(std::string value)
{
    std::replace(value.begin(), value.end(), '\\', '/');
    return std::filesystem::u8path(value);
}

float number(const std::string& value, float low, float high)
{
    size_t length = 0;
    float result = std::stof(value, &length);
    if (length != value.size() || !std::isfinite(result) || result < low || result > high)
        throw std::runtime_error("invalid value " + value);
    return result;
}

int64_t integer64(const std::string& value, int64_t low, int64_t high)
{
    size_t length = 0;
    auto result = std::stoll(value, &length);
    if (length != value.size() || result < low || result > high)
        throw std::runtime_error("invalid value " + value);
    return result;
}

int integer(const std::string& value, int low, int high)
{
    return static_cast<int>(integer64(value, low, high));
}

int key(const std::string& value)
{
    if (value.size() < 2 || std::toupper(static_cast<unsigned char>(value[0])) < 'A' ||
        std::toupper(static_cast<unsigned char>(value[0])) > 'G')
        return integer(value, 0, 127);
    constexpr int notes[] = {9, 11, 0, 2, 4, 5, 7};
    int note = notes[std::toupper(static_cast<unsigned char>(value[0])) - 'A'];
    size_t octave = 1;
    if (value[octave] == '#' || value[octave] == 'b')
        note += value[octave++] == '#' ? 1 : -1;
    note += 12 * (integer(value.substr(octave), -2, 10) + 1);
    if (note < 0 || note > 127)
        throw std::runtime_error("key outside MIDI range " + value);
    return note;
}

bool enabled(const std::string& value)
{
    if (value == "on" || value == "1")
        return true;
    if (value == "off" || value == "0")
        return false;
    throw std::runtime_error("invalid switch " + value);
}

bool powerCurve(const std::string& value)
{
    if (value == "power") return true;
    if (value == "gain") return false;
    throw std::runtime_error("unsupported crossfade curve " + value);
}

bool envelopeTarget(ModTarget target)
{
    return target >= ModTarget::envDelay && target <= ModTarget::envRelease;
}

int modulationController(std::string_view suffix)
{
    auto cc = integer(std::string(suffix), 0, 511);
    if (cc >= 128 && cc <= 130)
        throw std::runtime_error("unsupported extended controller CC" + std::to_string(cc));
    return cc;
}

bool modulationOpcode(Region& region, const std::string& op, const std::string& value,
                      int envelopeShift = 0, std::array<float, 7>* velocityEnvelope = nullptr)
{
    static const std::pair<std::string_view, ModTarget> parameters[] = {
        {"amplitude", ModTarget::amplitude}, {"volume", ModTarget::volume}, {"gain", ModTarget::volume},
        {"pan", ModTarget::pan}, {"width", ModTarget::width}, {"position", ModTarget::position},
        {"pitch", ModTarget::pitch}, {"tune", ModTarget::pitch},
        {"ampeg_delay", ModTarget::envDelay}, {"ampeg_attack", ModTarget::envAttack},
        {"ampeg_hold", ModTarget::envHold}, {"ampeg_decay", ModTarget::envDecay},
        {"ampeg_sustain", ModTarget::envSustain}, {"ampeg_start", ModTarget::envStart},
        {"ampeg_release", ModTarget::envRelease}, {"offset", ModTarget::offset}, {"delay", ModTarget::delay},
        {"cutoff", ModTarget::cutoff}, {"cutoff1", ModTarget::cutoff}, {"cutoff2", ModTarget::cutoff2},
        {"resonance", ModTarget::resonance}, {"resonance1", ModTarget::resonance}, {"resonance2", ModTarget::resonance2},
        {"fil_veltrack", ModTarget::filterVelocity}, {"fil1_veltrack", ModTarget::filterVelocity},
        {"fil2_veltrack", ModTarget::filterVelocity2},
        {"eq1_freq", ModTarget::eq1Frequency}, {"eq1_bw", ModTarget::eq1Bandwidth}, {"eq1_gain", ModTarget::eq1Gain},
        {"eq2_freq", ModTarget::eq2Frequency}, {"eq2_bw", ModTarget::eq2Bandwidth}, {"eq2_gain", ModTarget::eq2Gain},
        {"eq3_freq", ModTarget::eq3Frequency}, {"eq3_bw", ModTarget::eq3Bandwidth}, {"eq3_gain", ModTarget::eq3Gain},
    };
    for (const auto& parameter : parameters) {
        const auto name = parameter.first;
        const auto baseTarget = parameter.second;
        if (envelopeShift && !envelopeTarget(baseTarget))
            continue;
        const auto target = static_cast<ModTarget>(int(baseTarget) + envelopeShift);
        const std::string_view opcode(op);
        if (envelopeTarget(baseTarget) && baseTarget != ModTarget::envStart &&
            (opcode == std::string("ampeg_velto") + std::string(name.substr(6)) ||
             opcode == std::string("ampeg_vel2") + std::string(name.substr(6)))) {
            const auto index = int(baseTarget) - int(ModTarget::envDelay);
            auto depth = number(value, -100, 100);
            auto& velocity = velocityEnvelope ? *velocityEnvelope : region.velocityEnvelope;
            velocity[index] = baseTarget == ModTarget::envSustain ? depth / 100 : depth;
            return true;
        }
        if (opcode.substr(0, name.size()) != name)
            continue;
        auto suffix = opcode.substr(name.size());
        enum class Field { depth, curve, smooth };
        Field field;
        if (suffix.substr(0, 5) == "_oncc") { field = Field::depth; suffix.remove_prefix(5); }
        else if (suffix.substr(0, 3) == "_cc") { field = Field::depth; suffix.remove_prefix(3); }
        else if (suffix.substr(0, 2) == "cc") { field = Field::depth; suffix.remove_prefix(2); }
        else if (suffix.substr(0, 8) == "_curvecc") { field = Field::curve; suffix.remove_prefix(8); }
        else if (suffix.substr(0, 9) == "_smoothcc") { field = Field::smooth; suffix.remove_prefix(9); }
        else continue;
        if (target == ModTarget::offset && field == Field::curve) {
            (void)integer(std::string(suffix), 0, 511);
            (void)integer(value, 0, 255);
            return true; // sfizz 1.2.3 ignores offset_curveccN.
        }
        if ((field == Field::smooth && (envelopeShift || (target >= ModTarget::envDelay && target <= ModTarget::delay))) ||
            (field == Field::curve && target == ModTarget::delay))
            return false;
        auto cc = modulationController(suffix);
        auto it = std::find_if(region.modulation.begin(), region.modulation.end(),
            [&](const Modulation& m) { return m.target == target && m.cc == cc; });
        if (it == region.modulation.end()) {
            region.modulation.push_back({target, cc});
            it = std::prev(region.modulation.end());
        }
        if (field == Field::curve) it->curve = integer(value, 0, 255);
        else if (field == Field::smooth) it->smooth = integer(value, 0, 10000);
        else {
            const bool percent = target == ModTarget::amplitude || target == ModTarget::pan ||
                target == ModTarget::width || target == ModTarget::position ||
                baseTarget == ModTarget::envSustain || baseTarget == ModTarget::envStart;
            it->depth = number(value, -1e9f, 1e9f) / (percent ? 100 : 1);
        }
        if (target >= ModTarget::cutoff && target <= ModTarget::filterVelocity2) {
            const auto slot = target == ModTarget::cutoff2 || target == ModTarget::resonance2 ||
                target == ModTarget::filterVelocity2;
            if (region.filters[slot].type == FilterType::none)
                region.filters[slot].type = FilterType::low2;
        }
        if (target >= ModTarget::eq1Frequency)
            region.eq[(int(target) - int(ModTarget::eq1Frequency)) / 3].enabled = true;
        return true;
    }
    return false;
}

bool secondaryEnvelopeOpcode(Region& region, const std::string& op, const std::string& value)
{
    int slot;
    std::string_view name(op);
    if (name.substr(0, 6) == "fileg_") { slot = 0; name.remove_prefix(6); }
    else if (name.substr(0, 8) == "pitcheg_") { slot = 1; name.remove_prefix(8); }
    else return false;
    auto& env = region.secondary[slot];
    env.enabled = true;
    if (name == "depth") env.depth = number(value, -12000, 12000);
    else if (name == "vel2depth" || name == "veltodepth") env.velocityDepth = number(value, -12000, 12000);
    else if (name == "dynamic") env.dynamic = enabled(value);
    else if ((slot == 0 && name.substr(0, 7) == "depthcc") || (slot == 1 && name == "decay_shape")) {
        return true; // These spellings are unknown to sfizz 1.2.3.
    } else if (name.substr(0, 10) == "depth_oncc" || name.substr(0, 13) == "depth_curvecc") {
        const bool curve = name.substr(0, 13) == "depth_curvecc";
        const auto cc = modulationController(name.substr(curve ? 13 : 10));
        const auto target = slot == 0 ? ModTarget::filterEnvDepth : ModTarget::pitchEnvDepth;
        auto it = std::find_if(region.modulation.begin(), region.modulation.end(),
            [&](const Modulation& mod) { return mod.target == target && mod.cc == cc; });
        if (it == region.modulation.end()) {
            region.modulation.push_back({target, cc});
            it = std::prev(region.modulation.end());
        }
        if (curve) it->curve = integer(value, 0, 255);
        else it->depth = number(value, -12000, 12000);
    } else {
        constexpr std::string_view names[] = {"delay", "attack", "hold", "decay", "sustain", "start", "release"};
        const auto found = std::find(std::begin(names), std::end(names), name);
        if (found != std::end(names)) {
            auto i = found - std::begin(names);
            const bool percent = name == "sustain" || name == "start";
            env.values[i] = number(value, 0, 100) / (percent ? 100 : 1);
        } else {
            const auto start = slot == 0 ? ModTarget::filterEnvDelay : ModTarget::pitchEnvDelay;
            return modulationOpcode(region, "ampeg_" + std::string(name), value,
                int(start) - int(ModTarget::envDelay), &env.velocity);
        }
    }
    return true;
}

bool lfoOpcode(Region& region, const std::string& op, const std::string& value)
{
    std::string_view suffix(op);
    int index = 0;
    ModTarget legacyTarget = ModTarget::pitch;
    if (suffix.substr(0, 3) == "lfo") {
        suffix.remove_prefix(3);
        auto end = suffix.find('_');
        if (end == suffix.npos)
            return false;
        index = integer(std::string(suffix.substr(0, end)), 1, 64);
        suffix.remove_prefix(end + 1);
    } else if (suffix.substr(0, 9) == "pitchlfo_") {
        index = -1;
        suffix.remove_prefix(9);
    } else if (suffix.substr(0, 7) == "amplfo_") {
        index = -2;
        legacyTarget = ModTarget::volume;
        suffix.remove_prefix(7);
    } else if (suffix.substr(0, 7) == "fillfo_") {
        index = -3;
        legacyTarget = ModTarget::cutoff;
        suffix.remove_prefix(7);
    } else {
        return false;
    }
    // This inter-LFO CC form is unknown to sfizz 1.2.3.
    if (suffix.substr(0, 8) == "freq_lfo" && suffix.find("_oncc") != suffix.npos)
        return true;
    auto& lfo = region.lfos[index];
    if (suffix == "wave" || suffix == "wave1") {
        auto wave = integer(value, -1, 12);
        if (wave >= 8 && wave < 12)
            throw std::runtime_error("unsupported LFO waveform " + value);
        lfo.wave = std::max(0, wave);
        return true;
    }
    static const std::pair<std::string_view, ModTarget> parameters[] = {
        {"freq", ModTarget::lfoFrequency}, {"phase", ModTarget::lfoPhase},
        {"delay", ModTarget::lfoDelay}, {"fade", ModTarget::lfoFade},
        {"pitch", ModTarget::pitch}, {"volume", ModTarget::volume}, {"pan", ModTarget::pan},
        {"width", ModTarget::width}, {"position", ModTarget::position},
        {"cutoff2", ModTarget::cutoff2}, {"cutoff1", ModTarget::cutoff}, {"cutoff", ModTarget::cutoff},
        {"resonance2", ModTarget::resonance2}, {"resonance1", ModTarget::resonance}, {"resonance", ModTarget::resonance},
        {"eq1freq", ModTarget::eq1Frequency}, {"eq1bw", ModTarget::eq1Bandwidth}, {"eq1gain", ModTarget::eq1Gain},
        {"eq2freq", ModTarget::eq2Frequency}, {"eq2bw", ModTarget::eq2Bandwidth}, {"eq2gain", ModTarget::eq2Gain},
        {"eq3freq", ModTarget::eq3Frequency}, {"eq3bw", ModTarget::eq3Bandwidth}, {"eq3gain", ModTarget::eq3Gain},
    };
    const auto assign = [&](std::string_view field, ModTarget target) {
        auto unit = target == ModTarget::pan || target == ModTarget::width || target == ModTarget::position ? 100.f : 1.f;
        if (field.empty()) {
            auto parsed = number(value, -24000, 24000) / unit;
            if (target == ModTarget::lfoFrequency) lfo.frequency = parsed;
            else if (target == ModTarget::lfoPhase) lfo.phase = parsed;
            else if (target == ModTarget::lfoDelay) lfo.delay = parsed;
            else if (target == ModTarget::lfoFade) lfo.fade = parsed;
            else lfo.depths[target] = parsed;
            return true;
        }
        enum class Field { depth, curve, smooth };
        Field kind;
        if (field.substr(0, 5) == "_oncc") { kind = Field::depth; field.remove_prefix(5); }
        else if (field.substr(0, 3) == "_cc") { kind = Field::depth; field.remove_prefix(3); }
        else if (field.substr(0, 2) == "cc") { kind = Field::depth; field.remove_prefix(2); }
        else if (field.substr(0, 8) == "_curvecc") { kind = Field::curve; field.remove_prefix(8); }
        else if (field.substr(0, 9) == "_smoothcc") { kind = Field::smooth; field.remove_prefix(9); }
        else return false;
        const auto cc = modulationController(field);
        auto it = std::find_if(lfo.modulation.begin(), lfo.modulation.end(),
            [&](const Modulation& m) { return m.target == target && m.cc == cc; });
        if (it == lfo.modulation.end()) {
            lfo.modulation.push_back({target, cc});
            it = std::prev(lfo.modulation.end());
        }
        if (kind == Field::curve) it->curve = integer(value, 0, 255);
        else if (kind == Field::smooth) it->smooth = integer(value, 0, 10000);
        else it->depth = number(value, -24000, 24000) / unit;
        return true;
    };
    if (index < 0 && suffix.substr(0, 5) == "depth")
        return assign(suffix.substr(5), legacyTarget);
    for (const auto& parameter : parameters)
        if (suffix.substr(0, parameter.first.size()) == parameter.first &&
            assign(suffix.substr(parameter.first.size()), parameter.second))
            return true;
    return false;
}

bool equalizerOpcode(Region& region, const std::string& op, const std::string& value)
{
    if (op.size() < 5 || op.compare(0, 2, "eq") != 0 || op[2] < '1' || op[2] > '3' || op[3] != '_')
        return false;
    auto& eq = region.eq[op[2] - '1'];
    const auto field = std::string_view(op).substr(4);
    if (field == "freq") eq.frequency = number(value, 0, 20000);
    else if (field == "gain") eq.gain = number(value, -96, 96);
    else if (field == "bw") eq.bandwidth = number(value, .001f, 12);
    else return false;
    eq.enabled = true;
    return true;
}

bool flexEnvelopeOpcode(Region& region, const std::string& op, const std::string& value)
{
    if (op.compare(0, 2, "eg") != 0 || op.size() < 5 || op[2] < '0' || op[2] > '9')
        return false;
    const auto separator = op.find('_', 2);
    if (separator == op.npos)
        return false;
    const auto id = integer(op.substr(2, separator - 2), 1, 64);
    auto& env = region.pitchEnvelopes[id];
    auto field = std::string_view(op).substr(separator + 1);
    if (field == "sustain") env.sustain = integer(value, 0, 63);
    else if (field == "pitch") env.depth = number(value, -12000, 12000);
    else if (field.substr(0, 10) == "pitch_oncc")
        env.depthCC[modulationController(field.substr(10))] = number(value, -12000, 12000);
    else if (field.substr(0, 5) == "level")
        env.points[integer(std::string(field.substr(5)), 0, 63)].level = number(value, -1, 1);
    else if (field.substr(0, 4) == "time") {
        field.remove_prefix(4);
        const auto cc = field.find("_oncc");
        auto& point = env.points[integer(std::string(field.substr(0, cc)), 0, 63)];
        if (cc == field.npos) point.seconds = number(value, 0, 100);
        else point.timeCC[modulationController(field.substr(cc + 5))] = number(value, -100, 100);
    } else return false;
    return true;
}

bool filterOpcode(Region& region, const std::string& op, const std::string& v)
{
    auto name = op;
    size_t slot = 0;
    if (name.compare(0, 5, "fil2_") == 0) {
        name.erase(3, 1);
        slot = 1;
    } else if (name.compare(0, 5, "fil1_") == 0) {
        name.erase(3, 1);
    } else if (name == "cutoff2" || name == "resonance2") {
        name.pop_back();
        slot = 1;
    } else if (name == "cutoff1" || name == "resonance1") {
        name.pop_back();
    }
    auto& f = region.filters[slot];
    if (name == "cutoff") f.cutoff = number(v, 0, 20000);
    else if (name == "resonance") f.resonance = number(v, -60, 96);
    else if (name == "fil_keytrack") f.keyTrack = number(v, -12000, 12000);
    else if (name == "fil_keycenter") f.keyCenter = key(v);
    else if (name == "fil_veltrack") f.velocityTrack = number(v, -12000, 12000);
    else if (name == "fil_random" || name == "cutoff_random") f.random = number(v, -12000, 12000);
    else if (name == "fil_type") {
        static const std::map<std::string, FilterType> types{
            {"lpf_1p", FilterType::low1}, {"hpf_1p", FilterType::high1},
            {"lpf_2p", FilterType::low2}, {"hpf_2p", FilterType::high2},
            {"bpf_2p", FilterType::band2}, {"brf_2p", FilterType::notch2},
            {"lpf_4p", FilterType::low4}, {"hpf_4p", FilterType::high4}, {"bpf_4p", FilterType::band4},
            {"lpf_6p", FilterType::low6}, {"hpf_6p", FilterType::high6}, {"bpf_6p", FilterType::band6},
        };
        auto type = types.find(v);
        if (type == types.end())
            throw std::runtime_error("unsupported filter type " + v);
        f.type = type->second;
    } else {
        return false;
    }
    if (f.type == FilterType::none)
        f.type = FilterType::low2;
    return true;
}

bool controllerFadeOpcode(Region& region, const std::string& op, const std::string& v)
{
    if (op == "xf_cccurve") {
        region.powerControllerFade = powerCurve(v);
        return true;
    }
    static const std::pair<std::string_view, int Crossfade::*> fields[] = {
        {"xfin_locc", &Crossfade::inLow}, {"xfin_hicc", &Crossfade::inHigh},
        {"xfout_locc", &Crossfade::outLow}, {"xfout_hicc", &Crossfade::outHigh},
    };
    for (const auto& field : fields) {
        if (std::string_view(op).substr(0, field.first.size()) != field.first)
            continue;
        const auto cc = integer(op.substr(field.first.size()), 0, 127);
        region.controllerFades[cc].*field.second = integer(v, 0, 127);
        return true;
    }
    return false;
}

bool ignoredVariable(const std::string& op)
{
    if (op.compare(0, 3, "var") != 0)
        return false;
    const auto first = op.find_first_not_of("0123456789", 3);
    if (first == 3 || first == op.npos || op[first] != '_')
        return false;
    std::string_view target(op.data() + first + 1, op.size() - first - 1);
    return target == "cutoff" || target == "mod" || target.substr(0, 4) == "oncc" ||
        target.substr(0, 7) == "curvecc" || target.substr(0, 2) == "eq";
}

class Loader {
public:
    Loader(Path root, AudioBank& bank) : root_(std::move(root)), bank_(bank) {}

    Instrument load(const Path& path)
    {
        parse(path, 0);
        if (current_ == "curve") flushCurve();
        else flush();
        if (instrument_.regions.empty())
            throw std::runtime_error("no playable SFZ regions in " + path.u8string());
        return std::move(instrument_);
    }

private:
    void parse(const Path& file, int depth)
    {
        if (depth >= 32)
            throw std::runtime_error("too many nested SFZ includes");
        std::ifstream input(file, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot open SFZ " + file.u8string());
        std::string text{std::istreambuf_iterator<char>(input), {}};
        std::string_view src(text);
        size_t pos = src.substr(0, 3) == "\xef\xbb\xbf" ? 3 : 0;
        try {
            while (pos < src.size()) {
                pos = src.find_first_not_of(whitespace, pos);
                if (pos == src.npos)
                    break;
                if (src.substr(pos, 2) == "//") {
                    pos = src.find('\n', pos);
                } else if (src[pos] == '<') {
                    auto end = src.find('>', pos + 1);
                    if (end == src.npos)
                        throw std::runtime_error("unclosed SFZ header");
                    header(trim(src.substr(pos + 1, end - pos - 1)));
                    pos = end + 1;
                } else if (src[pos] == '#') {
                    auto end = src.find('\n', pos);
                    auto directive = src.substr(pos, end - pos);
                    if (directive.substr(0, 8) == "#include") {
                        auto filename = trim(directive.substr(8));
                        if (filename.size() < 2 || filename.front() != '"' || filename.back() != '"')
                            throw std::runtime_error("expected quoted #include path");
                        parse(file.parent_path() / filePath(unquote(expand(filename))), depth + 1);
                    } else if (directive.substr(0, 7) == "#define") {
                        auto definition = trim(directive.substr(7));
                        auto split = definition.find_first_of(whitespace);
                        if (definition.empty() || definition.front() != '$' || split == definition.npos)
                            throw std::runtime_error("invalid #define");
                        macros_[definition.substr(0, split)] = expand(trim(definition.substr(split)));
                    } else {
                        throw std::runtime_error("unsupported SFZ directive " + std::string(directive));
                    }
                    pos = end;
                } else {
                    size_t end = pos;
                    while (end < src.size() && (nameChar(src[end]) || src[end] == '$'))
                        ++end;
                    auto equals = src.find_first_not_of(" \t", end);
                    if (end == pos || equals == src.npos || src[equals] != '=')
                        throw std::runtime_error("expected opcode=value");
                    auto name = expand(std::string(src.substr(pos, end - pos)));
                    pos = equals + 1;
                    bool quoted = false;
                    while (pos < src.size()) {
                        char c = src[pos];
                        if (c == '"')
                            quoted = !quoted;
                        if (!quoted && (c == '\r' || c == '\n' || c == '<' ||
                            src.substr(pos, 2) == "//"))
                            break;
                        if (!quoted && (c == ' ' || c == '\t')) {
                            auto next = src.find_first_not_of(" \t", pos);
                            auto tail = next;
                            while (tail < src.size() && (nameChar(src[tail]) || src[tail] == '$'))
                                ++tail;
                            auto sep = src.find_first_not_of(" \t", tail);
                            if (tail != next && sep < src.size() && src[sep] == '=')
                                break;
                        }
                        ++pos;
                    }
                    if (quoted)
                        throw std::runtime_error("unclosed quoted value for " + name);
                    opcode(name, unquote(expand(trim(src.substr(equals + 1, pos - equals - 1)))));
                }
            }
        } catch (const std::exception& error) {
            auto line = 1 + std::count(src.begin(), src.begin() + std::min(pos, src.size()), '\n');
            throw std::runtime_error(file.u8string() + ":" + std::to_string(line) + ": " + error.what());
        }
    }

    std::string expand(std::string value) const
    {
        size_t pos = 0;
        while ((pos = value.find('$', pos)) != value.npos) {
            size_t end = pos + 1;
            while (end < value.size() && nameChar(value[end]))
                ++end;
            auto name = value.substr(pos, end - pos);
            auto it = macros_.find(name);
            if (it == macros_.end())
                throw std::runtime_error("undefined SFZ macro " + name);
            value.replace(pos, end - pos, it->second);
            pos += it->second.size();
        }
        return value;
    }

    void header(const std::string& name)
    {
        if (current_ == "curve") flushCurve();
        else flush();
        current_ = name;
        if (name == "global") {
            global_.clear();
            master_.clear();
            group_.clear();
        } else if (name == "master") {
            master_.clear();
            group_.clear();
        } else if (name == "group") {
            group_.clear();
        } else if (name != "region" && name != "control" && name != "curve") {
            throw std::runtime_error("unsupported SFZ header <" + name + ">");
        }
    }

    void opcode(const std::string& name, const std::string& value)
    {
        if (value.empty())
            throw std::runtime_error("empty value for " + name);
        if (current_ == "global") global_[name] = value;
        else if (current_ == "master") master_[name] = value;
        else if (current_ == "group") group_[name] = value;
        else if (current_ == "region") region_[name] = value;
        else if (current_ == "curve") curve_[name] = value;
        else if (current_ == "control") {
            if (name == "default_path") {
                defaultPath_ = filePath(value);
            } else if (name.compare(0, 6, "set_cc") == 0) {
                const auto cc = integer(name.substr(6), 0, 511);
                auto raw = number(value, 0, 127);
                if (cc < 128) instrument_.initialCC[cc] = raw;
                else instrument_.extendedCCDefaults[cc] = raw;
            } else if (name.compare(0, 8, "set_hdcc") == 0) {
                const auto cc = integer(name.substr(8), 0, 511);
                auto raw = 127 * number(value, 0, 1);
                if (cc < 128) instrument_.initialCC[cc] = raw;
                else instrument_.extendedCCDefaults[cc] = raw;
            } else if (name.compare(0, 8, "label_cc") == 0) {
                (void)integer(name.substr(8), 0, 511);
            } else if (name.compare(0, 9, "label_key") == 0) {
                (void)integer(name.substr(9), 0, 127);
            } else if (name == "image" || name == "image_controls") {
                return; // Display metadata has no offline render behavior.
            } else {
                throw std::runtime_error("unsupported control opcode " + name);
            }
        } else {
            throw std::runtime_error("opcode outside SFZ header " + name);
        }
    }

    void flushCurve()
    {
        const auto index = curve_.find("curve_index");
        if (index == curve_.end() && explicitCurves_) {
            curve_.clear();
            return;
        }
        const auto id = index == curve_.end() ? nextCurve_++ : integer(index->second, 0, 255);
        explicitCurves_ |= index != curve_.end();
        std::map<int, float> points{{0, 0}, {127, 1}};
        for (const auto& item : curve_) {
            if (item.first == "curve_index")
                continue;
            if (item.first.front() != 'v')
                throw std::runtime_error("unsupported <curve> opcode " + item.first);
            points[integer(item.first.substr(1), 0, 127)] = number(item.second, -1e16f, 1e16f);
        }
        auto& curve = instrument_.curves[id];
        for (auto first = points.begin(), last = std::next(first); last != points.end(); ++first, ++last) {
            for (int x = first->first; x <= last->first; ++x) {
                auto phase = float(x - first->first) / (last->first - first->first);
                curve[x] = std::clamp(first->second + phase * (last->second - first->second), -1.f, 1.f);
            }
        }
        curve_.clear();
    }

    void flush()
    {
        if (current_ != "region")
            return;
        Region region;
        std::string sample;
        std::map<int, Condition> conditions, triggers;
        for (const auto* scope : {&global_, &master_, &group_, &region_}) {
            // key sets all three key fields. Specific fields in the same scope win.
            if (auto it = scope->find("key"); it != scope->end()) {
                region.triggerOnNote = it->second != "-1";
                region.lowKey = region.highKey = region.rootKey =
                    region.triggerOnNote ? key(it->second) : 60;
            }
            for (const auto& item : *scope) {
                try {
                    apply(region, sample, conditions, triggers, item.first, item.second);
                } catch (const std::exception& error) {
                    throw std::runtime_error(item.first + "=" + item.second + ": " + error.what());
                }
            }
        }
        const bool emptyBlock = region_.empty();
        region_.clear();
        if (sample.empty() && emptyBlock)
            sample = "*silence";
        if (sample == "*sine") region.generator = Generator::sine;
        else if (sample == "*silence") region.generator = Generator::silence;
        else if (sample == "*noise") region.generator = Generator::noise;
        else if (sample.empty() || sample.front() == '*')
            throw std::runtime_error("missing or unsupported region sample " + sample);
        if ((region.lowKey > region.highKey && triggers.empty()) || region.lowVelocity > region.highVelocity ||
            region.randomLow > region.randomHigh ||
            region.switchLow > region.switchHigh)
            throw std::runtime_error("invalid region range for " + sample);
        if (!region.velocityCurve.empty()) {
            std::sort(region.velocityCurve.begin(), region.velocityCurve.end());
            if (region.velocityCurve.front().first != 0)
                region.velocityCurve.insert(region.velocityCurve.begin(), {0, 0});
            if (region.velocityCurve.back().first != 127)
                region.velocityCurve.emplace_back(127, 1);
        }
        if (region.generator == Generator::none) {
            region.sample = bank_.describe(root_ / defaultPath_ / filePath(sample));
            const Sample& audio = bank_.samples[region.sample];
            region.end = region.end < 0 ? audio.frames : std::min(region.end + 1, audio.frames);
            if (region.offset >= region.end)
                throw std::runtime_error("offset past end of sample " + sample);
            if (region.loop == Loop::automatic)
                region.loop = audio.loopEnd > 0 || region.loopEnd >= 0 ? Loop::continuous : Loop::none;
            if (region.loop == Loop::continuous || region.loop == Loop::sustain) {
                if (region.loopStart < 0) region.loopStart = audio.loopFirst;
                if (region.loopEnd < 0) region.loopEnd = audio.loopEnd > 0 ? audio.loopEnd : region.end;
                else region.loopEnd = std::min(region.loopEnd + 1, region.end);
                if (region.loopStart >= region.loopEnd || region.loopEnd > region.end || region.offset >= region.loopEnd)
                    throw std::runtime_error("invalid forward loop in " + sample);
            }
        }
        for (const auto& item : conditions) {
            if (item.second.low > item.second.high)
                throw std::runtime_error("invalid CC range for " + sample);
            region.conditions.push_back(item.second);
            instrument_.usedCC[item.first] = true;
        }
        for (auto& fade : region.controllerFades) {
            fade.second.power = region.powerControllerFade;
            instrument_.usedCC[fade.first] = true;
        }
        for (const auto& m : region.modulation)
            if (m.cc < 128)
                instrument_.usedCC[m.cc] = true;
        for (const auto& item : region.lfos) {
            const auto enableFilter = [&](ModTarget target) {
                const bool second = target == ModTarget::cutoff2 || target == ModTarget::resonance2;
                if (target >= ModTarget::cutoff && target <= ModTarget::resonance2 &&
                    region.filters[second].type == FilterType::none)
                    region.filters[second].type = FilterType::low2;
                if (target >= ModTarget::eq1Frequency)
                    region.eq[(int(target) - int(ModTarget::eq1Frequency)) / 3].enabled = true;
            };
            for (const auto& depth : item.second.depths)
                enableFilter(depth.first);
            for (const auto& m : item.second.modulation) {
                if (m.cc < 128)
                    instrument_.usedCC[m.cc] = true;
                enableFilter(m.target);
            }
        }
        for (const auto& item : region.pitchEnvelopes) {
            const auto& env = item.second;
            if (env.points.empty() || env.points.rbegin()->second.level != 0 ||
                (env.sustain >= 0 && env.sustain != env.points.rbegin()->first))
                throw std::runtime_error("flexible pitch envelope must end at a zero-level sustain point");
            for (const auto& cc : env.depthCC)
                if (cc.first < 128)
                    instrument_.usedCC[cc.first] = true;
            for (const auto& point : env.points)
                for (const auto& cc : point.second.timeCC)
                    if (cc.first < 128)
                        instrument_.usedCC[cc.first] = true;
        }
        if (region.sustainEnabled)
            instrument_.usedCC[region.sustainCC] = true;
        auto index = static_cast<int>(instrument_.regions.size());
        if (region.triggerOnNote) {
            for (int note = region.lowKey; note <= region.highKey; ++note)
                instrument_.byKey[note].push_back(index);
        }
        for (const auto& item : triggers) {
            if (item.second.low > item.second.high)
                throw std::runtime_error("invalid CC trigger range for " + sample);
            region.ccTriggers.push_back(item.second);
            instrument_.byCC[item.first].push_back(index);
            instrument_.usedCC[item.first] = true;
        }
        if (region.switchLow >= 0) {
            for (int note = region.switchLow; note <= region.switchHigh; ++note)
                instrument_.switchKeys[note] = true;
        }
        if (region.switchDefault >= 0)
            instrument_.initialSwitch = region.switchDefault;
        instrument_.hasOffBy |= region.chokedByGroup;
        instrument_.regions.push_back(std::move(region));
    }

    static void apply(Region& r, std::string& sample, std::map<int, Condition>& conditions,
                      std::map<int, Condition>& triggers,
                      const std::string& op, const std::string& v)
    {
        if (op == "sample") sample = v;
        else if (op == "key" || op == "group_label" || op == "master_label" ||
                 op == "global_label" || op == "region_label" || op == "sw_label") return;
        else if (op == "lokey") r.lowKey = v == "-1" ? 0 : key(v);
        else if (op == "hikey") {
            r.triggerOnNote = v != "-1";
            r.highKey = r.triggerOnNote ? key(v) : 127;
        }
        else if (op == "pitch_keycenter") r.rootKey = key(v);
        else if (op == "lovel") r.lowVelocity = integer(v, 0, 127);
        else if (op == "hivel") r.highVelocity = integer(v, 0, 127);
        else if (op == "lochan" || op == "hichan") { (void)integer(v, 1, 16); } // sfizz ignores these.
        else if (op == "volume") r.volume += number(v, -144, 60);
        else if (op == "amplitude") r.amplitude *= number(v, 0, 10000) / 100;
        else if (op == "amp_veltrack") r.velocityTrack = number(v, -100, 100) / 100;
        else if (op == "amp_random") r.ampRandom = number(v, -24, 24);
        else if (op == "pitch_random") r.pitchRandom = number(v, -12000, 12000);
        else if (op.compare(0, 13, "amp_velcurve_") == 0) {
            const auto velocity = integer(op.substr(13), 0, 127);
            const auto value = number(v, 0, 1);
            const auto point = std::find_if(r.velocityCurve.begin(), r.velocityCurve.end(),
                [&](const auto& p) { return p.first == velocity; });
            if (point == r.velocityCurve.end()) r.velocityCurve.emplace_back(velocity, value);
            else point->second = value;
        }
        else if (op == "pan") r.pan = number(v, -100, 100) / 100;
        else if (op == "width") r.width = number(v, -100, 100) / 100;
        else if (op == "position") r.position = number(v, -100, 100) / 100;
        else if (op == "xfin_lokey") r.keyFade.inLow = key(v);
        else if (op == "xfin_hikey") r.keyFade.inHigh = key(v);
        else if (op == "xfout_lokey") r.keyFade.outLow = key(v);
        else if (op == "xfout_hikey") r.keyFade.outHigh = key(v);
        else if (op == "xfin_lovel") r.velocityFade.inLow = integer(v, 0, 127);
        else if (op == "xfin_hivel") r.velocityFade.inHigh = integer(v, 0, 127);
        else if (op == "xfout_lovel") r.velocityFade.outLow = integer(v, 0, 127);
        else if (op == "xfout_hivel") r.velocityFade.outHigh = integer(v, 0, 127);
        else if (op == "xf_keycurve") r.keyFade.power = powerCurve(v);
        else if (op == "xf_velcurve") r.velocityFade.power = powerCurve(v);
        else if (op == "tune" || op == "pitch") r.tune = number(v, -1200, 1200);
        else if (op == "transpose") r.transpose = integer(v, -127, 127);
        else if (op == "pitch_keytrack") r.keyTrack = number(v, -1200, 1200);
        else if (op == "bend_up" || op == "bendup") r.bendUp = number(v, -12000, 12000);
        else if (op == "bend_down" || op == "benddown") r.bendDown = number(v, -12000, 12000);
        else if (op == "bend_smooth") r.bendSmooth = integer(v, 0, 10000);
        else if (op == "delay") r.delay = number(v, 0, 100);
        else if (op == "group") r.group = integer64(v, -INT32_MAX, UINT32_MAX);
        else if (op == "note_polyphony") r.notePolyphony = integer(v, 1, 256);
        else if (op == "note_selfmask") r.selfMask = v == "mask" || enabled(v);
        else if (op == "oscillator_phase") r.oscillatorPhase = number(v, -1000, 1000);
        else if (op == "off_by" || op == "offby") {
            r.offBy = integer64(v, -INT32_MAX, UINT32_MAX);
            r.chokedByGroup = true;
        } else if (op == "off_mode" || op == "offmode") {
            if (v == "fast") r.offMode = OffMode::fast;
            else if (v == "normal") r.offMode = OffMode::normal;
            else if (v == "time") r.offMode = OffMode::time;
            else throw std::runtime_error("unsupported off mode " + v);
        } else if (op == "off_time") {
            r.offTime = number(v, 0, 100);
            r.offMode = OffMode::time;
        }
        else if (op == "offset") r.offset = integer(v, 0, INT32_MAX - 1);
        else if (op == "offset_random") r.offsetRandom = integer(v, 0, INT32_MAX - 1);
        else if (op == "end") r.end = integer(v, 0, INT32_MAX - 1);
        else if (op == "loop_start" || op == "loopstart") r.loopStart = integer(v, 0, INT32_MAX - 1);
        else if (op == "loop_end" || op == "loopend") r.loopEnd = integer(v, 0, INT32_MAX - 1);
        else if (op == "loop_crossfade") r.loopCrossfade = std::max(.001f, number(v, 0, 100));
        else if (op == "loop_type" || op == "looptype") {
            if (v != "forward" && v != "alternate" && v != "backward")
                throw std::runtime_error("unknown loop direction hint " + v);
            // These hints have no effect in sfizz 1.2.3.
        }
        else if (op == "ampeg_delay") r.envelopeDelay = number(v, 0, 100);
        else if (op == "ampeg_attack") r.attack = number(v, 0, 100);
        else if (op == "ampeg_hold") r.hold = number(v, 0, 100);
        else if (op == "ampeg_decay") r.decay = number(v, 0, 100);
        else if (op == "ampeg_sustain") r.sustain = number(v, 0, 100) / 100;
        else if (op == "ampeg_start") r.start = number(v, 0, 100) / 100;
        else if (op == "ampeg_release") r.release = number(v, 0, 100);
        else if (op == "ampeg_dynamic") r.envelopeDynamic = enabled(v);
        else if (op == "rt_decay") r.releaseDecay = number(v, 0, 200);
        else if (op == "rt_dead") r.releaseDead = enabled(v);
        else if (op == "sustain_sw") r.sustainEnabled = enabled(v);
        else if (op == "sustain_cc") r.sustainCC = integer(v, 0, 127);
        else if (op == "sustain_lo") r.sustainLow = integer(v, 0, 127);
        else if (op == "seq_length") r.sequenceLength = integer(v, 1, 1000000);
        else if (op == "seq_position") {
            r.sequencePosition = integer(v, 1, 1000000);
            r.explicitSequencePosition = true;
        }
        else if (op == "sw_default") r.switchDefault = key(v);
        else if (op == "sw_down") r.switchDown = key(v);
        else if (op == "sw_up") r.switchUp = key(v);
        else if (op == "sw_previous") r.switchPrevious = key(v);
        else if (op == "sw_lokey" || op == "sw_hikey") { (void)key(v); } // Switch display range.
        else if (op == "sw_last") {
            if (!r.switchRange) r.switchLow = r.switchHigh = key(v);
        } else if (op == "sw_lolast" || op == "sw_hilast") {
            auto note = key(v);
            if (!r.switchRange) r.switchLow = r.switchHigh = note;
            (op == "sw_lolast" ? r.switchLow : r.switchHigh) = note;
            r.switchRange = true;
        }
        else if (op == "lorand") r.randomLow = number(v, 0, 1);
        else if (op == "hirand") r.randomHigh = number(v, 0, 1);
        else if (op == "trigger") {
            const std::map<std::string, Trigger> values{{"attack", Trigger::attack}, {"release", Trigger::release},
                {"release_key", Trigger::releaseKey}, {"first", Trigger::first}, {"legato", Trigger::legato}};
            r.trigger = values.at(v);
        } else if (op == "loop_mode") {
            const std::map<std::string, Loop> values{{"no_loop", Loop::none}, {"one_shot", Loop::oneShot},
                {"loop_continuous", Loop::continuous}, {"loop_sustain", Loop::sustain}};
            r.loop = values.at(v);
        } else {
            if (modulationOpcode(r, op, v) || filterOpcode(r, op, v) || equalizerOpcode(r, op, v) || lfoOpcode(r, op, v) ||
                secondaryEnvelopeOpcode(r, op, v) || flexEnvelopeOpcode(r, op, v) ||
                controllerFadeOpcode(r, op, v) || ignoredVariable(op) ||
                op == "ampeg_attack_shape" || op == "ampeg_release_shape")
                return;
            const bool trigger = op.compare(0, 3, "on_") == 0;
            const auto offset = trigger ? 3 : 0;
            if (op.compare(offset, 4, "locc") != 0 && op.compare(offset, 4, "hicc") != 0)
                throw std::runtime_error("unsupported SFZ opcode");
            auto cc = static_cast<uint8_t>(integer(op.substr(offset + 4), 0, 127));
            auto& ranges = trigger ? triggers : conditions;
            auto& condition = ranges.emplace(cc, Condition{cc, 0, 127}).first->second;
            (op[offset] == 'l' ? condition.low : condition.high) = static_cast<uint8_t>(integer(v, 0, 127));
        }
    }

    Path root_, defaultPath_;
    AudioBank& bank_;
    Instrument instrument_;
    std::string current_;
    Opcodes global_, master_, group_, region_, macros_, curve_;
    int nextCurve_ = 7;
    bool explicitCurves_ = false;
};

} // namespace

Instrument loadInstrument(const std::filesystem::path& path, AudioBank& bank)
{
    return Loader(path.parent_path(), bank).load(path);
}

float curveValue(const Instrument& instrument, int curve, float value)
{
    value = std::clamp(value, 0.f, 127.f);
    const auto lo = static_cast<int>(value);
    const auto hi = std::min(127, lo + 1);
    const auto fraction = value - lo;
    if (const auto custom = instrument.curves.find(curve); custom != instrument.curves.end())
        return custom->second[lo] + fraction * (custom->second[hi] - custom->second[lo]);
    const auto at = [curve](int key) {
        const auto x = key / 127.f;
        switch (curve) {
        case 1: return 2 * x - 1;
        case 2: return 1 - x;
        case 3: return 1 - 2 * x;
        case 4: return x * x;
        case 5: return std::sqrt(x);
        case 6: return std::sqrt(1 - x);
        default: return x;
        }
    };
    const auto a = at(lo);
    return a + fraction * (at(hi) - a);
}

} // namespace clay
