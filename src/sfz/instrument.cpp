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

class Loader {
public:
    Loader(Path root, AudioBank& bank) : root_(std::move(root)), bank_(bank) {}

    Instrument load(const Path& path)
    {
        parse(path, 0);
        flush();
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
        flush();
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
        } else if (name != "region" && name != "control") {
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
        else if (current_ == "control") {
            if (name == "default_path") {
                defaultPath_ = filePath(value);
            } else if (name.compare(0, 6, "set_cc") == 0) {
                const auto cc = integer(name.substr(6), 0, 127);
                instrument_.initialCC[cc] = number(value, 0, 127);
            } else if (name.compare(0, 8, "set_hdcc") == 0) {
                const auto cc = integer(name.substr(8), 0, 127);
                instrument_.initialCC[cc] = 127 * number(value, 0, 1);
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
        region_.clear();
        if (sample.empty() || sample.front() == '*')
            throw std::runtime_error("missing or unsupported region sample " + sample);
        if ((region.lowKey > region.highKey && triggers.empty()) || region.lowVelocity > region.highVelocity ||
            region.sequencePosition > region.sequenceLength || region.randomLow >= region.randomHigh ||
            region.switchLow > region.switchHigh)
            throw std::runtime_error("invalid region range for " + sample);
        if (!region.velocityCurve.empty()) {
            std::sort(region.velocityCurve.begin(), region.velocityCurve.end());
            if (region.velocityCurve.front().first != 0)
                region.velocityCurve.insert(region.velocityCurve.begin(), {0, 0});
            if (region.velocityCurve.back().first != 127)
                region.velocityCurve.emplace_back(127, 1);
        }
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
            else ++region.loopEnd;
            if (region.loopStart >= region.loopEnd || region.loopEnd > region.end || region.offset >= region.loopEnd)
                throw std::runtime_error("invalid forward loop in " + sample);
        }
        for (const auto& item : conditions) {
            if (item.second.low > item.second.high)
                throw std::runtime_error("invalid CC range for " + sample);
            region.conditions.push_back(item.second);
            instrument_.usedCC[item.first] = true;
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
        else if (op == "delay") r.delay = number(v, 0, 100);
        else if (op == "group") r.group = integer64(v, -INT32_MAX, UINT32_MAX);
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
        else if (op == "ampeg_dynamic") { (void)enabled(v); } // No dynamic modulation opcodes are admitted.
        else if (op == "rt_decay") r.releaseDecay = number(v, 0, 200);
        else if (op == "rt_dead") r.releaseDead = enabled(v);
        else if (op == "sustain_sw") r.sustainEnabled = enabled(v);
        else if (op == "sustain_cc") r.sustainCC = integer(v, 0, 127);
        else if (op == "sustain_lo") r.sustainLow = integer(v, 0, 127);
        else if (op == "seq_length") r.sequenceLength = integer(v, 1, 1000000);
        else if (op == "seq_position") r.sequencePosition = integer(v, 1, 1000000);
        else if (op == "sw_default") r.switchDefault = key(v);
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
    Opcodes global_, master_, group_, region_, macros_;
};

} // namespace

Instrument loadInstrument(const std::filesystem::path& path, AudioBank& bank)
{
    return Loader(path.parent_path(), bank).load(path);
}

} // namespace clay
