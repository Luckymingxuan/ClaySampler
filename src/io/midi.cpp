#include "midi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace clay {
namespace {

class Bytes {
public:
    explicit Bytes(std::string_view bytes) : bytes_(bytes) {}
    int byte() { return static_cast<unsigned char>(take(1).front()); }

    uint32_t big(int size)
    {
        uint32_t value = 0;
        while (size--)
            value = (value << 8) | byte();
        return value;
    }

    uint32_t variable()
    {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            auto b = byte();
            value = (value << 7) | (b & 127);
            if (b < 128)
                return value;
        }
        throw std::runtime_error("invalid MIDI variable-length value");
    }

    std::string_view take(size_t length)
    {
        if (length > bytes_.size())
            throw std::runtime_error("truncated MIDI");
        auto value = bytes_.substr(0, length);
        bytes_.remove_prefix(length);
        return value;
    }

    bool empty() const { return bytes_.empty(); }

private:
    std::string_view bytes_;
};

struct Message {
    uint64_t tick;
    int status, a, b;
};

void readTrack(Bytes track, std::vector<Message>& messages)
{
    uint64_t tick = 0;
    int running = 0;
    while (!track.empty()) {
        tick += track.variable();
        int status = track.byte();
        int a = -1;
        if (status < 128) {
            a = status;
            status = running;
            if (!status)
                throw std::runtime_error("MIDI running status has no channel message");
        }
        if (status == 255) {
            const auto type = track.byte();
            Bytes meta(track.take(track.variable()));
            if (type == 0x51) {
                const auto tempo = meta.big(3);
                if (!tempo || !meta.empty())
                    throw std::runtime_error("invalid MIDI tempo");
                messages.push_back({tick, 0xff, int(tempo), 0});
            } else if (type == 0x2f) {
                if (!meta.empty())
                    throw std::runtime_error("invalid MIDI end-of-track");
                messages.push_back({tick, 0, 0, 0});
                return;
            }
            running = 0;
            continue;
        }
        if (status >= 0xf0)
            throw std::runtime_error("unsupported MIDI system or SysEx message");
        running = status;
        if (a < 0)
            a = track.byte();
        int type = status >> 4;
        int b = type == 0xc || type == 0xd ? 0 : track.byte();
        if (a >= 128 || b >= 128)
            throw std::runtime_error("invalid MIDI data byte");
        messages.push_back({tick, status, a, b});
    }
    throw std::runtime_error("missing MIDI end-of-track");
}

int priority(const Message& message)
{
    if ((message.status >> 4) == 9 && message.b != 0)
        return 30;
    if ((message.status >> 4) == 8 || (message.status >> 4) == 9)
        return 25;
    return 15;
}

void prepareNatural(std::vector<Message>& messages)
{
    std::array<std::array<int, 128>, 16> down{};
    uint64_t lastNote = 0;
    int channel = -1;
    for (const auto& message : messages) {
        const int type = message.status >> 4;
        if (type != 8 && type != 9)
            continue;
        const int current = message.status & 15;
        auto& count = down[current][message.a];
        if (type == 9 && message.b) {
            ++count;
        } else if (count > 0) {
            --count;
            if (channel >= 0 && channel != current)
                throw std::runtime_error("multiple MIDI channels in one render job");
            channel = current;
            lastNote = message.tick;
        }
    }
    if (channel < 0)
        throw std::runtime_error("natural rendering requires a complete MIDI note");

    constexpr int pedals[] = {64, 66, 69};
    std::array<bool, 3> held{};
    auto end = lastNote;
    for (const auto& message : messages) {
        if ((message.status >> 4) != 0xb)
            continue;
        for (size_t p = 0; p < held.size(); ++p) {
            if (message.a != pedals[p])
                continue;
            if (message.tick <= end) {
                held[p] = message.b >= 64;
            } else if (held[p] && message.b < 64) {
                end = message.tick;
                held[p] = false;
            }
        }
    }
    messages.erase(std::remove_if(messages.begin(), messages.end(), [&](const Message& m) {
        if (!m.status || m.tick > end || (m.status >> 4) == 0xc)
            return true;
        if ((m.status >> 4) != 0xb)
            return false;
        switch (m.a) {
        case 0: case 7: case 10: case 32: case 91: case 93: return true;
        case 120: case 123: return m.tick >= lastNote;
        default: return false;
        }
    }), messages.end());
    const int control = 0xb0 | channel;
    messages.insert(messages.begin(), {{0, control, 7, 100}, {0, control, 10, 64}});
    std::stable_sort(messages.begin(), messages.end(), [](const Message& a, const Message& b) {
        return a.tick == b.tick ? priority(a) < priority(b) : a.tick < b.tick;
    });
    for (int cc : pedals)
        messages.push_back({end, control, cc, 0});
    messages.push_back({end, 0, 0, 0});
}

Midi readFile(std::string_view data, int rate, TailPolicy tail)
{
    Bytes file(data);
    if (file.take(4) != "MThd")
        throw std::runtime_error("expected a Standard MIDI File");
    Bytes header(file.take(file.big(4)));
    auto format = header.big(2);
    auto tracks = header.big(2);
    auto division = header.big(2);
    if (format > 1 || tracks == 0 || (format == 0 && tracks != 1) || division == 0 || division >= 32768)
        throw std::runtime_error("expected format 0/1 MIDI with PPQ timing");
    std::vector<Message> messages;
    for (uint32_t i = 0; i < tracks; ++i) {
        if (file.take(4) != "MTrk")
            throw std::runtime_error("missing MIDI track chunk");
        readTrack(Bytes(file.take(file.big(4))), messages);
    }
    std::stable_sort(messages.begin(), messages.end(),
        [](const Message& a, const Message& b) { return a.tick < b.tick; });
    if (tail == TailPolicy::natural)
        prepareNatural(messages);
    if (tail == TailPolicy::custom) {
        const auto last = std::find_if(messages.rbegin(), messages.rend(), [](const Message& m) {
            const int type = m.status >> 4;
            return type == 8 || type == 9 || type == 0xb || type == 0xe;
        });
        if (last != messages.rend()) {
            const auto tick = last->tick;
            const auto channel = last->status & 15;
            messages.erase(last.base(), messages.end());
            messages.push_back({tick, 0xb0 | channel, 64, 0});
            messages.push_back({tick, 0, 0, 0});
        }
    }
    Midi result;
    uint64_t previousTick = 0;
    double frames = 0;
    double scale = rate / (1000000.0 * division);
    int tempo = 500000;
    int channel = -1;
    for (const auto& message : messages) {
        frames += double(message.tick - previousTick) * tempo * scale;
        previousTick = message.tick;
        if (frames >= std::numeric_limits<int>::max() / 2)
            throw std::runtime_error("MIDI exceeds maximum render duration");
        auto frame = static_cast<int>(tail == TailPolicy::natural ? std::nearbyint(frames) : std::floor(frames + 1e-7));
        if (message.status == 0xff) {
            tempo = message.a;
            continue;
        }
        if (!message.status) {
            result.end = frame;
            continue;
        }
        auto type = message.status >> 4;
        auto current = message.status & 15;
        if (channel < 0)
            channel = current;
        if (channel != current)
            throw std::runtime_error("multiple MIDI channels in one render job");
        switch (type) {
        case 8:
        case 9:
            result.events.push_back({frame, type == 9 && message.b ? EventKind::on : EventKind::off,
                                     message.a, message.b});
            break;
        case 0xb:
            result.events.push_back({frame, EventKind::control, message.a, message.b});
            break;
        case 0xc: // The SFZ preset is selected by the job, as in sfizz_render.
            break;
        case 0xe:
            result.events.push_back({frame, EventKind::bend, (message.a | (message.b << 7)) - 8192, 0});
            break;
        default:
            throw std::runtime_error("unsupported MIDI channel message type " + std::to_string(type));
        }
    }
    return result;
}

} // namespace

Midi loadMidi(const std::filesystem::path& path, int rate, TailPolicy tail)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open MIDI " + path.u8string());
    const std::string data{std::istreambuf_iterator<char>(file), {}};
    try {
        return readFile(data, rate, tail);
    } catch (const std::exception& error) {
        throw std::runtime_error(path.u8string() + ": " + error.what());
    }
}

} // namespace clay
