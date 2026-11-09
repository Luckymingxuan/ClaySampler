#pragma once

#include <filesystem>
#include <vector>

namespace clay {

enum class TailPolicy { natural, orig, custom };

enum class EventKind { on, off, control, bend };

struct Event {
    int frame;
    EventKind kind;
    int a;
    int b;
};

struct Midi {
    std::vector<Event> events;
    int end = 0;
    int bodyFrames = 0;
    std::vector<int> onsets;
};

Midi loadMidi(const std::filesystem::path& path, int rate, TailPolicy tail, bool measure = false);

} // namespace clay
