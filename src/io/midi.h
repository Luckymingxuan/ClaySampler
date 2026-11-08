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
};

Midi loadMidi(const std::filesystem::path& path, int rate, TailPolicy tail);

} // namespace clay
