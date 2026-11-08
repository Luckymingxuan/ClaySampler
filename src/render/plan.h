#pragma once

#include "../cuda/render_types.h"
#include "../io/midi.h"
#include "../sfz/instrument.h"

namespace clay {

struct RenderPlan {
    std::vector<VoiceSpan> spans;
    std::vector<std::vector<ControlPoint>> controls;
    std::vector<VoiceModulation> modulation;
    int filterTracks = 0;
    int frames = 0;
};

RenderPlan planRender(const Instrument& instrument, AudioBank& bank, const Midi& midi,
                      int rate, unsigned seed);

} // namespace clay
