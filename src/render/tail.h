#pragma once

#include "../io/audio.h"

namespace clay {

constexpr int naturalTailBlock = 1024;
constexpr int naturalTailLimitSeconds = 120;

Quality writeNaturalTail(const std::filesystem::path& path, const float* stereo, int frames,
                         int musicEnd, int limit, int rate);
void fadeTail(int16_t* pcm, int frames, int channels, int count);

} // namespace clay
