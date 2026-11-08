#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace clay {

struct Sample {
    uint64_t offset;
    int frames;
    int rate;
    int channels;
    int loopFirst;
    int loopEnd;
};

class AudioBank {
public:
    explicit AudioBank(const std::filesystem::path& library);
    ~AudioBank();
    AudioBank(const AudioBank&) = delete;
    AudioBank& operator=(const AudioBank&) = delete;
    int describe(const std::filesystem::path& path);
    const Sample& decode(int id);

    std::vector<Sample> samples;
    std::vector<float> pcm;

private:
    struct Decoder;
    Decoder* decoder_;
};

void writeWave(const std::filesystem::path& path, const int16_t* pcm, int dataFrames, int frames,
               int rate, int channels);

} // namespace clay
