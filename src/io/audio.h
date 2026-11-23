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
    void releaseDecoded();
    bool completeOutput(const std::filesystem::path& path, bool flac, int rate, int channels) const;
    void writeFlac(const std::filesystem::path& path, const int16_t* pcm, int dataFrames, int frames,
                   int rate, int channels, int level) const;

    std::vector<Sample> samples;
    std::vector<float> pcm;

private:
    struct Decoder;
    Decoder* decoder_;
};

class PendingOutput {
public:
    explicit PendingOutput(const std::filesystem::path& destination);
    ~PendingOutput();
    PendingOutput(const PendingOutput&) = delete;
    PendingOutput& operator=(const PendingOutput&) = delete;
    const std::filesystem::path& path() const { return temporary_; }
    void publish();
private:
    std::filesystem::path destination_, temporary_;
};

void writeWave(const std::filesystem::path& path, const int16_t* pcm, int dataFrames, int frames,
               int rate, int channels);

} // namespace clay
