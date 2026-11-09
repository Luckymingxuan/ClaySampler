#include "audio.h"
#include "library.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>

namespace clay {
namespace {

struct SoundInfo {
    int64_t frames{};
    int rate{}, channels{}, format{}, sections{}, seekable{};
};

struct SoundInstrument {
    int gain;
    char baseNote, detune, velocityLow, velocityHigh, keyLow, keyHigh;
    int loopCount;
    struct Loop { int mode; unsigned start, end, count; };
    Loop loops[16];
};

constexpr int readMode = 0x10;
constexpr int getInstrument = 0x10d0;
constexpr int forwardLoop = 801;

} // namespace

struct AudioBank::Decoder {
    explicit Decoder(const std::filesystem::path& path)
        : library(path)
    {
    }

    Library library;
#ifdef _WIN32
    using Open = void*(*)(const wchar_t*, int, SoundInfo*);
    Open open = library.get<Open>("sf_wchar_open");
#else
    using Open = void*(*)(const char*, int, SoundInfo*);
    Open open = library.get<Open>("sf_open");
#endif
    int (*close)(void*) = library.get<int(*)(void*)>("sf_close");
    int64_t (*read)(void*, float*, int64_t) = library.get<int64_t(*)(void*, float*, int64_t)>("sf_readf_float");
    int (*command)(void*, int, void*, int) = library.get<int(*)(void*, int, void*, int)>("sf_command");
    std::map<std::filesystem::path, int> loaded;
    std::vector<std::filesystem::path> paths;
};

AudioBank::AudioBank(const std::filesystem::path& library)
    : decoder_(new Decoder(library))
{
}

AudioBank::~AudioBank() { delete decoder_; }

int AudioBank::describe(const std::filesystem::path& path)
{
    const auto key = std::filesystem::weakly_canonical(path);
    if (const auto it = decoder_->loaded.find(key); it != decoder_->loaded.end())
        return it->second;
    SoundInfo info;
    std::unique_ptr<void, int(*)(void*)> file(decoder_->open(key.c_str(), readMode, &info), decoder_->close);
    if (!file || info.frames <= 0 || info.frames > std::numeric_limits<int>::max() ||
        (info.channels != 1 && info.channels != 2) || info.rate <= 0)
        throw std::runtime_error("cannot decode mono or stereo sample " + path.u8string());

    SoundInstrument metadata{};
    int loopFirst = 0;
    int loopEnd = 0;
    if (decoder_->command(file.get(), getInstrument, &metadata, sizeof metadata) &&
        metadata.loopCount > 0 && metadata.loops[0].mode == forwardLoop &&
        metadata.loops[0].start < metadata.loops[0].end && metadata.loops[0].end <= info.frames) {
        loopFirst = static_cast<int>(metadata.loops[0].start);
        loopEnd = static_cast<int>(metadata.loops[0].end);
    }

    const auto id = static_cast<int>(samples.size());
    samples.push_back({UINT64_MAX, static_cast<int>(info.frames), info.rate, info.channels, loopFirst, loopEnd});
    decoder_->paths.push_back(key);
    decoder_->loaded.emplace(key, id);
    return id;
}

const Sample& AudioBank::decode(int id)
{
    auto& sample = samples.at(id);
    if (sample.offset != UINT64_MAX)
        return sample;
    const auto& path = decoder_->paths.at(id);
    SoundInfo info;
    std::unique_ptr<void, int(*)(void*)> file(decoder_->open(path.c_str(), readMode, &info), decoder_->close);
    if (!file || info.frames != sample.frames || info.rate != sample.rate || info.channels != sample.channels)
        throw std::runtime_error("sample changed since it was opened " + path.u8string());
    const auto offset = pcm.size();
    pcm.resize(offset + static_cast<size_t>(info.frames) * info.channels);
    if (decoder_->read(file.get(), pcm.data() + offset, info.frames) != info.frames ||
        !std::all_of(pcm.begin() + offset, pcm.end(), [](float value) { return std::isfinite(value); })) {
        pcm.resize(offset);
        throw std::runtime_error("invalid sample data in " + path.u8string());
    }
    sample.offset = offset;
    return sample;
}

void AudioBank::releaseDecoded()
{
    std::vector<float>().swap(pcm);
    for (auto& sample : samples)
        sample.offset = UINT64_MAX;
}

void writeWave(const std::filesystem::path& path, const int16_t* pcm, int dataFrames, int frames,
               int rate, int channels)
{
    const auto stride = channels * 2;
    if (frames <= 0 || dataFrames < 0 || dataFrames > frames || static_cast<uint64_t>(frames) * stride + 36 > UINT32_MAX)
        throw std::runtime_error("output exceeds PCM16 WAV limits");
    std::ofstream file(path, std::ios::binary);
    auto integer = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i)
            file.put(static_cast<char>(value >> (8 * i)));
    };
    const auto dataSize = static_cast<uint32_t>(frames) * stride;
    file.write("RIFF", 4);
    integer(dataSize + 36, 4);
    file.write("WAVEfmt ", 8);
    integer(16, 4);
    integer(1, 2);
    integer(channels, 2);
    integer(rate, 4);
    integer(rate * stride, 4);
    integer(stride, 2);
    integer(16, 2);
    file.write("data", 4);
    integer(dataSize, 4);
    if (dataFrames)
        file.write(reinterpret_cast<const char*>(pcm), size_t(dataFrames) * stride);
    static constexpr char silence[16384]{};
    for (auto remaining = size_t(frames - dataFrames) * stride; remaining;) {
        const auto count = std::min(remaining, sizeof(silence));
        file.write(silence, count);
        remaining -= count;
    }
    file.close();
    if (!file)
        throw std::runtime_error("cannot write WAV " + path.u8string());
}

} // namespace clay
