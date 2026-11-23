#include "audio.h"
#include "library.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

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

std::array<unsigned char, 44> waveHeader(uint32_t bytes, int rate, int channels)
{
    std::array<unsigned char, 44> header{};
    const auto put = [&](size_t offset, uint32_t value, int width) {
        for (int i = 0; i < width; ++i)
            header[offset + i] = static_cast<unsigned char>(value >> (8 * i));
    };
    std::memcpy(header.data(), "RIFF", 4);
    put(4, bytes + 36, 4);
    std::memcpy(header.data() + 8, "WAVEfmt ", 8);
    put(16, 16, 4);
    put(20, 1, 2);
    put(22, channels, 2);
    put(24, rate, 4);
    put(28, rate * channels * 2, 4);
    put(32, channels * 2, 2);
    put(34, 16, 2);
    std::memcpy(header.data() + 36, "data", 4);
    put(40, bytes, 4);
    return header;
}

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
    int64_t (*write)(void*, const int16_t*, int64_t) =
        library.get<decltype(write)>("sf_writef_short");
    int (*error)(void*) = library.get<decltype(error)>("sf_error");
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
    auto& loaded = decoder_->loaded;
    if (const auto it = loaded.find(path); it != loaded.end())
        return it->second;
    const auto key = std::filesystem::weakly_canonical(path);
    if (const auto it = loaded.find(key); it != loaded.end()) {
        loaded.emplace(path, it->second);
        return it->second;
    }
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
    loaded.emplace(key, id);
    if (path != key)
        loaded.emplace(path, id);
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

bool AudioBank::completeOutput(const std::filesystem::path& path, bool flac, int rate, int channels) const
{
    if (!flac) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        const std::streamoff size = input.tellg();
        if (size <= 44 || size - 44 > UINT32_MAX - 36 || (size - 44) % (channels * 2)) return false;
        std::array<unsigned char, 44> header;
        input.seekg(0);
        return input.read(reinterpret_cast<char*>(header.data()), header.size()) &&
            header == waveHeader(static_cast<uint32_t>(size - 44), rate, channels);
    }
    SoundInfo info;
    std::unique_ptr<void, int(*)(void*)> file(decoder_->open(path.c_str(), readMode, &info), decoder_->close);
    if (!file || info.format != (0x170000 | 0x0002) || info.frames <= 0 ||
        info.frames > std::numeric_limits<int>::max() || info.rate != rate || info.channels != channels)
        return false;
    std::array<float, 8192> buffer;
    int64_t total = 0;
    while (total < info.frames) {
        const auto count = std::min<int64_t>(buffer.size() / channels, info.frames - total);
        if (decoder_->read(file.get(), buffer.data(), count) != count) return false;
        total += count;
    }
    return decoder_->error(file.get()) == 0;
}

PendingOutput::PendingOutput(const std::filesystem::path& destination) : destination_(destination)
{
    static std::atomic<uint64_t> serial{0};
#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = getpid();
#endif
    for (;;) {
        temporary_ = destination_.parent_path() /
            ("claysampler." + std::to_string(pid) + "." + std::to_string(serial.fetch_add(1)) + ".part");
#ifdef _WIN32
        auto h = CreateFileW(temporary_.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); break; }
        if (GetLastError() == ERROR_FILE_EXISTS) continue;
#else
        int fd = ::open(temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd >= 0) { ::close(fd); break; }
        if (errno == EEXIST) continue;
#endif
        throw std::runtime_error("cannot create temporary output for " + destination_.u8string());
    }
}

PendingOutput::~PendingOutput()
{
    std::error_code ec;
    if (!temporary_.empty()) std::filesystem::remove(temporary_, ec);
}

void PendingOutput::publish()
{
#ifdef _WIN32
    auto h = CreateFileW(temporary_.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open completed output for flush");
    const bool synced = FlushFileBuffers(h) != 0;
    CloseHandle(h);
    if (!synced || !MoveFileExW(temporary_.c_str(), destination_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot publish " + destination_.u8string());
#else
    auto sync = [](const std::filesystem::path& path, int flags) {
        const int fd = ::open(path.c_str(), flags);
        if (fd < 0) throw std::runtime_error("cannot open for sync " + path.u8string());
        const int status = ::fsync(fd);
        const int closed = ::close(fd);
        if (status || closed) throw std::runtime_error("cannot sync " + path.u8string());
    };
    sync(temporary_, O_WRONLY);
    std::filesystem::rename(temporary_, destination_);
    sync(destination_.has_parent_path() ? destination_.parent_path() : ".", O_RDONLY | O_DIRECTORY);
#endif
    temporary_.clear();
}

void AudioBank::writeFlac(const std::filesystem::path& path, const int16_t* pcm, int dataFrames, int frames,
                          int rate, int channels, int level) const
{
    constexpr int flacPcm16 = 0x170000 | 0x0002;
    constexpr int writeMode = 0x20, setCompressionLevel = 0x1301;
    SoundInfo info{};
    info.rate = rate;
    info.channels = channels;
    info.format = flacPcm16;
    std::unique_ptr<void, int(*)(void*)> file(decoder_->open(path.c_str(), writeMode, &info), decoder_->close);
    if (!file)
        throw std::runtime_error("cannot create FLAC " + path.u8string());
    double compression = level / 8.0;
    if (!decoder_->command(file.get(), setCompressionLevel, &compression, sizeof compression))
        throw std::runtime_error("cannot set FLAC compression level");
    if (decoder_->write(file.get(), pcm, dataFrames) != dataFrames)
        throw std::runtime_error("cannot write FLAC " + path.u8string());
    static constexpr int16_t silence[8192]{};
    for (int remaining = frames - dataFrames; remaining > 0;) {
        const int count = std::min(remaining, 8192 / channels);
        if (decoder_->write(file.get(), silence, count) != count)
            throw std::runtime_error("cannot write FLAC " + path.u8string());
        remaining -= count;
    }
    if (decoder_->close(file.release()))
        throw std::runtime_error("cannot finish FLAC " + path.u8string());
}

void writeWave(const std::filesystem::path& path, const int16_t* pcm, int dataFrames, int frames,
               int rate, int channels)
{
    const auto stride = channels * 2;
    if (frames <= 0 || dataFrames < 0 || dataFrames > frames || static_cast<uint64_t>(frames) * stride + 36 > UINT32_MAX)
        throw std::runtime_error("output exceeds PCM16 WAV limits");
    std::ofstream file(path, std::ios::binary);
    const auto dataSize = static_cast<uint32_t>(frames) * stride;
    const auto header = waveHeader(dataSize, rate, channels);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
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
