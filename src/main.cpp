#include "cuda/renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cwctype>
#include <deque>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

using Path = std::filesystem::path;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

#ifdef _WIN32
using Char = wchar_t;
Path setting(const wchar_t* name, const wchar_t* fallback)
{
    const auto* value = _wgetenv(name);
    return value && *value ? value : fallback;
}
#else
using Char = char;
Path setting(const char* name, const char* fallback)
{
    const auto* value = std::getenv(name);
    return value && *value ? value : fallback;
}
#endif

struct Options {
    Path sfz, output;
#ifdef _WIN32
    Path nvrtc = setting(L"CLAY_NVRTC", L"nvrtc64_120_0.dll");
    Path sndfile = setting(L"CLAY_SNDFILE", L"libsndfile-1.dll");
#else
    Path nvrtc = setting("CLAY_NVRTC", "libnvrtc.so");
    Path sndfile = setting("CLAY_SNDFILE", "libsndfile.so.1");
#endif
    int device = 0, rate = 44100, passes = 1;
    clay::TailPolicy tail = clay::TailPolicy::natural;
    std::optional<double> tailMax, maxAudio;
    std::optional<int> tailFadeMs;
    std::vector<Path> midi;
};

int integer(const Path& input, int low, int high)
{
    auto value = input.u8string();
    size_t used = 0;
    auto number = std::stoi(value, &used);
    if (used != value.size() || number < low || number > high)
        throw std::runtime_error("invalid numeric argument " + value);
    return number;
}

double seconds(const Path& input)
{
    auto value = input.u8string();
    size_t used = 0;
    const auto number = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(number) || number < 0 || number > 3600)
        throw std::runtime_error("invalid seconds argument " + value);
    return number;
}

clay::TailPolicy tailPolicy(const Path& input)
{
    if (input == "natural") return clay::TailPolicy::natural;
    if (input == "orig") return clay::TailPolicy::orig;
    if (input == "custom") return clay::TailPolicy::custom;
    throw std::runtime_error("expected --tail-policy natural, orig, or custom");
}

Options arguments(int argc, Char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        Path option(argv[i]);
        auto name = option.u8string();
        if (name.empty() || name.front() != '-') {
            options.midi.push_back(std::move(option));
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("missing value for " + name);
        Path value(argv[++i]);
        if (name == "--sfz") options.sfz = value;
        else if (name == "--output") options.output = value;
        else if (name == "--nvrtc") options.nvrtc = value;
        else if (name == "--sndfile") options.sndfile = value;
        else if (name == "--device") options.device = integer(value, 0, 1024);
        else if (name == "--samplerate") options.rate = integer(value, 8000, 192000);
        else if (name == "--passes") options.passes = integer(value, 1, 1000);
        else if (name == "--tail-policy") options.tail = tailPolicy(value);
        else if (name == "--tail-max-seconds") options.tailMax = seconds(value);
        else if (name == "--tail-fade-ms") options.tailFadeMs = integer(value, 0, 3600000);
        else if (name == "--max-audio-seconds") options.maxAudio = seconds(value);
        else throw std::runtime_error("unknown option " + name);
    }
    if (options.sfz.empty() || options.output.empty() || options.midi.empty())
        throw std::runtime_error("provide --sfz, --output, and at least one MIDI path");
    if (options.tail == clay::TailPolicy::custom && !options.tailMax)
        throw std::runtime_error("custom tail requires --tail-max-seconds");
    if (options.tail != clay::TailPolicy::custom && (options.tailMax || options.tailFadeMs))
        throw std::runtime_error("tail length and fade options require --tail-policy custom");
    if (options.maxAudio && options.tail != clay::TailPolicy::natural)
        throw std::runtime_error("--max-audio-seconds requires --tail-policy natural");
    return options;
}

int checkPresets(int argc, Char** argv)
{
    Options options;
    int first = 2;
    if (argc > 3 && Path(argv[first]) == "--sndfile") {
        options.sndfile = argv[first + 1];
        first += 2;
    }
    if (first == argc)
        throw std::runtime_error("provide at least one SFZ path after --check-sfz");
    bool rejected = false;
    for (int i = first; i < argc; ++i) {
        const Path path(argv[i]);
        try {
            clay::AudioBank bank(options.sndfile);
            const auto instrument = clay::loadInstrument(path, bank);
            std::cout << "SFZ\tOK\t" << instrument.regions.size() << '\n';
        } catch (const std::exception& error) {
            rejected = true;
            std::cout << "SFZ\tREJECT\t" << error.what() << '\n';
        }
    }
    return rejected ? 2 : 0;
}

struct PlannedBatch {
    std::vector<clay::RenderPlan> plans;
    std::shared_ptr<std::vector<Path>> outputs = std::make_shared<std::vector<Path>>();
    std::vector<std::string> rejected;
    size_t next = 0;
    double seconds = 0;
};

PlannedBatch planBatch(const Options& options, const clay::Instrument& instrument, clay::AudioBank& bank, size_t first)
{
    const auto start = Clock::now();
    const bool natural = options.tail == clay::TailPolicy::natural;
    const int customFade = static_cast<int>(std::lround(options.tailFadeMs.value_or(100) * (options.rate / 1000.)));
    PlannedBatch batch;
    const auto count = std::min(size_t{64}, options.midi.size() - first);
    batch.plans.reserve(count);
    batch.outputs->reserve(count);
    size_t frames = 0;
    batch.next = first;
    // Give the CUDA and WAV stages work before planning the entire invocation.
    for (; batch.next - first < count && frames < 24 * size_t{1024} * 1024;
         ++batch.next) {
        const auto& path = options.midi[batch.next];
        try {
            auto midi = clay::loadMidi(path, options.rate, options.tail);
            const int musicEnd = midi.end + 1;
            int limit = std::numeric_limits<int>::max();
            if (natural) {
                limit = musicEnd + naturalTailLimitSeconds * options.rate;
                if (options.maxAudio)
                    limit = std::min(limit, static_cast<int>(std::floor(*options.maxAudio * options.rate)));
                if (limit < musicEnd)
                    throw std::runtime_error("natural render exceeds the audio duration limit");
            } else if (options.tailMax) {
                limit = musicEnd + static_cast<int>(std::floor(*options.tailMax * options.rate));
            }
            auto plan = clay::planRender(instrument, bank, midi, options.rate, 1, limit, options.tail);
            plan.musicEnd = musicEnd;
            plan.limit = limit;
            if (options.tail == clay::TailPolicy::custom && plan.tailClipped && plan.frames == limit)
                plan.fadeFrames = std::min(customFade, std::max(0, plan.frames - musicEnd));
            auto name = path.stem();
            name += ".wav";
            batch.outputs->push_back(options.output / name);
            frames += plan.frames;
            batch.plans.push_back(std::move(plan));
        } catch (const std::exception& error) {
            batch.rejected.push_back("REJECT\t" + path.u8string() + '\t' + error.what() + '\n');
        }
    }
    batch.seconds = elapsed(start);
    return batch;
}

const char* tailError(TailStatus status)
{
    switch (status) {
    case TailStatus::duration: return "natural tail exceeded the tail or audio duration limit";
    case TailStatus::nonfinite: return "natural render produced non-finite audio";
    case TailStatus::peak: return "excessive_peak: natural render requires more than 5 dB attenuation";
    case TailStatus::ok: return nullptr;
    }
    return "unknown CUDA tail status";
}

struct WrittenBatch {
    size_t passed = 0;
    std::string ok, rejected;
};

WrittenBatch writeBatch(const clay::RenderBatch& batch, const std::vector<Path>& outputs, size_t first,
                        int rate, int channels)
{
    std::ostringstream ok, rejected;
    size_t passed = 0;
    for (size_t i = 0; i < batch.results.size(); ++i) {
        const auto& result = batch.results[i];
        const auto& path = outputs[first + i];
        try {
            if (const auto* error = tailError(result.status))
                throw std::runtime_error(error);
            if (result.last < 0)
                throw std::runtime_error("rendered audio is silent " + path.u8string());
            clay::writeWave(path, batch.pcm.get() + result.offset * channels, result.dataFrames, result.frames, rate, channels);
            ok << "OK\t" << path.filename().u8string() << '\t' << result.frames
               << '\t' << result.peak << '\t' << result.first << '\t' << result.last << '\n';
            ++passed;
        } catch (const std::exception& error) {
            rejected << "REJECT\t" << path.u8string() << '\t' << error.what() << '\n';
        }
    }
    return {passed, ok.str(), rejected.str()};
}

int run(const Options& options)
{
    const bool natural = options.tail == clay::TailPolicy::natural;
    const int channels = natural ? 1 : 2;
    auto start = Clock::now();
    clay::AudioBank bank(options.sndfile);
    std::set<Path> names;
    for (const auto& path : options.midi) {
        auto name = path.stem();
#ifdef _WIN32
        auto normalized = name.wstring();
        for (auto& ch : normalized)
            ch = static_cast<wchar_t>(std::towlower(ch));
        name = normalized;
#endif
        if (!names.insert(name).second)
            throw std::runtime_error("duplicate output basename " + path.stem().u8string());
    }
    auto opening = std::async(std::launch::async, [&] {
        auto instrument = clay::loadInstrument(options.sfz, bank);
        auto first = planBatch(options, instrument, bank, 0);
        return std::make_pair(std::move(instrument), std::move(first));
    });
    clay::Renderer renderer(options.device, options.nvrtc, natural, options.rate);
    auto startup = opening.get();
    const auto& instrument = startup.first;
    std::cout << "DEVICE\t" << renderer.device() << '\n'
              << "INIT\t" << elapsed(start) << '\n';
    std::filesystem::create_directories(options.output);
    size_t passed = 0;
    for (int pass = 1; pass <= options.passes; ++pass) {
        auto passStart = Clock::now();
        double preparation = 0, kernelSeconds = 0;
        passed = 0;
        const auto prefetch = [&](size_t next) {
            return std::async(std::launch::async, [&, next] { return planBatch(options, instrument, bank, next); });
        };
        auto group = pass == 1 ? std::move(startup.second) : planBatch(options, instrument, bank, 0);
        std::future<PlannedBatch> next;
        std::deque<std::future<WrittenBatch>> writers;
        const auto drain = [&] {
            auto pending = std::move(writers.front());
            writers.pop_front();
            const auto result = pending.get();
            passed += result.passed;
            std::cout << result.ok;
            std::cerr << result.rejected;
        };
        for (;;) {
            for (const auto& line : group.rejected)
                std::cerr << line;
            const auto uploadStart = Clock::now();
            if (!group.plans.empty())
                renderer.uploadSamples(bank);
            preparation += group.seconds + elapsed(uploadStart);
            // Only the planner can grow the host bank, and the CUDA thread already uploaded this group's samples.
            if (group.next < options.midi.size())
                next = prefetch(group.next);
            for (size_t first = 0; first < group.plans.size();) {
                if (writers.size() >= 3)
                    drain();
                auto batch = renderer.render(group.plans, first);
                kernelSeconds += batch.kernelSeconds;
                const auto count = batch.results.size();
                writers.push_back(std::async(std::launch::async,
                    [batch = std::move(batch), outputs = group.outputs, first, rate = options.rate, channels] {
                        return writeBatch(batch, *outputs, first, rate, channels);
                    }));
                first += count;
            }
            if (!next.valid())
                break;
            group = next.get();
        }
        while (!writers.empty())
            drain();
        std::cout << "PASS\t" << pass << '\t' << elapsed(passStart) << '\t' << passed << '\t'
                  << renderer.peakBytes() << '\t' << preparation << '\t' << kernelSeconds << '\n';
    }
    return passed == options.midi.size() ? 0 : 2;
}

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    try {
        if (argc == 2 && Path(argv[1]) == "--help") {
            std::cout << "Usage: claysampler --sfz preset.sfz --output directory [options] file.mid ...\n"
                         "Options: --samplerate Hz --device index --passes count --nvrtc path --sndfile path\n"
                         "         --tail-policy natural|orig|custom (default natural)\n"
                         "         --max-audio-seconds seconds (natural)\n"
                         "         --tail-max-seconds seconds [--tail-fade-ms milliseconds] (custom)\n"
                         "       claysampler --check-sfz [--sndfile path] preset.sfz ...\n";
            return 0;
        }
        if (argc > 1 && Path(argv[1]) == "--check-sfz")
            return checkPresets(argc, argv);
        return run(arguments(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ERROR\t" << error.what() << '\n';
        return 1;
    }
}
