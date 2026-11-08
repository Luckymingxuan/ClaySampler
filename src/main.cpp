#include "cuda/renderer.h"
#include "render/tail.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cwctype>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

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

int run(const Options& options)
{
    const bool natural = options.tail == clay::TailPolicy::natural;
    const int customFade = static_cast<int>(std::lround(options.tailFadeMs.value_or(100) * (options.rate / 1000.)));
    auto start = Clock::now();
    clay::AudioBank bank(options.sndfile);
    auto instrument = clay::loadInstrument(options.sfz, bank);
    clay::Renderer renderer(options.device, options.nvrtc, natural);
    const auto initialization = elapsed(start);
    std::cout << "DEVICE\t" << renderer.device() << '\n'
              << "INIT\t" << initialization << '\n';
    std::filesystem::create_directories(options.output);
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
    size_t passed = 0;
    struct Output {
        Path path;
        int musicEnd, limit;
    };
    for (int pass = 1; pass <= options.passes; ++pass) {
        auto passStart = Clock::now();
        std::vector<clay::RenderPlan> plans;
        std::vector<Output> outputs;
        for (const auto& path : options.midi) {
            try {
                auto midi = clay::loadMidi(path, options.rate, options.tail);
                const int musicEnd = midi.end + 1;
                int limit = std::numeric_limits<int>::max();
                if (natural) {
                    limit = musicEnd + clay::naturalTailLimitSeconds * options.rate;
                    if (options.maxAudio)
                        limit = std::min(limit, static_cast<int>(std::floor(*options.maxAudio * options.rate)));
                    if (limit < musicEnd)
                        throw std::runtime_error("natural render exceeds the audio duration limit");
                } else if (options.tailMax) {
                    limit = musicEnd + static_cast<int>(std::floor(*options.tailMax * options.rate));
                }
                auto plan = clay::planRender(instrument, bank, midi, options.rate, 1, limit, options.tail);
                auto name = path.stem();
                name += ".wav";
                outputs.push_back({options.output / name, musicEnd, limit});
                plans.push_back(std::move(plan));
            } catch (const std::exception& error) {
                std::cerr << "REJECT\t" << path.u8string() << '\t' << error.what() << '\n';
            }
        }
        if (pass == 1 && !plans.empty())
            renderer.uploadSamples(bank);
        const auto preparation = elapsed(passStart);
        double kernelSeconds = 0;
        passed = 0;
        for (size_t first = 0; first < plans.size();) {
            auto batch = renderer.render(plans, first);
            kernelSeconds += batch.kernelSeconds;
            size_t offset = 0;
            for (size_t i = first; i < first + batch.count; ++i) {
                const auto& plan = plans[i];
                const auto& output = outputs[i];
                try {
                    clay::Quality qc;
                    if (natural) {
                        qc = clay::writeNaturalTail(output.path, batch.stereo.get() + offset, plan.frames,
                            output.musicEnd, output.limit, options.rate);
                    } else {
                        auto* pcm = batch.pcm.get() + offset;
                        if (options.tail == clay::TailPolicy::custom && plan.tailClipped && plan.frames == output.limit)
                            clay::fadeTail(pcm, plan.frames, 2,
                                std::min(customFade, std::max(0, plan.frames - output.musicEnd)));
                        qc = clay::writeWave(output.path, pcm, plan.frames, options.rate);
                    }
                    std::cout << "OK\t" << output.path.filename().u8string() << '\t' << qc.frames
                              << '\t' << qc.peak << '\t' << qc.first << '\t' << qc.last << '\n';
                    ++passed;
                } catch (const std::exception& error) {
                    std::cerr << "REJECT\t" << output.path.u8string() << '\t' << error.what() << '\n';
                }
                offset += size_t(plan.frames) * 2;
            }
            first += batch.count;
        }
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
