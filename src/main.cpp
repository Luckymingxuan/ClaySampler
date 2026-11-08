#include "cuda/renderer.h"

#include <chrono>
#include <cstdlib>
#include <cwctype>
#include <iostream>
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
        else throw std::runtime_error("unknown option " + name);
    }
    if (options.sfz.empty() || options.output.empty() || options.midi.empty())
        throw std::runtime_error("provide --sfz, --output, and at least one MIDI path");
    return options;
}

int run(const Options& options)
{
    auto start = Clock::now();
    clay::AudioBank bank(options.sndfile);
    auto instrument = clay::loadInstrument(options.sfz, bank);
    clay::Renderer renderer(options.device, options.nvrtc);
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
    for (int pass = 1; pass <= options.passes; ++pass) {
        auto passStart = Clock::now();
        std::vector<clay::RenderPlan> plans;
        std::vector<Path> outputs;
        for (const auto& path : options.midi) {
            try {
                auto midi = clay::loadMidi(path, options.rate);
                plans.push_back(clay::planRender(instrument, bank, midi, options.rate, 1));
                auto name = path.stem();
                name += ".wav";
                outputs.push_back(options.output / name);
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
                auto frames = plans[i].frames;
                try {
                    auto qc = clay::writeWave(outputs[i], batch.pcm.get() + offset, frames, options.rate);
                    std::cout << "OK\t" << outputs[i].filename().u8string() << '\t' << qc.frames
                              << '\t' << qc.peak << '\t' << qc.first << '\t' << qc.last << '\n';
                    ++passed;
                } catch (const std::exception& error) {
                    std::cerr << "REJECT\t" << outputs[i].u8string() << '\t' << error.what() << '\n';
                }
                offset += size_t(frames) * 2;
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
                         "Options: --samplerate Hz --device index --passes count --nvrtc path --sndfile path\n";
            return 0;
        }
        return run(arguments(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ERROR\t" << error.what() << '\n';
        return 1;
    }
}
