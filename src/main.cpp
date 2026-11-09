#include "cuda/renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cwctype>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
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
    Path sfz, output, jobs;
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
    std::string quality = "off";
    std::map<std::string, Path> thresholds;
    bool groupVolume = false;
    std::optional<int> groupMinimum, waitingMiB;
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

double number(const Path& input, double low, double high)
{
    auto value = input.u8string();
    size_t used = 0;
    const auto number = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(number) || number < low || number > high)
        throw std::runtime_error("invalid numeric argument " + value);
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
        else if (name == "--jobs") options.jobs = value;
        else if (name == "--nvrtc") options.nvrtc = value;
        else if (name == "--sndfile") options.sndfile = value;
        else if (name == "--device") options.device = integer(value, 0, 1024);
        else if (name == "--samplerate") options.rate = integer(value, 8000, 192000);
        else if (name == "--passes") options.passes = integer(value, 1, 1000);
        else if (name == "--tail-policy") options.tail = tailPolicy(value);
        else if (name == "--tail-max-seconds") options.tailMax = number(value, 0, 3600);
        else if (name == "--tail-fade-ms") options.tailFadeMs = integer(value, 0, 3600000);
        else if (name == "--max-audio-seconds") options.maxAudio = number(value, 0, 3600);
        else if (name == "--quality-gate") options.quality = value.u8string();
        else if (name.rfind("--qa-", 0) == 0) {
            if (!options.thresholds.emplace(name, value).second)
                throw std::runtime_error("duplicate option " + name);
        } else if (name == "--group-volume") {
            if (value != "off" && value != "mean")
                throw std::runtime_error("expected --group-volume off or mean");
            options.groupVolume = value == "mean";
        } else if (name == "--group-min-members") options.groupMinimum = integer(value, 2, 100000);
        else if (name == "--group-wait-mib") options.waitingMiB = integer(value, 0, 16384);
        else throw std::runtime_error("unknown option " + name);
    }
    if (options.jobs.empty() && (options.sfz.empty() || options.output.empty() || options.midi.empty()))
        throw std::runtime_error("provide --sfz, --output, and at least one MIDI path");
    if (!options.jobs.empty() && (!options.sfz.empty() || !options.output.empty() || !options.midi.empty()))
        throw std::runtime_error("--jobs supplies the SFZ, MIDI and output paths");
    if (options.quality != "off" && options.quality != "clayengine" && options.quality != "custom")
        throw std::runtime_error("expected --quality-gate off, clayengine, or custom");
    if (options.quality != "custom" && !options.thresholds.empty())
        throw std::runtime_error("quality thresholds require --quality-gate custom");
    if (options.groupVolume && options.jobs.empty())
        throw std::runtime_error("--group-volume mean requires --jobs");
    if (!options.groupVolume && (options.groupMinimum || options.waitingMiB))
        throw std::runtime_error("group options require --group-volume mean");
    if (options.tail == clay::TailPolicy::custom && !options.tailMax)
        throw std::runtime_error("custom tail requires --tail-max-seconds");
    if (options.tail != clay::TailPolicy::custom && (options.tailMax || options.tailFadeMs))
        throw std::runtime_error("tail length and fade options require --tail-policy custom");
    if (options.maxAudio && options.tail != clay::TailPolicy::natural)
        throw std::runtime_error("--max-audio-seconds requires --tail-policy natural");
    return options;
}

AudioConfig audioConfig(const Options& options)
{
    const auto rate = options.rate;
    const auto power = [](double dbfs) { return 4294967296. * std::pow(10., dbfs / 10); };
    AudioConfig config{rate, int(std::nearbyint(rate / 20.)), 375 * rate, 32, options.quality != "off",
        -38, -45, power(-70), .75, 10. * rate, .0001};
    for (const auto& [name, value] : options.thresholds) {
        if (name == "--qa-max-seconds") config.maxFrames = int(std::floor(number(value, 0, 3600) * rate));
        else if (name == "--qa-min-peak") config.minPeak = integer(value, 0, 32768);
        else if (name == "--qa-min-onset-median-dbfs") config.minMedian = number(value, -120, 0);
        else if (name == "--qa-min-onset-quartile-dbfs") config.minQuartile = number(value, -120, 0);
        else if (name == "--qa-silence-dbfs") config.silencePower = power(number(value, -120, 0));
        else if (name == "--qa-max-silence-fraction") config.maxSilentFraction = number(value, 0, 1);
        else if (name == "--qa-max-silence-seconds") config.maxSilentFrames = number(value, 0, 3600) * rate;
        else if (name == "--qa-max-clipped-fraction") config.maxClippedFraction = number(value, 0, 1);
        else throw std::runtime_error("unknown option " + name);
    }
    return config;
}

struct Task {
    Path sfz, midi, output;
    size_t group = 0;
};

std::vector<Task> tasksFor(const Options& options)
{
    std::vector<Task> tasks;
    if (options.jobs.empty()) {
        for (const auto& path : options.midi) {
            auto name = path.stem();
            name += ".wav";
            tasks.push_back({options.sfz, path, options.output / name});
        }
    } else {
        std::ifstream file(options.jobs, std::ios::binary);
        if (!file)
            throw std::runtime_error("cannot open job manifest " + options.jobs.u8string());
        const auto root = std::filesystem::absolute(options.jobs).parent_path();
        std::map<std::string, size_t> groupIndexes;
        std::vector<std::vector<Task>> schedule;
        size_t lineNumber = 0;
        for (std::string line; std::getline(file, line);) {
            if (++lineNumber == 1 && line.compare(0, 3, "\xef\xbb\xbf") == 0)
                line.erase(0, 3);
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.empty())
                continue;
            std::array<std::string, 4> columns;
            std::istringstream row(line);
            for (auto& column : columns) {
                if (!std::getline(row, column, '\t') || column.empty())
                    throw std::runtime_error("expected group, SFZ, MIDI and WAV on manifest line " + std::to_string(lineNumber));
            }
            if (!row.eof() || line.find('\0') != std::string::npos)
                throw std::runtime_error("invalid job manifest line " + std::to_string(lineNumber));
            Task task{root / std::filesystem::u8path(columns[1]), root / std::filesystem::u8path(columns[2]),
                      root / std::filesystem::u8path(columns[3])};
            if (task.output.extension() != ".wav")
                throw std::runtime_error("expected a .wav output on manifest line " + std::to_string(lineNumber));
            if (options.groupVolume && columns[0] != "-") {
                const auto [group, inserted] = groupIndexes.emplace(columns[0], schedule.size());
                if (inserted)
                    schedule.emplace_back();
                task.group = group->second + 1;
                schedule[group->second].push_back(std::move(task));
            } else {
                schedule.push_back({std::move(task)});
            }
        }
        if (!file.eof())
            throw std::runtime_error("cannot read job manifest " + options.jobs.u8string());
        for (auto& group : schedule)
            for (auto& task : group)
                tasks.push_back(std::move(task));
    }
    if (tasks.empty())
        throw std::runtime_error("job manifest is empty");
    std::set<Path> destinations;
    for (auto& task : tasks) {
        const auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(task.output));
#ifdef _WIN32
        auto normalized = path.wstring();
        for (auto& ch : normalized)
            ch = static_cast<wchar_t>(std::towlower(ch));
#else
        const auto& normalized = path;
#endif
        if (!destinations.insert(normalized).second)
            throw std::runtime_error("duplicate output path " + task.output.u8string());
    }
    return tasks;
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
    bool freshSamples = false;
};

PlannedBatch planBatch(const Options& options, const std::vector<Task>& tasks,
                       std::map<Path, clay::Instrument>& instruments, clay::AudioBank& bank, size_t first,
                       size_t sampleBudget, bool freshSamples)
{
    const auto start = Clock::now();
    const bool natural = options.tail == clay::TailPolicy::natural;
    const int customFade = static_cast<int>(std::lround(options.tailFadeMs.value_or(100) * (options.rate / 1000.)));
    PlannedBatch batch;
    batch.freshSamples = freshSamples;
    if (freshSamples)
        bank.releaseDecoded();
    const auto count = std::min(size_t{64}, tasks.size() - first);
    batch.plans.reserve(count);
    batch.outputs->reserve(count);
    size_t frames = 0;
    batch.next = first;
    Path parsedPath;
    clay::Midi midi;
    // Start work promptly, but do not leave a partially planned group behind a new group.
    for (; batch.next < tasks.size(); ++batch.next) {
        const auto& task = tasks[batch.next];
        const bool continuesGroup = task.group && batch.next > first && tasks[batch.next - 1].group == task.group;
        if (!continuesGroup && (batch.next - first >= count || frames >= 24 * size_t{1024} * 1024 ||
            (batch.next > first && bank.pcm.size() >= sampleBudget / sizeof(float))))
            break;
        try {
            auto instrument = instruments.find(task.sfz);
            if (instrument == instruments.end())
                instrument = instruments.emplace(task.sfz, clay::loadInstrument(task.sfz, bank)).first;
            if (task.midi != parsedPath) {
                midi = clay::loadMidi(task.midi, options.rate, options.tail, options.groupVolume || options.quality != "off");
                parsedPath = task.midi;
            }
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
            auto plan = clay::planRender(instrument->second, bank, midi, options.rate, 1, limit, options.tail);
            plan.musicEnd = musicEnd;
            plan.limit = limit;
            if (options.tail == clay::TailPolicy::custom && plan.tailClipped && plan.frames == limit)
                plan.fadeFrames = std::min(customFade, std::max(0, plan.frames - musicEnd));
            plan.volumeGroup = task.group;
            plan.bodyFrames = midi.bodyFrames;
            if (options.quality != "off")
                plan.onsets = midi.onsets;
            batch.outputs->push_back(task.output);
            frames += plan.frames;
            batch.plans.push_back(std::move(plan));
        } catch (const std::exception& error) {
            const auto& label = options.jobs.empty() ? task.midi : task.output;
            batch.rejected.push_back("REJECT\t" + label.u8string() + '\t' + error.what() + '\n');
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
    case TailStatus::audioDuration: return "rendered_audio_too_long";
    case TailStatus::quietPerformance: return "quiet_performance";
    case TailStatus::excessiveSilence: return "excessive_audio_silence";
    case TailStatus::silentAudio: return "effectively_silent_audio";
    case TailStatus::clipping: return "excessive_audio_clipping";
    case TailStatus::silentBody: return "silent_melody_body";
    case TailStatus::incompleteGroup: return "insufficient_qualified_group_members";
    case TailStatus::ok: return nullptr;
    }
    return "unknown CUDA tail status";
}

struct WrittenBatch {
    size_t passed = 0;
    std::string ok, rejected;
};

WrittenBatch writeBatch(const clay::RenderBatch& batch, const std::vector<Path>& outputs, size_t first,
                        int rate, int channels, bool manifest)
{
    std::ostringstream ok, rejected;
    size_t passed = 0;
    for (size_t i = 0; i < batch.results.size(); ++i) {
        const auto& result = batch.results[i];
        const auto& path = outputs[first + i];
        const auto& stats = result.audio;
        const auto* failure = tailError(result.status);
        if (stats.measured) {
            const auto* verdict = tailError(stats.verdict);
            ok << "QUALITY\t" << path.u8string() << '\t' << (verdict ? verdict : "accepted") << '\t' << result.frames
               << '\t' << stats.median << '\t' << stats.quartile << '\t' << stats.silentFraction << '\t'
               << double(stats.longestSilence) / rate << '\t' << stats.clippedFraction << '\n';
        }
        try {
            if (failure)
                throw std::runtime_error(failure);
            if (result.last < 0 && stats.gain <= 0)
                throw std::runtime_error("rendered audio is silent " + path.u8string());
            if (manifest)
                std::filesystem::create_directories(path.parent_path());
            clay::writeWave(path, batch.pcm.get() + result.offset * channels, result.dataFrames, result.frames, rate, channels);
            if (stats.gain > 0)
                ok << "VOLUME\t" << path.u8string() << '\t' << stats.bodyRms << '\t' << stats.targetRms << '\t'
                   << 20 * std::log10(stats.gain) << '\n';
            ok << "OK\t" << (manifest ? path : path.filename()).u8string() << '\t' << result.frames
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
    const bool manifest = !options.jobs.empty();
    const int channels = natural ? 1 : 2;
    const auto audio = audioConfig(options);
    const auto tasks = tasksFor(options);
    auto start = Clock::now();
    clay::AudioBank bank(options.sndfile);
    std::map<Path, clay::Instrument> instruments;
    const auto plan = [&](size_t first, size_t sampleBudget, bool fresh) {
        if (!manifest && instruments.empty())
            instruments.emplace(options.sfz, clay::loadInstrument(options.sfz, bank));
        return planBatch(options, tasks, instruments, bank, first, sampleBudget, fresh);
    };
    const auto prefetch = [&](size_t next, size_t sampleBudget, bool fresh) {
        return std::async(std::launch::async, plan, next, sampleBudget, fresh);
    };
    const auto unlimited = std::numeric_limits<size_t>::max();
    auto opening = prefetch(0, manifest ? 512 * size_t{1024} * 1024 : unlimited, false);
    clay::Renderer renderer(options.device, options.nvrtc, natural, audio,
        options.groupVolume ? options.groupMinimum.value_or(2) : 0, size_t(options.waitingMiB.value_or(256)) * 1024 * 1024);
    auto startup = opening.get();
    const auto sampleBudget = manifest ? renderer.sampleCacheBudget() : unlimited;
    const auto rotateSamples = [&] { return manifest && bank.pcm.size() >= sampleBudget / sizeof(float); };
    std::cout << "DEVICE\t" << renderer.device() << '\n'
              << "INIT\t" << elapsed(start) << '\n';
    if (!manifest)
        std::filesystem::create_directories(options.output);
    size_t passed = 0;
    for (int pass = 1; pass <= options.passes; ++pass) {
        auto passStart = Clock::now();
        double preparation = 0, kernelSeconds = 0;
        passed = 0;
        const auto spilledBefore = renderer.spilledBytes(), downloadedBefore = renderer.downloadedBytes();
        auto group = pass == 1 ? std::move(startup) : plan(0, sampleBudget, rotateSamples());
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
            if (!group.plans.empty() || group.freshSamples)
                renderer.uploadSamples(bank, group.freshSamples);
            preparation += group.seconds + elapsed(uploadStart);
            // Only the planner can grow the host bank, and the CUDA thread already uploaded this group's samples.
            if (group.next < tasks.size())
                next = prefetch(group.next, sampleBudget, rotateSamples());
            renderer.renderWindow(group.plans, [&](clay::RenderBatch batch, size_t first) {
                if (writers.size() >= 3)
                    drain();
                kernelSeconds += batch.kernelSeconds;
                writers.push_back(std::async(std::launch::async,
                    [batch = std::move(batch), outputs = group.outputs, first, rate = options.rate, channels, manifest] {
                        return writeBatch(batch, *outputs, first, rate, channels, manifest);
                    }));
            });
            if (!next.valid())
                break;
            group = next.get();
        }
        while (!writers.empty())
            drain();
        std::cout << "PASS\t" << pass << '\t' << elapsed(passStart) << '\t' << passed << '\t'
                  << renderer.peakBytes() << '\t' << preparation << '\t' << kernelSeconds << '\n';
        if (audio.enabled || options.groupVolume)
            std::cout << "TRANSFER\t" << pass << '\t' << renderer.downloadedBytes() - downloadedBefore << '\t'
                      << renderer.spilledBytes() - spilledBefore << '\t' << renderer.peakWaitingBytes() << '\n';
    }
    return passed == tasks.size() ? 0 : 2;
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
                         "       claysampler --jobs tasks.tsv [options]\n"
                         "Options: --samplerate Hz --device index --passes count --nvrtc path --sndfile path\n"
                         "         --tail-policy natural|orig|custom (default natural)\n"
                         "         --max-audio-seconds seconds (natural)\n"
                         "         --tail-max-seconds seconds [--tail-fade-ms milliseconds] (custom)\n"
                         "         --quality-gate off|clayengine|custom (default off)\n"
                         "         --group-volume off|mean (default off; mean uses groups in --jobs)\n"
                         "         --group-min-members count --group-wait-mib MiB\n"
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
