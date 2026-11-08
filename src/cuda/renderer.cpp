#include "renderer.h"
#include "../io/library.h"
#include "kernel_source.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

namespace clay {
namespace {

using DevicePtr = unsigned long long;
using Handle = void*;
using Path = std::filesystem::path;

#ifdef _WIN32
constexpr auto driverName = "nvcuda.dll";
#else
constexpr auto driverName = "libcuda.so.1";
#endif

struct Driver {
    Library lib{driverName};
    int (*init)(unsigned) = lib.get<int(*)(unsigned)>("cuInit");
    int (*device)(int*, int) = lib.get<int(*)(int*, int)>("cuDeviceGet");
    int (*name)(char*, int, int) = lib.get<int(*)(char*, int, int)>("cuDeviceGetName");
    int (*attribute)(int*, int, int) = lib.get<int(*)(int*, int, int)>("cuDeviceGetAttribute");
    int (*create)(Handle*, unsigned, int) = lib.get<int(*)(Handle*, unsigned, int)>("cuCtxCreate_v2");
    int (*destroy)(Handle) = lib.get<int(*)(Handle)>("cuCtxDestroy_v2");
    int (*load)(Handle*, const void*) = lib.get<int(*)(Handle*, const void*)>("cuModuleLoadData");
    int (*kernel)(Handle*, Handle, const char*) = lib.get<int(*)(Handle*, Handle, const char*)>("cuModuleGetFunction");
    int (*memory)(size_t*, size_t*) = lib.get<int(*)(size_t*, size_t*)>("cuMemGetInfo_v2");
    int (*allocate)(DevicePtr*, size_t) = lib.get<int(*)(DevicePtr*, size_t)>("cuMemAlloc_v2");
    int (*free)(DevicePtr) = lib.get<int(*)(DevicePtr)>("cuMemFree_v2");
    int (*put)(DevicePtr, const void*, size_t) = lib.get<int(*)(DevicePtr, const void*, size_t)>("cuMemcpyHtoD_v2");
    int (*get)(void*, DevicePtr, size_t) = lib.get<int(*)(void*, DevicePtr, size_t)>("cuMemcpyDtoH_v2");
    int (*copy)(DevicePtr, DevicePtr, size_t) = lib.get<int(*)(DevicePtr, DevicePtr, size_t)>("cuMemcpyDtoD_v2");
    int (*allocHost)(void**, size_t) = lib.get<int(*)(void**, size_t)>("cuMemAllocHost_v2");
    int (*freeHost)(void*) = lib.get<int(*)(void*)>("cuMemFreeHost");
    using Occupancy = int(*)(int*, int*, Handle, size_t(*)(int), size_t, int);
    Occupancy occupancy = lib.get<Occupancy>("cuOccupancyMaxPotentialBlockSize");
    using Launch = int(*)(Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                          unsigned, Handle, void**, void**);
    Launch launch = lib.get<Launch>("cuLaunchKernel");
    int (*newEvent)(Handle*, unsigned) = lib.get<int(*)(Handle*, unsigned)>("cuEventCreate");
    int (*dropEvent)(Handle) = lib.get<int(*)(Handle)>("cuEventDestroy_v2");
    int (*recordEvent)(Handle, Handle) = lib.get<int(*)(Handle, Handle)>("cuEventRecord");
    int (*elapsedEvents)(float*, Handle, Handle) = lib.get<int(*)(float*, Handle, Handle)>("cuEventElapsedTime");
    int (*message)(int, const char**) = lib.get<int(*)(int, const char**)>("cuGetErrorString");

    void check(int result, const char* operation) const
    {
        if (result) {
            const char* details = nullptr;
            message(result, &details);
            throw std::runtime_error(std::string(operation) + ": " +
                (details ? details : std::to_string(result)));
        }
    }
};

class TimingEvent {
public:
    explicit TimingEvent(Driver& driver) : driver_(driver)
    {
        driver_.check(driver_.newEvent(&address, 0), "CUDA timing event");
    }
    ~TimingEvent() { driver_.dropEvent(address); }
    TimingEvent(const TimingEvent&) = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;

    void record() const
    {
        driver_.check(driver_.recordEvent(address, nullptr), "CUDA timing record");
    }

    Handle address{};

private:
    Driver& driver_;
};

class Buffer {
public:
    explicit Buffer(Driver& driver) : driver_(driver) {}
    ~Buffer() { reset(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void allocate(size_t bytes)
    {
        bytes = std::max(bytes, size_t{1});
        if (bytes <= capacity)
            return;
        reset();
        driver_.check(driver_.allocate(&address, bytes), "CUDA allocation");
        capacity = bytes;
    }

    void reset()
    {
        if (address)
            driver_.free(address);
        address = 0;
        capacity = 0;
    }

    template<class T> void upload(const std::vector<T>& data)
    {
        auto bytes = data.size() * sizeof(T);
        allocate(bytes);
        if (bytes)
            driver_.check(driver_.put(address, data.data(), bytes), "CUDA upload");
    }

    DevicePtr address{};
    size_t capacity{};

private:
    Driver& driver_;
};

struct HostPcm {
    Driver& driver;
    void* address{};
    size_t capacity{};

    explicit HostPcm(Driver& owner) : driver(owner) {}
    ~HostPcm() { if (address) driver.freeHost(address); }
    bool reserve(size_t bytes)
    {
        if (bytes <= capacity)
            return true;
        if (address)
            driver.freeHost(address);
        address = nullptr;
        capacity = 0;
        void* allocation{};
        if (driver.allocHost(&allocation, bytes))
            return false;
        address = allocation;
        capacity = bytes;
        return true;
    }
};

struct ArenaLayout {
    size_t bytes = 0;

    size_t section(size_t size)
    {
        constexpr size_t alignment = 16;
        const auto offset = (bytes + alignment - 1) & ~(alignment - 1);
        bytes = offset + size;
        return offset;
    }
};

size_t blocksFor(size_t frames, size_t block) { return (frames + block - 1) / block; }

std::vector<char> compile(const Path& library, int architecture, bool floatingPoint)
{
    Library rtc(library);
    const auto version = rtc.get<int(*)(int*, int*)>("nvrtcVersion");
    int major{}, minor{};
    const bool hasVersion = version(&major, &minor) == 0;
#ifdef _WIN32
    std::unique_ptr<Library> builtins;
    if (library.has_parent_path() && hasVersion) {
        auto filename = "nvrtc-builtins64_" + std::to_string(major) + std::to_string(minor) + ".dll";
        auto path = std::filesystem::absolute(library).parent_path() / filename;
        if (std::filesystem::exists(path))
            builtins = std::make_unique<Library>(path);
    }
#endif
    auto create = rtc.get<int(*)(Handle*, const char*, const char*, int, const char* const*, const char* const*)>("nvrtcCreateProgram");
    auto destroy = rtc.get<int(*)(Handle*)>("nvrtcDestroyProgram");
    auto run = rtc.get<int(*)(Handle, int, const char* const*)>("nvrtcCompileProgram");
    auto logSize = rtc.get<int(*)(Handle, size_t*)>("nvrtcGetProgramLogSize");
    auto log = rtc.get<int(*)(Handle, char*)>("nvrtcGetProgramLog");
    auto archCount = rtc.get<int(*)(int*)>("nvrtcGetNumSupportedArchs");
    auto architectures = rtc.get<int(*)(int*)>("nvrtcGetSupportedArchs");
    auto error = rtc.get<const char*(*)(int)>("nvrtcGetErrorString");
    auto check = [&](int result) {
        if (result)
            throw std::runtime_error(error(result));
    };
    int count = 0;
    check(archCount(&count));
    std::vector<int> supported(count);
    check(architectures(supported.data()));
    auto found = std::upper_bound(supported.begin(), supported.end(), architecture);
    if (found == supported.begin())
        throw std::runtime_error("GPU architecture is older than the installed NVRTC supports");
    auto target = *std::prev(found);
    const bool native = target == architecture;
    auto sizeOf = rtc.get<int(*)(Handle, size_t*)>(native ? "nvrtcGetCUBINSize" : "nvrtcGetPTXSize");
    auto read = rtc.get<int(*)(Handle, char*)>(native ? "nvrtcGetCUBIN" : "nvrtcGetPTX");

    const char* header = kernelTypes;
    const char* headerName = "render_types.h";
    Handle program = nullptr;
    check(create(&program, kernelSource, "render.cu", 1, &header, &headerName));
    struct Program {
        Handle& handle;
        int (*destroy)(Handle*);
        ~Program() { destroy(&handle); }
    } scope{program, destroy};
    auto archFlag = std::string(native ? "--gpu-architecture=sm_" : "--gpu-architecture=compute_") +
        std::to_string(target);
    const bool parallel = hasVersion && (major > 12 || (major == 12 && minor >= 4)) &&
        std::thread::hardware_concurrency() >= 4;
    const char* flags[] = {archFlag.c_str(), "--std=c++17", floatingPoint ? "-DCLAY_FLOAT_OUTPUT=1" : "-DCLAY_FLOAT_OUTPUT=0",
        "--split-compile=4"};
    if (run(program, parallel ? 4 : 3, flags)) {
        size_t bytes{};
        check(logSize(program, &bytes));
        std::vector<char> diagnostics(bytes);
        check(log(program, diagnostics.data()));
        throw std::runtime_error("CUDA compilation: " + std::string(diagnostics.data()));
    }
    size_t bytes{};
    check(sizeOf(program, &bytes));
    std::vector<char> result(bytes);
    check(read(program, result.data()));
    return result;
}

std::vector<float> sincTable()
{
    constexpr int phases = 2048;
    constexpr int taps = 48;
    constexpr double beta = 8.5;
    constexpr double pi = 3.14159265358979323846;
    const auto bessel = [](double value) {
        double sum = 1;
        double term = 1;
        for (int i = 1; i < 30; ++i) {
            term *= value * value / (4 * i * i);
            sum += term;
            if (term < 1e-14)
                break;
        }
        return sum;
    };
    const auto normalization = bessel(beta);
    std::vector<float> table(taps * (phases + 1));
    for (int tap = 0; tap < taps; ++tap) {
        for (int phase = 0; phase <= phases; ++phase) {
            auto x = tap - 23 - double(phase) / phases;
            auto f = x / 24;
            auto window = bessel(beta * std::sqrt(1 - f * f)) / normalization;
            table[tap * (phases + 1) + phase] = static_cast<float>(
                window * (x == 0 ? 1 : std::sin(pi * x) / (pi * x)));
        }
    }
    return table;
}

struct BatchIndex {
    std::vector<VoiceSpan> voices;
    std::vector<Tile> tiles;
    std::vector<int> indices;
    std::vector<ControlPoint> controls;
    std::vector<VoiceModulation> modulation;
    std::vector<FilterTile> filterTiles;
    std::vector<int> filterHeads;
    size_t frames = 0;
    size_t filteredFrames = 0;
    size_t pitchFrames = 0;

    void append(const RenderPlan& plan, int tileFrames)
    {
        const auto count = (plan.frames + tileFrames - 1) / tileFrames;
        std::vector<int> offsets(count + 1);
        for (const auto& voice : plan.spans) {
            ++offsets[voice.begin / tileFrames];
            --offsets[(voice.end - 1) / tileFrames + 1];
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        std::vector<int> controlOffsets(plan.controls.size() + 1);
        controlOffsets[0] = static_cast<int>(controls.size());
        for (size_t i = 0; i < plan.controls.size(); ++i) {
            const auto& track = plan.controls[i];
            controls.insert(controls.end(), track.begin(), track.end());
            controlOffsets[i + 1] = static_cast<int>(controls.size());
        }
        const auto modulationOffset = static_cast<int>(modulation.size());
        for (auto binding : plan.modulation) {
            if (binding.first >= 0) {
                binding.last = controlOffsets[binding.first + 1];
                binding.first = controlOffsets[binding.first];
            }
            if (binding.shapeFirst >= 0) {
                binding.shapeLast = controlOffsets[binding.shapeFirst + 1];
                binding.shapeFirst = controlOffsets[binding.shapeFirst];
            }
            modulation.push_back(binding);
        }
        int indexCount = static_cast<int>(indices.size());
        for (int t = 0; t < count; ++t) {
            auto frame = t * tileFrames;
            Tile tile{frames + frame, frame, std::min(tileFrames, plan.frames - frame), indexCount, offsets[t], {}};
            std::copy_n(controlOffsets.begin(), 4, tile.controls);
            tiles.push_back(tile);
            indexCount += offsets[t];
            offsets[t] = tile.first;
        }
        indices.resize(indexCount);
        std::vector<int> tails(plan.filterTracks, -1);
        for (auto voice : plan.spans) {
            const auto index = static_cast<int>(voices.size());
            for (int t = voice.begin / tileFrames; t <= (voice.end - 1) / tileFrames; ++t)
                indices[offsets[t]++] = index;
            voice.firstModulation += modulationOffset;
            if (voice.filterTrack >= 0) {
                auto& tail = tails[voice.filterTrack];
                if (tail < 0)
                    filterHeads.push_back(index);
                else
                    voices[tail].nextFilteredSpan = index;
                tail = index;
                voice.nextFilteredSpan = -1;
                voice.filteredOffset = filteredFrames;
                filteredFrames += voice.end - voice.begin;
                if (voice.variablePitch) {
                    voice.phaseOffset = pitchFrames;
                    pitchFrames += voice.end - voice.begin;
                }
                std::copy_n(controlOffsets.begin(), 4, voice.filterControls);
                for (int frame = voice.begin; frame < voice.end; frame += tileFrames)
                    filterTiles.push_back({index, frame});
            }
            voices.push_back(voice);
        }
        frames += plan.frames;
    }

};

} // namespace

struct Renderer::Impl {
    Driver driver;
    struct Context {
        Driver& driver;
        Handle address{};
        ~Context() { if (address) driver.destroy(address); }
    } context{driver};
    Buffer samples{driver}, sinc{driver}, arena{driver};
    std::vector<std::shared_ptr<HostPcm>> hostPcm;
    Handle module{}, kernel{}, filterSource{}, filterPass{}, pitchPass{};
    Handle stereoScan{}, nativeEnd{}, monoQuantize{}, trimEnd{}, pcmLayout{}, pcmFinalize{};
    std::string deviceName;
    int threads{};
    bool natural;
    TailConfig tail;
    size_t uploaded{}, peak{};

    Impl(int ordinal, const Path& rtc, bool useNatural, int rate)
        : natural(useNatural), tail{int(std::nearbyint(rate / 20.)), 2 * rate,
            int(std::nearbyint(rate / 4.)), int(std::nearbyint(rate / 10.)), std::pow(10., -1.5),
            32768 * std::pow(10., -3.9), 32768 * std::pow(10., -3.)}
    {
        driver.check(driver.init(0), "CUDA initialization");
        int device{};
        driver.check(driver.device(&device, ordinal), "CUDA device");
        char name[256]{};
        driver.check(driver.name(name, sizeof name, device), "CUDA device name");
        deviceName = name;
        int major{}, minor{};
        driver.check(driver.attribute(&major, 75, device), "CUDA architecture major");
        driver.check(driver.attribute(&minor, 76, device), "CUDA architecture minor");
        driver.check(driver.create(&context.address, 0, device), "CUDA context");
        auto code = compile(rtc, 10 * major + minor, natural);
        driver.check(driver.load(&module, code.data()), "CUDA module");
        driver.check(driver.kernel(&kernel, module, "renderTiles"), "CUDA render kernel");
        driver.check(driver.kernel(&filterSource, module, "renderFilterSources"), "CUDA filter source kernel");
        driver.check(driver.kernel(&filterPass, module, "applyVoiceFilters"), "CUDA filter kernel");
        driver.check(driver.kernel(&pitchPass, module, "prepareVoicePitch"), "CUDA variable pitch kernel");
        driver.check(driver.kernel(&pcmLayout, module, "layoutPcm"), "CUDA PCM layout kernel");
        driver.check(driver.kernel(&pcmFinalize, module, "finalizePcm"), "CUDA PCM finalization kernel");
        if (natural) {
            driver.check(driver.kernel(&stereoScan, module, "scanStereo"), "CUDA stereo statistics kernel");
            driver.check(driver.kernel(&nativeEnd, module, "selectNative"), "CUDA natural endpoint kernel");
            driver.check(driver.kernel(&monoQuantize, module, "quantizeMono"), "CUDA mono conversion kernel");
            driver.check(driver.kernel(&trimEnd, module, "selectTrim"), "CUDA trim endpoint kernel");
        }
        int minBlocks{};
        driver.check(driver.occupancy(&minBlocks, &threads, kernel, nullptr, 0, 256), "CUDA launch sizing");
        auto table = sincTable();
        sinc.upload(table);
        peak = sinc.capacity;
    }

    void uploadSamples(const AudioBank& bank)
    {
        const auto bytes = bank.pcm.size() * sizeof(float);
        if (bytes <= uploaded && samples.address)
            return;
        if (bytes > samples.capacity || !samples.address) {
            size_t free{}, total{};
            driver.check(driver.memory(&free, &total), "CUDA free memory");
            const size_t slack = std::min(bytes / 2, size_t{128} * 1024 * 1024);
            Buffer replacement(driver);
            DevicePtr candidate{};
            const auto capacity = std::max(bytes + slack, size_t{1});
            if (capacity <= free && !driver.allocate(&candidate, capacity)) {
                replacement.address = candidate;
                replacement.capacity = capacity;
                peak = std::max(peak, sinc.capacity + arena.capacity + samples.capacity + replacement.capacity);
                if (uploaded)
                    driver.check(driver.copy(replacement.address, samples.address, uploaded), "CUDA sample bank growth");
                std::swap(samples.address, replacement.address);
                std::swap(samples.capacity, replacement.capacity);
            } else {
                arena.reset();
                samples.allocate(bytes);
                uploaded = 0;
            }
        }
        if (bytes > uploaded)
            driver.check(driver.put(samples.address + uploaded, bank.pcm.data() + uploaded / sizeof(float),
                bytes - uploaded), "CUDA sample upload");
        uploaded = bytes;
        peak = std::max(peak, sinc.capacity + samples.capacity + arena.capacity);
    }

    std::shared_ptr<int16_t> receivePcm(DevicePtr source, size_t samplesCount)
    {
        if (!samplesCount)
            return {};
        const auto bytes = samplesCount * sizeof(int16_t);
        std::shared_ptr<int16_t> pcm;
        // Bound locked host memory and retain each slot until its WAV writer finishes.
        if (bytes <= 128 * size_t{1024} * 1024) {
            auto slot = std::find_if(hostPcm.begin(), hostPcm.end(), [](const auto& item) { return item.use_count() == 1; });
            if (slot == hostPcm.end() && hostPcm.size() < 3)
                slot = hostPcm.insert(slot, std::make_shared<HostPcm>(driver));
            if (slot != hostPcm.end() && (*slot)->reserve(bytes))
                pcm = std::shared_ptr<int16_t>(*slot, static_cast<int16_t*>((*slot)->address));
        }
        if (!pcm)
            pcm = std::shared_ptr<int16_t>(new int16_t[samplesCount], std::default_delete<int16_t[]>());
        driver.check(driver.get(pcm.get(), source, bytes), "CUDA PCM download");
        return pcm;
    }

    template<class... Args>
    void launchChunks(Handle function, int blocks, int width, Args&... args)
    {
        constexpr int launchBlocks = 16384;
        for (int offset = 0; offset < blocks; offset += launchBlocks) {
            void* arguments[] = {&args..., &offset};
            driver.check(driver.launch(function, std::min(launchBlocks, blocks - offset), 1, 1,
                width, 1, 1, 0, nullptr, arguments, nullptr), "CUDA block launch");
        }
    }

    RenderBatch render(const std::vector<RenderPlan>& plans, size_t first)
    {
        const auto frameBytes = 2 * (natural ? sizeof(float) : sizeof(int16_t));
        size_t free{}, total{};
        driver.check(driver.memory(&free, &total), "CUDA free memory");
        constexpr size_t mib = 1024 * 1024;
        const auto available = free + arena.capacity;
        const auto budget = std::min(available / 2, std::clamp(total / 12, 512 * mib, 1024 * mib));
        const auto singleJobBudget = std::min(available - available / 3, budget + budget / 3);
        size_t estimated = 0;
        BatchIndex index;
        std::vector<TailJob> jobs;
        int blocks = 0, windows = 0;
        for (size_t job = first; job < plans.size(); ++job) {
            const auto& plan = plans[job];
            const auto tiles = blocksFor(plan.frames, threads);
            const auto jobBlocks = int(blocksFor(plan.frames, tailBlockFrames));
            const auto jobWindows = natural ? int(blocksFor(plan.frames, tail.window)) : 0;
            size_t bytes = size_t(plan.frames) * frameBytes + tiles * sizeof(Tile) + plan.spans.size() * sizeof(VoiceSpan);
            bytes += sizeof(TailJob) + sizeof(TailResult) + 256;
            if (natural)
                bytes += size_t(plan.frames) * sizeof(int16_t) + jobBlocks * sizeof(StereoStats) +
                    jobWindows * sizeof(MonoStats);
            for (const auto& span : plan.spans) {
                bytes += size_t((span.end - 1) / threads - span.begin / threads + 1) * sizeof(int);
                if (span.filterTrack >= 0) {
                    const size_t duration = span.end - span.begin;
                    bytes += duration * 2 * sizeof(float) + ((duration + threads - 1) / threads) * sizeof(FilterTile);
                    if (span.variablePitch)
                        bytes += duration * sizeof(double);
                }
            }
            bytes += plan.filterTracks * sizeof(int);
            for (const auto& track : plan.controls)
                bytes += track.size() * sizeof(ControlPoint);
            bytes += plan.modulation.size() * sizeof(VoiceModulation);
            const auto limit = jobs.empty() && natural ? singleJobBudget : budget;
            if (estimated >= limit || bytes > limit - estimated)
                break;
            jobs.push_back({index.frames, plan.frames, plan.musicEnd, plan.limit, plan.fadeFrames, blocks, windows});
            index.append(plan, threads);
            estimated += bytes;
            blocks += jobBlocks;
            windows += jobWindows;
        }
        if (jobs.empty())
            throw std::runtime_error("one render job exceeds the CUDA batch memory budget");
        int count = static_cast<int>(jobs.size());
        ArenaLayout layout;
        const auto reserve = [&](const auto& items) { return layout.section(items.size() * sizeof(items.front())); };
        const auto voiceOffset = reserve(index.voices), tileOffset = reserve(index.tiles), indexOffset = reserve(index.indices);
        const auto controlOffset = reserve(index.controls), modulationOffset = reserve(index.modulation);
        const auto filterTileOffset = reserve(index.filterTiles), filterHeadOffset = reserve(index.filterHeads);
        const auto filteredOffset = layout.section(index.filteredFrames * 2 * sizeof(float));
        const auto phaseOffset = layout.section(index.pitchFrames * sizeof(double));
        const auto mixOffset = layout.section(index.frames * frameBytes);
        const auto jobOffset = reserve(jobs), resultOffset = layout.section(count * sizeof(TailResult));
        const auto monoBytes = natural ? index.frames * sizeof(int16_t) : 0;
        const auto monoOffset = layout.section(monoBytes);
        const auto statsOffset = layout.section(natural ? blocks * sizeof(StereoStats) : 0);
        const auto windowOffset = layout.section(windows * sizeof(MonoStats));
        arena.allocate(layout.bytes);
        peak = std::max(peak, sinc.capacity + samples.capacity + arena.capacity);
        const auto upload = [&](size_t offset, const auto& items) {
            auto address = arena.address + offset;
            if (!items.empty())
                driver.check(driver.put(address, items.data(), items.size() * sizeof(items.front())), "CUDA batch upload");
            return address;
        };
        auto voices = upload(voiceOffset, index.voices), tiles = upload(tileOffset, index.tiles);
        auto indices = upload(indexOffset, index.indices), controls = upload(controlOffset, index.controls);
        auto modulation = upload(modulationOffset, index.modulation), jobData = upload(jobOffset, jobs);
        auto filterTiles = upload(filterTileOffset, index.filterTiles), filterHeads = upload(filterHeadOffset, index.filterHeads);
        auto filtered = arena.address + filteredOffset, phase = arena.address + phaseOffset;
        auto mix = arena.address + mixOffset, resultData = arena.address + resultOffset;
        auto mono = natural ? arena.address + monoOffset : mix;
        // After mono conversion the float mix is dead, so its allocation also holds the compact PCM.
        auto packed = mix;
        TimingEvent begin(driver), end(driver);
        begin.record();
        if (index.filteredFrames) {
            int heads = static_cast<int>(index.filterHeads.size());
            constexpr int filterThreads = 64;
            const auto voiceBlocks = (heads + filterThreads - 1) / filterThreads;
            if (index.pitchFrames) {
                void* args[] = {&voices, &filterHeads, &controls, &modulation, &phase, &heads};
                driver.check(driver.launch(pitchPass, voiceBlocks, 1, 1, filterThreads, 1, 1, 0,
                    nullptr, args, nullptr), "CUDA variable pitch launch");
            }
            launchChunks(filterSource, int(index.filterTiles.size()), threads, samples.address, sinc.address,
                voices, filterTiles, controls, modulation, phase, filtered);
            void* args[] = {&voices, &filterHeads, &controls, &modulation, &filtered, &heads};
            driver.check(driver.launch(filterPass, voiceBlocks, 1, 1,
                filterThreads, 1, 1, 0, nullptr, args, nullptr), "CUDA filter launch");
        }
        launchChunks(kernel, int(index.tiles.size()), threads, samples.address, sinc.address, voices, tiles,
            indices, controls, modulation, filtered, mix);
        if (natural) {
            auto stats = arena.address + statsOffset, windowStats = arena.address + windowOffset;
            launchChunks(stereoScan, blocks, 128, jobData, count, mix, stats);
            void* nativeArgs[] = {&jobData, &count, &mix, &stats, &resultData};
            driver.check(driver.launch(nativeEnd, (count + 31) / 32, 1, 1, 32, 1, 1, 0,
                nullptr, nativeArgs, nullptr), "CUDA native endpoint launch");
            launchChunks(monoQuantize, windows, 256, jobData, count, mix, resultData, mono, windowStats, tail.window);
            void* trimArgs[] = {&jobData, &count, &windowStats, &resultData, &tail};
            driver.check(driver.launch(trimEnd, (count + 31) / 32, 1, 1, 32, 1, 1, 0,
                nullptr, trimArgs, nullptr), "CUDA tail trim launch");
        }
        void* layoutArgs[] = {&jobData, &count, &resultData};
        driver.check(driver.launch(pcmLayout, 1, 1, 1, 1, 1, 1, 0, nullptr, layoutArgs, nullptr), "CUDA PCM layout launch");
        launchChunks(pcmFinalize, blocks, 128, jobData, count, mono, packed, resultData);
        end.record();
        RenderBatch batch;
        batch.results.resize(count);
        driver.check(driver.get(batch.results.data(), resultData, count * sizeof(TailResult)), "CUDA tail metadata download");
        const auto& last = batch.results.back();
        batch.pcm = receivePcm(packed, (last.offset + last.dataFrames) * (natural ? 1 : 2));
        float milliseconds{};
        driver.check(driver.elapsedEvents(&milliseconds, begin.address, end.address), "CUDA kernel timing");
        batch.kernelSeconds = milliseconds / 1000.;
        return batch;
    }
};

Renderer::Renderer(int device, const Path& nvrtc, bool natural, int rate)
    : impl_(std::make_unique<Impl>(device, nvrtc, natural, rate))
{
}

Renderer::~Renderer() = default;

void Renderer::uploadSamples(const AudioBank& bank)
{
    impl_->uploadSamples(bank);
}

RenderBatch Renderer::render(const std::vector<RenderPlan>& plans, size_t first)
{
    return impl_->render(plans, first);
}
const std::string& Renderer::device() const { return impl_->deviceName; }
size_t Renderer::peakBytes() const { return impl_->peak; }

} // namespace clay
