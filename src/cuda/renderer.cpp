#include "renderer.h"
#include "../io/library.h"
#include "kernel_source.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace clay {
namespace {

using DevicePtr = unsigned long long;
using Handle = void*;
using Path = std::filesystem::path;
constexpr int cudaOutOfMemory = 2;

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
    int (*synchronize)() = lib.get<int(*)()>("cuCtxSynchronize");
    int (*load)(Handle*, const void*) = lib.get<int(*)(Handle*, const void*)>("cuModuleLoadData");
    int (*kernel)(Handle*, Handle, const char*) = lib.get<int(*)(Handle*, Handle, const char*)>("cuModuleGetFunction");
    int (*memory)(size_t*, size_t*) = lib.get<int(*)(size_t*, size_t*)>("cuMemGetInfo_v2");
    int (*allocate)(DevicePtr*, size_t) = lib.get<int(*)(DevicePtr*, size_t)>("cuMemAlloc_v2");
    int (*free)(DevicePtr) = lib.get<int(*)(DevicePtr)>("cuMemFree_v2");
    int (*put)(DevicePtr, const void*, size_t) = lib.get<int(*)(DevicePtr, const void*, size_t)>("cuMemcpyHtoD_v2");
    int (*get)(void*, DevicePtr, size_t) = lib.get<int(*)(void*, DevicePtr, size_t)>("cuMemcpyDtoH_v2");
    int (*copy)(DevicePtr, DevicePtr, size_t) = lib.get<int(*)(DevicePtr, DevicePtr, size_t)>("cuMemcpyDtoD_v2");
    int (*allocHost)(void**, size_t) = lib.get<int(*)(void**, size_t)>("cuMemAllocHost_v2");
    int (*allocMapped)(void**, size_t, unsigned) = lib.get<int(*)(void**, size_t, unsigned)>("cuMemHostAlloc");
    int (*mappedAddress)(DevicePtr*, void*, unsigned) = lib.get<int(*)(DevicePtr*, void*, unsigned)>("cuMemHostGetDevicePointer_v2");
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
        driver_.check(tryAllocate(bytes), "CUDA allocation");
    }

    int tryAllocate(size_t bytes)
    {
        bytes = std::max(bytes, size_t{1});
        if (bytes <= capacity)
            return 0;
        reset();
        DevicePtr allocation{};
        const auto error = driver_.allocate(&allocation, bytes);
        if (error)
            return error;
        address = allocation;
        capacity = bytes;
        return 0;
    }

    void swap(Buffer& other)
    {
        std::swap(address, other.address);
        std::swap(capacity, other.capacity);
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
    ~HostPcm() { reset(); }
    void reset()
    {
        if (address)
            driver.freeHost(address);
        address = nullptr;
        capacity = 0;
    }
    bool reserve(size_t bytes)
    {
        if (bytes <= capacity)
            return true;
        reset();
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

Path kernelCache()
{
#ifdef _WIN32
    const auto* local = _wgetenv(L"LOCALAPPDATA");
    return local && *local ? Path(local) / "ClaySampler" / "kernels" : Path{};
#else
    if (const auto* local = std::getenv("XDG_CACHE_HOME"); local && *local)
        return Path(local) / "claysampler" / "kernels";
    const auto* home = std::getenv("HOME");
    return home && *home ? Path(home) / ".cache" / "claysampler" / "kernels" : Path{};
#endif
}

std::vector<char> readKernelCache(const Path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes > 32 * 1024 * 1024)
        return {};
    std::vector<char> code(static_cast<size_t>(bytes) + 1);
    if (!file.seekg(0).read(code.data(), static_cast<std::streamsize>(bytes)))
        return {};
    return code;
}

void writeKernelCache(const Path& path, const std::vector<char>& code)
{
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error)
        return;
#ifdef _WIN32
    const auto process = GetCurrentProcessId();
#else
    const auto process = getpid();
#endif
    auto temporary = path;
    temporary += "." + std::to_string(process) + "." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(code.data(), static_cast<std::streamsize>(code.size()));
    file.close();
    if (file)
        std::filesystem::rename(temporary, path, error);
    std::filesystem::remove(temporary, error);
}

void loadKernels(const Driver& driver, Handle& module, const Path& library, int architecture,
                 bool floatingPoint, bool quality, bool grouping)
{
    Library rtc(library);
    const auto version = rtc.get<int(*)(int*, int*)>("nvrtcVersion");
    int major{}, minor{};
    const bool hasVersion = version(&major, &minor) == 0;
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

    auto archFlag = std::string(native ? "--gpu-architecture=sm_" : "--gpu-architecture=compute_") +
        std::to_string(target);
    const bool parallel = hasVersion && (major > 12 || (major == 12 && minor >= 4)) &&
        std::thread::hardware_concurrency() >= 4;
    const char* flags[] = {archFlag.c_str(), "--std=c++17", floatingPoint ? "-DCLAY_FLOAT_OUTPUT=1" : "-DCLAY_FLOAT_OUTPUT=0",
        quality ? "-DCLAY_AUDIO_QA=1" : "-DCLAY_AUDIO_QA=0",
        grouping ? "-DCLAY_GROUP_VOLUME=1" : "-DCLAY_GROUP_VOLUME=0", "--split-compile=4"};
    auto cache = hasVersion ? kernelCache() : Path{};
    if (!cache.empty()) {
        uint64_t hash = 14695981039346656037ull;
        const auto append = [&](std::string_view value) {
            for (unsigned char byte : value)
                hash = (hash ^ byte) * 1099511628211ull;
            hash = (hash ^ 0xff) * 1099511628211ull;
        };
        append(kernelSource);
        append(kernelTypes);
        for (int i = 0; i < (parallel ? 6 : 5); ++i)
            append(flags[i]);
        append(library.u8string());
        if (library.has_parent_path()) {
            std::error_code error;
            const auto updated = std::filesystem::last_write_time(library, error);
            if (!error)
                append(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                    updated.time_since_epoch()).count()));
        }
        cache /= std::to_string(major) + "-" + std::to_string(minor) + "-" + std::to_string(hash) + ".bin";
        const auto cached = readKernelCache(cache);
        if (!cached.empty()) {
            if (!driver.load(&module, cached.data()))
                return;
            std::error_code error;
            std::filesystem::remove(cache, error);
        }
    }
#ifdef _WIN32
    std::unique_ptr<Library> builtins;
    if (library.has_parent_path() && hasVersion) {
        auto filename = "nvrtc-builtins64_" + std::to_string(major) + std::to_string(minor) + ".dll";
        auto path = std::filesystem::absolute(library).parent_path() / filename;
        if (std::filesystem::exists(path))
            builtins = std::make_unique<Library>(path);
    }
#endif
    const char* header = kernelTypes;
    const char* headerName = "render_types.h";
    Handle program = nullptr;
    check(create(&program, kernelSource, "render.cu", 1, &header, &headerName));
    struct Program {
        Handle& handle;
        int (*destroy)(Handle*);
        ~Program() { destroy(&handle); }
    } scope{program, destroy};
    if (run(program, parallel ? 6 : 5, flags)) {
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
    driver.check(driver.load(&module, result.data()), "CUDA module");
    if (!cache.empty())
        writeKernelCache(cache, result);
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
    std::vector<int> filterHeads, pitchHeads;
    size_t frames = 0;
    size_t filteredSamples = 0;
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
                if (tail < 0) {
                    if (voice.variablePitch)
                        pitchHeads.push_back(index);
                    if (std::any_of(std::begin(voice.filters), std::end(voice.filters),
                        [](const auto& filter) { return filter.type != FilterType::none; }))
                        filterHeads.push_back(index);
                } else {
                    voices[tail].nextFilteredSpan = index;
                }
                tail = index;
                voice.nextFilteredSpan = -1;
                voice.filteredOffset = filteredSamples;
                filteredSamples += size_t(voice.end - voice.begin) * voice.channels;
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
    Buffer samples{driver}, sinc{driver}, arena{driver}, groupControl{driver};
    HostPcm mappedScratch{driver}, mappedSamples{driver};
    DevicePtr mappedBase{}, sampleData{};
    std::vector<std::shared_ptr<HostPcm>> hostPcm;
    Handle module{}, kernel{}, filterSource{}, filterPass{}, pitchPass{};
    Handle scheduledMix{}, scheduledSource{}, scheduledPitch{}, scheduledFilter{}, scheduledFinish{}, scheduledPackLayout{};
    size_t residentBytes{};
    Handle stereoScan{}, nativeEnd{}, monoQuantize{}, trimEnd{}, pcmLayout{}, pcmFinalize{};
    Handle bodyMeasure{}, onsetMeasure{}, audioSummary{}, volumeMatch{}, acceptedLayout{}, acceptedPack{};
    std::string deviceName;
    int threads{}, multiprocessors{};
    bool natural;
    TailConfig tail;
    AudioConfig audio;
    int groupMinimum;
    size_t waitingLimit, batchJobs;
    size_t uploaded{}, cacheBudget{}, peak{}, waiting{}, peakWaiting{}, spilled{}, downloaded{};

    Impl(int ordinal, const Path& rtc, bool useNatural, AudioConfig config, int minimum, size_t limit, size_t jobs)
        : natural(useNatural), tail{config.window, 2 * config.rate,
            int(std::nearbyint(config.rate / 4.)), int(std::nearbyint(config.rate / 10.)), std::pow(10., -1.5),
            32768 * std::pow(10., -3.9), 32768 * std::pow(10., -3.)},
          audio(config), groupMinimum(minimum), waitingLimit(limit), batchJobs(jobs)
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
        driver.check(driver.attribute(&multiprocessors, 16, device), "CUDA multiprocessor count");
        constexpr unsigned mapHost = 8; // CU_CTX_MAP_HOST
        driver.check(driver.create(&context.address, mapHost, device), "CUDA context");
        loadKernels(driver, module, rtc, 10 * major + minor, natural, audio.enabled, groupMinimum > 0);
        driver.check(driver.kernel(&kernel, module, "renderTiles"), "CUDA render kernel");
        driver.check(driver.kernel(&filterSource, module, "renderFilterSources"), "CUDA filter source kernel");
        driver.check(driver.kernel(&filterPass, module, "applyVoiceFilters"), "CUDA filter kernel");
        driver.check(driver.kernel(&pitchPass, module, "prepareVoicePitch"), "CUDA variable pitch kernel");
        driver.check(driver.kernel(&scheduledMix, module, "scheduledTiles"), "CUDA scheduled mix kernel");
        driver.check(driver.kernel(&scheduledSource, module, "scheduledSources"), "CUDA scheduled source kernel");
        driver.check(driver.kernel(&scheduledPitch, module, "scheduledPitch"), "CUDA scheduled pitch kernel");
        driver.check(driver.kernel(&scheduledFilter, module, "scheduledFilters"), "CUDA scheduled filter kernel");
        driver.check(driver.kernel(&scheduledFinish, module, "scheduledFinish"), "CUDA scheduled finishing kernel");
        driver.check(driver.kernel(&scheduledPackLayout, module, "scheduledPackLayout"), "CUDA completion layout kernel");
        driver.check(driver.kernel(&pcmLayout, module, "layoutPcm"), "CUDA PCM layout kernel");
        driver.check(driver.kernel(&pcmFinalize, module, "finalizePcm"), "CUDA PCM finalization kernel");
        if (audio.enabled || groupMinimum) {
            driver.check(driver.kernel(&bodyMeasure, module, "measureBody"), "CUDA body RMS kernel");
            driver.check(driver.kernel(&audioSummary, module, "summarizeAudio"), "CUDA audio statistics kernel");
            driver.check(driver.kernel(&acceptedLayout, module, "layoutAccepted"), "CUDA accepted PCM layout kernel");
            driver.check(driver.kernel(&acceptedPack, module, "packAccepted"), "CUDA accepted PCM packing kernel");
        }
        if (audio.enabled)
            driver.check(driver.kernel(&onsetMeasure, module, "measureOnsets"), "CUDA onset RMS kernel");
        if (groupMinimum)
            driver.check(driver.kernel(&volumeMatch, module, "matchVolume"), "CUDA group volume kernel");
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
        size_t free{}, total{};
        driver.check(driver.memory(&free, &total), "CUDA sample cache budget");
        cacheBudget = std::min(free - free / 3, total / 2);
    }

    void recordPeak(size_t extra = 0)
    {
        peak = std::max(peak, sinc.capacity + samples.capacity + arena.capacity + groupControl.capacity + waiting + residentBytes + extra);
    }

    void uploadSamples(const AudioBank& bank, bool fresh)
    {
        if (fresh) {
            samples.reset();
            mappedSamples.reset();
            sampleData = 0;
            uploaded = 0;
        }
        const auto bytes = bank.pcm.size() * sizeof(float);
        if (bytes <= uploaded && sampleData)
            return;
        if (!mappedSamples.address && (bytes > samples.capacity || !samples.address)) {
            size_t free{}, total{};
            driver.check(driver.memory(&free, &total), "CUDA free memory");
            const size_t slack = std::min(bytes / 2, size_t{128} * 1024 * 1024);
            Buffer replacement(driver);
            DevicePtr candidate{};
            const auto capacity = std::max(bytes + slack, size_t{1});
            if (capacity <= free && !driver.allocate(&candidate, capacity)) {
                replacement.address = candidate;
                replacement.capacity = capacity;
                recordPeak(replacement.capacity);
                if (uploaded)
                    driver.check(driver.copy(replacement.address, samples.address, uploaded), "CUDA sample bank growth");
                std::swap(samples.address, replacement.address);
                std::swap(samples.capacity, replacement.capacity);
            } else {
                arena.reset();
                const auto status = samples.tryAllocate(bytes);
                if (status != cudaOutOfMemory)
                    driver.check(status, "CUDA sample bank allocation");
                uploaded = 0;
            }
        }
        if (!samples.address && bytes > mappedSamples.capacity) {
            mappedSamples.reset();
            driver.check(tryMapHost(mappedSamples, bytes, sampleData), "CUDA mapped sample bank");
            uploaded = 0;
        }
        if (bytes > uploaded) {
            const auto* source = bank.pcm.data() + uploaded / sizeof(float);
            if (mappedSamples.address)
                std::memcpy(static_cast<char*>(mappedSamples.address) + uploaded, source, bytes - uploaded);
            else
                driver.check(driver.put(samples.address + uploaded, source, bytes - uploaded), "CUDA sample upload");
        }
        if (samples.address)
            sampleData = samples.address;
        uploaded = bytes;
        recordPeak();
    }

    int tryMapHost(HostPcm& buffer, size_t bytes, DevicePtr& address)
    {
        constexpr unsigned deviceMap = 2; // CU_MEMHOSTALLOC_DEVICEMAP
        void* host{};
        if (const auto status = driver.allocMapped(&host, bytes, deviceMap))
            return status;
        buffer.address = host;
        DevicePtr mapped{};
        if (const auto status = driver.mappedAddress(&mapped, host, 0)) {
            buffer.reset();
            return status;
        }
        buffer.capacity = bytes;
        address = mapped;
        return 0;
    }

    DevicePtr mapScratch(size_t bytes)
    {
        const auto status = tryMapHost(mappedScratch, bytes, mappedBase);
        if (status == cudaOutOfMemory)
            return 0;
        driver.check(status, "CUDA mapped scratch allocation");
        return mappedBase;
    }

    void releaseMappedScratch()
    {
        mappedScratch.reset();
        mappedBase = 0;
    }

    char* mappedPointer(DevicePtr address) const
    {
        if (address >= mappedBase && address - mappedBase < mappedScratch.capacity)
            return static_cast<char*>(mappedScratch.address) + (address - mappedBase);
        return nullptr;
    }

    void uploadScratch(DevicePtr destination, const void* source, size_t bytes, const char* operation)
    {
        if (auto* host = mappedPointer(destination))
            std::memcpy(host, source, bytes);
        else
            driver.check(driver.put(destination, source, bytes), operation);
    }

    void downloadScratch(void* destination, DevicePtr source, size_t bytes, const char* operation)
    {
        if (const auto* host = mappedPointer(source)) {
            driver.check(driver.synchronize(), operation);
            std::memcpy(destination, host, bytes);
        } else {
            driver.check(driver.get(destination, source, bytes), operation);
        }
    }

    void copyScratch(DevicePtr destination, DevicePtr source, size_t bytes, const char* operation)
    {
        if (const auto* host = mappedPointer(source)) {
            driver.check(driver.synchronize(), operation);
            driver.check(driver.put(destination, host, bytes), operation);
        } else {
            driver.check(driver.copy(destination, source, bytes), operation);
        }
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
        downloadScratch(pcm.get(), source, bytes, "CUDA PCM download");
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

    void launchRecurrences(Handle function, int count, DevicePtr voices, DevicePtr heads,
                           DevicePtr controls, DevicePtr modulation, DevicePtr scratch, DevicePtr states,
                           int segmentFirst, int segmentEnd, int segmentFrames)
    {
        if (!count)
            return;
        // Each lane advances an independent serial recurrence. Spread small sets
        // across SMs, then fill warps as the number of independent tracks grows.
        const int blocks = 2 * multiprocessors;
        const int width = std::clamp(1 + (count - 1) / blocks, 1, 32);
        void* args[] = {&voices, &heads, &controls, &modulation, &scratch, &states,
                        &segmentFirst, &segmentEnd, &segmentFrames, &count};
        driver.check(driver.launch(function, 1 + (count - 1) / width, 1, 1,
            width, 1, 1, 0, nullptr, args, nullptr), "CUDA voice recurrence launch");
    }

    struct Synthesis {
        std::vector<TailJob> jobs;
        DevicePtr jobData{}, resultData{}, source{}, output{}, pcmJobs{};
        int blocks{};
        size_t failedScratchBytes{};
    };

    struct Pending {
        BatchIndex index;
        ScheduledJob device{};
        FinishJob finishing{};
        Synthesis work;
        int frame = 0;
    };

    int channels() const { return natural ? 1 : 2; }

    int bodyWindows(int frames) const
    {
        const int chunk = 2 * audio.rate, span = 5 * audio.window;
        return (frames / chunk) * int(blocksFor(chunk, span)) + int(blocksFor(frames % chunk, span));
    }

    size_t footprint(const RenderPlan& plan) const
    {
        const auto frameBytes = 2 * (natural ? sizeof(float) : sizeof(int16_t));
        const auto tiles = blocksFor(plan.frames, threads);
        size_t bytes = size_t(plan.frames) * frameBytes + tiles * sizeof(Tile) + plan.spans.size() * sizeof(VoiceSpan);
        bytes += sizeof(TailJob) + sizeof(TailResult) + 256;
        size_t finishing = 0, filtering = 0;
        if (natural)
            finishing += size_t(plan.frames) * sizeof(int16_t) + blocksFor(plan.frames, tailBlockFrames) * sizeof(StereoStats) +
                blocksFor(plan.frames, tail.window) * sizeof(MonoStats);
        if (audio.enabled || groupMinimum) {
            const int body = audio.enabled || plan.volumeGroup ? plan.bodyFrames : 0;
            bytes += sizeof(AudioJob) + plan.onsets.size() * sizeof(int);
            finishing += sizeof(PcmJob) + (size_t(bodyWindows(body)) + plan.onsets.size()) * sizeof(unsigned long long);
            if (!natural)
                finishing += size_t(plan.frames) * frameBytes;
        }
        for (const auto& span : plan.spans) {
            bytes += size_t((span.end - 1) / threads - span.begin / threads + 1) * sizeof(int);
            if (span.filterTrack >= 0) {
                const size_t duration = span.end - span.begin;
                filtering += duration * span.channels * sizeof(float);
                bytes += blocksFor(duration, threads) * sizeof(FilterTile);
                if (span.variablePitch)
                    filtering += duration * sizeof(double);
            }
        }
        bytes += plan.filterTracks * 2 * sizeof(int);
        for (const auto& track : plan.controls)
            bytes += track.size() * sizeof(ControlPoint);
        return bytes + plan.modulation.size() * sizeof(VoiceModulation) + std::max(finishing, std::min(filtering, size_t{64} * 1024 * 1024));
    }

    Synthesis synthesize(const std::vector<RenderPlan>& plans, size_t first, size_t stop, const TimingEvent& begin,
                         const std::vector<size_t>& sizes = {})
    {
        size_t maxJobs = batchJobs;
        size_t scratchLimit = std::numeric_limits<size_t>::max();
        for (;;) {
            auto work = trySynthesize(plans, first, stop, begin, sizes, maxJobs, scratchLimit);
            if (!work.jobs.empty())
                return work;
        }
    }

    Synthesis trySynthesize(const std::vector<RenderPlan>& plans, size_t first, size_t stop, const TimingEvent& begin,
                            const std::vector<size_t>& sizes, size_t& maxJobs, size_t& scratchLimit, Pending* pending = nullptr)
    {
        const auto frameBytes = 2 * (natural ? sizeof(float) : sizeof(int16_t));
        size_t free{}, total{};
        driver.check(driver.memory(&free, &total), "CUDA free memory");
        const auto available = free + arena.capacity;
        // Keep routine batches bounded by both the device size and current memory pressure.
        // A large job gets its own batch and tries the exact arena allocation below.
        const auto budget = std::min(available - available / 4, total / 4);
        size_t estimated = 0;
        BatchIndex index;
        Synthesis synthesis;
        auto& jobs = synthesis.jobs;
        const bool assess = audio.enabled || groupMinimum;
        std::vector<AudioJob> audioJobs;
        std::vector<int> onsets;
        int blocks = 0, windows = 0, bodyBlocks = 0;
        for (size_t job = first; job < stop && jobs.size() < maxJobs; ++job) {
            const auto& plan = plans[job];
            const auto bytes = sizes.empty() ? footprint(plan) : sizes[job];
            if (!jobs.empty() && (estimated >= budget || bytes > budget - estimated))
                break;
            if (!sizes.empty() && plan.volumeGroup && (job == first || plans[job - 1].volumeGroup != plan.volumeGroup)) {
                auto end = job;
                size_t groupBytes = 0;
                for (; end < stop && plans[end].volumeGroup == plan.volumeGroup; ++end)
                    groupBytes += sizes[end];
                if (end - first > maxJobs || groupBytes > budget - std::min(budget, estimated)) {
                    if (!jobs.empty())
                        break;
                    // A split starts with this group and never keeps unrelated PCM waiting.
                    stop = end;
                }
            }
            const auto jobBlocks = int(blocksFor(plan.frames, tailBlockFrames));
            const auto jobWindows = natural ? int(blocksFor(plan.frames, tail.window)) : 0;
            const int body = audio.enabled || plan.volumeGroup ? plan.bodyFrames : 0;
            const int jobBodyBlocks = bodyWindows(body);
            jobs.push_back({index.frames, plan.frames, plan.musicEnd, plan.limit, plan.fadeFrames, blocks, windows});
            if (assess) {
                audioJobs.push_back({body, bodyBlocks, int(onsets.size()), int(plan.onsets.size())});
                onsets.insert(onsets.end(), plan.onsets.begin(), plan.onsets.end());
                bodyBlocks += jobBodyBlocks;
            }
            index.append(plan, threads);
            estimated += bytes;
            blocks += jobBlocks;
            windows += jobWindows;
        }
        std::vector<VoiceState> states(index.voices.size());
        int count = static_cast<int>(jobs.size());
        ArenaLayout layout;
        const auto reserve = [&](const auto& items) { return layout.section(items.size() * sizeof(items.front())); };
        const auto voiceOffset = reserve(index.voices), tileOffset = reserve(index.tiles), indexOffset = reserve(index.indices);
        const auto controlOffset = reserve(index.controls), modulationOffset = reserve(index.modulation);
        const auto filterTileOffset = reserve(index.filterTiles), filterHeadOffset = reserve(index.filterHeads);
        const auto pitchHeadOffset = reserve(index.pitchHeads);
        const auto jobOffset = reserve(jobs), audioOffset = reserve(audioJobs), onsetOffset = reserve(onsets);
        const auto stateOffset = reserve(states);
        const auto mixOffset = layout.section(index.frames * frameBytes);
        const auto resultOffset = layout.section(count * sizeof(TailResult));
        // Once the mix kernel finishes, PCM and quality buffers can reuse filter and pitch scratch.
        const auto scratchOffset = layout.section(0);
        ArenaLayout finishing;
        const auto monoOffset = finishing.section(natural ? index.frames * sizeof(int16_t) : 0);
        const auto statsOffset = finishing.section(natural ? blocks * sizeof(StereoStats) : 0);
        const auto windowOffset = finishing.section(windows * sizeof(MonoStats));
        const auto bodyPowerOffset = finishing.section(size_t(bodyBlocks) * sizeof(unsigned long long));
        const auto onsetPowerOffset = finishing.section(onsets.size() * sizeof(unsigned long long));
        const auto pcmJobOffset = finishing.section(assess ? jobs.size() * sizeof(PcmJob) : 0);
        const auto stereoOutput = finishing.section(assess && !natural ? index.frames * frameBytes : 0);
        // Reserve the mix, metadata, and finishing buffers before budgeting reusable synthesis scratch.
        const auto fixedBytes = scratchOffset + finishing.bytes;
        const auto usable = available - available / 8;
        const auto scratchBudget = std::min(scratchLimit, std::max(size_t{1024} * 1024,
            usable > fixedBytes ? std::min(budget, usable - scratchOffset) : size_t{0}));
        struct Activity { int frame, channels, pitch; };
        struct Usage { int frame; size_t samples, phases; int channels, pitch; };
        struct Segment { int first, end; size_t samples, phases; };
        std::vector<Activity> activity;
        std::vector<Usage> usage{{0, 0, 0, 0, 0}};
        int lastFrame = 0;
        for (const auto& job : jobs)
            lastFrame = std::max(lastFrame, job.frames);
        const auto fullFilterBytes = index.filteredSamples * sizeof(float);
        const auto fullPhaseOffset = (fullFilterBytes + 15) & ~size_t{15};
        const auto fullScratchBytes = fullPhaseOffset + index.pitchFrames * sizeof(double);
        if (!pending && fullScratchBytes > scratchBudget) {
            for (const auto& voice : index.voices) {
                if (voice.filterTrack < 0)
                    continue;
                activity.push_back({voice.begin, voice.channels, voice.variablePitch});
                activity.push_back({voice.end, -voice.channels, -voice.variablePitch});
            }
            std::sort(activity.begin(), activity.end(), [](const auto& a, const auto& b) { return a.frame < b.frame; });
            for (size_t i = 0; i < activity.size();) {
                auto point = usage.back();
                const auto frame = activity[i].frame;
                point.samples += size_t(frame - point.frame) * point.channels;
                point.phases += size_t(frame - point.frame) * point.pitch;
                point.frame = frame;
                do {
                    point.channels += activity[i].channels;
                    point.pitch += activity[i].pitch;
                    ++i;
                } while (i < activity.size() && activity[i].frame == frame);
                usage.push_back(point);
            }
        }
        const auto usageAt = [&](int frame) {
            auto point = *std::prev(std::upper_bound(usage.begin(), usage.end(), frame,
                [](int frame, const auto& point) { return frame < point.frame; }));
            point.samples += size_t(frame - point.frame) * point.channels;
            point.phases += size_t(frame - point.frame) * point.pitch;
            return point;
        };
        std::vector<Segment> segments;
        size_t filterBytes = 0;
        for (int firstFrame = 0; firstFrame < lastFrame;) {
            if (pending || fullScratchBytes <= scratchBudget) {
                segments.push_back({0, lastFrame, index.filteredSamples, index.pitchFrames});
                filterBytes = fullScratchBytes;
                break;
            }
            const auto firstUsage = usageAt(firstFrame);
            const auto fits = [&](int end) {
                const auto last = usageAt(end);
                const auto bytes = ((last.samples - firstUsage.samples) * sizeof(float) + 15) & ~size_t{15};
                return bytes <= scratchBudget && (last.phases - firstUsage.phases) * sizeof(double) <= scratchBudget - bytes;
            };
            int low = firstFrame, high = lastFrame;
            while (low < high) {
                const auto middle = low + (high - low + 1) / 2;
                if (fits(middle)) low = middle;
                else high = middle - 1;
            }
            // If even one frame cannot fit, use allocation recovery without constructing one segment per frame.
            const auto end = low == firstFrame ? lastFrame : low;
            const auto last = usageAt(end);
            Segment segment{firstFrame, end, last.samples - firstUsage.samples, last.phases - firstUsage.phases};
            const auto bytes = ((segment.samples * sizeof(float) + 15) & ~size_t{15}) + segment.phases * sizeof(double);
            filterBytes = std::max(filterBytes, bytes);
            segments.push_back(segment);
            firstFrame = end;
        }
        const bool segmented = segments.size() > 1;
        if (segmented) {
            std::cerr << "SEGMENT\tfirst=" << first << "\tjobs=" << jobs.size()
                      << "\tcount=" << segments.size() << "\tscratch_bytes=" << filterBytes
                      << "\tfull_scratch_bytes=" << fullScratchBytes << '\n';
            std::sort(index.tiles.begin(), index.tiles.end(), [](const auto& a, const auto& b) { return a.frame < b.frame; });
            std::sort(index.filterTiles.begin(), index.filterTiles.end(),
                [](const auto& a, const auto& b) { return a.frame < b.frame; });
        }
        if (pending) {
            std::sort(index.tiles.begin(), index.tiles.end(), [](const auto& a, const auto& b) { return a.frame < b.frame; });
            std::sort(index.filterTiles.begin(), index.filterTiles.end(),
                [](const auto& a, const auto& b) { return a.frame < b.frame; });
        }
        const auto bytes = scratchOffset + (pending ? finishing.bytes : std::max(finishing.bytes, filterBytes));
        auto allocation = arena.tryAllocate(bytes);
        if (pending && allocation == cudaOutOfMemory)
            return {};
        if (allocation == cudaOutOfMemory && filterBytes > std::max(finishing.bytes, size_t{2} * 1024 * 1024)) {
            scratchLimit = filterBytes / 2;
            return {};
        }
        if (allocation == cudaOutOfMemory && jobs.size() > 1) {
            // Destroy the failed batch's host metadata before assembling a smaller one.
            maxJobs = jobs.size() / 2;
            return {};
        }
        if (allocation == cudaOutOfMemory) {
            std::cerr << "MEMORY\tjob=" << first << "\trequested_bytes=" << bytes
                      << "\tmix_bytes=" << index.frames * frameBytes
                      << "\tfilter_bytes=" << filterBytes
                      << "\tpitch_bytes=" << index.pitchFrames * sizeof(double)
                      << "\tfinishing_bytes=" << finishing.bytes
                      << "\tmetadata_bytes=" << scratchOffset - index.frames * frameBytes
                      << "\tsample_bytes=" << uploaded
                      << "\tfree_device_bytes=" << free << '\n';
        }
        DevicePtr filterScratch{};
        if (allocation == cudaOutOfMemory && filterBytes > finishing.bytes) {
            // Filter and pitch scratch account for most of a dense job's memory.
            // Kernels can access it directly in host memory while the mix stays on the device.
            allocation = arena.tryAllocate(scratchOffset + finishing.bytes);
            if (!allocation) {
                filterScratch = mapScratch(filterBytes);
                if (!filterScratch) {
                    synthesis.failedScratchBytes = filterBytes;
                    return synthesis;
                }
            }
        }
        if (allocation != cudaOutOfMemory)
            driver.check(allocation, "CUDA synthesis scratch allocation");
        const auto base = allocation == cudaOutOfMemory ? mapScratch(bytes) : arena.address;
        if (!base) {
            synthesis.failedScratchBytes = bytes;
            return synthesis;
        }
        const auto finishScratch = base + scratchOffset;
        if (!filterScratch)
            filterScratch = finishScratch;
        recordPeak();
        std::vector<unsigned char> staging(pending ? mixOffset : 0);
        const auto upload = [&](size_t offset, const auto& items) {
            auto address = base + offset;
            if (!items.empty()) {
                const auto bytes = items.size() * sizeof(items.front());
                if (pending) std::memcpy(staging.data() + offset, items.data(), bytes);
                else uploadScratch(address, items.data(), bytes, "CUDA batch upload");
            }
            return address;
        };
        auto voices = upload(voiceOffset, index.voices);
        auto tiles = base + tileOffset;
        auto stateData = upload(stateOffset, states);
        auto indices = upload(indexOffset, index.indices), controls = upload(controlOffset, index.controls);
        auto modulation = upload(modulationOffset, index.modulation), jobData = upload(jobOffset, jobs);
        auto filterTiles = base + filterTileOffset;
        auto filterHeads = upload(filterHeadOffset, index.filterHeads);
        auto pitchHeads = upload(pitchHeadOffset, index.pitchHeads);
        auto filtered = filterScratch;

        auto mix = base + mixOffset, resultData = base + resultOffset;
        auto mono = natural ? finishScratch + monoOffset : mix;
        // After mono conversion the float mix is dead, so its allocation also holds the compact PCM.
        auto packed = mix;
        auto assessment = upload(audioOffset, audioJobs), starts = upload(onsetOffset, onsets);
        synthesis.jobData = jobData;
        synthesis.resultData = resultData;
        synthesis.source = packed;
        synthesis.output = natural ? mono : finishScratch + stereoOutput;
        synthesis.pcmJobs = finishScratch + pcmJobOffset;
        synthesis.blocks = blocks;
        if (pending) {
            upload(tileOffset, index.tiles);
            upload(filterTileOffset, index.filterTiles);
            uploadScratch(base, staging.data(), staging.size(), "CUDA resident metadata upload");
            pending->device = {reinterpret_cast<const VoiceSpan*>(voices), reinterpret_cast<const Tile*>(tiles),
                reinterpret_cast<const FilterTile*>(filterTiles), reinterpret_cast<const int*>(indices),
                reinterpret_cast<const int*>(pitchHeads), reinterpret_cast<const int*>(filterHeads),
                reinterpret_cast<const ControlPoint*>(controls), reinterpret_cast<const VoiceModulation*>(modulation),
                reinterpret_cast<VoiceState*>(stateData), nullptr, nullptr, reinterpret_cast<void*>(mix),
                0, 0, 0, 0, 0, 0, 0, 0};
            pending->finishing = {reinterpret_cast<const TailJob*>(jobData), reinterpret_cast<const AudioJob*>(assessment),
                reinterpret_cast<const int*>(starts), reinterpret_cast<const void*>(mix), reinterpret_cast<short*>(mono),
                reinterpret_cast<short*>(packed), reinterpret_cast<short*>(synthesis.output),
                reinterpret_cast<StereoStats*>(finishScratch + statsOffset), reinterpret_cast<MonoStats*>(finishScratch + windowOffset),
                reinterpret_cast<unsigned long long*>(finishScratch + bodyPowerOffset),
                reinterpret_cast<unsigned long long*>(finishScratch + onsetPowerOffset), reinterpret_cast<TailResult*>(resultData),
                reinterpret_cast<PcmJob*>(synthesis.pcmJobs), blocks, windows, bodyBlocks, int(onsets.size()), nullptr, 0};
            pending->index = std::move(index);
            pending->work = synthesis;
            return synthesis;
        }
        begin.record();
        for (const auto& segment : segments) {
            auto segmentFirst = segment.first, segmentEnd = segment.end;
            auto segmentFrames = segmentEnd - segmentFirst;
            auto phase = filtered + ((segment.samples * sizeof(float) + 15) & ~size_t{15});
            size_t filteredOffset = 0, pitchOffset = 0;
            for (auto& voice : index.voices) {
                if (voice.filterTrack < 0)
                    continue;
                const auto frames = std::max(0, std::min(voice.end, segmentEnd) - std::max(voice.begin, segmentFirst));
                voice.filteredOffset = filteredOffset;
                filteredOffset += size_t(frames) * voice.channels;
                voice.phaseOffset = pitchOffset;
                if (voice.variablePitch)
                    pitchOffset += frames;
            }
            if (index.filteredSamples && segmented)
                uploadScratch(voices, index.voices.data(), index.voices.size() * sizeof(VoiceSpan), "CUDA segment voices");
            const auto activeHeads = [&](const auto& heads, DevicePtr destination) {
                if (!segmented)
                    return int(heads.size());
                std::vector<int> active;
                for (auto head : heads) {
                    for (int span = head; span >= 0; span = index.voices[span].nextFilteredSpan) {
                        const auto& voice = index.voices[span];
                        if (voice.begin >= segmentEnd)
                            break;
                        if (voice.end > segmentFirst) {
                            active.push_back(head);
                            break;
                        }
                    }
                }
                if (!active.empty())
                    uploadScratch(destination, active.data(), active.size() * sizeof(int), "CUDA active voice heads");
                return int(active.size());
            };
            const auto pitchCount = activeHeads(index.pitchHeads, pitchHeads);
            const auto filterCount = activeHeads(index.filterHeads, filterHeads);
            const auto uploadTiles = [&](const auto& items, DevicePtr destination, int margin) {
                const auto first = !segmented ? items.begin() : std::lower_bound(items.begin(), items.end(), segmentFirst - margin,
                    [](const auto& item, int frame) { return item.frame < frame; });
                const auto end = !segmented ? items.end() : std::lower_bound(first, items.end(), segmentEnd,
                    [](const auto& item, int frame) { return item.frame < frame; });
                const auto count = int(end - first);
                if (count)
                    uploadScratch(destination, &*first, size_t(count) * sizeof(*first), "CUDA segment tile upload");
                return count;
            };
            if (index.filteredSamples) {
                launchRecurrences(pitchPass, pitchCount, voices, pitchHeads, controls, modulation,
                    phase, stateData, segmentFirst, segmentEnd, segmentFrames);
                const auto count = uploadTiles(index.filterTiles, filterTiles, threads - 1);
                launchChunks(filterSource, count, threads, sampleData, sinc.address,
                    voices, filterTiles, controls, modulation, phase, filtered, segmentFirst, segmentEnd, segmentFrames);
                launchRecurrences(filterPass, filterCount, voices, filterHeads, controls, modulation,
                    filtered, stateData, segmentFirst, segmentEnd, segmentFrames);
            }
            const auto count = uploadTiles(index.tiles, tiles, threads - 1);
            launchChunks(kernel, count, threads, sampleData, sinc.address, voices, tiles,
                indices, controls, modulation, filtered, mix, segmentFirst, segmentEnd, segmentFrames);
        }
        if (natural) {
            auto stats = finishScratch + statsOffset, windowStats = finishScratch + windowOffset;
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
        if (assess) {
            auto bodyPower = finishScratch + bodyPowerOffset, onsetPower = finishScratch + onsetPowerOffset;
            launchChunks(bodyMeasure, bodyBlocks, 128, assessment, count, packed, resultData, bodyPower, audio);
            if (audio.enabled)
                launchChunks(onsetMeasure, int(onsets.size()), 160, assessment, count, starts, packed,
                    resultData, onsetPower, audio.window);
            void* args[] = {&assessment, &count, &bodyPower, &onsetPower, &resultData, &audio};
            driver.check(driver.launch(audioSummary, (count + 31) / 32, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr),
                "CUDA audio assessment launch");
        }
        return synthesis;
    }

    void pack(Synthesis& work)
    {
        int count = static_cast<int>(work.jobs.size());
        void* args[] = {&work.jobData, &count, &work.resultData, &work.pcmJobs};
        driver.check(driver.launch(acceptedLayout, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
            "CUDA accepted PCM layout launch");
        launchChunks(acceptedPack, work.blocks, 128, work.pcmJobs, count, work.source, work.output, work.resultData);
    }

    void match(DevicePtr groups, int count, int base, DevicePtr results)
    {
        void* args[] = {&groups, &count, &base, &groupMinimum, &results};
        driver.check(driver.launch(volumeMatch, (count + 31) / 32, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr),
            "CUDA group volume launch");
    }

    std::vector<TailResult> readResults(const Synthesis& work)
    {
        std::vector<TailResult> results(work.jobs.size());
        downloadScratch(results.data(), work.resultData, results.size() * sizeof(TailResult), "CUDA render metadata download");
        return results;
    }

    size_t packedSamples(const std::vector<TailResult>& results) const
    {
        const auto& last = results.back();
        return (last.offset + last.dataFrames) * channels();
    }

    double timing(const TimingEvent& begin, const TimingEvent& end)
    {
        float milliseconds{};
        driver.check(driver.elapsedEvents(&milliseconds, begin.address, end.address), "CUDA kernel timing");
        return milliseconds / 1000.;
    }

    RenderBatch download(const Synthesis& work, DevicePtr pcm, const TimingEvent& begin, const TimingEvent& end)
    {
        auto results = readResults(work);
        const auto count = packedSamples(results);
        auto samples = receivePcm(pcm, count);
        releaseMappedScratch();
        downloaded += count * sizeof(int16_t);
        return {std::move(samples), std::move(results), timing(begin, end)};
    }

    RenderBatch render(const std::vector<RenderPlan>& plans, size_t first)
    {
        TimingEvent begin(driver), end(driver);
        auto work = synthesize(plans, first, plans.size(), begin);
        if (work.failedScratchBytes)
            return {{}, {}, 0, work.failedScratchBytes};
        const bool assess = audio.enabled || groupMinimum;
        if (assess)
            pack(work);
        end.record();
        return download(work, assess ? work.output : work.source, begin, end);
    }

    struct Retained {
        std::vector<TailJob> jobs;
        size_t first, bytes;
        int blocks;
        std::unique_ptr<Buffer> device;
        std::shared_ptr<int16_t> host;
    };

    Retained retain(Synthesis& work, size_t first, DevicePtr destination, const TimingEvent& end, bool packed = false)
    {
        if (!packed) pack(work);
        const auto resultBytes = work.jobs.size() * sizeof(TailResult);
        copyScratch(destination, work.resultData, resultBytes, "CUDA retain group metadata");
        end.record();
        TailResult last{};
        driver.check(driver.get(&last, destination + resultBytes - sizeof(last), sizeof(last)), "CUDA retained PCM size");
        const auto count = (last.offset + last.dataFrames) * channels();
        const auto bytes = count * sizeof(int16_t);
        Retained chunk{std::move(work.jobs), first, bytes, work.blocks, {}, {}};
        if (bytes) {
            size_t free{}, total{};
            driver.check(driver.memory(&free, &total), "CUDA waiting memory budget");
            const auto limit = std::min(waitingLimit, total / 16);
            if (waiting <= limit && bytes <= std::min(limit - waiting, free / 4)) {
                auto device = std::make_unique<Buffer>(driver);
                if (!driver.allocate(&device->address, bytes)) {
                    device->capacity = bytes;
                    waiting += bytes;
                    peakWaiting = std::max(peakWaiting, waiting);
                    recordPeak();
                    copyScratch(device->address, work.output, bytes, "CUDA retain group PCM");
                    chunk.device = std::move(device);
                }
            }
            if (!chunk.device) {
                chunk.host = receivePcm(work.output, count);
                spilled += bytes;
            }
        }
        releaseMappedScratch();
        return chunk;
    }

    RenderBatch packRetained(Retained& chunk, DevicePtr resultData, size_t index = 0, const TailResult* single = nullptr)
    {
        const auto count = single ? size_t{1} : chunk.jobs.size();
        const auto offset = single ? single->offset * channels() : 0;
        const auto bytes = single ? single->dataFrames * channels() * sizeof(int16_t) : chunk.bytes;
        ArenaLayout layout;
        const auto jobBytes = count * sizeof(TailJob);
        const auto jobOffset = layout.section(jobBytes);
        const auto pcmOffset = layout.section(count * sizeof(PcmJob));
        const auto sourceOffset = layout.section(chunk.host ? bytes : 0);
        const auto outputOffset = layout.section(bytes);
        const auto allocation = arena.tryAllocate(layout.bytes);
        if (allocation != cudaOutOfMemory)
            driver.check(allocation, "CUDA retained group scratch allocation");
        else if (count > 1)
            return {{}, {}, 0, layout.bytes};
        const auto base = allocation == cudaOutOfMemory ? mapScratch(layout.bytes) : arena.address;
        if (!base)
            return {{}, {}, 0, layout.bytes};
        recordPeak();
        Synthesis work{{}, base + jobOffset, resultData,
            chunk.device ? chunk.device->address + offset * sizeof(int16_t) : base + sourceOffset,
            base + outputOffset, base + pcmOffset, chunk.blocks};
        if (single) {
            work.jobs.push_back(chunk.jobs[index]);
            work.jobs.front().blockFirst = 0;
            work.blocks = int(blocksFor(work.jobs.front().frames, tailBlockFrames));
            auto rebased = *single;
            rebased.offset = 0;
            driver.check(driver.put(resultData, &rebased, sizeof(rebased)), "CUDA retained job metadata upload");
        } else {
            work.jobs = std::move(chunk.jobs);
        }
        uploadScratch(work.jobData, work.jobs.data(), jobBytes, "CUDA retained job upload");
        if (chunk.host && bytes)
            uploadScratch(work.source, chunk.host.get() + offset, bytes, "CUDA spilled PCM upload");
        TimingEvent begin(driver), end(driver);
        begin.record();
        pack(work);
        end.record();
        return download(work, work.output, begin, end);
    }

    size_t finishSplit(const std::vector<RenderPlan>& plans, const std::vector<size_t>& sizes, Synthesis work,
                       const AudioGroup& group, DevicePtr groupData, DevicePtr resultData, const TimingEvent& begin,
                       const std::function<void(RenderBatch, size_t)>& publish)
    {
        std::vector<Retained> chunks;
        TimingEvent end(driver);
        double synthSeconds = 0;
        for (size_t first = group.first;;) {
            const auto next = first + work.jobs.size();
            const auto destination = resultData + (first - group.first) * sizeof(TailResult);
            const bool last = next == size_t(group.end);
            if (work.failedScratchBytes) {
                TailResult rejected{};
                rejected.status = TailStatus::scratchMemory;
                driver.check(driver.put(destination, &rejected, sizeof(rejected)), "CUDA rejected group job upload");
                publish({{}, {}, 0, work.failedScratchBytes}, first);
                if (last && !chunks.empty())
                    match(groupData, 1, group.first, resultData);
            } else if (last) {
                copyScratch(destination, work.resultData, work.jobs.size() * sizeof(TailResult), "CUDA complete group metadata");
                work.resultData = destination;
                match(groupData, 1, group.first, resultData);
                pack(work);
                end.record();
                auto batch = download(work, work.output, begin, end);
                batch.kernelSeconds += synthSeconds;
                synthSeconds = 0;
                publish(std::move(batch), first);
            } else {
                chunks.push_back(retain(work, first, destination, end));
                synthSeconds += timing(begin, end);
            }
            if (last)
                break;
            first = next;
            work = synthesize(plans, first, group.end, begin, sizes);
        }
        for (auto& chunk : chunks) {
            const auto sourceResults = resultData + (chunk.first - group.first) * sizeof(TailResult);
            const auto publishChunk = [&](RenderBatch batch, size_t first) {
                batch.kernelSeconds += synthSeconds;
                synthSeconds = 0;
                publish(std::move(batch), first);
            };
            auto batch = packRetained(chunk, sourceResults);
            if (!batch.failedScratchBytes || chunk.jobs.size() == 1) {
                publishChunk(std::move(batch), chunk.first);
            } else {
                // Retry the retained PCM one job at a time, using the matched gain already computed.
                std::vector<TailResult> results(chunk.jobs.size());
                driver.check(driver.get(results.data(), sourceResults, results.size() * sizeof(TailResult)),
                    "CUDA retained job metadata download");
                for (size_t i = 0; i < results.size(); ++i) {
                    RenderBatch single;
                    if (results[i].status == TailStatus::ok) {
                        single = packRetained(chunk, sourceResults + i * sizeof(TailResult), i, &results[i]);
                    } else {
                        single.results.push_back(results[i]);
                    }
                    publishChunk(std::move(single), chunk.first + i);
                }
            }
            if (chunk.device) {
                waiting -= chunk.bytes;
                chunk.device.reset();
            }
            chunk.host.reset();
        }
        return group.end;
    }

    // Keep independent job state resident and fuse each scheduling round into four launches.
    // Completed jobs release their synthesis storage before the next queue admission.
    void renderRolling(std::vector<RenderPlan>& plans, const Refill& refill,
                       const std::function<void(RenderBatch, size_t)>& publish)
    {
        struct Resident {
            Buffer storage;
            Pending pending;
            size_t first;
            Resident(Driver& driver, size_t index) : storage(driver), first(index) {}
        };
        std::vector<std::unique_ptr<Resident>> active;
        Buffer scratch(driver), descriptors(driver);
        struct Group {
            AudioGroup range;
            int remaining;
            std::unique_ptr<Buffer> metadata;
            std::vector<Retained> retained;
        };
        std::vector<Group> groups;
        std::vector<int> groupOf;
        constexpr size_t resultOffset = 16;
        const auto registerJobs = [&] {
            const auto firstNew = groupOf.size();
            groupOf.resize(plans.size(), -1);
            for (size_t first = firstNew, end; first < plans.size(); first = end) {
                end = first + 1;
                const auto id = plans[first].volumeGroup;
                if (!id) continue;
                while (end < plans.size() && plans[end].volumeGroup == id) ++end;
                const auto g = int(groups.size());
                groups.push_back({{int(first), int(end)}, int(end - first), {}, {}});
                std::fill(groupOf.begin() + first, groupOf.begin() + end, g);
            }
        };
        registerJobs();
        arena.reset();
        double seconds = 0;
        size_t admitted = 0, rounds = 0, refills = 0;
        const auto account = [&] {
            residentBytes = scratch.capacity + descriptors.capacity;
            for (const auto& item : active)
                residentBytes += item->storage.capacity;
            for (const auto& group : groups)
                if (group.metadata) residentBytes += group.metadata->capacity;
            recordPeak();
        };
        const auto emit = [&](RenderBatch batch, size_t first) {
            batch.kernelSeconds += seconds;
            seconds = 0;
            publish(std::move(batch), first);
        };
        const auto complete = [&](Synthesis& work, size_t first, const TimingEvent& begin, bool packed = false) {
            TimingEvent end(driver);
            const auto g = groupOf[first];
            DevicePtr groupData = 0, resultData = 0, destination = 0;
            if (g >= 0) {
                auto& group = groups[g];
                if (!group.metadata) {
                    group.metadata = std::make_unique<Buffer>(driver);
                    group.metadata->allocate(resultOffset + (group.range.end - group.range.first) * sizeof(TailResult));
                    driver.check(driver.put(group.metadata->address, &group.range, sizeof(AudioGroup)), "CUDA rolling group");
                }
                groupData = group.metadata->address;
                resultData = groupData + resultOffset;
                destination = resultData + (first - group.range.first) * sizeof(TailResult);
            }
            if (work.failedScratchBytes) {
                if (g >= 0) {
                    TailResult rejected{};
                    rejected.status = TailStatus::scratchMemory;
                    driver.check(driver.put(destination, &rejected, sizeof(rejected)),
                        "CUDA rolling rejected metadata");
                }
                emit({{}, {}, 0, work.failedScratchBytes}, first);
            } else if (g >= 0) {
                groups[g].retained.push_back(retain(work, first, destination, end, packed));
                seconds += timing(begin, end);
            } else {
                const bool assess = audio.enabled || groupMinimum;
                if (assess && !packed) pack(work);
                end.record();
                emit(download(work, assess ? work.output : work.source, begin, end), first);
            }
            if (g < 0 || --groups[g].remaining)
                return;
            match(groupData, 1, groups[g].range.first, resultData);
            for (auto& chunk : groups[g].retained) {
                emit(packRetained(chunk, resultData + (chunk.first - groups[g].range.first) * sizeof(TailResult)), chunk.first);
                if (chunk.device) waiting -= chunk.bytes;
            }
            groups[g].retained.clear();
            groups[g].metadata.reset();
            // Group repacking must not reserve memory needed by newly admitted synthesis jobs.
            arena.reset();
        };
        for (;;) {
            while (active.size() < batchJobs) {
                if (admitted == plans.size()) {
                    if (!refill || !refill(plans, active.empty())) break;
                    registerJobs();
                }
                size_t free{}, total{};
                driver.check(driver.memory(&free, &total), "CUDA rolling admission budget");
                if (!active.empty() && footprint(plans[admitted]) > free / 2)
                    break;
                auto item = std::make_unique<Resident>(driver, admitted);
                TimingEvent begin(driver);
                size_t cap = 1, limit = std::numeric_limits<size_t>::max();
                // The ordinary arena is temporarily borrowed for preparation only.
                arena.swap(item->storage);
                auto work = trySynthesize(plans, admitted, admitted + 1, begin, {}, cap, limit, &item->pending);
                arena.swap(item->storage);
                if (work.jobs.empty()) {
                    if (!active.empty()) break;
                    // No live synthesis state is at risk: retain the established single-job recovery path.
                    scratch.reset();
                    descriptors.reset();
                    account();
                    auto fallback = synthesize(plans, admitted, admitted + 1, begin);
                    complete(fallback, admitted, begin);
                    plans[admitted] = {};
                    arena.reset();
                    ++admitted;
                    continue;
                }
                if (rounds && !active.empty()) ++refills;
                active.push_back(std::move(item));
                ++admitted;
                account();
            }
            if (active.empty()) {
                if (admitted == plans.size()) break;
                continue;
            }
            int shortest = std::numeric_limits<int>::max(), longest = 0;
            size_t tracks = 0;
            for (const auto& item : active) {
                const int remaining = plans[item->first].frames - item->pending.frame;
                shortest = std::min(shortest, remaining);
                longest = std::max(longest, remaining);
                const auto& index = item->pending.index;
                tracks += std::max(index.filterHeads.size(), index.pitchHeads.size());
            }
            int quantum = admitted < plans.size() || refill ? std::min(longest, std::max(shortest, audio.rate / 4)) : longest;
            // Preserve parallelism across independent notes before shortening for memory.
            // A single long voice cannot gain more parallel tracks merely by using a longer segment.
            const auto target = tracks;
            while (quantum < longest && target) {
                size_t parallel = 0;
                for (const auto& item : active) {
                    const auto& pending = item->pending;
                    const auto& index = pending.index;
                    const auto& heads = index.filterHeads.size() >= index.pitchHeads.size() ? index.filterHeads : index.pitchHeads;
                    const int end = pending.frame + std::min(quantum, plans[item->first].frames - pending.frame);
                    for (auto head : heads) {
                        if (parallel >= target) break;
                        for (int span = head; span >= 0; span = index.voices[span].nextFilteredSpan) {
                            const auto& voice = index.voices[span];
                            if (voice.begin >= end) break;
                            if (voice.end > pending.frame) { ++parallel; break; }
                        }
                    }
                }
                if (parallel >= target) break;
                quantum += std::min(quantum, longest - quantum);
            }
            size_t free{}, total{};
            driver.check(driver.memory(&free, &total), "CUDA rolling scratch budget");
            const auto available = free + scratch.capacity;
            size_t budget = std::min(total / 4, available - available / 8);
            std::vector<ScheduledJob> jobs;
            size_t bytes = 0;
            bool recover = false;
            for (;;) {
                jobs.clear();
                bytes = 0;
                for (auto& item : active) {
                    auto& pending = item->pending;
                    auto job = pending.device;
                    job.first = pending.frame;
                    job.end = job.first + std::min(quantum, plans[item->first].frames - job.first);
                    size_t samples = 0, phases = 0;
                    for (auto& voice : pending.index.voices) {
                        if (voice.filterTrack < 0) continue;
                        const auto frames = std::max(0, std::min(voice.end, job.end) - std::max(voice.begin, job.first));
                        voice.filteredOffset = samples;
                        voice.phaseOffset = phases;
                        samples += size_t(frames) * voice.channels;
                        if (voice.variablePitch) phases += frames;
                    }
                    job.filtered = reinterpret_cast<float*>(bytes);
                    bytes = (bytes + samples * sizeof(float) + 15) & ~size_t{15};
                    job.phase = reinterpret_cast<double*>(bytes);
                    bytes = (bytes + phases * sizeof(double) + 15) & ~size_t{15};
                    jobs.push_back(job);
                }
                if (bytes <= budget) {
                    const auto status = scratch.tryAllocate(bytes);
                    if (status != cudaOutOfMemory) {
                        driver.check(status, "CUDA rolling scratch allocation");
                        break;
                    }
                    budget = bytes / 2;
                }
                if (quantum == 1) {
                    recover = true;
                    break;
                }
                quantum = std::max(1, quantum / 2);
            }
            if (recover) {
                std::vector<size_t> retry;
                for (const auto& item : active) retry.push_back(item->first);
                active.clear();
                scratch.reset();
                descriptors.reset();
                account();
                for (auto first : retry) {
                    TimingEvent begin(driver);
                    auto work = synthesize(plans, first, first + 1, begin);
                    complete(work, first, begin);
                    plans[first] = {};
                    arena.reset();
                }
                continue;
            }
            int tiles = 0, sources = 0, pitches = 0, filters = 0;
            for (size_t i = 0; i < active.size(); ++i) {
                auto& pending = active[i]->pending;
                auto& job = jobs[i];
                job.filtered = reinterpret_cast<float*>(scratch.address + reinterpret_cast<size_t>(job.filtered));
                job.phase = reinterpret_cast<double*>(scratch.address + reinterpret_cast<size_t>(job.phase));
                const auto& index = pending.index;
                if (index.filteredSamples && (job.first || job.end < plans[active[i]->first].frames))
                    uploadScratch(reinterpret_cast<DevicePtr>(job.voices), index.voices.data(),
                        index.voices.size() * sizeof(VoiceSpan), "CUDA rolling voice offsets");
                const auto range = [&](const auto& items, int& first) {
                    auto start = std::lower_bound(items.begin(), items.end(), job.first - threads + 1,
                        [](const auto& tile, int frame) { return tile.frame < frame; });
                    auto stop = std::lower_bound(start, items.end(), job.end,
                        [](const auto& tile, int frame) { return tile.frame < frame; });
                    first = int(start - items.begin());
                    return int(stop - start);
                };
                const auto heads = [&](const auto& items, const int* destination) {
                    std::vector<int> selected;
                    for (auto head : items) {
                        for (int span = head; span >= 0; span = index.voices[span].nextFilteredSpan) {
                            const auto& voice = index.voices[span];
                            if (voice.begin >= job.end) break;
                            if (voice.end > job.first) { selected.push_back(head); break; }
                        }
                    }
                    if (!selected.empty())
                        uploadScratch(reinterpret_cast<DevicePtr>(destination), selected.data(), selected.size() * sizeof(int),
                            "CUDA rolling active tracks");
                    return int(selected.size());
                };
                job.tileEnd = (tiles += range(index.tiles, job.tileFirst));
                job.sourceEnd = (sources += range(index.filterTiles, job.sourceFirst));
                job.pitchEnd = (pitches += heads(index.pitchHeads, job.pitchHeads));
                job.filterEnd = (filters += heads(index.filterHeads, job.filterHeads));
            }
            descriptors.upload(jobs);
            account();
            auto deviceJobs = descriptors.address;
            int count = int(jobs.size());
            TimingEvent begin(driver), end(driver);
            begin.record();
            const auto recurrence = [&](Handle kernel, int tracks) {
                if (!tracks) return;
                const int width = std::clamp(1 + (tracks - 1) / (2 * multiprocessors), 1, 32);
                void* args[] = {&deviceJobs, &count, &tracks};
                driver.check(driver.launch(kernel, 1 + (tracks - 1) / width, 1, 1, width, 1, 1, 0,
                    nullptr, args, nullptr), "CUDA scheduled recurrence");
            };
            recurrence(scheduledPitch, pitches);
            launchChunks(scheduledSource, sources, threads, sampleData, sinc.address, deviceJobs, count);
            recurrence(scheduledFilter, filters);
            launchChunks(scheduledMix, tiles, threads, sampleData, sinc.address, deviceJobs, count);
            end.record();
            driver.check(driver.synchronize(), "CUDA rolling round completion");
            const auto elapsed = timing(begin, end);
            seconds += elapsed;
            ++rounds;
            std::vector<FinishJob> finishing;
            for (size_t i = 0; i < active.size(); ++i) {
                if (jobs[i].end != plans[active[i]->first].frames) continue;
                auto job = active[i]->pending.finishing;
                if (!finishing.empty()) {
                    job.blocks += finishing.back().blocks;
                    job.windowsEnd += finishing.back().windowsEnd;
                    job.bodyEnd += finishing.back().bodyEnd;
                    job.onsetEnd += finishing.back().onsetEnd;
                }
                finishing.push_back(job);
            }
            bool combined = !finishing.empty();
            size_t outputFrames = 0;
            for (size_t i = 0; i < active.size(); ++i) {
                if (jobs[i].end != plans[active[i]->first].frames) continue;
                combined &= groupOf[active[i]->first] < 0;
                outputFrames += plans[active[i]->first].frames;
            }
            if (combined) {
                const auto status = scratch.tryAllocate(outputFrames * channels() * sizeof(int16_t));
                if (status != cudaOutOfMemory) driver.check(status, "CUDA completion transfer allocation");
                else combined = false;
            }
            if (!finishing.empty()) {
                descriptors.allocate(finishing.size() * (sizeof(FinishJob) + sizeof(TailResult)));
                auto data = descriptors.address;
                const auto summary = data + finishing.size() * sizeof(FinishJob);
                size_t offset = 0, j = 0;
                for (size_t i = 0; i < active.size(); ++i) {
                    if (jobs[i].end != plans[active[i]->first].frames) continue;
                    auto& job = finishing[j];
                    job.summary = reinterpret_cast<TailResult*>(summary + j * sizeof(TailResult));
                    job.outputOffset = offset;
                    if (combined) {
                        job.output = reinterpret_cast<short*>(scratch.address + offset * channels() * sizeof(int16_t));
                        if (!audio.enabled && !groupMinimum) job.packed = job.output;
                    }
                    offset += plans[active[i]->first].frames;
                    ++j;
                }
                driver.check(driver.put(data, finishing.data(), finishing.size() * sizeof(FinishJob)), "CUDA finishing descriptors");
                account();
                int finished = int(finishing.size());
                TimingEvent finishBegin(driver), finishEnd(driver);
                finishBegin.record();
                const auto finishStage = [&](int stage, int blocks, int width) {
                    launchChunks(scheduledFinish, blocks, width, data, finished, stage, tail, audio);
                };
                const auto compact = [&] {
                    auto output = scratch.address;
                    void* args[] = {&data, &finished, &output};
                    driver.check(driver.launch(scheduledPackLayout, 1, 1, 1, 1, 1, 1, 0,
                        nullptr, args, nullptr), "CUDA compact completion layout");
                };
                const auto& last = finishing.back();
                if (natural) {
                    finishStage(0, last.blocks, 128);
                    finishStage(1, finished, 1);
                    finishStage(2, last.windowsEnd, 256);
                    finishStage(3, finished, 1);
                }
                finishStage(4, finished, 1);
                if (combined && !audio.enabled && !groupMinimum) compact();
                finishStage(5, last.blocks, 128);
                if (audio.enabled || groupMinimum) {
                    finishStage(6, last.bodyEnd, 128);
                    if (audio.enabled) finishStage(7, last.onsetEnd, 160);
                    finishStage(8, finished, 1);
                    finishStage(9, finished, 1);
                    if (combined) compact();
                    finishStage(10, last.blocks, 128);
                }
                if (combined) finishStage(11, finished, 1);
                finishEnd.record();
                driver.check(driver.synchronize(), "CUDA scheduled finishing completion");
                seconds += timing(finishBegin, finishEnd);
                if (combined) {
                    RenderBatch batch;
                    batch.results.resize(finishing.size());
                    driver.check(driver.get(batch.results.data(), summary, batch.results.size() * sizeof(TailResult)),
                        "CUDA completion metadata download");
                    size_t samplesCount = 0;
                    for (const auto& result : batch.results)
                        if (result.dataFrames) samplesCount = std::max(samplesCount, size_t(result.offset + result.dataFrames) * channels());
                    batch.pcm = receivePcm(scratch.address, samplesCount);
                    downloaded += samplesCount * sizeof(int16_t);
                    for (size_t i = 0; i < active.size(); ++i)
                        if (jobs[i].end == plans[active[i]->first].frames) batch.jobIndices.push_back(active[i]->first);
                    const auto first = batch.jobIndices.front();
                    emit(std::move(batch), first);
                }
            }
            for (size_t i = active.size(); i-- > 0;) {
                auto& item = *active[i];
                item.pending.frame = jobs[i].end;
                if (item.pending.frame != plans[item.first].frames) continue;
                if (!combined) {
                    TimingEvent finishBegin(driver);
                    finishBegin.record();
                    complete(item.pending.work, item.first, finishBegin, true);
                }
                plans[item.first] = {};
                active.erase(active.begin() + i);
            }
            account();
        }
        residentBytes = 0;
        std::cerr << "SCHEDULE\tjobs=" << plans.size() << "\trounds=" << rounds << "\trefills=" << refills << '\n';
    }

    void renderWindow(std::vector<RenderPlan>& plans, const std::function<void(RenderBatch, size_t)>& publish,
                      const Refill& refill)
    {
        std::vector<AudioGroup> groups;
        size_t largest = 0;
        for (size_t first = 0, end = 0; first < plans.size(); first = end) {
            const auto id = plans[first].volumeGroup;
            end = first + 1;
            if (!id)
                continue;
            while (end < plans.size() && plans[end].volumeGroup == id)
                ++end;
            groups.push_back({int(first), int(end)});
            largest = std::max(largest, end - first);
        }
        const bool recurrences = std::any_of(plans.begin(), plans.end(), [](const auto& plan) { return plan.filterTracks != 0; });
        if (batchJobs > 1 && (refill || plans.size() > batchJobs) && recurrences) {
            renderRolling(plans, refill, publish);
            return;
        }
        if (groups.empty()) {
            for (size_t first = 0; first < plans.size();) {
                auto batch = render(plans, first);
                const auto next = first + (batch.failedScratchBytes ? 1 : batch.results.size());
                publish(std::move(batch), first);
                first = next;
            }
            return;
        }
        std::vector<size_t> sizes;
        sizes.reserve(plans.size());
        for (const auto& plan : plans)
            sizes.push_back(footprint(plan));
        ArenaLayout control;
        const auto groupOffset = control.section(groups.size() * sizeof(AudioGroup));
        const auto resultOffset = control.section(largest * sizeof(TailResult));
        const auto allocation = groupControl.tryAllocate(control.bytes);
        if (allocation == cudaOutOfMemory) {
            arena.reset();
            groupControl.allocate(control.bytes);
        } else {
            driver.check(allocation, "CUDA group control allocation");
        }
        recordPeak();
        const auto groupData = groupControl.address + groupOffset;
        const auto resultData = groupControl.address + resultOffset;
        driver.check(driver.put(groupData, groups.data(), groups.size() * sizeof(AudioGroup)), "CUDA group upload");
        size_t group = 0;
        for (size_t first = 0; first < plans.size();) {
            TimingEvent begin(driver), end(driver);
            auto work = synthesize(plans, first, plans.size(), begin, sizes);
            const auto next = first + work.jobs.size();
            const auto groupStart = group;
            while (group < groups.size() && size_t(groups[group].end) <= next)
                ++group;
            if (group < groups.size() && size_t(groups[group].first) < next) {
                first = finishSplit(plans, sizes, std::move(work), groups[group], groupData + group * sizeof(AudioGroup),
                    resultData, begin, publish);
                ++group;
                continue;
            }
            if (work.failedScratchBytes) {
                publish({{}, {}, 0, work.failedScratchBytes}, first);
            } else {
                if (group != groupStart)
                    match(groupData + groupStart * sizeof(AudioGroup), int(group - groupStart), int(first), work.resultData);
                pack(work);
                end.record();
                publish(download(work, work.output, begin, end), first);
            }
            first = next;
        }
    }
};

Renderer::Renderer(int device, const Path& nvrtc, bool natural, AudioConfig audio, int groupMinimum,
                   size_t waitingLimit, size_t batchJobs)
    : impl_(std::make_unique<Impl>(device, nvrtc, natural, audio, groupMinimum, waitingLimit, batchJobs))
{
}

Renderer::~Renderer() = default;

void Renderer::uploadSamples(const AudioBank& bank, bool fresh)
{
    impl_->uploadSamples(bank, fresh);
}

void Renderer::renderWindow(std::vector<RenderPlan>& plans, const std::function<void(RenderBatch, size_t)>& publish,
                            const Refill& refill)
{
    impl_->renderWindow(plans, publish, refill);
}
const std::string& Renderer::device() const { return impl_->deviceName; }
size_t Renderer::peakBytes() const { return impl_->peak; }
size_t Renderer::sampleCacheBudget() const { return impl_->cacheBudget; }
size_t Renderer::peakWaitingBytes() const { return impl_->peakWaiting; }
size_t Renderer::spilledBytes() const { return impl_->spilled; }
size_t Renderer::downloadedBytes() const { return impl_->downloaded; }

} // namespace clay
