#include "renderer.h"
#include "../io/library.h"
#include "kernel_source.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
    ~Buffer() { if (address) driver_.free(address); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void allocate(size_t bytes)
    {
        driver_.check(driver_.allocate(&address, std::max(bytes, size_t{1})), "CUDA allocation");
    }

    template<class T> void upload(const std::vector<T>& data)
    {
        auto bytes = data.size() * sizeof(T);
        allocate(bytes);
        if (bytes)
            driver_.check(driver_.put(address, data.data(), bytes), "CUDA upload");
    }

    DevicePtr address{};

private:
    Driver& driver_;
};

std::vector<char> compile(const Path& library, int architecture)
{
    Library rtc(library);
#ifdef _WIN32
    std::unique_ptr<Library> builtins;
    if (library.has_parent_path()) {
        auto version = rtc.get<int(*)(int*, int*)>("nvrtcVersion");
        int major{}, minor{};
        if (!version(&major, &minor)) {
            auto filename = "nvrtc-builtins64_" + std::to_string(major) + std::to_string(minor) + ".dll";
            auto path = std::filesystem::absolute(library).parent_path() / filename;
            if (std::filesystem::exists(path))
                builtins = std::make_unique<Library>(path);
        }
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
    const char* flags[] = {archFlag.c_str(), "--std=c++17"};
    if (run(program, 2, flags)) {
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
    size_t frames = 0;

    void append(const RenderPlan& plan, int tileFrames)
    {
        const auto count = (plan.frames + tileFrames - 1) / tileFrames;
        std::vector<int> offsets(count + 1);
        for (const auto& voice : plan.spans) {
            ++offsets[voice.begin / tileFrames];
            --offsets[(voice.end - 1) / tileFrames + 1];
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        int controlOffsets[4] = {static_cast<int>(controls.size())};
        for (size_t i = 0; i < plan.controls.size(); ++i) {
            const auto& track = plan.controls[i];
            controls.insert(controls.end(), track.begin(), track.end());
            controlOffsets[i + 1] = static_cast<int>(controls.size());
        }
        int indexCount = static_cast<int>(indices.size());
        for (int t = 0; t < count; ++t) {
            auto frame = t * tileFrames;
            Tile tile{frames + frame, frame, std::min(tileFrames, plan.frames - frame), indexCount, offsets[t], {}};
            std::copy(std::begin(controlOffsets), std::end(controlOffsets), tile.controls);
            tiles.push_back(tile);
            indexCount += offsets[t];
            offsets[t] = tile.first;
        }
        indices.resize(indexCount);
        for (const auto& voice : plan.spans) {
            const auto index = static_cast<int>(voices.size());
            for (int t = voice.begin / tileFrames; t <= (voice.end - 1) / tileFrames; ++t)
                indices[offsets[t]++] = index;
            voices.push_back(voice);
        }
        frames += plan.frames;
    }

    size_t bytes() const
    {
        return voices.size() * sizeof(VoiceSpan) + tiles.size() * sizeof(Tile) +
            indices.size() * sizeof(int) + controls.size() * sizeof(ControlPoint) + frames * 2 * sizeof(int16_t);
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
    Buffer samples{driver}, sinc{driver};
    Handle module{}, kernel{};
    std::string deviceName;
    int threads{};
    size_t resident{}, peak{};

    Impl(int ordinal, const Path& rtc)
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
        auto code = compile(rtc, 10 * major + minor);
        driver.check(driver.load(&module, code.data()), "CUDA module");
        driver.check(driver.kernel(&kernel, module, "renderTiles"), "CUDA render kernel");
        int minBlocks{};
        driver.check(driver.occupancy(&minBlocks, &threads, kernel, nullptr, 0, 256), "CUDA launch sizing");
        auto table = sincTable();
        sinc.upload(table);
        resident = table.size() * sizeof(float);
        peak = resident;
    }

    RenderBatch render(const std::vector<RenderPlan>& plans, size_t first)
    {
        size_t free{}, total{};
        driver.check(driver.memory(&free, &total), "CUDA free memory");
        // Leave headroom for other CUDA users and cap host PCM staging on large-memory GPUs.
        const size_t budget = std::min(free / 2, size_t{512} * 1024 * 1024);
        size_t count = 0;
        size_t estimated = 0;
        BatchIndex index;
        for (size_t job = first; job < plans.size(); ++job) {
            const auto& plan = plans[job];
            size_t tiles = (size_t(plan.frames) + threads - 1) / threads;
            size_t bytes = size_t(plan.frames) * 4 + tiles * sizeof(Tile) + plan.spans.size() * sizeof(VoiceSpan);
            for (const auto& span : plan.spans)
                bytes += size_t((span.end - 1) / threads - span.begin / threads + 1) * sizeof(int);
            for (const auto& track : plan.controls)
                bytes += track.size() * sizeof(ControlPoint);
            if (bytes > budget - estimated)
                break;
            index.append(plan, threads);
            estimated += bytes;
            ++count;
        }
        if (!count)
            throw std::runtime_error("one render job exceeds the CUDA batch memory budget");
        peak = std::max(peak, resident + index.bytes());
        Buffer voices(driver), tiles(driver), indices(driver), controls(driver), output(driver);
        voices.upload(index.voices);
        tiles.upload(index.tiles);
        indices.upload(index.indices);
        controls.upload(index.controls);
        output.allocate(index.frames * 4);
        TimingEvent begin(driver), end(driver);
        begin.record();
        const int countTiles = static_cast<int>(index.tiles.size());
        constexpr int launchTiles = 16384;
        for (int offset = 0; offset < countTiles; offset += launchTiles) {
            void* args[] = {&samples.address, &sinc.address, &voices.address, &tiles.address,
                            &indices.address, &controls.address, &output.address, &offset};
            driver.check(driver.launch(kernel, std::min(launchTiles, countTiles - offset), 1, 1,
                threads, 1, 1, 0, nullptr, args, nullptr), "CUDA render launch");
        }
        end.record();
        RenderBatch batch{std::unique_ptr<int16_t[]>(new int16_t[index.frames * 2]), count, 0};
        driver.check(driver.get(batch.pcm.get(), output.address, index.frames * 4), "CUDA audio download");
        float milliseconds{};
        driver.check(driver.elapsedEvents(&milliseconds, begin.address, end.address), "CUDA kernel timing");
        batch.kernelSeconds = milliseconds / 1000.;
        return batch;
    }
};

Renderer::Renderer(int device, const Path& nvrtc)
    : impl_(std::make_unique<Impl>(device, nvrtc))
{
}

Renderer::~Renderer() = default;

void Renderer::uploadSamples(const AudioBank& bank)
{
    impl_->samples.upload(bank.pcm);
    impl_->resident += bank.pcm.size() * sizeof(float);
    impl_->peak = impl_->resident;
}

RenderBatch Renderer::render(const std::vector<RenderPlan>& plans, size_t first) { return impl_->render(plans, first); }
const std::string& Renderer::device() const { return impl_->deviceName; }
size_t Renderer::peakBytes() const { return impl_->peak; }

} // namespace clay
