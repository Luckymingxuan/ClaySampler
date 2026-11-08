
struct Sum {
    template<class T> __device__ T operator()(T a, T b) const { return a + b; }
};
struct Maximum {
    template<class T> __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
struct Minimum {
    __device__ int operator()(int a, int b) const { return min(a, b); }
};

// The tail kernels use a fixed 128 or 256 threads per block.
template<class T, class Operation>
__device__ T tailReduce(T value, T identity, Operation operation)
{
    __shared__ T warps[8];
    const auto lane = threadIdx.x & 31;
    for (int distance = 16; distance; distance >>= 1)
        value = operation(value, __shfl_down_sync(0xffffffff, value, distance));
    if (!lane)
        warps[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = lane < blockDim.x / 32 ? warps[lane] : identity;
        for (int distance = 16; distance; distance >>= 1)
            value = operation(value, __shfl_down_sync(0xffffffff, value, distance));
    }
    __syncthreads();
    return value;
}

template<int TailJob::*First>
__device__ int tailJobFor(const TailJob* jobs, int count, int block)
{
    int first = 0;
    while (first + 1 < count) {
        const int middle = (first + count) / 2;
        if (jobs[middle].*First <= block)
            first = middle;
        else
            count = middle;
    }
    return first;
}

__device__ short fadeSample(short value, int index, int count)
{
    constexpr double pi = 3.14159265358979323846;
    const double gain = count == 1 ? 0 : .5 * (1 + cos(pi * index / (count - 1)));
    return static_cast<short>(__double2int_rn(value * gain));
}

#if CLAY_FLOAT_OUTPUT
extern "C" __global__ void scanStereo(const TailJob* jobs, int count, const float2* mix,
                                     StereoStats* stats, int firstBlock)
{
    const int block = firstBlock + blockIdx.x;
    const auto job = jobs[tailJobFor<&TailJob::blockFirst>(jobs, count, block)];
    const int first = (block - job.blockFirst) * tailBlockFrames;
    const int end = min(first + tailBlockFrames, job.frames);
    double power = 0;
    float peak = 0;
    for (int frame = first + threadIdx.x; frame < end; frame += blockDim.x) {
        const auto value = mix[job.offset + frame];
        if (!isfinite(value.x) || !isfinite(value.y)) {
            peak = __int_as_float(0x7f800000);
            continue;
        }
        peak = fmaxf(peak, fmaxf(fabsf(value.x), fabsf(value.y)));
        power += double(__fmul_rn(value.x, value.x)) + double(__fmul_rn(value.y, value.y));
    }
    power = tailReduce(power, 0., Sum{});
    peak = tailReduce(peak, 0.f, Maximum{});
    if (!threadIdx.x)
        stats[block] = {power, peak};
}

extern "C" __global__ void selectNative(const TailJob* jobs, int count, const float2* mix,
                                       const StereoStats* stats, TailResult* results)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const auto job = jobs[index];
    TailResult result{};
    float peak = 0;
    int end = 0;
    for (int first = 0; first < job.frames; first += tailBlockFrames) {
        const auto stat = stats[job.blockFirst + first / tailBlockFrames];
        if (!isfinite(stat.peak)) {
            result.status = TailStatus::nonfinite;
            results[index] = result;
            return;
        }
        peak = fmaxf(peak, stat.peak);
        const int blockEnd = min(job.limit, first + tailBlockFrames);
        if (blockEnd < job.musicEnd)
            continue;
        const double threshold = 1e-12 * (blockEnd - first) * 2;
        double power = stat.power;
        // Preserve the host's summation order when roundoff could change a stop decision.
        if (fabs(power - threshold) <= threshold * 1e-12) {
            power = 0;
            for (int frame = first; frame < min(blockEnd, job.frames); ++frame) {
                const auto value = mix[job.offset + frame];
                power += __fmul_rn(value.x, value.x);
                power += __fmul_rn(value.y, value.y);
            }
        }
        if (power <= threshold) {
            end = blockEnd;
            break;
        }
    }
    if (!end) {
        // The mix beyond the last planned frame is silent, including long MIDI gaps.
        const int silent = (job.frames + tailBlockFrames - 1) / tailBlockFrames * tailBlockFrames;
        if (silent < job.limit)
            end = min(job.limit, max(silent + tailBlockFrames,
                (job.musicEnd + tailBlockFrames - 1) / tailBlockFrames * tailBlockFrames));
    }
    if (!end) {
        result.status = TailStatus::duration;
    } else {
        const double attenuation = peak >= 1 ? 20 * log10(double(peak)) + 1 : 0;
        if (attenuation > 5 + 1e-6) {
            result.status = TailStatus::peak;
        } else {
            result.frames = end;
            result.dataFrames = min(end, job.frames);
            result.scale = float(32768 * pow(10., -attenuation / 20));
        }
    }
    results[index] = result;
}

extern "C" __global__ void quantizeMono(const TailJob* jobs, int count, const float2* mix,
                                       const TailResult* results, short* mono, MonoStats* stats,
                                       int window, int firstBlock)
{
    const int block = firstBlock + blockIdx.x;
    const int index = tailJobFor<&TailJob::windowFirst>(jobs, count, block);
    const auto job = jobs[index];
    const auto result = results[index];
    const int first = (block - job.windowFirst) * window;
    const int end = min(first + window, result.dataFrames);
    if (first >= end)
        return;
    double power = 0;
    int peak = 0;
    for (int frame = first + threadIdx.x; frame < end; frame += blockDim.x) {
        const auto sample = mix[job.offset + frame];
        const int left = __float2int_rn(fminf(32767, fmaxf(-32768, sample.x * result.scale)));
        const int right = __float2int_rn(fminf(32767, fmaxf(-32768, sample.y * result.scale)));
        const int sum = left + right;
        const int value = (sum >> 1) + ((sum & 1) & ((sum >> 1) & 1));
        mono[job.offset + frame] = static_cast<short>(value);
        power += double(value) * value;
        peak = max(peak, abs(value));
    }
    power = tailReduce(power, 0., Sum{});
    peak = tailReduce(peak, 0, Maximum{});
    if (!threadIdx.x)
        stats[block] = {power, peak};
}

extern "C" __global__ void selectTrim(const TailJob* jobs, int count, const MonoStats* stats,
                                     TailResult* results, TailConfig config)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || results[index].status != TailStatus::ok)
        return;
    auto& result = results[index];
    const auto* windows = stats + jobs[index].windowFirst;
    const int countWindows = (result.dataFrames + config.window - 1) / config.window;
    double maxPower = 0;
    for (int i = 0; i < countWindows; ++i)
        maxPower = fmax(maxPower, windows[i].power);
    const auto threshold = fmin(config.ceiling, fmax(config.floor,
        sqrt(maxPower / config.window) * config.relative));
    for (int i = countWindows - 1; i >= 0; --i) {
        if (sqrt(windows[i].power / config.window) > threshold || windows[i].peak > threshold * 4) {
            const int lastSound = min(result.frames, (i + 1) * config.window);
            if (result.frames - lastSound >= config.quietFrames) {
                result.frames = lastSound + config.padding;
                result.dataFrames = min(result.frames, result.dataFrames);
                result.fade = min(result.frames, config.fade);
            }
            break;
        }
    }
}
#endif

extern "C" __global__ void layoutPcm(const TailJob* jobs, int count, TailResult* results)
{
    unsigned long long offset = 0;
    for (int i = 0; i < count; ++i) {
        auto& result = results[i];
#if !CLAY_FLOAT_OUTPUT
        result = {};
        result.frames = jobs[i].frames;
        result.dataFrames = jobs[i].frames;
        result.fade = jobs[i].fade;
#endif
        result.offset = offset;
        result.peak = 0;
        result.first = result.frames;
        result.last = -1;
        offset += result.dataFrames;
    }
}

extern "C" __global__ void finalizePcm(const TailJob* jobs, int count, const short* input,
                                      short* output, TailResult* results, int firstBlock)
{
    const int block = firstBlock + blockIdx.x;
    const int index = tailJobFor<&TailJob::blockFirst>(jobs, count, block);
    const auto job = jobs[index];
    const auto outputOffset = results[index].offset;
    const int frames = results[index].frames;
    const int fade = results[index].fade;
    const int first = (block - job.blockFirst) * tailBlockFrames;
    const int end = min(first + tailBlockFrames, results[index].dataFrames);
    if (first >= end)
        return;
    constexpr int channels = CLAY_FLOAT_OUTPUT ? 1 : 2;
    int peak = 0, audibleFirst = frames, audibleLast = -1;
    const int fadeStart = frames - fade;
    for (int frame = first + threadIdx.x; frame < end; frame += blockDim.x) {
        int framePeak = 0;
        for (int channel = 0; channel < channels; ++channel) {
            const auto source = (job.offset + frame) * channels + channel;
            auto value = input[source];
            if (frame >= fadeStart)
                value = fadeSample(value, frame - fadeStart, fade);
            if (CLAY_FLOAT_OUTPUT || frame >= fadeStart)
                output[(outputOffset + frame) * channels + channel] = value;
            framePeak = max(framePeak, abs(int(value)));
        }
        peak = max(peak, framePeak);
        if (framePeak > 3) {
            audibleFirst = min(audibleFirst, frame);
            audibleLast = frame;
        }
    }
    peak = tailReduce(peak, 0, Maximum{});
    audibleFirst = tailReduce(audibleFirst, frames, Minimum{});
    audibleLast = tailReduce(audibleLast, -1, Maximum{});
    if (!threadIdx.x && audibleLast >= 0) {
        atomicMax(&results[index].peak, peak);
        atomicMin(&results[index].first, audibleFirst);
        atomicMax(&results[index].last, audibleLast);
    }
}
