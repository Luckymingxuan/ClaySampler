#if CLAY_AUDIO_QA || CLAY_GROUP_VOLUME

constexpr int pcmChannels = CLAY_FLOAT_OUTPUT ? 1 : 2;
using Power = unsigned long long;

__device__ Power monoPower(const short* pcm, int frame)
{
    const int value = pcmChannels == 1 ? 2 * int(pcm[frame])
        : int(pcm[2 * frame]) + int(pcm[2 * frame + 1]);
    return Power(static_cast<long long>(value) * value);
}

__device__ int bodyBlockStart(int block, AudioConfig config)
{
    const int chunk = 2 * config.rate;
    const int span = 5 * config.window;
    const int windows = (chunk + span - 1) / span;
    return (block / windows) * chunk + (block % windows) * span;
}

extern "C" __global__ void measureBody(const AudioJob* jobs, int count, const short* pcm,
                                       const TailResult* results, Power* powers, AudioConfig config, int firstBlock)
{
    const int block = firstBlock + blockIdx.x;
    const int index = tailJobFor<&AudioJob::bodyFirst>(jobs, count, block);
    const auto& result = results[index];
    if (result.status != TailStatus::ok)
        return;
    const auto job = jobs[index];
    const int window = block - job.bodyFirst;
    const int first = bodyBlockStart(window, config);
    const int end = min(min(job.bodyFrames, result.dataFrames), bodyBlockStart(window + 1, config));
    const auto* audio = pcm + result.offset * pcmChannels;
    Power power = 0;
    for (int frame = first + threadIdx.x; frame < end; frame += blockDim.x)
        power += monoPower(audio, frame);
    power = tailReduce(power, Power{0}, Sum{});
    if (!threadIdx.x)
        powers[block] = power;
}

#if CLAY_AUDIO_QA
extern "C" __global__ void measureOnsets(const AudioJob* jobs, int count, const int* onsets, const short* pcm,
                                         const TailResult* results, Power* powers, int window, int firstBlock)
{
    const int block = firstBlock + blockIdx.x;
    const int index = tailJobFor<&AudioJob::onsetFirst>(jobs, count, block);
    const auto& result = results[index];
    if (result.status != TailStatus::ok)
        return;
    // One warp per 50 ms window. Missing output frames contribute zero.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int first = onsets[block] + warp * window;
    const int end = min(first + window, result.dataFrames);
    const auto* audio = pcm + result.offset * pcmChannels;
    Power power = 0;
    for (int frame = first + lane; frame < end; frame += 32)
        power += monoPower(audio, frame);
    for (int distance = 16; distance; distance >>= 1)
        power += __shfl_down_sync(0xffffffff, power, distance);
    __shared__ Power maxima[5];
    if (!lane)
        maxima[warp] = power;
    __syncthreads();
    if (!threadIdx.x)
        powers[block] = max(max(maxima[0], maxima[1]), max(max(maxima[2], maxima[3]), maxima[4]));
}

__device__ void siftPower(Power* values, int root, int count)
{
    const auto value = values[root];
    for (int child = 2 * root + 1; child < count; child = 2 * root + 1) {
        if (child + 1 < count && values[child] < values[child + 1])
            ++child;
        if (value >= values[child])
            break;
        values[root] = values[child];
        root = child;
    }
    values[root] = value;
}

__device__ double level(Power power, int window)
{
    return power ? fmax(-120., 10 * log10(double(power) / (window * 4294967296.))) : -120.;
}

__device__ double percentile(const Power* values, int count, double fraction, int window)
{
    const double position = (count - 1) * fraction;
    const int first = int(position);
    const auto lower = level(values[first], window);
    return first + 1 < count ? lower + (level(values[first + 1], window) - lower) * (position - first) : lower;
}
#endif

extern "C" __global__ void summarizeAudio(const AudioJob* jobs, int count, const Power* body, Power* onsets,
                                          TailResult* results, AudioConfig config)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || results[index].status != TailStatus::ok)
        return;
    const auto job = jobs[index];
    auto& result = results[index];
    auto& stats = result.audio;
    const int end = min(result.frames, job.bodyFrames);
    Power power = 0;
    int silent = 0, run = 0;
    for (int window = 0, first = 0; first < end; ++window) {
        const int next = min(end, bodyBlockStart(window + 1, config));
        const auto windowPower = body[job.bodyFirst + window];
        power += windowPower;
        if (config.enabled) {
            if (double(windowPower) < (next - first) * config.silencePower) {
                silent += next - first;
                run += next - first;
                stats.longestSilence = max(run, stats.longestSilence);
            } else {
                run = 0;
            }
        }
        first = next;
    }
    stats.bodyRms = job.bodyFrames ? sqrt(double(power) / job.bodyFrames) * .5 : 0;
#if CLAY_AUDIO_QA
    stats.measured = true;
    stats.median = stats.quartile = -120;
    stats.silentFraction = double(silent) / max(1, end);
    stats.clippedFraction = double(stats.clipped) / (double(max(1, result.frames)) * pcmChannels);
    if (job.onsetCount) {
        auto* levels = onsets + job.onsetFirst;
        for (int root = job.onsetCount / 2; root-- > 0;)
            siftPower(levels, root, job.onsetCount);
        for (int last = job.onsetCount - 1; last > 0; --last) {
            const auto top = levels[0];
            levels[0] = levels[last];
            levels[last] = top;
            siftPower(levels, 0, last);
        }
        stats.median = percentile(levels, job.onsetCount, .5, config.window);
        stats.quartile = percentile(levels, job.onsetCount, .25, config.window);
    }
    if (result.frames > config.maxFrames)
        result.status = TailStatus::audioDuration;
    else if (!job.onsetCount || stats.median < config.minMedian || stats.quartile < config.minQuartile)
        result.status = TailStatus::quietPerformance;
    else if (stats.silentFraction > config.maxSilentFraction || stats.longestSilence > config.maxSilentFrames)
        result.status = TailStatus::excessiveSilence;
    else if (result.peak < config.minPeak)
        result.status = TailStatus::silentAudio;
    else if (stats.clippedFraction > config.maxClippedFraction)
        result.status = TailStatus::clipping;
    stats.verdict = result.status;
#endif
}

#if CLAY_GROUP_VOLUME
extern "C" __global__ void matchVolume(const AudioGroup* groups, int count, int base, int minimum, TailResult* results)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const auto bounds = groups[index];
    const AudioGroup group{bounds.first - base, bounds.end - base};
    double desired = 0, safe = 1e30;
    int accepted = 0;
    for (int i = group.first; i < group.end; ++i) {
        auto& result = results[i];
        if (result.status != TailStatus::ok)
            continue;
        const double rms = result.audio.bodyRms;
        if (rms <= 0 || result.last < 0) {
            result.status = TailStatus::silentBody;
            continue;
        }
        desired += rms;
        constexpr double ceiling = 32767 * .8912509381337456; // -1 dBFS
        safe = fmin(safe, ceiling * rms / max(1, result.peak));
        ++accepted;
    }
    const double target = fmin(desired / max(1, accepted), safe);
    for (int i = group.first; i < group.end; ++i) {
        auto& result = results[i];
        if (result.status != TailStatus::ok)
            continue;
        if (accepted < minimum) {
            result.status = TailStatus::incompleteGroup;
        } else {
            result.audio.targetRms = target;
            result.audio.gain = target / result.audio.bodyRms;
        }
    }
}
#endif

extern "C" __global__ void layoutAccepted(const TailJob* jobs, int count, TailResult* results, PcmJob* pcm)
{
    unsigned long long offset = 0;
    for (int i = 0; i < count; ++i) {
        auto& result = results[i];
        pcm[i] = {result.offset, jobs[i].blockFirst};
        if (result.status != TailStatus::ok || result.last < 0)
            result.dataFrames = 0;
        result.offset = offset;
        offset += result.dataFrames;
        if (result.audio.gain > 0) {
            result.peak = 0;
            result.first = result.frames;
            result.last = -1;
        }
    }
}

extern "C" __global__ void packAccepted(const PcmJob* jobs, int count, const short* source, short* output,
                                        TailResult* results, int firstBlock)
{
    const int block = firstBlock + blockIdx.x;
    const int index = tailJobFor<&PcmJob::blockFirst>(jobs, count, block);
    const auto job = jobs[index];
    const auto& result = results[index];
    const int first = (block - job.blockFirst) * tailBlockFrames;
    const int end = min(first + tailBlockFrames, result.dataFrames);
    if (first >= end)
        return;
    const double gain = result.audio.gain;
    const auto* input = source + job.sourceOffset * pcmChannels;
    auto* pcm = output + result.offset * pcmChannels;
    int peak = 0, audibleFirst = result.frames, audibleLast = -1;
    for (int frame = first + threadIdx.x; frame < end; frame += blockDim.x) {
        int framePeak = 0;
        for (int channel = 0; channel < pcmChannels; ++channel) {
            const auto sample = frame * pcmChannels + channel;
            const short value = gain > 0 ? __double2int_rn(input[sample] * gain) : input[sample];
            pcm[sample] = value;
            framePeak = max(framePeak, abs(int(value)));
        }
        peak = max(peak, framePeak);
        if (framePeak > 3) {
            audibleFirst = min(audibleFirst, frame);
            audibleLast = frame;
        }
    }
    if (gain > 0) {
        peak = tailReduce(peak, 0, Maximum{});
        audibleFirst = tailReduce(audibleFirst, result.frames, Minimum{});
        audibleLast = tailReduce(audibleLast, -1, Maximum{});
        if (!threadIdx.x) {
            atomicMax(&results[index].peak, peak);
            if (audibleLast >= 0) {
                atomicMin(&results[index].first, audibleFirst);
                atomicMax(&results[index].last, audibleLast);
            }
        }
    }
}

#endif
