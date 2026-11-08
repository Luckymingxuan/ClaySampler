#include "tail.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace clay {
namespace {

int16_t quantize(double value)
{
    return static_cast<int16_t>(std::nearbyint(std::clamp(value, -32768., 32767.)));
}

} // namespace

void fadeTail(int16_t* pcm, int frames, int channels, int count)
{
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < count; ++i) {
        const double gain = count == 1 ? 0 : .5 * (1 + std::cos(pi * i / (count - 1)));
        auto* frame = pcm + size_t(frames - count + i) * channels;
        for (int c = 0; c < channels; ++c)
            frame[c] = quantize(frame[c] * gain);
    }
}

Quality writeNaturalTail(const std::filesystem::path& path, const float* stereo, int frames,
                         int musicEnd, int limit, int rate)
{
    if (limit < musicEnd)
        throw std::runtime_error("natural render exceeds the audio duration limit");
    // Analyze only generated samples. Beyond the last planned voice, the mix is zero.
    int nativeEnd = 0;
    float peak = 0;
    for (int start = 0; start < limit;) {
        const int end = std::min(start + naturalTailBlock, limit);
        if (start >= frames && end < musicEnd) {
            start = (musicEnd - 1) / naturalTailBlock * naturalTailBlock;
            continue;
        }
        double power = 0;
        for (size_t i = size_t(start) * 2, last = size_t(std::min(frames, end)) * 2; i < last; ++i) {
            const float value = stereo[i];
            if (!std::isfinite(value))
                throw std::runtime_error("natural render produced non-finite audio");
            peak = std::max(peak, std::abs(value));
            power += value * value;
        }
        if (end >= musicEnd && power <= 1e-12 * (end - start) * 2) {
            nativeEnd = end;
            break;
        }
        start = end;
    }
    if (!nativeEnd)
        throw std::runtime_error("natural tail exceeded the tail or audio duration limit");

    const double attenuation = peak >= 1 ? 20 * std::log10(double(peak)) + 1 : 0;
    if (attenuation > 5 + 1e-6)
        throw std::runtime_error("excessive_peak: natural render requires more than 5 dB attenuation");
    const float scale = static_cast<float>(32768 * std::pow(10., -attenuation / 20));
    const int window = static_cast<int>(std::nearbyint(rate / 20.));
    struct Window { double rms; int peak; };
    std::vector<Window> windows;
    const int generated = std::min(frames, nativeEnd);
    windows.reserve((generated + window - 1) / window);
    std::vector<int16_t> mono(generated);
    double maxRms = 0;
    for (int start = 0; start < generated; start += window) {
        double power = 0;
        int windowPeak = 0;
        for (int i = start, end = std::min(start + window, generated); i < end; ++i) {
            const auto left = quantize(stereo[size_t(i) * 2] * scale);
            const auto right = quantize(stereo[size_t(i) * 2 + 1] * scale);
            // Average PCM16 with ties-to-even, using exact integer arithmetic.
            const int sum = int(left) + int(right);
            const int half = sum / 2;
            const int value = half + ((sum % 2 && half % 2) ? (sum > 0 ? 1 : -1) : 0);
            mono[i] = value;
            power += double(value) * value;
            windowPeak = std::max(windowPeak, std::abs(int(value)));
        }
        const auto rms = std::sqrt(power / window);
        windows.push_back({rms, windowPeak});
        maxRms = std::max(maxRms, rms);
    }
    const double threshold = std::clamp(maxRms * std::pow(10., -1.5),
        32768 * std::pow(10., -3.9), 32768 * std::pow(10., -3.));
    const auto active = std::find_if(windows.rbegin(), windows.rend(), [&](const Window& stats) {
        return stats.rms > threshold || stats.peak > threshold * 4;
    });
    int accepted = nativeEnd;
    if (active != windows.rend()) {
        const int lastSound = std::min(nativeEnd, int(active.base() - windows.begin()) * window);
        if (nativeEnd - lastSound >= 2 * rate) {
            accepted = lastSound + static_cast<int>(std::nearbyint(rate / 4.));
        }
    }
    mono.resize(accepted);
    if (accepted < nativeEnd)
        fadeTail(mono.data(), accepted, 1, std::min(accepted, static_cast<int>(std::nearbyint(rate / 10.))));
    return writeWave(path, mono.data(), accepted, rate, 1);
}

} // namespace clay
