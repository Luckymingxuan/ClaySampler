#pragma once

#include "../render/plan.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace clay {

struct RenderBatch {
    std::shared_ptr<int16_t> pcm;
    std::vector<TailResult> results;
    double kernelSeconds = 0;
    size_t failedScratchBytes = 0; // Single rejected job at the published index.
    std::vector<size_t> jobIndices{}; // Noncontiguous completions from the rolling queue.
};

class Renderer {
public:
    Renderer(int device, const std::filesystem::path& nvrtc, bool natural, AudioConfig audio, int groupMinimum,
             size_t waitingLimit, size_t batchJobs);
    ~Renderer();
    void uploadSamples(const AudioBank& bank, bool fresh);
    using Refill = std::function<bool(std::vector<RenderPlan>&, bool wait)>;
    void renderWindow(std::vector<RenderPlan>& plans, const std::function<void(RenderBatch, size_t)>& publish,
                      const Refill& refill = {});
    const std::string& device() const;
    size_t peakBytes() const;
    size_t sampleCacheBudget() const;
    size_t peakWaitingBytes() const;
    size_t spilledBytes() const;
    size_t downloadedBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clay
