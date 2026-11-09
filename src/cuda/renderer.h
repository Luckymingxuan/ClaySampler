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
};

class Renderer {
public:
    Renderer(int device, const std::filesystem::path& nvrtc, bool natural, AudioConfig audio, int groupMinimum,
             size_t waitingLimit);
    ~Renderer();
    void uploadSamples(const AudioBank& bank, bool fresh);
    void renderWindow(const std::vector<RenderPlan>& plans, const std::function<void(RenderBatch, size_t)>& publish);
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
