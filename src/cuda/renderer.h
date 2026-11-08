#pragma once

#include "../render/plan.h"

#include <filesystem>
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
    Renderer(int device, const std::filesystem::path& nvrtc, bool natural, int rate);
    ~Renderer();
    void uploadSamples(const AudioBank& bank);
    RenderBatch render(const std::vector<RenderPlan>& plans, size_t first);
    const std::string& device() const;
    size_t peakBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clay
