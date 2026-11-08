#pragma once

#include "../render/plan.h"

#include <filesystem>
#include <memory>
#include <string>

namespace clay {

struct RenderBatch {
    std::unique_ptr<int16_t[]> pcm;
    std::unique_ptr<float[]> stereo;
    size_t count;
    double kernelSeconds;
};

class Renderer {
public:
    Renderer(int device, const std::filesystem::path& nvrtc, bool floatingPoint);
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
