#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkOtsuThresholdImageFilter.h>
#include <chrono>

namespace ctseg {

// Global (binary) Otsu thresholding: single threshold separating
// foreground from background by maximizing between-class variance.
class GlobalOtsuSegmenter : public ISegmenter {
public:
    std::string name() const override { return "GlobalOtsu"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap&) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        using FilterType = itk::OtsuThresholdImageFilter<ImageType, LabelImageType>;
        auto filter = FilterType::New();
        filter->SetInput(input);
        filter->SetInsideValue(0);
        filter->SetOutsideValue(1);
        filter->Update();

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = filter->GetOutput();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = 2;
        result.method_name = name();
        lastThreshold = filter->GetThreshold();
        return result;
    }

    double lastThreshold = 0.0;
};

} 
