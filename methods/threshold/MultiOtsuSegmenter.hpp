#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkOtsuMultipleThresholdsImageFilter.h>
#include <chrono>

namespace ctseg {

// Multi-level Otsu thresholding via ITK's native OtsuMultipleThresholdsImageFilter,
// which generalizes Otsu's method to N classes by maximizing between-class
// variance over the image histogram. Good baseline for data with visually
// distinct intensity bands (background / soft tissue / bone in brain CT).
class MultiOtsuSegmenter : public ISegmenter {
public:
    std::string name() const override { return "MultiOtsu"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        int numClasses = 3; // default: background / tissue / bone
        auto it = params.find("num_classes");
        if (it != params.end()) numClasses = static_cast<int>(it->second);

        using FilterType = itk::OtsuMultipleThresholdsImageFilter<ImageType, LabelImageType>;
        auto filter = FilterType::New();
        filter->SetInput(input);
        filter->SetNumberOfThresholds(numClasses - 1);
        filter->Update();

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = filter->GetOutput();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = numClasses;
        result.method_name = name();
        computedThresholds = filter->GetThresholds();
        return result;
    }

    // Exposed for reporting/debugging — the actual intensity values ITK chose.
    std::vector<double> lastThresholds() const { return computedThresholds; }

private:
    std::vector<double> computedThresholds;
};

} // namespace ctseg
