#pragma once
#include "../volume/VolumeTypes.hpp"
#include <itkOtsuMultipleThresholdsImageFilter.h>
#include <chrono>

namespace ctseg {

// 3D counterpart of methods/threshold/MultiOtsuSegmenter.hpp. Compare the
// two files side by side: the only difference is VolumeType/LabelVolumeType
// in place of ImageType/LabelImageType. This is the concrete proof that
// "3D support is a direct extension, not a redesign" — same filter
// (itk::OtsuMultipleThresholdsImageFilter), same histogram-based algorithm,
// now operating on the full voxel volume so thresholds are chosen using
// the ENTIRE volume's intensity distribution at once, rather than each
// slice separately (this can give more consistent results slice-to-slice
// than running the 2D version independently on each slice).
class MultiOtsuVolumeSegmenter : public IVolumeSegmenter {
public:
    std::string name() const override { return "MultiOtsu3D"; }

    VolumeSegmentationResult segment(VolumeType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        int numClasses = 3;
        if (auto it = params.find("num_classes"); it != params.end()) numClasses = static_cast<int>(it->second);

        using FilterType = itk::OtsuMultipleThresholdsImageFilter<VolumeType, LabelVolumeType>;
        auto filter = FilterType::New();
        filter->SetInput(input);
        filter->SetNumberOfThresholds(numClasses - 1);
        filter->Update();

        auto t1 = std::chrono::high_resolution_clock::now();

        VolumeSegmentationResult result;
        result.labels = filter->GetOutput();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = numClasses;
        result.method_name = name();
        return result;
    }
};

} // namespace ctseg
