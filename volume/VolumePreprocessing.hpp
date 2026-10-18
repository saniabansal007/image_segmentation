#pragma once
#include "VolumeTypes.hpp"
#include <itkMedianImageFilter.h>

namespace ctseg {

// Same MedianImageFilter as the 2D pipeline, just instantiated at
// VolumeType (3D) instead of ImageType (2D) — the algorithm and its
// tuning (radius) are unchanged; ITK does the dimension generalization.
// Note: a true 3D median filter considers neighbors in the slice-to-slice
// (z) direction too, not just within each slice — this is the actual
// difference in *behavior* from running 2D median filtering slice-by-slice.
class VolumePreprocessing {
public:
    static VolumeType::Pointer denoiseMedian(VolumeType::Pointer input, unsigned int radius = 2) {
        using FilterType = itk::MedianImageFilter<VolumeType, VolumeType>;
        auto filter = FilterType::New();
        VolumeType::SizeType r;
        r.Fill(radius);
        filter->SetRadius(r);
        filter->SetInput(input);
        filter->Update();
        return filter->GetOutput();
    }
};

} // namespace ctseg
