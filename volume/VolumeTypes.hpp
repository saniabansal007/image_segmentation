#pragma once
#include <itkImage.h>
#include "../core/ISegmenter.hpp"  // reuse PixelType, LabelPixelType

namespace ctseg {

// 3D counterparts of ImageType/LabelImageType. Same pixel types as the 2D
// pipeline (unsigned short raw intensity, unsigned char labels) — only the
// dimension changes, since every ITK filter used in this project is
// templated on dimension. This is why 3D support is an extension of the
// existing code rather than a rewrite.
using VolumeType = itk::Image<PixelType, 3>;
using LabelVolumeType = itk::Image<LabelPixelType, 3>;

struct VolumeSegmentationResult {
    LabelVolumeType::Pointer labels;
    double runtime_ms = 0.0;
    int num_regions = 0;
    std::string method_name;
};

// 3D counterpart of ISegmenter. Kept as a separate interface rather than
// templating ISegmenter<Dim> itself, so the existing 2D method classes and
// SegmenterFactory registry don't need to change — this interface is
// additive. A method that wants to support both dimensions implements
// both ISegmenter and IVolumeSegmenter (see MultiOtsuVolumeSegmenter).
class IVolumeSegmenter {
public:
    virtual ~IVolumeSegmenter() = default;
    virtual VolumeSegmentationResult segment(VolumeType::Pointer input, const ParamMap& params) = 0;
    virtual std::string name() const = 0;
};

} // namespace ctseg
