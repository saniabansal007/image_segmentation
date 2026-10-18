#pragma once
#include <itkImage.h>
#include <string>
#include <map>

namespace ctseg {

// Standard pixel/image types used across the whole module.
// 2D for now; a 3D counterpart (itk::Image<PixelType, 3>) is a direct
// extension since ITK filters are dimension-templated, noted in README.
using PixelType = unsigned short;              // raw 16-bit CT intensity
using ImageType = itk::Image<PixelType, 2>;

using LabelPixelType = unsigned char;          // up to 255 segmentation classes
using LabelImageType = itk::Image<LabelPixelType, 2>;

// Generic parameter bag so each method can accept its own tunables
// without changing the common interface.
using ParamMap = std::map<std::string, double>;

struct SegmentationResult {
    LabelImageType::Pointer labels;
    double runtime_ms = 0.0;
    int num_regions = 0;
    std::string method_name;
};

// Common interface every segmentation approach must implement.
// Every method plugs in here without touching the pipeline that calls it.
class ISegmenter {
public:
    virtual ~ISegmenter() = default;

    // input: preprocessed single-channel ITK image.
    virtual SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) = 0;

    virtual std::string name() const = 0;
};

} // namespace ctseg
