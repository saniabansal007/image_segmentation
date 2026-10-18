#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkCastImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkWatershedImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <chrono>
#include <set>

namespace ctseg {

// Watershed segmentation: treats the gradient-magnitude image as a
// topographic surface and "floods" it from local minima, with catchment
// basins becoming segments. Particularly good at splitting touching
// structures (e.g. the adjacent tissue lobes in the brain slices) that
// intensity-only methods like thresholding tend to merge together.
class WatershedSegmenter : public ISegmenter {
public:
    std::string name() const override { return "Watershed"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        double level = params.count("level") ? params.at("level") : 0.2;
        double threshold = params.count("threshold") ? params.at("threshold") : 0.005;

        using FloatImageType = itk::Image<float, 2>;
        using CastType = itk::CastImageFilter<ImageType, FloatImageType>;
        auto caster = CastType::New();
        caster->SetInput(input);

        using GradientType = itk::GradientMagnitudeImageFilter<FloatImageType, FloatImageType>;
        auto gradient = GradientType::New();
        gradient->SetInput(caster->GetOutput());

        using WatershedType = itk::WatershedImageFilter<FloatImageType>;
        auto watershed = WatershedType::New();
        watershed->SetInput(gradient->GetOutput());
        watershed->SetLevel(level);
        watershed->SetThreshold(threshold);
        watershed->Update();

        // Watershed output labels are itk::IdentifierType (unsigned long)
        // and can number in the hundreds/thousands for noisy gradients.
        // We wrap them into our 0-255 LabelImageType via modulo for
        // visualization purposes, and separately report the TRUE region
        // count (num_regions) computed from the unwrapped label set.
        using WatershedOutputType = WatershedType::OutputImageType;
        WatershedOutputType::Pointer rawLabels = watershed->GetOutput();

        auto output = LabelImageType::New();
        output->SetRegions(rawLabels->GetLargestPossibleRegion());
        output->SetSpacing(rawLabels->GetSpacing());
        output->SetOrigin(rawLabels->GetOrigin());
        output->Allocate();

        std::set<WatershedOutputType::PixelType> uniqueLabels;
        itk::ImageRegionConstIterator<WatershedOutputType> inIt(rawLabels, rawLabels->GetLargestPossibleRegion());
        itk::ImageRegionIterator<LabelImageType> outIt(output, output->GetLargestPossibleRegion());
        for (inIt.GoToBegin(), outIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++outIt) {
            uniqueLabels.insert(inIt.Get());
            outIt.Set(static_cast<LabelPixelType>(inIt.Get() % 256));
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = output;
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = static_cast<int>(uniqueLabels.size());
        result.method_name = name();
        return result;
    }
};

} // namespace ctseg
