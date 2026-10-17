#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkBoxMeanImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <chrono>

namespace ctseg {

// Adaptive (local) thresholding: instead of one global threshold, each
// pixel is compared to the mean intensity of its local neighborhood minus
// an offset. Handles gradual illumination/intensity drift across an image
// better than a single global threshold — useful when scanner artifacts
// or beam-hardening cause slow intensity gradients across the FOV.
class AdaptiveThresholdSegmenter : public ISegmenter {
public:
    std::string name() const override { return "AdaptiveThreshold"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        unsigned int radius = 15;
        double offset = 0.0;
        if (auto it = params.find("radius"); it != params.end()) radius = static_cast<unsigned int>(it->second);
        if (auto it = params.find("offset"); it != params.end()) offset = it->second;

        // BoxMeanImageFilter is a separable (1D-pass-per-axis) box filter,
        // which is dramatically faster than a naive full-neighborhood mean
        // filter at larger radii (O(N) vs O(N*r^2) per pixel) - the naive
        // MeanImageFilter took 73+ seconds at radius=15 on a 1000x1000
        // slice during testing; this is the fix for that.
        using MeanFilterType = itk::BoxMeanImageFilter<ImageType, ImageType>;
        auto meanFilter = MeanFilterType::New();
        ImageType::SizeType r;
        r.Fill(radius);
        meanFilter->SetRadius(r);
        meanFilter->SetInput(input);
        meanFilter->Update();
        ImageType::Pointer localMean = meanFilter->GetOutput();

        auto output = LabelImageType::New();
        output->SetRegions(input->GetLargestPossibleRegion());
        output->SetSpacing(input->GetSpacing());
        output->SetOrigin(input->GetOrigin());
        output->Allocate();

        itk::ImageRegionConstIterator<ImageType> inIt(input, input->GetLargestPossibleRegion());
        itk::ImageRegionConstIterator<ImageType> meanIt(localMean, localMean->GetLargestPossibleRegion());
        itk::ImageRegionIterator<LabelImageType> outIt(output, output->GetLargestPossibleRegion());

        for (inIt.GoToBegin(), meanIt.GoToBegin(), outIt.GoToBegin();
             !inIt.IsAtEnd(); ++inIt, ++meanIt, ++outIt) {
            bool foreground = static_cast<double>(inIt.Get()) > (static_cast<double>(meanIt.Get()) + offset);
            outIt.Set(foreground ? 1 : 0);
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = output;
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = 2;
        result.method_name = name();
        return result;
    }
};

} 
