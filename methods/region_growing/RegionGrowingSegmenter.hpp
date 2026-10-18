#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkConfidenceConnectedImageFilter.h>
#include <itkCastImageFilter.h>
#include <chrono>

namespace ctseg {

// Region growing via ITK's ConfidenceConnectedImageFilter: starting from a
// seed point, repeatedly expands the region to include neighboring pixels
// whose intensity is within [mean - f*stddev, mean + f*stddev] of the
// already-accepted region, re-estimating mean/stddev each iteration.
// Seed defaults to image center; pass "seed_x"/"seed_y" in params to
// target a specific structure (e.g. one tissue lobe).
class RegionGrowingSegmenter : public ISegmenter {
public:
    std::string name() const override { return "RegionGrowing"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto size = input->GetLargestPossibleRegion().GetSize();
        ImageType::IndexType seed;
        seed[0] = (params.count("seed_x")) ? static_cast<long>(params.at("seed_x")) : size[0] / 2;
        seed[1] = (params.count("seed_y")) ? static_cast<long>(params.at("seed_y")) : size[1] / 2;

        double multiplier = params.count("multiplier") ? params.at("multiplier") : 2.5;
        int iterations = params.count("iterations") ? static_cast<int>(params.at("iterations")) : 5;
        unsigned int initialRadius = params.count("initial_radius") ? static_cast<unsigned int>(params.at("initial_radius")) : 2;

        using FilterType = itk::ConfidenceConnectedImageFilter<ImageType, ImageType>;
        auto filter = FilterType::New();
        filter->SetInput(input);
        filter->AddSeed(seed);
        filter->SetMultiplier(multiplier);
        filter->SetNumberOfIterations(iterations);
        filter->SetInitialNeighborhoodRadius(initialRadius);
        filter->SetReplaceValue(1);
        filter->Update();

        using CastType = itk::CastImageFilter<ImageType, LabelImageType>;
        auto caster = CastType::New();
        caster->SetInput(filter->GetOutput());
        caster->Update();

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = caster->GetOutput();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = 2; // grown region vs. background
        result.method_name = name();
        return result;
    }
};

} // namespace ctseg
