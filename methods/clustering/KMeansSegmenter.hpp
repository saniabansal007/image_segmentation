#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkScalarImageKmeansImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <chrono>

namespace ctseg {

// K-means clustering segmenter: groups pixels into k intensity clusters
// by iteratively assigning pixels to the nearest cluster mean and
// recomputing means, without assuming any particular histogram shape
// (unlike Otsu, which assumes well-separated peaks).
class KMeansSegmenter : public ISegmenter {
public:
    std::string name() const override { return "KMeansClustering"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        int k = params.count("num_clusters") ? static_cast<int>(params.at("num_clusters")) : 3;

        using CalcType = itk::MinimumMaximumImageCalculator<ImageType>;
        auto calc = CalcType::New();
        calc->SetImage(input);
        calc->Compute();
        double minV = calc->GetMinimum();
        double maxV = calc->GetMaximum();

        using FilterType = itk::ScalarImageKmeansImageFilter<ImageType>;
        auto filter = FilterType::New();
        filter->SetInput(input);
        // Evenly spaced initial means across the observed intensity range —
        // a simple, deterministic init; k-means refines them iteratively.
        for (int i = 0; i < k; ++i) {
            double mean = minV + (maxV - minV) * (i + 0.5) / k;
            filter->AddClassWithInitialMean(mean);
        }
        filter->Update();

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = filter->GetOutput();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = k;
        result.method_name = name();
        auto means = filter->GetFinalMeans();
        finalMeans.assign(means.begin(), means.end());
        return result;
    }

    std::vector<double> lastMeans() const { return finalMeans; }

private:
    std::vector<double> finalMeans;
};

} // namespace ctseg
