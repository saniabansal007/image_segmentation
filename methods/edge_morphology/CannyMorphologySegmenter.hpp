#pragma once
#include "../../core/ISegmenter.hpp"
#include <itkCastImageFilter.h>
#include <itkCannyEdgeDetectionImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryMorphologicalClosingImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryFillholeImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <chrono>
#include <vector>
#include <algorithm>

namespace ctseg {

// Edge-based + morphology segmenter: detects edges with the Canny
// operator, closes small gaps in the edge contours with morphological
// closing, then fills enclosed regions to produce solid segmented areas.
// Complements intensity-based methods by using boundary/gradient
// information instead of raw pixel values.
class CannyMorphologySegmenter : public ISegmenter {
public:
    std::string name() const override { return "CannyMorphology"; }

    SegmentationResult segment(ImageType::Pointer input, const ParamMap& params) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        double variance = params.count("canny_variance") ? params.at("canny_variance") : 2.0;
        unsigned int closingRadius = params.count("closing_radius") ? static_cast<unsigned int>(params.at("closing_radius")) : 3;
        // Thresholds are chosen as PERCENTILES of this image's own gradient
        // magnitude distribution, not fixed absolute numbers. Raw CT slices
        // are 16-bit with gradient magnitudes in the hundreds-to-thousands
        // range (very different from an 8-bit photo), so a fixed threshold
        // like "6.0" is meaningless here and picks up background noise as
        // edges everywhere (this was a real bug found while testing this
        // method on the brain dataset - see README for details).
        double lowerPct = params.count("canny_lower_percentile") ? params.at("canny_lower_percentile") : 90.0;
        double upperPct = params.count("canny_upper_percentile") ? params.at("canny_upper_percentile") : 99.0;

        using FloatImageType = itk::Image<float, 2>;
        using CastType = itk::CastImageFilter<ImageType, FloatImageType>;
        auto caster = CastType::New();
        caster->SetInput(input);
        caster->Update();

        auto [lowerThresh, upperThresh] = estimateGradientThresholds(caster->GetOutput(), lowerPct, upperPct);

        using CannyType = itk::CannyEdgeDetectionImageFilter<FloatImageType, FloatImageType>;
        auto canny = CannyType::New();
        canny->SetInput(caster->GetOutput());
        canny->SetVariance(variance);
        canny->SetLowerThreshold(lowerThresh);
        canny->SetUpperThreshold(upperThresh);

        using ThreshType = itk::BinaryThresholdImageFilter<FloatImageType, LabelImageType>;
        auto thresh = ThreshType::New();
        thresh->SetInput(canny->GetOutput());
        thresh->SetLowerThreshold(0.01);
        thresh->SetUpperThreshold(itk::NumericTraits<float>::max());
        thresh->SetInsideValue(1);
        thresh->SetOutsideValue(0);

        using StructElemType = itk::BinaryBallStructuringElement<LabelPixelType, 2>;
        StructElemType structElem;
        structElem.SetRadius(closingRadius);
        structElem.CreateStructuringElement();

        using ClosingType = itk::BinaryMorphologicalClosingImageFilter<LabelImageType, LabelImageType, StructElemType>;
        auto closing = ClosingType::New();
        closing->SetInput(thresh->GetOutput());
        closing->SetKernel(structElem);
        closing->SetForegroundValue(1);

        using FillHoleType = itk::BinaryFillholeImageFilter<LabelImageType>;
        auto fillHole = FillHoleType::New();
        fillHole->SetInput(closing->GetOutput());
        fillHole->SetForegroundValue(1);
        fillHole->Update();

        auto t1 = std::chrono::high_resolution_clock::now();

        SegmentationResult result;
        result.labels = fillHole->GetOutput();
        result.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.num_regions = 2;
        result.method_name = name();
        return result;
    }

private:
    // Computes gradient-magnitude percentiles on this specific image so
    // Canny's thresholds scale with the data instead of being fixed
    // magic numbers tuned for 8-bit images.
    static std::pair<double, double> estimateGradientThresholds(
            itk::Image<float, 2>::Pointer floatImg, double lowerPct, double upperPct) {
        using FloatImageType = itk::Image<float, 2>;
        using GradType = itk::GradientMagnitudeImageFilter<FloatImageType, FloatImageType>;
        auto grad = GradType::New();
        grad->SetInput(floatImg);
        grad->Update();

        std::vector<float> vals;
        vals.reserve(floatImg->GetLargestPossibleRegion().GetNumberOfPixels());
        itk::ImageRegionConstIterator<FloatImageType> it(grad->GetOutput(), grad->GetOutput()->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) vals.push_back(it.Get());
        std::sort(vals.begin(), vals.end());

        auto pct = [&](double p) {
            size_t idx = static_cast<size_t>(p / 100.0 * (vals.size() - 1));
            return static_cast<double>(vals[idx]);
        };
        return {pct(lowerPct), pct(upperPct)};
    }
};

} // namespace ctseg
