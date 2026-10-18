#include "../io/SliceIO.hpp"
#include "../preprocessing/Preprocessing.hpp"
#include <itkCastImageFilter.h>
#include <itkCannyEdgeDetectionImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace ctseg;

int main(int argc, char** argv) {
    ImageType::Pointer img = SliceIO::loadSlice(argv[1]);
    ImageType::Pointer processed = Preprocessing::denoiseMedian(img, 2);
    processed = Preprocessing::denoiseGaussian(processed, std::stod(argv[5]));

    using FloatImageType = itk::Image<float, 2>;
    using CastType = itk::CastImageFilter<ImageType, FloatImageType>;
    auto caster = CastType::New();
    caster->SetInput(processed);
    caster->Update();
    std::cout << "cast ok\n";

    using CannyType = itk::CannyEdgeDetectionImageFilter<FloatImageType, FloatImageType>;
    auto canny = CannyType::New();
    canny->SetInput(caster->GetOutput());
    canny->SetVariance(std::stod(argv[2]));
    canny->SetLowerThreshold(std::stod(argv[3]));
    canny->SetUpperThreshold(std::stod(argv[4]));
    canny->Update();
    std::cout << "canny ok\n";

    // Inspect canny output range
    using RescaleType = itk::RescaleIntensityImageFilter<FloatImageType, itk::Image<unsigned char,2>>;
    auto rescaler = RescaleType::New();
    rescaler->SetInput(canny->GetOutput());
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(255);
    rescaler->Update();

    using WriterType = itk::ImageFileWriter<itk::Image<unsigned char,2>>;
    auto writer = WriterType::New();
    writer->SetFileName("debug_canny_edges.png");
    writer->SetInput(rescaler->GetOutput());
    writer->Update();
    std::cout << "saved debug_canny_edges.png\n";

    // Check raw canny pixel value range without rescale
    itk::ImageRegionConstIterator<FloatImageType> it(canny->GetOutput(), canny->GetOutput()->GetLargestPossibleRegion());
    float minV = 1e30f, maxV = -1e30f;
    long countNonzero = 0, total = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float v = it.Get();
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        if (v > 0.0001f) countNonzero++;
        total++;
    }
    std::cout << "canny raw range: [" << minV << ", " << maxV << "]  nonzero=" << countNonzero << "/" << total << "\n";

    // Also inspect the raw gradient magnitude BEFORE Canny's internal
    // thresholding, to pick meaningful thresholds for this data's actual
    // intensity scale (raw 16-bit values in the thousands, not 0-255).
    using GradType = itk::GradientMagnitudeImageFilter<FloatImageType, FloatImageType>;
    auto grad = GradType::New();
    grad->SetInput(caster->GetOutput());
    grad->Update();

    std::vector<float> gvals;
    itk::ImageRegionConstIterator<FloatImageType> git(grad->GetOutput(), grad->GetOutput()->GetLargestPossibleRegion());
    for (git.GoToBegin(); !git.IsAtEnd(); ++git) gvals.push_back(git.Get());
    std::sort(gvals.begin(), gvals.end());
    auto pct = [&](double p) { return gvals[static_cast<size_t>(p / 100.0 * (gvals.size() - 1))]; };
    std::cout << "gradient magnitude percentiles: p50=" << pct(50) << " p90=" << pct(90)
              << " p99=" << pct(99) << " p99.9=" << pct(99.9) << " max=" << gvals.back() << "\n";

    return 0;
}
