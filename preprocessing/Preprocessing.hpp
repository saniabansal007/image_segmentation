#pragma once
#include "../core/ISegmenter.hpp"
#include <itkMedianImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkCastImageFilter.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace ctseg {

class Preprocessing {
public:
    // Median filter denoising: replaces a pixel value with median of pixels within radius r.
    static ImageType::Pointer denoiseMedian(ImageType::Pointer input, unsigned int radius = 2) {
        using FilterType = itk::MedianImageFilter<ImageType, ImageType>;
        auto filter = FilterType::New();
        ImageType::SizeType r;
        r.Fill(radius);
        filter->SetRadius(r);
        filter->SetInput(input);
        filter->Update();
        return filter->GetOutput();
    }

    // Discrete Gaussian denoising — smoother than median, better for
    // preserving gradual intensity transitions; useful as an alternative
    // to compare against median filtering.
    static ImageType::Pointer denoiseGaussian(ImageType::Pointer input, double variance = 1.0) {
        using FilterType = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
        auto filter = FilterType::New();
        filter->SetVariance(variance);
        filter->SetInput(input);
        filter->Update();
        return filter->GetOutput();
    }

    // Ring-artifact suppression via polar-domain radial bias subtraction.
    // Ring artifacts come from per-detector-pixel gain error, which in a
    // CT reconstruction shows up as a bias that is roughly CONSTANT across
    // angle for a given radius. Approach: resample the image on a polar
    // grid (angle x radius) around the image center, compute the per-radius
    // median across all angles (the "radial profile"), separate that
    // profile into a smooth low-frequency trend (real anatomy, which
    // varies gradually with radius) and a high-frequency residual (the
    // ring bias), then subtract only the residual back out in Cartesian
    // space. This is pure ITK pixel access (no external image library).
    static ImageType::Pointer suppressRingArtifacts(ImageType::Pointer input) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        const int W = static_cast<int>(size[0]);
        const int H = static_cast<int>(size[1]);
        const double cx = W / 2.0, cy = H / 2.0;
        const double maxRadius = std::min(W, H) / 2.0;
        const int numAngles = 360;
        const int numRadii = static_cast<int>(maxRadius);

        // Sample each (angle, radius) cell via nearest-neighbor lookup,
        // building the radial median profile directly (no explicit polar
        // image needed, saving a resampling pass).
        std::vector<std::vector<PixelType>> samplesByRadius(numRadii);
        for (int r = 0; r < numRadii; ++r) samplesByRadius[r].reserve(numAngles);

        for (int a = 0; a < numAngles; ++a) {
            double theta = a * 2.0 * M_PI / numAngles;
            for (int r = 0; r < numRadii; ++r) {
                int x = static_cast<int>(std::round(cx + r * std::cos(theta)));
                int y = static_cast<int>(std::round(cy + r * std::sin(theta)));
                if (x < 0 || x >= W || y < 0 || y >= H) continue;
                ImageType::IndexType idx;
                idx[0] = x; idx[1] = y;
                samplesByRadius[r].push_back(input->GetPixel(idx));
            }
        }

        std::vector<double> radialProfile(numRadii, 0.0);
        for (int r = 0; r < numRadii; ++r) {
            auto& v = samplesByRadius[r];
            if (v.empty()) continue;
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            radialProfile[r] = static_cast<double>(v[v.size() / 2]);
        }

        // Smooth radial profile (low-frequency real trend) via a wide
        // moving average along radius.
        int win = std::max(5, numRadii / 25);
        if (win % 2 == 0) win++;
        int half = win / 2;
        std::vector<double> smoothProfile(numRadii, 0.0);
        for (int r = 0; r < numRadii; ++r) {
            int lo = std::max(0, r - half), hi = std::min(numRadii - 1, r + half);
            double sum = 0.0;
            for (int k = lo; k <= hi; ++k) sum += radialProfile[k];
            smoothProfile[r] = sum / (hi - lo + 1);
        }

        std::vector<double> residual(numRadii);
        for (int r = 0; r < numRadii; ++r) residual[r] = radialProfile[r] - smoothProfile[r];

        // Apply correction in Cartesian space: for every pixel, find its
        // radius from center and subtract that radius's residual bias.
        auto output = ImageType::New();
        output->SetRegions(region);
        output->SetSpacing(input->GetSpacing());
        output->SetOrigin(input->GetOrigin());
        output->Allocate();

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double dx = x - cx, dy = y - cy;
                double rad = std::sqrt(dx * dx + dy * dy);
                int ri = static_cast<int>(std::round(rad));
                double bias = (ri >= 0 && ri < numRadii) ? residual[ri] : 0.0;

                ImageType::IndexType idx; idx[0] = x; idx[1] = y;
                double v = static_cast<double>(input->GetPixel(idx)) - bias;
                v = std::clamp(v, 0.0, 65535.0);
                output->SetPixel(idx, static_cast<PixelType>(v));
            }
        }
        return output;
    }
};

} 
