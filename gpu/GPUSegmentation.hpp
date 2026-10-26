#pragma once
#include "../volume/VolumeTypes.hpp"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <vector>
#include <string>

namespace ctseg_gpu {
    void thresholdRange3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        unsigned short lower, unsigned short upper,
        unsigned int& outFgCount);

    void globalOtsu3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        unsigned short minVal, unsigned short maxVal,
        unsigned short& outThreshold,
        unsigned int& outFgCount);

    void multiOtsu3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        unsigned short minVal, unsigned short maxVal,
        int numClasses, int targetClass,
        std::vector<unsigned short>& outThresholds,
        unsigned int& outFgCount);

    void connectedThreshold3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        int seedX, int seedY, int seedZ,
        unsigned short lower, unsigned short upper,
        unsigned int& outFgCount);

#include "GPUPreprocessing.hpp"
}

namespace ctseg {

class GPUSegmentation {
public:
    static LabelVolumeType::Pointer thresholdGPU(
            VolumeType::Pointer input,
            unsigned short lower, unsigned short upper,
            unsigned int& outFgCount) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = static_cast<int>(size[0]), h = static_cast<int>(size[1]), d = static_cast<int>(size[2]);

        std::vector<unsigned short> inBuf(static_cast<size_t>(w) * h * d);
        const unsigned short* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, inBuf.size() * sizeof(unsigned short));
        } else {
            itk::ImageRegionConstIterator<VolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned char> outBuf;
        ctseg_gpu::thresholdRange3DGPU(inBuf, outBuf, w, h, d, lower, upper, outFgCount);

        return makeLabelVolume(region, input->GetSpacing(), input->GetOrigin(), input->GetDirection(), outBuf);
    }

    static LabelVolumeType::Pointer otsuGPU(
            VolumeType::Pointer input,
            unsigned short minVal, unsigned short maxVal,
            unsigned short& outThreshold,
            unsigned int& outFgCount) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = static_cast<int>(size[0]), h = static_cast<int>(size[1]), d = static_cast<int>(size[2]);

        std::vector<unsigned short> inBuf(static_cast<size_t>(w) * h * d);
        const unsigned short* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, inBuf.size() * sizeof(unsigned short));
        } else {
            itk::ImageRegionConstIterator<VolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned char> outBuf;
        ctseg_gpu::globalOtsu3DGPU(inBuf, outBuf, w, h, d, minVal, maxVal, outThreshold, outFgCount);

        return makeLabelVolume(region, input->GetSpacing(), input->GetOrigin(), input->GetDirection(), outBuf);
    }

    static LabelVolumeType::Pointer multiOtsuGPU(
            VolumeType::Pointer input,
            unsigned short minVal, unsigned short maxVal,
            int numClasses, int targetClass,
            std::vector<unsigned short>& outThresholds,
            unsigned int& outFgCount) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = static_cast<int>(size[0]), h = static_cast<int>(size[1]), d = static_cast<int>(size[2]);

        std::vector<unsigned short> inBuf(static_cast<size_t>(w) * h * d);
        const unsigned short* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, inBuf.size() * sizeof(unsigned short));
        } else {
            itk::ImageRegionConstIterator<VolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned char> outBuf;
        ctseg_gpu::multiOtsu3DGPU(inBuf, outBuf, w, h, d, minVal, maxVal, numClasses, targetClass, outThresholds, outFgCount);

        return makeLabelVolume(region, input->GetSpacing(), input->GetOrigin(), input->GetDirection(), outBuf);
    }

    static LabelVolumeType::Pointer connectedThresholdGPU(
            VolumeType::Pointer input,
            int seedX, int seedY, int seedZ,
            unsigned short lower, unsigned short upper,
            unsigned int& outFgCount) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = static_cast<int>(size[0]), h = static_cast<int>(size[1]), d = static_cast<int>(size[2]);

        std::vector<unsigned short> inBuf(static_cast<size_t>(w) * h * d);
        const unsigned short* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, inBuf.size() * sizeof(unsigned short));
        } else {
            itk::ImageRegionConstIterator<VolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned char> outBuf;
        ctseg_gpu::connectedThreshold3DGPU(inBuf, outBuf, w, h, d, seedX, seedY, seedZ, lower, upper, outFgCount);

        return makeLabelVolume(region, input->GetSpacing(), input->GetOrigin(), input->GetDirection(), outBuf);
    }

    static LabelVolumeType::Pointer cannyMorphologyGPU(
            VolumeType::Pointer input,
            unsigned short minVal, unsigned short maxVal,
            int morphRadius,
            unsigned int& outFgCount) {
        unsigned short otsuThresh = 0;
        unsigned int initialFg = 0;
        auto mask = otsuGPU(input, minVal, maxVal, otsuThresh, initialFg);

        auto region = mask->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = static_cast<int>(size[0]), h = static_cast<int>(size[1]), d = static_cast<int>(size[2]);

        std::vector<unsigned char> inBuf(static_cast<size_t>(w) * h * d);
        std::memcpy(inBuf.data(), mask->GetBufferPointer(), inBuf.size());

        std::vector<unsigned char> outBuf;
        ctseg_gpu::morphologyClose3D(inBuf, outBuf, w, h, d, morphRadius);

        outFgCount = 0;
        for (unsigned char b : outBuf) if (b > 0) outFgCount++;

        return makeLabelVolume(region, input->GetSpacing(), input->GetOrigin(), input->GetDirection(), outBuf);
    }

private:
    static LabelVolumeType::Pointer makeLabelVolume(
            const LabelVolumeType::RegionType& region,
            const VolumeType::SpacingType& spacing,
            const VolumeType::PointType& origin,
            const VolumeType::DirectionType& direction,
            const std::vector<unsigned char>& buffer) {
        auto output = LabelVolumeType::New();
        output->SetRegions(region);
        output->SetSpacing(spacing);
        output->SetOrigin(origin);
        output->SetDirection(direction);
        output->Allocate();
        uint8_t* outPtr = output->GetBufferPointer();
        if (outPtr && !buffer.empty()) {
            std::memcpy(outPtr, buffer.data(), buffer.size() * sizeof(unsigned char));
        }
        return output;
    }
};

} // namespace ctseg
