#pragma once
#include "../volume/VolumeTypes.hpp"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <vector>

// Declarations matching the .cu files' functions - this header is plain
// C++ (no __global__/CUDA syntax), so it can be included from ordinary
// .cpp files without needing nvcc. The actual kernel calls are compiled
// separately by nvcc into MedianFilter3D.cu.o / Morphology3D.cu.o and
namespace ctseg_gpu {
    void medianFilter3D(const std::vector<unsigned short>& input, std::vector<unsigned short>& output,
                         int width, int height, int depth, int radius);

    void gaussianFilter3D(const std::vector<unsigned short>& input, std::vector<unsigned short>& output,
                          int width, int height, int depth, double sigma);

    enum class MorphOp { Erode, Dilate };
    void morphology3D(const std::vector<unsigned char>& input, std::vector<unsigned char>& output,
                       int width, int height, int depth, int radius, MorphOp op);
    void morphologyOpen3D(const std::vector<unsigned char>& input, std::vector<unsigned char>& output,
                           int width, int height, int depth, int radius);
    void morphologyClose3D(const std::vector<unsigned char>& input, std::vector<unsigned char>& output,
                            int width, int height, int depth, int radius);
}

namespace ctseg {

// Drop-in GPU-accelerated counterparts of VolumePreprocessing's and
// ROIProcessor's CPU filters, same ITK-in/ITK-out signatures so they can
// be swapped in without changing any calling code. All the ITK<->raw
// buffer copying happens here; the .cu files never see an ITK type.
class GPUPreprocessing {
public:
    static VolumeType::Pointer denoiseMedianGPU(VolumeType::Pointer input, int radius = 1) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = size[0], h = size[1], d = size[2];
        size_t numVoxels = static_cast<size_t>(w) * h * d;

        std::vector<unsigned short> inBuf(numVoxels);
        const unsigned short* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, numVoxels * sizeof(unsigned short));
        } else {
            itk::ImageRegionConstIterator<VolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned short> outBuf;
        ctseg_gpu::medianFilter3D(inBuf, outBuf, w, h, d, radius);

        auto output = VolumeType::New();
        output->SetRegions(region);
        output->SetSpacing(input->GetSpacing());
        output->SetOrigin(input->GetOrigin());
        output->Allocate();
        unsigned short* dstPtr = output->GetBufferPointer();
        if (dstPtr) {
            std::memcpy(dstPtr, outBuf.data(), numVoxels * sizeof(unsigned short));
        } else {
            itk::ImageRegionIterator<VolumeType> outIt(output, region);
            size_t i = 0;
            for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt) outIt.Set(outBuf[i++]);
        }
        return output;
    }

    static VolumeType::Pointer gaussianSmoothGPU(VolumeType::Pointer input, double sigma = 1.0) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = size[0], h = size[1], d = size[2];
        size_t numVoxels = static_cast<size_t>(w) * h * d;

        std::vector<unsigned short> inBuf(numVoxels);
        const unsigned short* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, numVoxels * sizeof(unsigned short));
        } else {
            itk::ImageRegionConstIterator<VolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned short> outBuf;
        ctseg_gpu::gaussianFilter3D(inBuf, outBuf, w, h, d, sigma);

        auto output = VolumeType::New();
        output->SetRegions(region);
        output->SetSpacing(input->GetSpacing());
        output->SetOrigin(input->GetOrigin());
        output->Allocate();
        unsigned short* dstPtr = output->GetBufferPointer();
        if (dstPtr) {
            std::memcpy(dstPtr, outBuf.data(), numVoxels * sizeof(unsigned short));
        } else {
            itk::ImageRegionIterator<VolumeType> outIt(output, region);
            size_t i = 0;
            for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt) outIt.Set(outBuf[i++]);
        }
        return output;
    }

    static LabelVolumeType::Pointer erodeGPU(LabelVolumeType::Pointer input, int radius) {
        return runMorphGPU(input, radius, ctseg_gpu::MorphOp::Erode);
    }
    static LabelVolumeType::Pointer dilateGPU(LabelVolumeType::Pointer input, int radius) {
        return runMorphGPU(input, radius, ctseg_gpu::MorphOp::Dilate);
    }

private:
    static LabelVolumeType::Pointer runMorphGPU(LabelVolumeType::Pointer input, int radius, ctseg_gpu::MorphOp op) {
        auto region = input->GetLargestPossibleRegion();
        auto size = region.GetSize();
        int w = size[0], h = size[1], d = size[2];
        size_t numVoxels = static_cast<size_t>(w) * h * d;

        std::vector<unsigned char> inBuf(numVoxels);
        const unsigned char* srcPtr = input->GetBufferPointer();
        if (srcPtr) {
            std::memcpy(inBuf.data(), srcPtr, numVoxels * sizeof(unsigned char));
        } else {
            itk::ImageRegionConstIterator<LabelVolumeType> it(input, region);
            size_t i = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) inBuf[i++] = it.Get();
        }

        std::vector<unsigned char> outBuf;
        ctseg_gpu::morphology3D(inBuf, outBuf, w, h, d, radius, op);

        auto output = LabelVolumeType::New();
        output->SetRegions(region);
        output->SetSpacing(input->GetSpacing());
        output->SetOrigin(input->GetOrigin());
        output->Allocate();
        unsigned char* dstPtr = output->GetBufferPointer();
        if (dstPtr) {
            std::memcpy(dstPtr, outBuf.data(), numVoxels * sizeof(unsigned char));
        } else {
            itk::ImageRegionIterator<LabelVolumeType> outIt(output, region);
            size_t i = 0;
            for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt) outIt.Set(outBuf[i++]);
        }
        return output;
    }
};

} // namespace ctseg
