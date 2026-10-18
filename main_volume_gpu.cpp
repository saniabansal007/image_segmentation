// GPU-accelerated counterpart of main_volume.cpp. Only built when a CUDA
// compiler is detected (see CMakeLists.txt's check_language(CUDA)).
//
// NOT run anywhere in this project's development - no CUDA GPU was
// available in the sandbox this was built in. Build and test this on
// your own machine; see gpu/README.md for what to check first.
#include "volume/VolumeIO.hpp"
#include "volume/VolumePreprocessing.hpp"
#include "gpu/GPUPreprocessing.hpp"
#include "methods3d/MultiOtsuVolumeSegmenter.hpp"
#include <iostream>
#include <chrono>

using namespace ctseg;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <slice_directory> [num_classes] [--compare-cpu]\n";
        std::cerr << "  --compare-cpu also runs the CPU median filter and prints both timings.\n";
        return 1;
    }
    std::string dir = argv[1];
    int numClasses = (argc > 2) ? std::stoi(argv[2]) : 3;
    bool compareCpu = false;
    for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--compare-cpu") compareCpu = true;

    std::cout << "Loading slice stack from: " << dir << "\n";
    VolumeType::Pointer volume = VolumeIO::loadVolumeFromDirectory(dir);
    auto size = volume->GetLargestPossibleRegion().GetSize();
    std::cout << "  volume size: " << size[0] << " x " << size[1] << " x " << size[2] << "\n";

    std::cout << "Preprocessing: GPU 3D median denoising...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    VolumeType::Pointer processed = GPUPreprocessing::denoiseMedianGPU(volume, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  GPU median filter: " << gpuMs << " ms\n";

    if (compareCpu) {
        std::cout << "Preprocessing: CPU 3D median denoising (for comparison)...\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        VolumeType::Pointer cpuResult = VolumePreprocessing::denoiseMedian(volume, 1);
        auto t3 = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "  CPU median filter: " << cpuMs << " ms\n";
        std::cout << "  Speedup: " << (cpuMs / gpuMs) << "x\n";
        // NOTE: does not currently verify GPU and CPU outputs match
        // voxel-for-voxel - add that check before trusting GPU results
        // match the CPU reference on your data.
    }

    std::cout << "Segmenting with 3D Multi-Otsu (" << numClasses << " classes)...\n";
    MultiOtsuVolumeSegmenter segmenter;
    ParamMap params = {{"num_classes", static_cast<double>(numClasses)}};
    VolumeSegmentationResult result = segmenter.segment(processed, params);
    std::cout << "  runtime: " << result.runtime_ms << " ms (CPU - multi-Otsu wasn't GPU-ported, see README)\n";

    VolumeIO::saveVolumeMHA(processed, "output_volume_gpu_preprocessed.mha");
    VolumeIO::saveLabelVolumeMHA(result.labels, "output_volume_gpu_segmented.mha");
    std::cout << "Saved: output_volume_gpu_preprocessed.mha, output_volume_gpu_segmented.mha\n";

    return 0;
}
