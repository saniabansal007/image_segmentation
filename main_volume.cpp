#include "volume/VolumeIO.hpp"
#include "volume/VolumePreprocessing.hpp"
#include "methods3d/MultiOtsuVolumeSegmenter.hpp"
#include <iostream>

using namespace ctseg;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <slice_directory> [num_classes] [max_slices]\n";
        std::cerr << "  Stacks every .tif in the directory (sorted by filename) into one volume.\n";
        return 1;
    }
    std::string dir = argv[1];
    int numClasses = (argc > 2) ? std::stoi(argv[2]) : 3;
    size_t maxSlices = (argc > 3) ? static_cast<size_t>(std::stoi(argv[3])) : 0;

    std::cout << "Loading slice stack from: " << dir << "\n";
    VolumeType::Pointer volume = VolumeIO::loadVolumeFromDirectory(dir, ".tif", maxSlices);
    auto size = volume->GetLargestPossibleRegion().GetSize();
    std::cout << "  volume size: " << size[0] << " x " << size[1] << " x " << size[2] << " (X x Y x Z slices)\n";

    std::cout << "Preprocessing: 3D median denoising...\n";
    VolumeType::Pointer processed = VolumePreprocessing::denoiseMedian(volume, 1);
    // Note: radius=1 (not 2, as in the 2D pipeline) because a 3D median
    // filter's neighborhood grows as (2r+1)^3, not (2r+1)^2 - radius=2
    // would mean a 125-voxel window per voxel, which is considerably
    // more expensive across a full volume than the 2D case.

    std::cout << "Segmenting with 3D Multi-Otsu (" << numClasses << " classes)...\n";
    MultiOtsuVolumeSegmenter segmenter;
    ParamMap params = {{"num_classes", static_cast<double>(numClasses)}};
    VolumeSegmentationResult result = segmenter.segment(processed, params);

    std::cout << "  runtime: " << result.runtime_ms << " ms\n";
    std::cout << "  method: " << result.method_name << "\n";

    VolumeIO::saveVolumeMHA(processed, "output_volume_preprocessed.mha");
    VolumeIO::saveLabelVolumeMHA(result.labels, "output_volume_segmented.mha");
    std::cout << "Saved full volumes: output_volume_preprocessed.mha, output_volume_segmented.mha\n";

    // Also dump a few individual axial slices as PNGs so results can be
    // eyeballed without a 3D viewer like 3D Slicer/ParaView.
    unsigned int numZ = static_cast<unsigned int>(size[2]);
    std::vector<unsigned int> sampleSlices = {0, numZ / 2, numZ > 1 ? numZ - 1 : 0};
    for (unsigned int z : sampleSlices) {
        VolumeIO::saveSliceAsPNG<VolumeType, itk::Image<PixelType, 2>>(
            processed, z, "output_volume_slice" + std::to_string(z) + "_preprocessed.png");
        VolumeIO::saveSliceAsPNG<LabelVolumeType, itk::Image<LabelPixelType, 2>>(
            result.labels, z, "output_volume_slice" + std::to_string(z) + "_segmented.png");
    }
    std::cout << "Saved sample slice PNGs at z = ";
    for (auto z : sampleSlices) std::cout << z << " ";
    std::cout << "\n";

    return 0;
}
