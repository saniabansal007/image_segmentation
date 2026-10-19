#include "../volume/SliceStore.hpp"
#include "../volume/VolumeIO.hpp"
#include "../io/SliceIO.hpp"

#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageRegionConstIterator.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <vector>
#include <string>

using namespace ctseg;

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <input_dataset_dir> [options]\n\n"
              << "Arguments:\n"
              << "  input_dataset_dir       Path to folder containing CT slice TIFF images (e.g. ./data)\n\n"
              << "Options:\n"
              << "  -o, --output <dir>      Output preprocessed directory (default: <input_dir>_preprocessed)\n"
              << "  -f, --force             Force re-preprocessing even if valid preprocessed data exists\n"
              << "  --max-slices <N>        Process at most first N slices (useful for quick testing)\n"
              << "  -h, --help              Show this help message\n";
}

// Computes 1% and 99% percentile window/level
static void computeWindowLevel(VolumeType::Pointer vol, double minV, double maxV, double& outWindow, double& outLevel) {
    double range = maxV - minV;
    if (range <= 0) { outWindow = 1000; outLevel = minV; return; }

    const int numBins = 2000;
    std::vector<uint64_t> hist(numBins, 0);
    itk::ImageRegionConstIterator<VolumeType> it(vol, vol->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        int bin = static_cast<int>((static_cast<double>(it.Get()) - minV) / range * (numBins - 1));
        bin = std::max(0, std::min(numBins - 1, bin));
        hist[bin]++;
    }

    uint64_t total = vol->GetLargestPossibleRegion().GetNumberOfPixels();
    uint64_t loTarget = static_cast<uint64_t>(total * 0.01);
    uint64_t hiTarget = static_cast<uint64_t>(total * 0.99);

    uint64_t cum = 0; int loBin = 0;
    for (int b = 0; b < numBins; ++b) { cum += hist[b]; if (cum >= loTarget) { loBin = b; break; } }
    cum = 0; int hiBin = numBins - 1;
    for (int b = 0; b < numBins; ++b) { cum += hist[b]; if (cum >= hiTarget) { hiBin = b; break; } }

    double lo = minV + (static_cast<double>(loBin) / (numBins - 1)) * range;
    double hi = minV + (static_cast<double>(hiBin) / (numBins - 1)) * range;
    if (hi <= lo) hi = lo + 1;

    outWindow = hi - lo;
    outLevel = (hi + lo) / 2.0;
}

int main(int argc, char** argv) {
    std::string inputDir = "";
    std::string outputDir = "";
    bool force = false;
    size_t maxSlices = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "-f" || arg == "--force") {
            force = true;
        } else if (arg == "--max-slices" && i + 1 < argc) {
            maxSlices = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg.rfind("-", 0) != 0 && inputDir.empty()) {
            inputDir = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (inputDir.empty()) {
        std::cerr << "Error: input_dataset_dir is required.\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir)) {
        std::cerr << "Error: Directory does not exist: " << inputDir << "\n";
        return 1;
    }

    if (outputDir.empty()) {
        outputDir = SliceStore::getDefaultPreprocessedDir(inputDir);
    }

    std::cout << "=============================================================\n";
    std::cout << " CT Preprocessing Tool: One-Time Multi-Plane Slice Extractor \n";
    std::cout << "=============================================================\n";
    std::cout << "Input directory : " << inputDir << "\n";
    std::cout << "Output directory: " << outputDir << "\n";
    if (maxSlices > 0) std::cout << "Max slices limit: " << maxSlices << "\n";

    // 1. Validate original TIFF stack
    std::vector<std::string> sliceFiles = SliceIO::listSlices(inputDir, ".tif");
    if (sliceFiles.empty()) {
        std::cerr << "Error: No .tif slices found in input directory: " << inputDir << "\n";
        return 1;
    }
    std::cout << "Found " << sliceFiles.size() << " raw TIFF slice files.\n";

    // 2. Compute dataset fingerprint
    std::string fingerprint = SliceStore::computeDatasetFingerprint(inputDir);
    std::cout << "Dataset Fingerprint: " << fingerprint << "\n";

    // 3. Check for valid existing preprocessed data (reuse check)
    if (!force && SliceStore::isPreprocessed(outputDir)) {
        SliceStore store;
        if (store.open(outputDir) && store.manifest().datasetFingerprint == fingerprint) {
            std::string err;
            if (store.validateAllSlices(err)) {
                std::cout << "\n[ct_preprocess] Dataset is ALREADY preprocessed and up-to-date at:\n"
                          << "  " << outputDir << "\n"
                          << "  Volume dimensions: " << store.dimX() << " x " << store.dimY() << " x " << store.dimZ() << "\n"
                          << "  Slices: " << store.manifest().axialCount << " axial, "
                          << store.manifest().coronalCount << " coronal, "
                          << store.manifest().sagittalCount << " sagittal\n"
                          << "  Reusing valid preprocessed data. (Use --force to regenerate).\n";
                return 0;
            }
        }
    }

    // 4. Load full volume into memory for one-time extraction
    auto startTime = std::chrono::steady_clock::now();
    std::cout << "\n[1/4] Loading volume from original TIFFs...\n";
    VolumeType::Pointer volume = nullptr;
    try {
        volume = VolumeIO::loadVolumeFromDirectory(inputDir, ".tif", maxSlices);
    } catch (const std::exception& e) {
        std::cerr << "Error loading volume: " << e.what() << "\n";
        return 1;
    }

    auto size = volume->GetLargestPossibleRegion().GetSize();
    auto spacing = volume->GetSpacing();
    auto origin = volume->GetOrigin();
    std::cout << "  Loaded volume geometry:\n"
              << "    Dimensions: " << size[0] << " x " << size[1] << " x " << size[2] << " (voxels)\n"
              << "    Spacing   : [" << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << "] mm\n"
              << "    Origin    : [" << origin[0] << ", " << origin[1] << ", " << origin[2] << "]\n";

    // 5. Compute volume intensity range and auto window/level
    std::cout << "\n[2/4] Analyzing intensity statistics...\n";
    using CalcType = itk::MinimumMaximumImageCalculator<VolumeType>;
    auto calc = CalcType::New();
    calc->SetImage(volume);
    calc->Compute();
    double minV = calc->GetMinimum();
    double maxV = calc->GetMaximum();

    double autoWindow = 1000.0, autoLevel = 500.0;
    computeWindowLevel(volume, minV, maxV, autoWindow, autoLevel);
    std::cout << "    Min intensity: " << minV << ", Max intensity: " << maxV << "\n";
    std::cout << "    Auto Window  : " << autoWindow << ", Auto Level: " << autoLevel << "\n";

    // 6. Generate orthogonal slices
    std::cout << "\n[3/4] Generating disk-backed Axial, Coronal, and Sagittal slices...\n";
    auto progressCallback = [](int percent, const std::string& msg) {
        std::cout << "  [" << std::setw(3) << percent << "%] " << msg << "\r" << std::flush;
    };

    bool ok = SliceStore::generateFromVolume(
        volume,
        inputDir,
        outputDir,
        fingerprint,
        autoWindow,
        autoLevel,
        minV,
        maxV,
        progressCallback
    );

    std::cout << "\n";
    if (!ok) {
        std::cerr << "Error: Preprocessing failed during slice generation or validation!\n";
        return 1;
    }

    // 7. Verify generated outputs
    std::cout << "\n[4/4] Validating preprocessed dataset integrity...\n";
    SliceStore verifier;
    if (!verifier.open(outputDir)) {
        std::cerr << "Error: Failed to open preprocessed dataset at " << outputDir << "\n";
        return 1;
    }

    std::string err;
    if (!verifier.validateAllSlices(err)) {
        std::cerr << "Error: Output validation failed: " << err << "\n";
        return 1;
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // Calculate total disk size
    uint64_t totalBytes = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(outputDir)) {
        if (entry.is_regular_file()) totalBytes += entry.file_size();
    }
    double totalMB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);

    std::cout << "\n=============================================================\n";
    std::cout << " PREPROCESSING COMPLETED SUCCESSFULLY!\n";
    std::cout << "=============================================================\n";
    std::cout << "Preprocessed Path : " << outputDir << "\n";
    std::cout << "Manifest File     : " << outputDir << "/manifest.json\n";
    std::cout << "Dimensions        : " << verifier.dimX() << " x " << verifier.dimY() << " x " << verifier.dimZ() << "\n";
    std::cout << "Slices Generated  :\n";
    std::cout << "  - Axial    (Z)  : " << verifier.manifest().axialCount << " slices (" << verifier.dimX() << "x" << verifier.dimY() << " uint16)\n";
    std::cout << "  - Coronal  (Y)  : " << verifier.manifest().coronalCount << " slices (" << verifier.dimX() << "x" << verifier.dimZ() << " uint16)\n";
    std::cout << "  - Sagittal (X)  : " << verifier.manifest().sagittalCount << " slices (" << verifier.dimY() << "x" << verifier.dimZ() << " uint16)\n";
    std::cout << "Total Disk Storage: " << std::fixed << std::setprecision(2) << totalMB << " MB\n";
    std::cout << "Total Elapsed Time: " << (elapsedMs / 1000.0) << " seconds\n";
    std::cout << "=============================================================\n";

    return 0;
}
