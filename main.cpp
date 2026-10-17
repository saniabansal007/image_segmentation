#include "io/SliceIO.hpp"
#include "preprocessing/Preprocessing.hpp"
#include "core/SegmenterFactory.hpp"

#include "methods/threshold/GlobalOtsuSegmenter.hpp"
#include "methods/threshold/MultiOtsuSegmenter.hpp"
#include "methods/threshold/AdaptiveThresholdSegmenter.hpp"
#include "methods/region_growing/RegionGrowingSegmenter.hpp"

#include <iostream>
#include <iomanip>

using namespace ctseg;

static void registerAllMethods() {
    auto& f = SegmenterFactory::instance();
    f.registerMethod("global_otsu", [] { return std::make_unique<GlobalOtsuSegmenter>(); });
    f.registerMethod("multi_otsu", [] { return std::make_unique<MultiOtsuSegmenter>(); });
    f.registerMethod("adaptive_threshold", [] { return std::make_unique<AdaptiveThresholdSegmenter>(); });
    f.registerMethod("region_growing", [] { return std::make_unique<RegionGrowingSegmenter>(); });
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ct_segmentation <slice.tif> [--method=name|all] [--classes=N] [--ring-correct]\n";
        return 1;
    }
    registerAllMethods();
    auto& factory = SegmenterFactory::instance();

    std::string slicePath = argv[1];
    std::string method = "all";
    int numClasses = 3;
    bool ringCorrect = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--method=", 0) == 0) method = arg.substr(9);
        else if (arg.rfind("--classes=", 0) == 0) numClasses = std::stoi(arg.substr(10));
        else if (arg == "--ring-correct") ringCorrect = true;
    }

    std::cout << "Loading slice: " << slicePath << "\n";
    ImageType::Pointer img = SliceIO::loadSlice(slicePath);
    auto size = img->GetLargestPossibleRegion().GetSize();
    std::cout << "  size: " << size[0] << "x" << size[1] << "\n";

    std::cout << "Preprocessing: denoising (median filter)...\n";
    ImageType::Pointer processed = Preprocessing::denoiseMedian(img, 2);
    if (ringCorrect) {
        std::cout << "Preprocessing: ring artifact suppression...\n";
        processed = Preprocessing::suppressRingArtifacts(processed);
    }

    std::string base = slicePath.substr(slicePath.find_last_of("/\\") + 1);
    base = base.substr(0, base.find_last_of('.'));
    SliceIO::saveAsPNG<ImageType>(img, "output_" + base + "_original.png");
    SliceIO::saveAsPNG<ImageType>(processed, "output_" + base + "_preprocessed.png");

    std::vector<std::string> toRun;
    if (method == "all") toRun = factory.availableMethods();
    else toRun = {method};

    std::cout << "\n=== Running " << toRun.size() << " method(s) ===\n";
    struct Row { std::string name; double ms; int regions; };
    std::vector<Row> summary;

    for (const auto& key : toRun) {
        std::cout << "\n--- " << key << " ---\n";
        auto segmenter = factory.create(key);
        ParamMap params = {{"num_classes", static_cast<double>(numClasses)}};
        SegmentationResult r = segmenter->segment(processed, params);
        std::cout << "  runtime: " << r.runtime_ms << " ms\n  regions: " << r.num_regions << "\n";
        std::string out = "output_" + base + "_" + key + "_segmented.png";
        SliceIO::saveAsPNG<LabelImageType>(r.labels, out);
        std::cout << "  saved: " << out << "\n";
        summary.push_back({key, r.runtime_ms, r.num_regions});
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << std::left << std::setw(22) << "Method" << std::setw(14) << "Runtime(ms)" << "Regions\n";
    for (auto& row : summary)
        std::cout << std::left << std::setw(22) << row.name << std::setw(14) << row.ms << row.regions << "\n";
    return 0;
}