#include "io/SliceIO.hpp"
#include "preprocessing/Preprocessing.hpp"
#include "core/SegmenterFactory.hpp"
#include "evaluation/Evaluator.hpp"

#include "methods/threshold/GlobalOtsuSegmenter.hpp"
#include "methods/threshold/MultiOtsuSegmenter.hpp"
#include "methods/threshold/AdaptiveThresholdSegmenter.hpp"
#include "methods/region_growing/RegionGrowingSegmenter.hpp"
#include "methods/clustering/KMeansSegmenter.hpp"
#include "methods/edge_morphology/CannyMorphologySegmenter.hpp"
#include "methods/watershed/WatershedSegmenter.hpp"
#include "methods/ml/MLSegmenter.hpp"

#include <iostream>
#include <iomanip>

using namespace ctseg;

static void registerAllMethods() {
    auto& f = SegmenterFactory::instance();
    f.registerMethod("global_otsu", [] { return std::make_unique<GlobalOtsuSegmenter>(); });
    f.registerMethod("multi_otsu", [] { return std::make_unique<MultiOtsuSegmenter>(); });
    f.registerMethod("adaptive_threshold", [] { return std::make_unique<AdaptiveThresholdSegmenter>(); });
    f.registerMethod("region_growing", [] { return std::make_unique<RegionGrowingSegmenter>(); });
    f.registerMethod("kmeans", [] { return std::make_unique<KMeansSegmenter>(); });
    f.registerMethod("canny_morphology", [] { return std::make_unique<CannyMorphologySegmenter>(); });
    f.registerMethod("watershed", [] { return std::make_unique<WatershedSegmenter>(); });
    f.registerMethod("ml", [] { return std::make_unique<MLSegmenter>(); });
}

struct Args {
    std::string slicePath;
    std::string method = "all";
    int numClasses = 3;
    std::string groundTruthPath;
    bool useRingCorrection = false;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        throw std::runtime_error("Usage: ct_segmentation <slice.tif> [--method=name|all] "
                                  "[--classes=N] [--groundtruth=mask.png] [--ring-correct]");
    }
    a.slicePath = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto eq = arg.find('=');
        std::string key = (eq != std::string::npos) ? arg.substr(0, eq) : arg;
        std::string val = (eq != std::string::npos) ? arg.substr(eq + 1) : "";
        if (key == "--method") a.method = val;
        else if (key == "--classes") a.numClasses = std::stoi(val);
        else if (key == "--groundtruth") a.groundTruthPath = val;
        else if (key == "--ring-correct") a.useRingCorrection = true;
    }
    return a;
}

int main(int argc, char** argv) {
    registerAllMethods();
    auto& factory = SegmenterFactory::instance();

    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        std::cerr << "Available methods: ";
        for (auto& m : factory.availableMethods()) std::cerr << m << " ";
        std::cerr << "\n";
        return 1;
    }

    std::cout << "Loading slice: " << args.slicePath << "\n";
    ImageType::Pointer img = SliceIO::loadSlice(args.slicePath);
    auto size = img->GetLargestPossibleRegion().GetSize();
    std::cout << "  size: " << size[0] << "x" << size[1] << "\n";

    std::cout << "Preprocessing: denoising (median filter)...\n";
    ImageType::Pointer processed = Preprocessing::denoiseMedian(img, 2);

    if (args.useRingCorrection) {
        std::cout << "Preprocessing: ring artifact suppression...\n";
        processed = Preprocessing::suppressRingArtifacts(processed);
    }

    LabelImageType::Pointer groundTruth;
    if (!args.groundTruthPath.empty()) {
        std::cout << "Loading ground truth: " << args.groundTruthPath << "\n";
        groundTruth = SliceIO::loadLabelPNG(args.groundTruthPath);
    }

    std::string base = args.slicePath.substr(args.slicePath.find_last_of("/\\") + 1);
    base = base.substr(0, base.find_last_of('.'));

    SliceIO::saveAsPNG<ImageType>(img, "output_" + base + "_original.png");
    SliceIO::saveAsPNG<ImageType>(processed, "output_" + base + "_preprocessed.png");

    std::vector<std::string> methodsToRun;
    if (args.method == "all") methodsToRun = factory.availableMethods();
    else methodsToRun = {args.method};

    std::cout << "\n=== Running " << methodsToRun.size() << " method(s) ===\n";
    struct Row { std::string name; double ms; int regions; bool ok; std::string error; };
    std::vector<Row> summary;

    for (const auto& key : methodsToRun) {
        std::cout << "\n--- " << key << " ---\n";
        auto segmenter = factory.create(key);
        ParamMap params = {{"num_classes", static_cast<double>(args.numClasses)},
                            {"num_clusters", static_cast<double>(args.numClasses)}};
        try {
            SegmentationResult result = segmenter->segment(processed, params);
            std::cout << "  method: " << result.method_name << "\n";
            std::cout << "  runtime: " << result.runtime_ms << " ms\n";
            std::cout << "  regions: " << result.num_regions << "\n";

            std::string outPath = "output_" + base + "_" + key + "_segmented.png";
            SliceIO::saveAsPNG<LabelImageType>(result.labels, outPath);
            std::cout << "  saved: " << outPath << "\n";

            if (groundTruth) {
                EvalMetrics m = Evaluator::compare(result.labels, groundTruth);
                Evaluator::printMetrics(key, m);
            }
            summary.push_back({key, result.runtime_ms, result.num_regions, true, ""});
        } catch (const std::exception& e) {
            std::cout << "  SKIPPED: " << e.what() << "\n";
            summary.push_back({key, 0.0, 0, false, e.what()});
        }
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << std::left << std::setw(20) << "Method" << std::setw(12) << "Runtime(ms)"
              << std::setw(10) << "Regions" << "Status\n";
    for (auto& row : summary) {
        std::cout << std::left << std::setw(20) << row.name;
        if (row.ok) {
            std::cout << std::setw(12) << row.ms << std::setw(10) << row.regions << "OK\n";
        } else {
            std::cout << std::setw(12) << "-" << std::setw(10) << "-" << "SKIPPED\n";
        }
    }

    return 0;
}
