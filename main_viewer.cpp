// Complete Interactive ITK/VTK/Qt Medical CT Image Visualization and Segmentation Application
#include "viewer/CTViewerWindow.hpp"
#include <QApplication>
#include <iostream>
#include <string>
#include <filesystem>

using namespace ctseg;

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("CT Interactive Segmentation Viewer");
    app.setOrganizationName("MedicalImageProcessing");

    CTViewerWindow window;

    std::string inputPath = "";
    bool testRoi = false;
    int testSeed[3] = {500, 500, 50};
    std::string screenshotPath = "";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test-roi" && i + 3 < argc) {
            testRoi = true;
            testSeed[0] = std::stoi(argv[i + 1]);
            testSeed[1] = std::stoi(argv[i + 2]);
            testSeed[2] = std::stoi(argv[i + 3]);
            i += 3;
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshotPath = argv[i + 1];
            i += 1;
        } else if ((arg == "--preprocessed" || arg == "--lazy") && i + 1 < argc) {
            inputPath = argv[i + 1];
            i += 1;
        } else if (arg.rfind("--", 0) != 0 && inputPath.empty()) {
            inputPath = arg;
        }
    }

    // Default: Check data_preprocessed first, then ./data, then ../data
    if (inputPath.empty()) {
        if (std::filesystem::exists("./data_preprocessed/manifest.json")) inputPath = "./data_preprocessed";
        else if (std::filesystem::exists("data_preprocessed/manifest.json")) inputPath = "data_preprocessed";
        else if (std::filesystem::exists("./data")) inputPath = "./data";
        else if (std::filesystem::exists("../data")) inputPath = "../data";
    }

    if (!inputPath.empty()) {
        std::cout << "[CTViewer] Loading dataset from: " << inputPath << "\n";
        if (std::filesystem::is_directory(inputPath)) {
            window.loadVolumeFromDirectory(inputPath);
        } else {
            window.loadSingleSliceFile(inputPath);
        }
    }

    window.show();

    if (testRoi) {
        std::cout << "[CTViewer] Running test region grow at seed: (" 
                  << testSeed[0] << ", " << testSeed[1] << ", " << testSeed[2] << ")\n";
        window.runTestSeedAndGrow(testSeed[0], testSeed[1], testSeed[2]);
    }

    if (!screenshotPath.empty()) {
        std::cout << "[CTViewer] Headless screenshot requested -> saving to " << screenshotPath << "\n";
        window.saveScreenshot(screenshotPath);
        std::cout << "[CTViewer] Screenshot saved successfully. Exiting test mode.\n";
        return 0;
    }

    return app.exec();
}
