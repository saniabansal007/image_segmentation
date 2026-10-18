// Standalone test of ROIProcessor's core logic, with no VTK/GUI involved.
// Run after building: ./test_roi_processor <slice_dir> seedX seedY seedZ
#include "../volume/VolumeIO.hpp"
#include "../viewer/ROIProcessor.hpp"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <iostream>

using namespace ctseg;

static long countForeground(LabelVolumeType::Pointer mask, VolumeType::RegionType region) {
    itk::ImageRegionConstIterator<LabelVolumeType> it(mask, region);
    long count = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) if (it.Get() > 0) count++;
    return count;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <slice_dir> seedX seedY seedZ\n";
        return 1;
    }
    std::string dir = argv[1];
    VolumeType::IndexType seed;
    seed[0] = std::stol(argv[2]);
    seed[1] = std::stol(argv[3]);
    seed[2] = std::stol(argv[4]);

    std::cout << "Loading volume from: " << dir << "\n";
    VolumeType::Pointer volume = VolumeIO::loadVolumeFromDirectory(dir);
    auto size = volume->GetLargestPossibleRegion().GetSize();
    std::cout << "  volume: " << size[0] << "x" << size[1] << "x" << size[2] << "\n";

    // NOTE: our only test data is 2 non-adjacent slices (not a real
    // coherent scan - see README), so a 3D crop spanning both Z indices
    // would mix two unrelated images and break region growing's local
    // statistics. Using a single-slice-thick crop (z half-size 0) here
    // keeps this test meaningful until a real sequential slice stack is
    // used, at which point the default 3D half-size is appropriate.
    VolumeType::SizeType halfSize;
    halfSize[0] = 60; halfSize[1] = 60; halfSize[2] = (size[2] <= 2) ? 0 : 15;
    ROIProcessor proc(volume, halfSize);
    proc.setSeed(seed);

    std::cout << "\n--- regionGrow() ---\n";
    bool ok = proc.regionGrow();
    std::cout << "  success: " << (ok ? "yes" : "no") << "\n";
    long countGrow = countForeground(proc.getMask(), proc.lastCropRegion());
    std::cout << "  foreground voxels in crop after grow: " << countGrow << "\n";
    if (countGrow == 0) {
        std::cerr << "  FAIL: region growing produced an empty mask - check seed location.\n";
        return 1;
    }

    std::cout << "\n--- dilate(2) ---\n";
    proc.dilate(2);
    long countDilate = countForeground(proc.getMask(), proc.lastCropRegion());
    std::cout << "  foreground voxels after dilate: " << countDilate
               << (countDilate >= countGrow ? "  [OK: grew or stayed same, as expected]" : "  [FAIL: should not shrink]") << "\n";

    std::cout << "\n--- erode(2) on the dilated result ---\n";
    proc.erode(2);
    long countErode = countForeground(proc.getMask(), proc.lastCropRegion());
    std::cout << "  foreground voxels after erode: " << countErode
               << (countErode <= countDilate ? "  [OK: shrank or stayed same, as expected]" : "  [FAIL: should not grow]") << "\n";

    std::cout << "\n--- re-grow, then open(1) ---\n";
    proc.setSeed(seed);
    proc.regionGrow();
    long beforeOpen = countForeground(proc.getMask(), proc.lastCropRegion());
    proc.open(1);
    long afterOpen = countForeground(proc.getMask(), proc.lastCropRegion());
    std::cout << "  foreground before/after open: " << beforeOpen << " -> " << afterOpen
               << (afterOpen <= beforeOpen ? "  [OK: opening should not add voxels]" : "  [FAIL]") << "\n";

    std::cout << "\n--- close(1) ---\n";
    long beforeClose = afterOpen;
    proc.close(1);
    long afterClose = countForeground(proc.getMask(), proc.lastCropRegion());
    std::cout << "  foreground before/after close: " << beforeClose << " -> " << afterClose
               << (afterClose >= beforeClose ? "  [OK: closing should not remove voxels]" : "  [FAIL]") << "\n";

    std::cout << "\n--- saveAsTiffStack() ---\n";
    proc.saveAsTiffStack("./test_roi_output");
    std::cout << "  check ./test_roi_output/ for roi_slice_*.tif files\n";

    std::cout << "\n--- circle (ball) ROI ---\n";
    {
        const double R = 20.0;
        ROIProcessor p2(volume, halfSize);
        p2.setBallROI(seed, R);
        auto crop = p2.lastCropRegion();
        // brute-force expected size of the discrete ball, clipped to the volume
        long expected = 0, outside = 0, fg = 0;
        itk::ImageRegionConstIteratorWithIndex<LabelVolumeType> it(p2.getMask(), crop);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            auto i = it.GetIndex();
            double dx = double(i[0]) - seed[0], dy = double(i[1]) - seed[1], dz = double(i[2]) - seed[2];
            bool in = dx*dx + dy*dy + dz*dz <= R*R;
            if (in) expected++;
            if (it.Get() > 0) { fg++; if (!in) outside++; }
        }
        std::cout << "  ball voxels: " << fg << " (expected " << expected << ")"
                  << (fg == expected && outside == 0 ? "  [OK]" : "  [FAIL]") << "\n";
        if (fg != expected || outside != 0) return 1;

        p2.regionGrow();
        long grownFg = 0, leaked = 0;
        itk::ImageRegionConstIteratorWithIndex<LabelVolumeType> g(p2.getMask(), p2.lastCropRegion());
        for (g.GoToBegin(); !g.IsAtEnd(); ++g) {
            if (g.Get() == 0) continue;
            grownFg++;
            auto i = g.GetIndex();
            double dx = double(i[0]) - seed[0], dy = double(i[1]) - seed[1], dz = double(i[2]) - seed[2];
            if (dx*dx + dy*dy + dz*dz > R*R) leaked++;
        }
        std::cout << "  region grow inside circle: " << grownFg << " voxels, " << leaked << " outside the circle"
                  << (leaked == 0 && grownFg > 0 ? "  [OK: confined to the circle]" : "  [FAIL]") << "\n";
        if (leaked != 0 || grownFg == 0) return 1;

        p2.setBallROI(seed, R);               // fresh ball again
        long b0 = countForeground(p2.getMask(), p2.lastCropRegion());
        p2.erode(2);
        long b1 = countForeground(p2.getMask(), p2.lastCropRegion());
        p2.dilate(2);
        long b2 = countForeground(p2.getMask(), p2.lastCropRegion());
        std::cout << "  ball erode->dilate: " << b0 << " -> " << b1 << " -> " << b2
                  << (b1 < b0 && b1 > 0 && b2 > b1 ? "  [OK: ball survives 3D erosion]" : "  [FAIL]") << "\n";
        if (!(b1 < b0 && b1 > 0 && b2 > b1)) return 1;
    }

    std::cout << "\n--- one ROI at a time: new circle replaces a previous seed ROI ---\n";
    {
        ROIProcessor p3(volume, halfSize);
        p3.setSeed(seed);
        p3.regionGrow();                    // seed ROI (large)
        VolumeType::IndexType c2 = seed; c2[0] += 70;   // a circle elsewhere
        p3.setBallROI(c2, 10.0);
        long total = countForeground(p3.getMask(), volume->GetLargestPossibleRegion());
        long ballOnly = 0;
        itk::ImageRegionConstIteratorWithIndex<LabelVolumeType> it(p3.getMask(), p3.lastCropRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) if (it.Get() > 0) ballOnly++;
        std::cout << "  foreground in whole volume: " << total << ", inside circle window: " << ballOnly
                  << (total == ballOnly ? "  [OK: no leftovers from the previous ROI]" : "  [FAIL: stale ROI left behind]") << "\n";
        if (total != ballOnly) return 1;
        p3.clearROI();
        long afterClear = countForeground(p3.getMask(), volume->GetLargestPossibleRegion());
        std::cout << "  after clearROI(): " << afterClear << (afterClear == 0 ? "  [OK]" : "  [FAIL]") << "\n";
        if (afterClear != 0) return 1;
    }

    std::cout << "\nAll ROIProcessor checks completed.\n";
    return 0;
}
