#pragma once
#include "../volume/VolumeTypes.hpp"
#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkConfidenceConnectedImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryMorphologicalOpeningImageFilter.h>
#include <itkBinaryMorphologicalClosingImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace ctseg {

// Drives everything the interactive viewer needs around a single
// user-selected region:
//   1. User clicks a point in any view -> setSeed()
//   2. 'r' -> regionGrow() grows a binary ROI from that seed, within a
//      crop window around it (not the whole volume - keeps every
//      operation fast regardless of how large the full scan is)
//   3. 'e'/'d'/'o'/'c' -> erode/dilate/open/close refine that ROI, still
//      scoped to the same crop window
//   4. 's' -> saveAsTiffStack() writes the resulting mask as a TIFF
//      slice stack
//
// Kept entirely free of any GUI/VTK code so it can be built and tested
// on its own (see tools/test_roi_processor.cpp).
class ROIProcessor {
public:
    explicit ROIProcessor(VolumeType::Pointer volume,
                           VolumeType::SizeType cropHalfSize = makeDefaultHalfSize())
        : volume_(volume), cropHalfSize_(cropHalfSize) {
        mask_ = LabelVolumeType::New();
        mask_->SetRegions(volume_->GetLargestPossibleRegion());
        mask_->SetSpacing(volume_->GetSpacing());
        mask_->SetOrigin(volume_->GetOrigin());
        mask_->Allocate();
        mask_->FillBuffer(0);
    }

    void setSeed(VolumeType::IndexType seed) {
        seed_ = seed;
        hasSeed_ = true;
        hasBall_ = false; // a plain seed click cancels any circle limit
        lastCrop_ = computeCropRegion(seed);
        std::cout << "[ROIProcessor] seed set at (" << seed[0] << "," << seed[1] << "," << seed[2] << ")\n";
    }

    // Circle ROI. The user draws a circle in any view (center + radius in
    // voxels); it is stored as a BALL (sphere) around that center so it
    // appears as a circle in all three orthogonal views and survives the 3D
    // erode/open filters (a one-slice-thick disc would be erased by any 3D
    // erosion). The mask inside the working window is replaced by the ball.
    // Region growing afterwards starts at the center and is confined to
    // the ball; erode/dilate/open/close then refine whatever the mask holds.
    void setBallROI(VolumeType::IndexType center, double radius) {
        if (radius < 1.0) radius = 1.0;
        clearROI();
        seed_ = center;
        hasSeed_ = true;
        hasBall_ = true;
        ballRadius_ = radius;

        // Working window = ball bounding box plus a margin so that dilation
        // can still grow beyond the original circle.
        VolumeType::SizeType half;
        itk::SizeValueType h = static_cast<itk::SizeValueType>(std::ceil(radius)) + 8;
        half.Fill(h);
        lastCrop_ = cropAround(center, half);

        itk::ImageRegionIteratorWithIndex<LabelVolumeType> it(mask_, lastCrop_);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            it.Set(insideBall(it.GetIndex()) ? 1 : 0);
        }
        extendFootprint(lastCrop_);
        std::cout << "[ROIProcessor] circle ROI center (" << center[0] << "," << center[1] << "," << center[2]
                  << ") radius " << radius << " voxels\n";
    }

    // Clears the ENTIRE current ROI (everywhere it was ever written), not
    // just the working window - a new ROI must never leave pieces of the
    // previous one behind (seen as a stale square frame around a new circle).
    void clearROI() {
        if (!hasFootprint_) return;
        itk::ImageRegionIterator<LabelVolumeType> it(mask_, footprint_);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) it.Set(0);
        hasFootprint_ = false;
    }

    bool hasBall() const { return hasBall_; }
    double ballRadius() const { return ballRadius_; }
    VolumeType::IndexType seed() const { return seed_; }

    bool hasSeed() const { return hasSeed_; }
    LabelVolumeType::Pointer getMask() const { return mask_; }
    VolumeType::RegionType lastCropRegion() const { return lastCrop_; }

    // Grows a fresh ROI from the current seed (confidence-connected region
    // growing), OVERWRITING whatever was in the crop window before -
    // selecting a new region is meant to start clean.
    bool regionGrow(double multiplier = 2.5, int iterations = 5, unsigned int initRadius = 2) {
        if (!hasSeed_) { warnNoSeed(); return false; }
        clearROI(); // grow a fresh ROI; never leave the previous one behind

        using ExtractType = itk::ExtractImageFilter<VolumeType, VolumeType>;
        auto extractor = ExtractType::New();
        extractor->SetExtractionRegion(lastCrop_);
        extractor->SetInput(volume_);
        extractor->SetDirectionCollapseToIdentity();
        extractor->Update();
        VolumeType::Pointer cropGray = extractor->GetOutput();

        using GrowType = itk::ConfidenceConnectedImageFilter<VolumeType, VolumeType>;
        auto grow = GrowType::New();
        grow->SetInput(cropGray);
        // NOTE: itk::ExtractImageFilter does NOT reindex its output to
        // (0,0,0) - cropGray keeps the SAME global index as lastCrop_
        // (verified while debugging this). So the seed passed here must
        // be the original global seed_, not a crop-relative shift - an
        // earlier version of this code subtracted lastCrop_'s index and
        // silently produced an out-of-range/zero-value lookup instead.
        grow->AddSeed(seed_);
        grow->SetMultiplier(multiplier);
        grow->SetNumberOfIterations(iterations);
        grow->SetInitialNeighborhoodRadius(initRadius);
        grow->SetReplaceValue(1);
        grow->Update();

        using CastType = itk::CastImageFilter<VolumeType, LabelVolumeType>;
        auto caster = CastType::New();
        caster->SetInput(grow->GetOutput());
        caster->Update();

        LabelVolumeType::Pointer grown = caster->GetOutput();
        if (hasBall_) {
            // Confine the grown region to the user's circle.
            itk::ImageRegionIteratorWithIndex<LabelVolumeType> gIt(grown, grown->GetLargestPossibleRegion());
            for (gIt.GoToBegin(); !gIt.IsAtEnd(); ++gIt) {
                if (gIt.Get() != 0 && !insideBall(gIt.GetIndex())) gIt.Set(0);
            }
        }
        pasteCropIntoMask(grown);
        std::cout << "[ROIProcessor] region grow done, crop size "
                  << lastCrop_.GetSize()[0] << "x" << lastCrop_.GetSize()[1] << "x" << lastCrop_.GetSize()[2] << "\n";
        return true;
    }

    bool erode(unsigned int radius = 1) { return applyMorphology(MorphOp::Erode, radius); }
    bool dilate(unsigned int radius = 1) { return applyMorphology(MorphOp::Dilate, radius); }
    bool open(unsigned int radius = 1) { return applyMorphology(MorphOp::Open, radius); }
    bool close(unsigned int radius = 1) { return applyMorphology(MorphOp::Close, radius); }

    // Clears the ROI within the last crop window (not the whole volume).
    // Clears the whole current ROI (kept under this name for the viewer's 'x' key).
    void resetCrop() {
        clearROI();
        std::cout << "[ROIProcessor] ROI cleared\n";
    }

    // Writes the current mask as a TIFF per Z-slice (0/255 for
    // visibility), named roi_slice_0000.tif, roi_slice_0001.tif, ...
    // Only writes slices that actually intersect the last crop window
    // by default, to avoid dumping hundreds of empty slices on a large
    // volume; pass wholeVolume=true to export every slice instead.
    void saveAsTiffStack(const std::string& outDir, bool wholeVolume = false) const {
        std::filesystem::create_directories(outDir);

        auto region = mask_->GetLargestPossibleRegion();
        unsigned int zStart = 0, zEnd = static_cast<unsigned int>(region.GetSize()[2]);
        if (!wholeVolume && hasSeed_) {
            zStart = static_cast<unsigned int>(lastCrop_.GetIndex()[2]);
            zEnd = zStart + static_cast<unsigned int>(lastCrop_.GetSize()[2]);
        }

        using Label2DType = itk::Image<LabelPixelType, 2>;
        using ExtractType = itk::ExtractImageFilter<LabelVolumeType, Label2DType>;
        using RescaleType = itk::RescaleIntensityImageFilter<Label2DType, Label2DType>;
        using WriterType = itk::ImageFileWriter<Label2DType>;

        int written = 0;
        for (unsigned int z = zStart; z < zEnd; ++z) {
            LabelVolumeType::RegionType sliceRegion = region;
            LabelVolumeType::SizeType size = sliceRegion.GetSize();
            size[2] = 0;
            LabelVolumeType::IndexType idx = sliceRegion.GetIndex();
            idx[2] = z;
            sliceRegion.SetSize(size);
            sliceRegion.SetIndex(idx);

            auto extractor = ExtractType::New();
            extractor->SetExtractionRegion(sliceRegion);
            extractor->SetInput(mask_);
            extractor->SetDirectionCollapseToIdentity();

            auto rescaler = RescaleType::New();
            rescaler->SetInput(extractor->GetOutput());
            rescaler->SetOutputMinimum(0);
            rescaler->SetOutputMaximum(255);

            std::ostringstream fname;
            fname << outDir << "/roi_slice_" << std::setfill('0') << std::setw(4) << z << ".tif";

            auto writer = WriterType::New();
            writer->SetFileName(fname.str());
            writer->SetInput(rescaler->GetOutput());
            writer->Update();
            written++;
        }
        std::cout << "[ROIProcessor] saved " << written << " TIFF slice(s) to " << outDir << "\n";
    }

private:
    enum class MorphOp { Erode, Dilate, Open, Close };

    bool applyMorphology(MorphOp op, unsigned int radius) {
        if (!hasSeed_) { warnNoSeed(); return false; }

        using ExtractType = itk::ExtractImageFilter<LabelVolumeType, LabelVolumeType>;
        auto extractor = ExtractType::New();
        extractor->SetExtractionRegion(lastCrop_);
        extractor->SetInput(mask_);
        extractor->SetDirectionCollapseToIdentity();
        extractor->Update();
        LabelVolumeType::Pointer cropMask = extractor->GetOutput();

        using StructElemType = itk::BinaryBallStructuringElement<LabelPixelType, 3>;
        StructElemType structElem;
        structElem.SetRadius(radius);
        structElem.CreateStructuringElement();

        LabelVolumeType::Pointer result;
        switch (op) {
            case MorphOp::Erode: {
                using FilterType = itk::BinaryErodeImageFilter<LabelVolumeType, LabelVolumeType, StructElemType>;
                auto f = FilterType::New();
                f->SetInput(cropMask); f->SetKernel(structElem);
                f->SetErodeValue(1); f->Update();
                result = f->GetOutput();
                break;
            }
            case MorphOp::Dilate: {
                using FilterType = itk::BinaryDilateImageFilter<LabelVolumeType, LabelVolumeType, StructElemType>;
                auto f = FilterType::New();
                f->SetInput(cropMask); f->SetKernel(structElem);
                f->SetDilateValue(1); f->Update();
                result = f->GetOutput();
                break;
            }
            case MorphOp::Open: {
                using FilterType = itk::BinaryMorphologicalOpeningImageFilter<LabelVolumeType, LabelVolumeType, StructElemType>;
                auto f = FilterType::New();
                f->SetInput(cropMask); f->SetKernel(structElem);
                f->SetForegroundValue(1); f->Update();
                result = f->GetOutput();
                break;
            }
            case MorphOp::Close: {
                using FilterType = itk::BinaryMorphologicalClosingImageFilter<LabelVolumeType, LabelVolumeType, StructElemType>;
                auto f = FilterType::New();
                f->SetInput(cropMask); f->SetKernel(structElem);
                f->SetForegroundValue(1); f->Update();
                result = f->GetOutput();
                break;
            }
        }
        pasteCropIntoMask(result);
        std::cout << "[ROIProcessor] morphology op applied (radius=" << radius << ")\n";
        return true;
    }

    // Pastes a crop-sized label image back into the full-size mask_ at
    // lastCrop_'s location. `source`'s index may be either lastCrop_'s
    // real index (as ExtractImageFilter preserves) or 0-based (as a
    // freshly-allocated small image) - we handle both by re-indexing.
    void pasteCropIntoMask(LabelVolumeType::Pointer source) {
        // Copies the crop-sized result into the full mask IN PLACE. An earlier
        // version used itk::PasteImageFilter, which allocates and fills a whole
        // NEW full-size mask on every operation (and keeps the old one alive
        // until the swap) - on a 900-slice volume that is a ~0.9 GB copy per
        // keypress, and doubles the mask's memory at its peak. Writing only
        // the crop's voxels touches just the working window.
        LabelVolumeType::RegionType srcRegion = source->GetLargestPossibleRegion();
        LabelVolumeType::RegionType dstRegion;
        dstRegion.SetIndex(lastCrop_.GetIndex());
        dstRegion.SetSize(srcRegion.GetSize());
        itk::ImageRegionConstIterator<LabelVolumeType> sIt(source, srcRegion);
        itk::ImageRegionIterator<LabelVolumeType> dIt(mask_, dstRegion);
        for (sIt.GoToBegin(), dIt.GoToBegin(); !sIt.IsAtEnd() && !dIt.IsAtEnd(); ++sIt, ++dIt) {
            dIt.Set(sIt.Get());
        }
        extendFootprint(dstRegion);
    }

    // Bounding box of every voxel range ever written into the mask for the
    // current ROI, so clearROI() can wipe it all without touching the rest.
    void extendFootprint(const LabelVolumeType::RegionType& r) {
        if (!hasFootprint_) { footprint_ = r; hasFootprint_ = true; return; }
        LabelVolumeType::IndexType lo, hi;
        for (unsigned int d = 0; d < 3; ++d) {
            long a0 = footprint_.GetIndex()[d], a1 = a0 + static_cast<long>(footprint_.GetSize()[d]) - 1;
            long b0 = r.GetIndex()[d],          b1 = b0 + static_cast<long>(r.GetSize()[d]) - 1;
            lo[d] = std::min(a0, b0);
            hi[d] = std::max(a1, b1);
        }
        LabelVolumeType::SizeType size;
        for (unsigned int d = 0; d < 3; ++d) size[d] = static_cast<itk::SizeValueType>(hi[d] - lo[d] + 1);
        footprint_.SetIndex(lo);
        footprint_.SetSize(size);
    }

    VolumeType::RegionType computeCropRegion(VolumeType::IndexType seed) const {
        return cropAround(seed, cropHalfSize_);
    }

    VolumeType::RegionType cropAround(VolumeType::IndexType center, VolumeType::SizeType half) const {
        auto fullRegion = volume_->GetLargestPossibleRegion();
        VolumeType::IndexType start;
        VolumeType::SizeType size;
        for (unsigned int d = 0; d < 3; ++d) {
            long lo = static_cast<long>(center[d]) - static_cast<long>(half[d]);
            long hi = static_cast<long>(center[d]) + static_cast<long>(half[d]);
            long fullLo = fullRegion.GetIndex()[d];
            long fullHi = fullLo + static_cast<long>(fullRegion.GetSize()[d]) - 1;
            lo = std::max(lo, fullLo);
            hi = std::min(hi, fullHi);
            start[d] = lo;
            size[d] = static_cast<itk::SizeValueType>(hi - lo + 1);
        }
        VolumeType::RegionType region;
        region.SetIndex(start);
        region.SetSize(size);
        return region;
    }

    bool insideBall(const VolumeType::IndexType& idx) const {
        double dx = static_cast<double>(idx[0]) - static_cast<double>(seed_[0]);
        double dy = static_cast<double>(idx[1]) - static_cast<double>(seed_[1]);
        double dz = static_cast<double>(idx[2]) - static_cast<double>(seed_[2]);
        return dx * dx + dy * dy + dz * dz <= ballRadius_ * ballRadius_;
    }

    static VolumeType::SizeType makeDefaultHalfSize() {
        VolumeType::SizeType s;
        s[0] = 60; s[1] = 60; s[2] = 15; // voxels either side of the seed
        return s;
    }

    void warnNoSeed() const {
        std::cout << "[ROIProcessor] no seed set yet - click a point in a view first.\n";
    }

    VolumeType::Pointer volume_;
    LabelVolumeType::Pointer mask_;
    VolumeType::SizeType cropHalfSize_;
    VolumeType::IndexType seed_{};
    bool hasSeed_ = false;
    bool hasBall_ = false;
    bool hasFootprint_ = false;
    LabelVolumeType::RegionType footprint_;
    double ballRadius_ = 0.0;
    VolumeType::RegionType lastCrop_;
};

} // namespace ctseg
