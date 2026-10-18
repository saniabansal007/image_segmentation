#pragma once
#include "VolumeTypes.hpp"
#include "../io/SliceIO.hpp"
#include <itkImageSeriesReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

namespace ctseg {

class VolumeIO {
public:
    // Stacks every slice in a directory (sorted by filename, so files must
    // be named so that sorting = scan order, e.g. recon0001.tif,
    // recon0002.tif, ...) into one 3D volume using ITK's ImageSeriesReader.
    //
    // Slices with the malformed-resolution-tag problem (see SliceIO) are
    // sanitized individually first, same as the 2D loader does, since
    // ImageSeriesReader hits the same ITK exception otherwise.
    static VolumeType::Pointer loadVolumeFromDirectory(const std::string& dir,
                                                         const std::string& extension = ".tif",
                                                         size_t maxSlices = 0) {
        std::vector<std::string> files = SliceIO::listSlices(dir, extension);
        if (files.empty()) {
            throw std::runtime_error("No slices found in directory: " + dir);
        }
        if (maxSlices > 0 && files.size() > maxSlices) {
            files.resize(maxSlices); // useful for a quick test on a subset
        }

        // Try direct load first (fast path - avoids double-reading 800+ slices)
        try {
            using ReaderType = itk::ImageSeriesReader<VolumeType>;
            auto reader = ReaderType::New();
            reader->SetFileNames(files);
            reader->Update();
            VolumeType::Pointer volume = reader->GetOutput();
            auto size = volume->GetLargestPossibleRegion().GetSize();
            if (size[2] == files.size()) {
                return volume;
            }
        } catch (const itk::ExceptionObject&) {
            // Direct load failed; fall back to pre-sanitizing files below
        }

        std::vector<std::string> cleanedFiles;
        cleanedFiles.reserve(files.size());
        for (const auto& f : files) {
            cleanedFiles.push_back(ensureReadable(f));
        }

        using ReaderType = itk::ImageSeriesReader<VolumeType>;
        auto reader = ReaderType::New();
        reader->SetFileNames(cleanedFiles);
        reader->Update();

        VolumeType::Pointer volume = reader->GetOutput();
        auto size = volume->GetLargestPossibleRegion().GetSize();
        if (size[2] != cleanedFiles.size()) {
            throw std::runtime_error("Volume stacking mismatch: expected " +
                std::to_string(cleanedFiles.size()) + " slices, got " + std::to_string(size[2]));
        }
        return volume;
    }

    // Saves a single 2D axial slice (index z) as a 16-bit TIFF image,
    // preserving exact pixel intensities and physical spacing.
    static void saveSliceAsTIFF(VolumeType::Pointer volume, unsigned int z, const std::string& path) {
        auto region = volume->GetLargestPossibleRegion();
        auto size = region.GetSize();
        size[2] = 0;
        VolumeType::IndexType start = region.GetIndex();
        start[2] = z;

        VolumeType::RegionType extractRegion;
        extractRegion.SetSize(size);
        extractRegion.SetIndex(start);

        using ExtractType = itk::ExtractImageFilter<VolumeType, itk::Image<PixelType, 2>>;
        auto extractor = ExtractType::New();
        extractor->SetExtractionRegion(extractRegion);
        extractor->SetInput(volume);
        extractor->SetDirectionCollapseToIdentity();
        extractor->Update();

        using WriterType = itk::ImageFileWriter<itk::Image<PixelType, 2>>;
        auto writer = WriterType::New();
        writer->SetFileName(path);
        writer->SetInput(extractor->GetOutput());
        writer->Update();
    }

    // Saves a single 2D axial label mask slice as an 8-bit TIFF image.
    static void saveLabelSliceAsTIFF(LabelVolumeType::Pointer volume, unsigned int z, const std::string& path) {
        auto region = volume->GetLargestPossibleRegion();
        auto size = region.GetSize();
        size[2] = 0;
        LabelVolumeType::IndexType start = region.GetIndex();
        start[2] = z;

        LabelVolumeType::RegionType extractRegion;
        extractRegion.SetSize(size);
        extractRegion.SetIndex(start);

        using ExtractType = itk::ExtractImageFilter<LabelVolumeType, itk::Image<LabelPixelType, 2>>;
        auto extractor = ExtractType::New();
        extractor->SetExtractionRegion(extractRegion);
        extractor->SetInput(volume);
        extractor->SetDirectionCollapseToIdentity();
        extractor->Update();

        using WriterType = itk::ImageFileWriter<itk::Image<LabelPixelType, 2>>;
        auto writer = WriterType::New();
        writer->SetFileName(path);
        writer->SetInput(extractor->GetOutput());
        writer->Update();
    }

    // Saves a single 2D slice with the segmentation mask applied (intensities preserved inside ROI, 0 outside) as 16-bit TIFF.
    static void saveMaskedSliceAsTIFF(VolumeType::Pointer volume, LabelVolumeType::Pointer mask, unsigned int z, const std::string& path) {
        auto region = volume->GetLargestPossibleRegion();
        auto size = region.GetSize();
        size[2] = 0;
        VolumeType::IndexType start = region.GetIndex();
        start[2] = z;

        VolumeType::RegionType extractRegion;
        extractRegion.SetSize(size);
        extractRegion.SetIndex(start);

        using ExtractType = itk::ExtractImageFilter<VolumeType, itk::Image<PixelType, 2>>;
        auto extractor = ExtractType::New();
        extractor->SetExtractionRegion(extractRegion);
        extractor->SetInput(volume);
        extractor->SetDirectionCollapseToIdentity();
        extractor->Update();

        using ExtractMaskType = itk::ExtractImageFilter<LabelVolumeType, itk::Image<LabelPixelType, 2>>;
        auto maskExtractor = ExtractMaskType::New();
        maskExtractor->SetExtractionRegion(extractRegion);
        maskExtractor->SetInput(mask);
        maskExtractor->SetDirectionCollapseToIdentity();
        maskExtractor->Update();

        auto ctSlice = extractor->GetOutput();
        auto maskSlice = maskExtractor->GetOutput();

        auto outSlice = itk::Image<PixelType, 2>::New();
        outSlice->SetRegions(ctSlice->GetLargestPossibleRegion());
        outSlice->SetSpacing(ctSlice->GetSpacing());
        outSlice->SetOrigin(ctSlice->GetOrigin());
        outSlice->Allocate();

        itk::ImageRegionConstIterator<itk::Image<PixelType, 2>> ctIt(ctSlice, ctSlice->GetLargestPossibleRegion());
        itk::ImageRegionConstIterator<itk::Image<LabelPixelType, 2>> maskIt(maskSlice, maskSlice->GetLargestPossibleRegion());
        itk::ImageRegionIterator<itk::Image<PixelType, 2>> outIt(outSlice, outSlice->GetLargestPossibleRegion());

        for (ctIt.GoToBegin(), maskIt.GoToBegin(), outIt.GoToBegin();
             !ctIt.IsAtEnd(); ++ctIt, ++maskIt, ++outIt) {
            outIt.Set(maskIt.Get() > 0 ? ctIt.Get() : 0);
        }

        using WriterType = itk::ImageFileWriter<itk::Image<PixelType, 2>>;
        auto writer = WriterType::New();
        writer->SetFileName(path);
        writer->SetInput(outSlice);
        writer->Update();
    }

    // Exports an entire 3D volume as an ordered series of TIFF slice files in a directory.
    static void saveVolumeAsTiffSeries(VolumeType::Pointer volume, const std::string& outDir, const std::string& prefix = "filtered_recon_") {
        std::filesystem::create_directories(outDir);
        unsigned int numZ = static_cast<unsigned int>(volume->GetLargestPossibleRegion().GetSize()[2]);
        for (unsigned int z = 0; z < numZ; ++z) {
            std::ostringstream ss;
            ss << outDir << "/" << prefix << std::setw(4) << std::setfill('0') << z << ".tif";
            saveSliceAsTIFF(volume, z, ss.str());
        }
    }

    // Exports an entire 3D label mask as an ordered series of TIFF slice files in a directory.
    static void saveLabelVolumeAsTiffSeries(LabelVolumeType::Pointer mask, const std::string& outDir, const std::string& prefix = "mask_slice_") {
        std::filesystem::create_directories(outDir);
        unsigned int numZ = static_cast<unsigned int>(mask->GetLargestPossibleRegion().GetSize()[2]);
        for (unsigned int z = 0; z < numZ; ++z) {
            std::ostringstream ss;
            ss << outDir << "/" << prefix << std::setw(4) << std::setfill('0') << z << ".tif";
            saveLabelSliceAsTIFF(mask, z, ss.str());
        }
    }

    // Exports an entire 3D masked CT volume (original intensities inside ROI, 0 outside) as a TIFF series.
    static void saveMaskedCTVolumeAsTiffSeries(VolumeType::Pointer volume, LabelVolumeType::Pointer mask, const std::string& outDir, const std::string& prefix = "roi_ct_slice_") {
        std::filesystem::create_directories(outDir);
        unsigned int numZ = static_cast<unsigned int>(volume->GetLargestPossibleRegion().GetSize()[2]);
        for (unsigned int z = 0; z < numZ; ++z) {
            std::ostringstream ss;
            ss << outDir << "/" << prefix << std::setw(4) << std::setfill('0') << z << ".tif";
            saveMaskedSliceAsTIFF(volume, mask, z, ss.str());
        }
    }

    // Saves a 3D volume in ITK's native MetaImage format (.mha)
    static void saveVolumeMHA(VolumeType::Pointer volume, const std::string& path) {
        using WriterType = itk::ImageFileWriter<VolumeType>;
        auto writer = WriterType::New();
        writer->SetFileName(path);
        writer->SetInput(volume);
        writer->Update();
    }

    static void saveLabelVolumeMHA(LabelVolumeType::Pointer volume, const std::string& path) {
        using WriterType = itk::ImageFileWriter<LabelVolumeType>;
        auto writer = WriterType::New();
        writer->SetFileName(path);
        writer->SetInput(volume);
        writer->Update();
    }

    // Extracts a single 2D axial slice (index z) from a 3D volume as a PNG
    template <typename TVolume, typename TSliceImage>
    static void saveSliceAsPNG(typename TVolume::Pointer volume, unsigned int z, const std::string& path) {
        auto region = volume->GetLargestPossibleRegion();
        auto size = region.GetSize();
        size[2] = 0;
        typename TVolume::IndexType start = region.GetIndex();
        start[2] = z;

        typename TVolume::RegionType extractRegion;
        extractRegion.SetSize(size);
        extractRegion.SetIndex(start);

        using ExtractType = itk::ExtractImageFilter<TVolume, TSliceImage>;
        auto extractor = ExtractType::New();
        extractor->SetExtractionRegion(extractRegion);
        extractor->SetInput(volume);
        extractor->SetDirectionCollapseToIdentity();
        extractor->Update();

        SliceIO::template saveAsPNG<TSliceImage>(extractor->GetOutput(), path);
    }

private:
    static std::string ensureReadable(const std::string& path) {
        try {
            using ReaderType = itk::ImageFileReader<itk::Image<PixelType, 2>>;
            auto reader = ReaderType::New();
            reader->SetFileName(path);
            reader->Update();
            return path;
        } catch (const itk::ExceptionObject&) {
            return SliceIO::sanitizeResolutionTags(path);
        }
    }
};

} // namespace ctseg
