#pragma once
#include "../core/ISegmenter.hpp"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <functional>

namespace ctseg {

class SliceIO {
public:
    // Loads a single 16-bit grayscale TIFF slice. Some reconstruction
    // software writes malformed XResolution/YResolution tags (e.g. "N/0",
    // a rational with zero denominator) which ITK's TIFF reader refuses,
    // since it would imply zero pixel spacing. We detect that failure and
    // transparently retry after stripping resolution tags from a temp copy,
    // rather than requiring every caller to pre-clean their files.
    static ImageType::Pointer loadSlice(const std::string& path) {
        using ReaderType = itk::ImageFileReader<ImageType>;
        auto reader = ReaderType::New();
        reader->SetFileName(path);
        try {
            reader->Update();
            return reader->GetOutput();
        } catch (const itk::ExceptionObject&) {
            std::string cleaned = sanitizeResolutionTags(path);
            auto reader2 = ReaderType::New();
            reader2->SetFileName(cleaned);
            reader2->Update();
            return reader2->GetOutput();
        }
    }

    // Lists all slice files in a directory matching an extension,
    // sorted in natural order — needed for building a 3D volume.
    //
    // Defensively skips any "*.cleaned.tif" file even though sanitized
    // copies are no longer written into the slice directory at all (see
    // sanitizeResolutionTags) - this guards against stray .cleaned.tif
    // files left behind by an earlier version of this code, which really
    // did write them here and would otherwise get double-counted as
    // extra slices if this directory is reused.
    static std::vector<std::string> listSlices(const std::string& dir,
                                                const std::string& extension = ".tif") {
        std::vector<std::string> files;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const std::string name = entry.path().filename().string();
            if (entry.path().extension() == extension &&
                name.find(".cleaned.tif") == std::string::npos) {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    // Saves any scalar image as a viewable 8-bit PNG, rescaling its
    // intensity range to 0-255 first (handles both raw 16-bit slices
    // and small-range label masks the same way).
    template <typename TImage>
    static void saveAsPNG(typename TImage::Pointer image, const std::string& path) {
        using PNGPixelType = unsigned char;
        using PNGImageType = itk::Image<PNGPixelType, 2>;
        using RescaleType = itk::RescaleIntensityImageFilter<TImage, PNGImageType>;

        auto rescaler = RescaleType::New();
        rescaler->SetInput(image);
        rescaler->SetOutputMinimum(0);
        rescaler->SetOutputMaximum(255);

        using WriterType = itk::ImageFileWriter<PNGImageType>;
        auto writer = WriterType::New();
        writer->SetFileName(path);
        writer->SetInput(rescaler->GetOutput());
        writer->Update();
    }

    // Loads a label mask stored as an 8-bit PNG (e.g. ground truth for
    // evaluation). Values are used as-is; caller decides class meaning.
    static LabelImageType::Pointer loadLabelPNG(const std::string& path) {
        using ReaderType = itk::ImageFileReader<LabelImageType>;
        auto reader = ReaderType::New();
        reader->SetFileName(path);
        reader->Update();
        return reader->GetOutput();
    }

    // Strips XResolution/YResolution/ResolutionUnit tags by invoking a
    // small Python helper and writes a cleaned copy to the SYSTEM TEMP
    // DIRECTORY, keyed by a hash of the original path so repeated calls
    // for the same file reuse the cached copy instead of re-cleaning it.
    //
    // IMPORTANT: this deliberately does NOT write next to the original
    // file anymore. An earlier version wrote "<path>.cleaned.tif" right
    // next to the source slice - since that filename still ends in
    // ".tif", listSlices() (and VolumeIO's directory stacking, which
    // calls this same sanitize path) picked the cleaned copies up as
    // EXTRA slices, interleaving them with the originals in sorted order
    // and corrupting the volume's slice order and count. Writing to a
    // separate temp directory makes that bug structurally impossible,
    // rather than relying on every caller to filter ".cleaned." names.
    static std::string sanitizeResolutionTags(const std::string& path) {
        auto tempDir = std::filesystem::temp_directory_path() / "ctseg_cleaned_tiffs";
        std::filesystem::create_directories(tempDir);

        size_t h = std::hash<std::string>{}(std::filesystem::absolute(path).string());
        std::string cleaned = (tempDir / (std::to_string(h) + ".tif")).string();

        if (std::filesystem::exists(cleaned)) return cleaned; // already cleaned, reuse it

        std::string cmd = "python3 -c \"import tifffile; a = tifffile.imread('" + path +
                           "'); tifffile.imwrite('" + cleaned + "', a)\"";
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
            throw std::runtime_error("Failed to sanitize TIFF metadata for: " + path);
        }
        return cleaned;
    }

};

} // namespace ctseg
