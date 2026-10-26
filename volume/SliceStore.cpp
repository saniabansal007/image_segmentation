#include "SliceStore.hpp"
#include "../io/SliceIO.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>

namespace ctseg {

static std::string formatPad4(int index) {
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << index;
    return ss.str();
}

// Simple JSON string value extractor
static std::string extractJsonString(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    auto quoteStart = json.find('"', pos);
    if (quoteStart == std::string::npos) return "";
    auto quoteEnd = json.find('"', quoteStart + 1);
    if (quoteEnd == std::string::npos) return "";
    return json.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
}

// Simple JSON number value extractor
static double extractJsonNumber(const std::string& json, const std::string& key, double defaultVal = 0.0) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return defaultVal;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return defaultVal;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n')) {
        pos++;
    }
    size_t endPos = pos;
    while (endPos < json.size() && (std::isdigit(json[endPos]) || json[endPos] == '.' || json[endPos] == '-' || json[endPos] == '+')) {
        endPos++;
    }
    if (endPos > pos) {
        try {
            return std::stod(json.substr(pos, endPos - pos));
        } catch (...) {}
    }
    return defaultVal;
}

// Simple JSON boolean extractor
static bool extractJsonBool(const std::string& json, const std::string& key, bool defaultVal = false) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return defaultVal;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return defaultVal;
    auto truePos = json.find("true", pos);
    auto falsePos = json.find("false", pos);
    auto commaPos = json.find_first_of(",}\n", pos);
    if (truePos != std::string::npos && (commaPos == std::string::npos || truePos < commaPos)) return true;
    if (falsePos != std::string::npos && (commaPos == std::string::npos || falsePos < commaPos)) return false;
    return defaultVal;
}

// -------------------------------------------------------------------------
// SliceCache Implementation
// -------------------------------------------------------------------------

SliceCache::SliceCache(size_t maxBytes) : maxBytes_(maxBytes) {}

void SliceCache::setMaxBytes(size_t maxBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxBytes_ = maxBytes;
    evictToFit(0);
}

size_t SliceCache::maxBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return maxBytes_;
}

size_t SliceCache::currentBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return currentBytes_;
}

size_t SliceCache::entryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.size();
}

std::shared_ptr<const std::vector<PixelType>> SliceCache::get(const SliceKey& key, int& outWidth, int& outHeight) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
        misses_++;
        return nullptr;
    }
    hits_++;
    // Move to front (most recently used)
    lruList_.splice(lruList_.begin(), lruList_, it->second);
    outWidth = it->second->width;
    outHeight = it->second->height;
    return it->second->data;
}

void SliceCache::insert(const SliceKey& key, std::shared_ptr<std::vector<PixelType>> data, int width, int height) {
    if (!data) return;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t itemBytes = data->size() * sizeof(PixelType);
    if (itemBytes > maxBytes_) return; // Item exceeds total cache capacity

    auto it = map_.find(key);
    if (it != map_.end()) {
        currentBytes_ -= it->second->byteSize();
        lruList_.erase(it->second);
        map_.erase(it);
    }

    evictToFit(itemBytes);

    lruList_.push_front({key, data, width, height});
    map_[key] = lruList_.begin();
    currentBytes_ += itemBytes;
}

void SliceCache::evictToFit(size_t neededBytes) {
    while (!lruList_.empty() && (currentBytes_ + neededBytes > maxBytes_)) {
        auto last = std::prev(lruList_.end());
        currentBytes_ -= last->byteSize();
        map_.erase(last->key);
        lruList_.pop_back();
        evictions_++;
    }
}

void SliceCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    lruList_.clear();
    map_.clear();
    currentBytes_ = 0;
}

double SliceCache::hitRate() const {
    uint64_t h = hits_.load();
    uint64_t m = misses_.load();
    return (h + m == 0) ? 0.0 : static_cast<double>(h) / (h + m) * 100.0;
}

// -------------------------------------------------------------------------
// SliceStoreManifest Implementation
// -------------------------------------------------------------------------

bool SliceStoreManifest::saveToFile(const std::string& filePath) const {
    std::string tempPath = filePath + ".tmp";
    std::ofstream out(tempPath);
    if (!out.is_open()) return false;

    out << "{\n";
    out << "  \"format_version\": " << formatVersion << ",\n";
    out << "  \"dataset_fingerprint\": \"" << datasetFingerprint << "\",\n";
    out << "  \"source_directory\": \"" << sourceDirectory << "\",\n";
    out << "  \"dim_x\": " << dimX << ",\n";
    out << "  \"dim_y\": " << dimY << ",\n";
    out << "  \"dim_z\": " << dimZ << ",\n";
    out << "  \"spacing_x\": " << std::setprecision(6) << spacingX << ",\n";
    out << "  \"spacing_y\": " << std::setprecision(6) << spacingY << ",\n";
    out << "  \"spacing_z\": " << std::setprecision(6) << spacingZ << ",\n";
    out << "  \"origin_x\": " << std::setprecision(6) << originX << ",\n";
    out << "  \"origin_y\": " << std::setprecision(6) << originY << ",\n";
    out << "  \"origin_z\": " << std::setprecision(6) << originZ << ",\n";
    out << "  \"pixel_type\": \"" << pixelType << "\",\n";
    out << "  \"min_intensity\": " << minIntensity << ",\n";
    out << "  \"max_intensity\": " << maxIntensity << ",\n";
    out << "  \"auto_window\": " << autoWindow << ",\n";
    out << "  \"auto_level\": " << autoLevel << ",\n";
    out << "  \"axial_count\": " << axialCount << ",\n";
    out << "  \"coronal_count\": " << coronalCount << ",\n";
    out << "  \"sagittal_count\": " << sagittalCount << ",\n";
    out << "  \"is_complete\": " << (isComplete ? "true" : "false") << "\n";
    out << "}\n";
    out.close();

    std::error_code ec;
    std::filesystem::rename(tempPath, filePath, ec);
    return !ec;
}

bool SliceStoreManifest::loadFromFile(const std::string& filePath, SliceStoreManifest& outManifest) {
    std::ifstream in(filePath);
    if (!in.is_open()) return false;

    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string json = buffer.str();

    outManifest.formatVersion = static_cast<int>(extractJsonNumber(json, "format_version", 1));
    outManifest.datasetFingerprint = extractJsonString(json, "dataset_fingerprint");
    outManifest.sourceDirectory = extractJsonString(json, "source_directory");
    outManifest.dimX = static_cast<int>(extractJsonNumber(json, "dim_x", 0));
    outManifest.dimY = static_cast<int>(extractJsonNumber(json, "dim_y", 0));
    outManifest.dimZ = static_cast<int>(extractJsonNumber(json, "dim_z", 0));
    outManifest.spacingX = extractJsonNumber(json, "spacing_x", 1.0);
    outManifest.spacingY = extractJsonNumber(json, "spacing_y", 1.0);
    outManifest.spacingZ = extractJsonNumber(json, "spacing_z", 1.0);
    outManifest.originX = extractJsonNumber(json, "origin_x", 0.0);
    outManifest.originY = extractJsonNumber(json, "origin_y", 0.0);
    outManifest.originZ = extractJsonNumber(json, "origin_z", 0.0);
    outManifest.pixelType = extractJsonString(json, "pixel_type");
    if (outManifest.pixelType.empty()) outManifest.pixelType = "uint16";
    outManifest.minIntensity = extractJsonNumber(json, "min_intensity", 0.0);
    outManifest.maxIntensity = extractJsonNumber(json, "max_intensity", 65535.0);
    outManifest.autoWindow = extractJsonNumber(json, "auto_window", 1000.0);
    outManifest.autoLevel = extractJsonNumber(json, "auto_level", 500.0);
    outManifest.axialCount = static_cast<int>(extractJsonNumber(json, "axial_count", 0));
    outManifest.coronalCount = static_cast<int>(extractJsonNumber(json, "coronal_count", 0));
    outManifest.sagittalCount = static_cast<int>(extractJsonNumber(json, "sagittal_count", 0));
    outManifest.isComplete = extractJsonBool(json, "is_complete", false);

    return (outManifest.dimX > 0 && outManifest.dimY > 0 && outManifest.dimZ > 0 && outManifest.isComplete);
}

// -------------------------------------------------------------------------
// SliceStore Implementation
// -------------------------------------------------------------------------

SliceStore::SliceStore(size_t cacheMaxBytes) : cache_(cacheMaxBytes) {}

bool SliceStore::isPreprocessed(const std::string& dirPath) {
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        return false;
    }
    std::string manifestFile = (std::filesystem::path(dirPath) / "manifest.json").string();
    if (!std::filesystem::exists(manifestFile)) return false;

    SliceStoreManifest manifest;
    if (!SliceStoreManifest::loadFromFile(manifestFile, manifest)) return false;

    return manifest.isComplete;
}

std::string SliceStore::getDefaultPreprocessedDir(const std::string& datasetPath) {
    std::filesystem::path p(datasetPath);
    if (p.filename().empty()) p = p.parent_path();
    std::string name = p.filename().string();
    return (p.parent_path() / (name + "_preprocessed")).string();
}

std::string SliceStore::computeDatasetFingerprint(const std::string& datasetDir) {
    if (!std::filesystem::exists(datasetDir)) return "";

    std::vector<std::string> files = SliceIO::listSlices(datasetDir, ".tif");
    if (files.empty()) return "";

    uint64_t totalBytes = 0;
    uint64_t hashAcc = 14695981039346656037ULL; // FNV-1a 64-bit basis
    auto fnvAppend = [&hashAcc](const void* data, size_t len) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            hashAcc ^= bytes[i];
            hashAcc *= 1099511628211ULL;
        }
    };

    size_t count = files.size();
    fnvAppend(&count, sizeof(count));

    for (const auto& f : files) {
        std::filesystem::path fp(f);
        std::string fname = fp.filename().string();
        fnvAppend(fname.data(), fname.size());

        std::error_code ec;
        uintmax_t sz = std::filesystem::file_size(fp, ec);
        if (!ec) {
            totalBytes += sz;
            fnvAppend(&sz, sizeof(sz));
        }
    }

    // Sample content from first, middle, and last files for robust detection of modified data
    auto sampleContent = [&](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            char buf[512];
            in.read(buf, sizeof(buf));
            std::streamsize bytesRead = in.gcount();
            if (bytesRead > 0) fnvAppend(buf, static_cast<size_t>(bytesRead));
        }
    };

    sampleContent(files.front());
    sampleContent(files[files.size() / 2]);
    sampleContent(files.back());

    std::ostringstream ss;
    ss << "fp_" << files.size() << "_" << totalBytes << "_" << std::hex << hashAcc;
    return ss.str();
}

bool SliceStore::open(const std::string& preprocessedDir) {
    close();

    std::string manifestPath = (std::filesystem::path(preprocessedDir) / "manifest.json").string();
    if (!SliceStoreManifest::loadFromFile(manifestPath, manifest_)) {
        return false;
    }

    preprocessedDir_ = preprocessedDir;
    isOpen_ = true;
    cache_.clear();
    return true;
}

void SliceStore::close() {
    isOpen_ = false;
    preprocessedDir_.clear();
    manifest_ = SliceStoreManifest();
    cache_.clear();
}

int SliceStore::sliceCount(int orientation) const {
    if (!isOpen_) return 0;
    if (orientation == 2) return manifest_.dimZ;      // Axial
    if (orientation == 1) return manifest_.dimY;      // Coronal
    if (orientation == 0) return manifest_.dimX;      // Sagittal
    return 0;
}

void SliceStore::getSpacing(double outSpacing[3]) const {
    outSpacing[0] = manifest_.spacingX;
    outSpacing[1] = manifest_.spacingY;
    outSpacing[2] = manifest_.spacingZ;
}

void SliceStore::getOrigin(double outOrigin[3]) const {
    outOrigin[0] = manifest_.originX;
    outOrigin[1] = manifest_.originY;
    outOrigin[2] = manifest_.originZ;
}

void SliceStore::getSliceDimensions(int orientation, int& outWidth, int& outHeight) const {
    if (!isOpen_) { outWidth = 0; outHeight = 0; return; }
    if (orientation == 2) { // Axial
        outWidth = manifest_.dimX;
        outHeight = manifest_.dimY;
    } else if (orientation == 1) { // Coronal
        outWidth = manifest_.dimX;
        outHeight = manifest_.dimZ;
    } else if (orientation == 0) { // Sagittal
        outWidth = manifest_.dimY;
        outHeight = manifest_.dimZ;
    } else {
        outWidth = 0;
        outHeight = 0;
    }
}

size_t SliceStore::sliceByteSize(int orientation) const {
    int w = 0, h = 0;
    getSliceDimensions(orientation, w, h);
    return static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(PixelType);
}

std::string SliceStore::getSliceFilePath(int orientation, int index) const {
    std::string subDir;
    if (orientation == 2) subDir = "axial";
    else if (orientation == 1) subDir = "coronal";
    else subDir = "sagittal";

    return (std::filesystem::path(preprocessedDir_) / subDir / ("slice_" + formatPad4(index) + ".raw")).string();
}

std::shared_ptr<const std::vector<PixelType>> SliceStore::getSlice(int orientation, int index) {
    if (!isOpen_) return nullptr;
    if (index < 0 || index >= sliceCount(orientation)) return nullptr;

    SliceKey key{orientation, index};
    int w = 0, h = 0;
    auto cached = cache_.get(key, w, h);
    if (cached) return cached;

    // Cache miss -> read from disk
    auto data = std::make_shared<std::vector<PixelType>>();
    getSliceDimensions(orientation, w, h);
    if (w <= 0 || h <= 0) return nullptr;

    bool ok = readSliceDirect(orientation, index, *data);
    if (!ok) {
        // Safe fallback for missing or corrupted slice
        data->assign(w * h, 0);
    }

    cache_.insert(key, data, w, h);
    return data;
}

bool SliceStore::readSliceDirect(int orientation, int index, std::vector<PixelType>& outBuffer) {
    if (!isOpen_) return false;
    if (index < 0 || index >= sliceCount(orientation)) return false;

    int w = 0, h = 0;
    getSliceDimensions(orientation, w, h);
    if (w <= 0 || h <= 0) return false;

    size_t expectedVoxels = static_cast<size_t>(w) * static_cast<size_t>(h);
    outBuffer.resize(expectedVoxels);

    std::string path = getSliceFilePath(orientation, index);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::fill(outBuffer.begin(), outBuffer.end(), 0);
        return false;
    }

    in.read(reinterpret_cast<char*>(outBuffer.data()), expectedVoxels * sizeof(PixelType));
    if (!in || static_cast<size_t>(in.gcount()) != expectedVoxels * sizeof(PixelType)) {
        // Corrupted / truncated file -> pad remainder with 0 safely
        size_t voxelsRead = in.gcount() / sizeof(PixelType);
        for (size_t i = voxelsRead; i < expectedVoxels; ++i) outBuffer[i] = 0;
        return false;
    }

    return true;
}

double SliceStore::getVoxelValue(int x, int y, int z) {
    if (!isOpen_) return 0.0;
    if (x < 0 || x >= manifest_.dimX ||
        y < 0 || y >= manifest_.dimY ||
        z < 0 || z >= manifest_.dimZ) {
        return 0.0;
    }

    // Check if axial slice z is in cache
    int w = 0, h = 0;
    auto cachedAxial = cache_.get({2, z}, w, h);
    if (cachedAxial && cachedAxial->size() >= static_cast<size_t>(manifest_.dimX * manifest_.dimY)) {
        return static_cast<double>((*cachedAxial)[x + y * manifest_.dimX]);
    }

    // Check coronal slice y
    auto cachedCoronal = cache_.get({1, y}, w, h);
    if (cachedCoronal && cachedCoronal->size() >= static_cast<size_t>(manifest_.dimX * manifest_.dimZ)) {
        return static_cast<double>((*cachedCoronal)[x + z * manifest_.dimX]);
    }

    // Check sagittal slice x
    auto cachedSagittal = cache_.get({0, x}, w, h);
    if (cachedSagittal && cachedSagittal->size() >= static_cast<size_t>(manifest_.dimY * manifest_.dimZ)) {
        return static_cast<double>((*cachedSagittal)[y + z * manifest_.dimY]);
    }

    // Fetch axial slice (which will also cache it for subsequent probes)
    auto slice = getSlice(2, z);
    if (slice && slice->size() >= static_cast<size_t>(manifest_.dimX * manifest_.dimY)) {
        return static_cast<double>((*slice)[x + y * manifest_.dimX]);
    }

    return 0.0;
}

bool SliceStore::validateAllSlices(std::string& outErrorMsg) const {
    if (!isOpen_) {
        outErrorMsg = "SliceStore is not open.";
        return false;
    }

    auto checkCategory = [&](int orientation, int count, const std::string& name) -> bool {
        size_t expectedBytes = sliceByteSize(orientation);
        for (int i = 0; i < count; ++i) {
            std::string path = getSliceFilePath(orientation, i);
            std::error_code ec;
            if (!std::filesystem::exists(path, ec)) {
                outErrorMsg = "Missing " + name + " slice file: " + path;
                return false;
            }
            uintmax_t sz = std::filesystem::file_size(path, ec);
            if (ec || sz != expectedBytes) {
                outErrorMsg = "Corrupted/invalid size for " + name + " slice: " + path +
                              " (expected " + std::to_string(expectedBytes) + ", found " + std::to_string(sz) + ")";
                return false;
            }
        }
        return true;
    };

    if (!checkCategory(2, manifest_.axialCount, "axial")) return false;
    if (!checkCategory(1, manifest_.coronalCount, "coronal")) return false;
    if (!checkCategory(0, manifest_.sagittalCount, "sagittal")) return false;

    return true;
}

VolumeType::Pointer SliceStore::loadFullVolume(std::function<void(int percent, const std::string& msg)> progressCallback) {
    if (!isOpen_ || manifest_.dimX <= 0 || manifest_.dimY <= 0 || manifest_.dimZ <= 0) {
        return nullptr;
    }

    VolumeType::Pointer volume = VolumeType::New();
    VolumeType::RegionType region;
    VolumeType::IndexType start = {0, 0, 0};
    VolumeType::SizeType size = {
        static_cast<VolumeType::SizeValueType>(manifest_.dimX),
        static_cast<VolumeType::SizeValueType>(manifest_.dimY),
        static_cast<VolumeType::SizeValueType>(manifest_.dimZ)
    };
    region.SetSize(size);
    region.SetIndex(start);
    volume->SetRegions(region);

    VolumeType::SpacingType sp;
    sp[0] = manifest_.spacingX;
    sp[1] = manifest_.spacingY;
    sp[2] = manifest_.spacingZ;
    volume->SetSpacing(sp);

    VolumeType::PointType orig;
    orig[0] = manifest_.originX;
    orig[1] = manifest_.originY;
    orig[2] = manifest_.originZ;
    volume->SetOrigin(orig);

    volume->Allocate();

    PixelType* outBuffer = volume->GetBufferPointer();
    if (!outBuffer) return nullptr;

    size_t sliceVoxels = static_cast<size_t>(manifest_.dimX) * static_cast<size_t>(manifest_.dimY);
    size_t sliceBytes = sliceVoxels * sizeof(PixelType);

    unsigned int numThreads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::atomic<int> completedSlices{0};
    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (unsigned int t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            int zStart = static_cast<int>(t * manifest_.dimZ / numThreads);
            int zEnd = static_cast<int>((t + 1) * manifest_.dimZ / numThreads);
            for (int z = zStart; z < zEnd; ++z) {
                std::string slicePath = getSliceFilePath(2, z);
                std::ifstream in(slicePath, std::ios::binary);
                PixelType* destSlice = outBuffer + static_cast<size_t>(z) * sliceVoxels;
                if (in.is_open()) {
                    in.read(reinterpret_cast<char*>(destSlice), sliceBytes);
                    if (!in || static_cast<size_t>(in.gcount()) != sliceBytes) {
                        size_t readVoxels = in.gcount() / sizeof(PixelType);
                        for (size_t i = readVoxels; i < sliceVoxels; ++i) {
                            destSlice[i] = 0;
                        }
                    }
                } else {
                    std::memset(destSlice, 0, sliceBytes);
                }

                int done = ++completedSlices;
                if (progressCallback && (done % 50 == 0 || done == manifest_.dimZ)) {
                    int pct = done * 100 / manifest_.dimZ;
                    progressCallback(pct, "Loading 3D volume: " + std::to_string(done) + "/" + std::to_string(manifest_.dimZ) + " slices");
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }

    return volume;
}

bool SliceStore::generateFromVolume(VolumeType::Pointer volume,
                                   const std::string& sourceDir,
                                   const std::string& outputDir,
                                   const std::string& fingerprint,
                                   double autoWindow,
                                   double autoLevel,
                                   double minIntensity,
                                   double maxIntensity,
                                   std::function<void(int percent, const std::string& msg)> progressCallback) {
    if (!volume) return false;

    auto size = volume->GetLargestPossibleRegion().GetSize();
    auto spacing = volume->GetSpacing();
    auto origin = volume->GetOrigin();

    int dimX = static_cast<int>(size[0]);
    int dimY = static_cast<int>(size[1]);
    int dimZ = static_cast<int>(size[2]);

    if (dimX <= 0 || dimY <= 0 || dimZ <= 0) return false;

    std::filesystem::path outPath(outputDir);
    std::filesystem::create_directories(outPath / "axial");
    std::filesystem::create_directories(outPath / "coronal");
    std::filesystem::create_directories(outPath / "sagittal");

    const PixelType* rawBuffer = volume->GetBufferPointer();
    if (!rawBuffer) return false;

    int totalSteps = dimZ + dimY + dimX;
    int currentStep = 0;

    auto report = [&](const std::string& msg) {
        if (progressCallback) {
            int pct = (totalSteps > 0) ? (currentStep * 100 / totalSteps) : 0;
            progressCallback(pct, msg);
        }
    };

    // 1. Generate Axial Slices (Orientation 2): each slice is width=dimX, height=dimY
    report("Writing axial slices...");
    size_t axialSlicePixels = static_cast<size_t>(dimX) * static_cast<size_t>(dimY);
    for (int z = 0; z < dimZ; ++z) {
        std::string filePath = (outPath / "axial" / ("slice_" + formatPad4(z) + ".raw")).string();
        std::ofstream out(filePath, std::ios::binary);
        if (!out.is_open()) return false;

        const PixelType* slicePtr = rawBuffer + static_cast<size_t>(z) * axialSlicePixels;
        out.write(reinterpret_cast<const char*>(slicePtr), axialSlicePixels * sizeof(PixelType));
        currentStep++;
        if (z % 50 == 0) report("Writing axial slice " + std::to_string(z) + " / " + std::to_string(dimZ));
    }

    // 2. Generate Coronal Slices (Orientation 1): each slice is width=dimX, height=dimZ
    report("Writing coronal slices...");
    size_t coronalSlicePixels = static_cast<size_t>(dimX) * static_cast<size_t>(dimZ);
    std::vector<PixelType> coronalBuffer(coronalSlicePixels);
    for (int y = 0; y < dimY; ++y) {
        std::string filePath = (outPath / "coronal" / ("slice_" + formatPad4(y) + ".raw")).string();
        std::ofstream out(filePath, std::ios::binary);
        if (!out.is_open()) return false;

        for (int z = 0; z < dimZ; ++z) {
            const PixelType* srcRow = rawBuffer + (static_cast<size_t>(z) * dimY + y) * dimX;
            PixelType* dstRow = coronalBuffer.data() + static_cast<size_t>(z) * dimX;
            std::memcpy(dstRow, srcRow, dimX * sizeof(PixelType));
        }

        out.write(reinterpret_cast<const char*>(coronalBuffer.data()), coronalSlicePixels * sizeof(PixelType));
        currentStep++;
        if (y % 50 == 0) report("Writing coronal slice " + std::to_string(y) + " / " + std::to_string(dimY));
    }

    // 3. Generate Sagittal Slices (Orientation 0): each slice is width=dimY, height=dimZ
    report("Writing sagittal slices...");
    size_t sagittalSlicePixels = static_cast<size_t>(dimY) * static_cast<size_t>(dimZ);
    std::vector<PixelType> sagittalBuffer(sagittalSlicePixels);
    for (int x = 0; x < dimX; ++x) {
        std::string filePath = (outPath / "sagittal" / ("slice_" + formatPad4(x) + ".raw")).string();
        std::ofstream out(filePath, std::ios::binary);
        if (!out.is_open()) return false;

        for (int z = 0; z < dimZ; ++z) {
            size_t zOffset = static_cast<size_t>(z) * dimY * dimX;
            for (int y = 0; y < dimY; ++y) {
                sagittalBuffer[static_cast<size_t>(z) * dimY + y] = rawBuffer[zOffset + static_cast<size_t>(y) * dimX + x];
            }
        }

        out.write(reinterpret_cast<const char*>(sagittalBuffer.data()), sagittalSlicePixels * sizeof(PixelType));
        currentStep++;
        if (x % 50 == 0) report("Writing sagittal slice " + std::to_string(x) + " / " + std::to_string(dimX));
    }

    // 4. Validate output files
    report("Validating preprocessed slices...");
    SliceStoreManifest manifest;
    manifest.formatVersion = 1;
    manifest.datasetFingerprint = fingerprint;
    manifest.sourceDirectory = sourceDir;
    manifest.dimX = dimX;
    manifest.dimY = dimY;
    manifest.dimZ = dimZ;
    manifest.spacingX = spacing[0];
    manifest.spacingY = spacing[1];
    manifest.spacingZ = spacing[2];
    manifest.originX = origin[0];
    manifest.originY = origin[1];
    manifest.originZ = origin[2];
    manifest.pixelType = "uint16";
    manifest.minIntensity = minIntensity;
    manifest.maxIntensity = maxIntensity;
    manifest.autoWindow = autoWindow;
    manifest.autoLevel = autoLevel;
    manifest.axialCount = dimZ;
    manifest.coronalCount = dimY;
    manifest.sagittalCount = dimX;
    manifest.isComplete = true;

    std::string manifestFile = (outPath / "manifest.json").string();
    if (!manifest.saveToFile(manifestFile)) {
        return false;
    }

    // Temporary SliceStore verification
    SliceStore verifier;
    if (!verifier.open(outputDir)) return false;

    std::string validationError;
    if (!verifier.validateAllSlices(validationError)) {
        std::cerr << "[SliceStore::generateFromVolume] Validation failed: " << validationError << "\n";
        return false;
    }

    // Sample voxel cross-validation check
    int sampleSeeds[3][3] = {
        {dimX / 2, dimY / 2, dimZ / 2},
        {std::min(10, dimX - 1), std::min(10, dimY - 1), std::min(10, dimZ - 1)},
        {dimX - 1, dimY - 1, dimZ - 1}
    };

    for (int i = 0; i < 3; ++i) {
        int sx = sampleSeeds[i][0], sy = sampleSeeds[i][1], sz = sampleSeeds[i][2];
        PixelType origVal = rawBuffer[(static_cast<size_t>(sz) * dimY + sy) * dimX + sx];
        double storeVal = verifier.getVoxelValue(sx, sy, sz);
        if (static_cast<PixelType>(std::round(storeVal)) != origVal) {
            std::cerr << "[SliceStore::generateFromVolume] Sample voxel mismatch at ("
                      << sx << ", " << sy << ", " << sz << "): original=" << origVal
                      << " vs store=" << storeVal << "\n";
            return false;
        }
    }

    if (progressCallback) progressCallback(100, "Preprocessing completed successfully");
    return true;
}

} // namespace ctseg
