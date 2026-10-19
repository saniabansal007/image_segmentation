#pragma once

#include "VolumeTypes.hpp"
#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <atomic>
#include <functional>
#include <filesystem>
#include <cstdint>

namespace ctseg {

// Orientation constants matching OrthoSliceView:
// 0 = Sagittal (plane perpendicular to X axis)
// 1 = Coronal  (plane perpendicular to Y axis)
// 2 = Axial    (plane perpendicular to Z axis)
enum class SliceOrientation : int {
    Sagittal = 0,
    Coronal = 1,
    Axial = 2
};

struct SliceKey {
    int orientation = 2; // 0, 1, or 2
    int index = 0;       // Slice index along orientation axis

    bool operator==(const SliceKey& other) const {
        return orientation == other.orientation && index == other.index;
    }
};

struct SliceKeyHash {
    std::size_t operator()(const SliceKey& k) const {
        return std::hash<int>()(k.orientation * 1000000 + k.index);
    }
};

struct CachedSlice {
    SliceKey key;
    std::shared_ptr<std::vector<PixelType>> data;
    int width = 0;
    int height = 0;

    size_t byteSize() const {
        return data ? data->size() * sizeof(PixelType) : 0;
    }
};

// Thread-safe, bounded, byte-based Least Recently Used (LRU) cache for 2D slices.
class SliceCache {
public:
    explicit SliceCache(size_t maxBytes = 128 * 1024 * 1024); // Default: 128 MB

    void setMaxBytes(size_t maxBytes);
    size_t maxBytes() const;
    size_t currentBytes() const;
    size_t entryCount() const;

    // Retrieves a cached slice if present; returns nullptr on miss.
    std::shared_ptr<const std::vector<PixelType>> get(const SliceKey& key, int& outWidth, int& outHeight);

    // Inserts or updates a slice, evicting LRU items if byte bounds would be exceeded.
    void insert(const SliceKey& key, std::shared_ptr<std::vector<PixelType>> data, int width, int height);

    void clear();

    uint64_t hits() const { return hits_.load(); }
    uint64_t misses() const { return misses_.load(); }
    uint64_t evictions() const { return evictions_.load(); }
    double hitRate() const;

private:
    void evictToFit(size_t neededBytes);

    mutable std::mutex mutex_;
    size_t maxBytes_;
    size_t currentBytes_ = 0;

    std::list<CachedSlice> lruList_; // Front: Most Recently Used; Back: Least Recently Used
    std::unordered_map<SliceKey, std::list<CachedSlice>::iterator, SliceKeyHash> map_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};
};

// Metadata manifest describing preprocessed dataset geometry, pixel type, and fingerprint.
struct SliceStoreManifest {
    int formatVersion = 1;
    std::string datasetFingerprint;
    std::string sourceDirectory;
    int dimX = 0;
    int dimY = 0;
    int dimZ = 0;
    double spacingX = 1.0;
    double spacingY = 1.0;
    double spacingZ = 1.0;
    double originX = 0.0;
    double originY = 0.0;
    double originZ = 0.0;
    std::string pixelType = "uint16";
    double minIntensity = 0.0;
    double maxIntensity = 65535.0;
    double autoWindow = 1000.0;
    double autoLevel = 500.0;
    int axialCount = 0;
    int coronalCount = 0;
    int sagittalCount = 0;
    bool isComplete = false;

    bool saveToFile(const std::string& filePath) const;
    static bool loadFromFile(const std::string& filePath, SliceStoreManifest& outManifest);
};

// Manages disk-backed slice storage and lazy retrieval for orthogonal viewing.
class SliceStore {
public:
    explicit SliceStore(size_t cacheMaxBytes = 128 * 1024 * 1024);
    ~SliceStore() = default;

    // Checks if directory contains a valid, completed preprocessed dataset.
    static bool isPreprocessed(const std::string& dirPath);

    // Default preprocessed directory name for a given dataset path.
    static std::string getDefaultPreprocessedDir(const std::string& datasetPath);

    // Computes a dataset fingerprint (file count, sorted names, timestamps, and sample data).
    static std::string computeDatasetFingerprint(const std::string& datasetDir);

    // Opens an existing preprocessed dataset directory.
    bool open(const std::string& preprocessedDir);

    // Closes current preprocessed store.
    void close();

    bool isOpen() const { return isOpen_; }
    const std::string& directory() const { return preprocessedDir_; }
    const SliceStoreManifest& manifest() const { return manifest_; }
    SliceCache& cache() { return cache_; }

    int dimX() const { return manifest_.dimX; }
    int dimY() const { return manifest_.dimY; }
    int dimZ() const { return manifest_.dimZ; }
    int sliceCount(int orientation) const;

    void getSpacing(double outSpacing[3]) const;
    void getOrigin(double outOrigin[3]) const;

    // Slice dimensions:
    // Axial (2):    width = dimX, height = dimY
    // Coronal (1):  width = dimX, height = dimZ
    // Sagittal (0): width = dimY, height = dimZ
    void getSliceDimensions(int orientation, int& outWidth, int& outHeight) const;
    size_t sliceByteSize(int orientation) const;

    // Gets slice data, using LRU cache on hit or reading from disk on miss.
    std::shared_ptr<const std::vector<PixelType>> getSlice(int orientation, int index);

    // Reads directly into caller's buffer. Returns false if slice is missing or corrupted.
    bool readSliceDirect(int orientation, int index, std::vector<PixelType>& outBuffer);

    // Reads single voxel intensity at (x, y, z) without loading the full 3D volume into memory.
    double getVoxelValue(int x, int y, int z);

    // Validates that all disk-backed slice files exist and have the exact expected byte sizes.
    bool validateAllSlices(std::string& outErrorMsg) const;

    // Reconstructs and loads the entire 3D ITK volume directly from preprocessed raw axial slices.
    // Extremely fast (<1s) and completely bypasses original TIFF parsing / metadata issues.
    VolumeType::Pointer loadFullVolume(std::function<void(int percent, const std::string& msg)> progressCallback = nullptr);

    // Generates preprocessed slice files from an in-memory 3D VolumeType.
    static bool generateFromVolume(VolumeType::Pointer volume,
                                   const std::string& sourceDir,
                                   const std::string& outputDir,
                                   const std::string& fingerprint,
                                   double autoWindow,
                                   double autoLevel,
                                   double minIntensity,
                                   double maxIntensity,
                                   std::function<void(int percent, const std::string& msg)> progressCallback = nullptr);

    std::string getSliceFilePath(int orientation, int index) const;

private:
    std::string preprocessedDir_;
    SliceStoreManifest manifest_;
    SliceCache cache_;
    bool isOpen_ = false;
};

} // namespace ctseg
