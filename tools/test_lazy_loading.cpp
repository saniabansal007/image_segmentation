#include "volume/SliceStore.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <filesystem>
#include <fstream>
#include <cstring>

namespace fs = std::filesystem;
using namespace ctseg;

static int g_failures = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_failures++; \
        } else { \
            std::cout << "  [PASS] " << msg << "\n"; \
        } \
    } while (0)

void test_lru_cache_basic() {
    std::cout << "\n=== Test 1: SliceCache Basic & LRU Eviction ===\n";
    
    // Create cache with 4000 bytes capacity (~4 slices of 1000 bytes each)
    SliceCache cache(4000);
    TEST_ASSERT(cache.maxBytes() == 4000, "Cache capacity is 4000 bytes");
    TEST_ASSERT(cache.currentBytes() == 0, "Cache starts empty");

    // Insert 4 dummy slices of 1000 bytes each (500 uint16 pixels)
    for (int i = 0; i < 4; ++i) {
        auto data = std::make_shared<std::vector<PixelType>>(500, static_cast<PixelType>(i * 100));
        cache.insert({2, i}, data, 250, 2);
    }

    TEST_ASSERT(cache.currentBytes() == 4000, "Cache is full at 4000 bytes");
    TEST_ASSERT(cache.entryCount() == 4, "Cache contains 4 slices");

    // Access slice 0, 1, 2 to make slice 3 the least recently used
    int w, h;
    auto s0 = cache.get({2, 0}, w, h);
    auto s1 = cache.get({2, 1}, w, h);
    auto s2 = cache.get({2, 2}, w, h);
    TEST_ASSERT(s0 != nullptr && s1 != nullptr && s2 != nullptr, "Slices 0, 1, 2 retrieved successfully");

    // Now insert slice 4 (1000 bytes). It should evict slice 3 (the LRU)
    auto data4 = std::make_shared<std::vector<PixelType>>(500, 400);
    cache.insert({2, 4}, data4, 250, 2);

    TEST_ASSERT(cache.currentBytes() == 4000, "Cache size still bounded at 4000 bytes");
    TEST_ASSERT(cache.get({2, 3}, w, h) == nullptr, "Slice 3 was evicted");
    TEST_ASSERT(cache.get({2, 0}, w, h) != nullptr, "Slice 0 is still retained");
    TEST_ASSERT(cache.get({2, 4}, w, h) != nullptr, "Slice 4 is present");

    TEST_ASSERT(cache.hits() >= 3, "Cache recorded hit stats");
    TEST_ASSERT(cache.misses() >= 1, "Cache recorded miss stats");
    TEST_ASSERT(cache.evictions() == 1, "Cache recorded 1 eviction");
}

void test_lru_cache_concurrency() {
    std::cout << "\n=== Test 2: SliceCache Thread-Safety ===\n";
    SliceCache cache(50000); // 50 KB

    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&cache, t]() {
            int w, h;
            for (int i = 0; i < 50; ++i) {
                int sliceIdx = (t * 50 + i) % 30;
                SliceKey key{2, sliceIdx};
                auto existing = cache.get(key, w, h);
                if (!existing) {
                    auto data = std::make_shared<std::vector<PixelType>>(500, static_cast<PixelType>(sliceIdx));
                    cache.insert(key, data, 50, 10);
                }
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    TEST_ASSERT(cache.currentBytes() <= 50000, "Cache remains strictly bounded under concurrent access");
    std::cout << "  Cache after concurrency test: " << cache.entryCount() << " slices, " 
              << cache.currentBytes() << " bytes\n";
}

void test_synthetic_slice_store() {
    std::cout << "\n=== Test 3: SliceStore Metadata, Reading & Geometry ===\n";

    fs::path tempDir = fs::temp_directory_path() / "ct_test_slicestore";
    fs::create_directories(tempDir / "axial");
    fs::create_directories(tempDir / "coronal");
    fs::create_directories(tempDir / "sagittal");

    // Dimensions: X=10, Y=20, Z=30
    const int dimX = 10, dimY = 20, dimZ = 30;
    
    // Create manifest.json
    SliceStoreManifest manifest;
    manifest.formatVersion = 1;
    manifest.dimX = dimX;
    manifest.dimY = dimY;
    manifest.dimZ = dimZ;
    manifest.spacingX = 0.5;
    manifest.spacingY = 0.5;
    manifest.spacingZ = 1.0;
    manifest.originX = 10.0;
    manifest.originY = 20.0;
    manifest.originZ = 30.0;
    manifest.minIntensity = 50.0;
    manifest.maxIntensity = 1050.0;
    manifest.autoWindow = 1000.0;
    manifest.autoLevel = 550.0;
    manifest.axialCount = dimZ;
    manifest.coronalCount = dimY;
    manifest.sagittalCount = dimX;
    manifest.datasetFingerprint = "test_synthetic_fp";
    manifest.isComplete = true;

    TEST_ASSERT(manifest.saveToFile((tempDir / "manifest.json").string()), "Manifest saved successfully");

    auto pad4 = [](int idx) {
        std::ostringstream ss;
        ss << std::setw(4) << std::setfill('0') << idx;
        return ss.str();
    };

    // Synthesize volume data: V(x, y, z) = x + y*10 + z*200
    // Generate axial slices (orientation 2): size = dimX * dimY
    for (int z = 0; z < dimZ; ++z) {
        std::vector<uint16_t> axial(dimX * dimY);
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                axial[x + y * dimX] = static_cast<uint16_t>(x + y * 10 + z * 200);
            }
        }
        std::ofstream out(tempDir / "axial" / ("slice_" + pad4(z) + ".raw"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(axial.data()), axial.size() * sizeof(uint16_t));
    }

    // Generate coronal slices (orientation 1): size = dimX * dimZ
    for (int y = 0; y < dimY; ++y) {
        std::vector<uint16_t> coronal(dimX * dimZ);
        for (int z = 0; z < dimZ; ++z) {
            for (int x = 0; x < dimX; ++x) {
                coronal[x + z * dimX] = static_cast<uint16_t>(x + y * 10 + z * 200);
            }
        }
        std::ofstream out(tempDir / "coronal" / ("slice_" + pad4(y) + ".raw"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(coronal.data()), coronal.size() * sizeof(uint16_t));
    }

    // Generate sagittal slices (orientation 0): size = dimY * dimZ
    for (int x = 0; x < dimX; ++x) {
        std::vector<uint16_t> sagittal(dimY * dimZ);
        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                sagittal[y + z * dimY] = static_cast<uint16_t>(x + y * 10 + z * 200);
            }
        }
        std::ofstream out(tempDir / "sagittal" / ("slice_" + pad4(x) + ".raw"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(sagittal.data()), sagittal.size() * sizeof(uint16_t));
    }

    // Initialize SliceStore with 1 MB cache
    SliceStore store(1 * 1024 * 1024);
    TEST_ASSERT(store.open(tempDir.string()), "SliceStore opened successfully");
    TEST_ASSERT(store.isOpen(), "SliceStore is open");

    // Verify metadata
    const auto& meta = store.manifest();
    TEST_ASSERT(meta.dimX == dimX && meta.dimY == dimY && meta.dimZ == dimZ, "Dimensions match manifest");
    TEST_ASSERT(store.sliceCount(2) == dimZ, "Axial slice count matches dimZ");
    TEST_ASSERT(store.sliceCount(1) == dimY, "Coronal slice count matches dimY");
    TEST_ASSERT(store.sliceCount(0) == dimX, "Sagittal slice count matches dimX");

    // Fetch and check axial slice at z=15 (orientation 2)
    auto ax15 = store.getSlice(2, 15);
    TEST_ASSERT(ax15 != nullptr, "Axial slice 15 retrieved");
    TEST_ASSERT(ax15->size() == static_cast<size_t>(dimX * dimY), "Axial slice 15 size correct");
    // Verify voxel at x=3, y=7 on z=15: expected 3 + 7*10 + 15*200 = 3 + 70 + 3000 = 3073
    TEST_ASSERT((*ax15)[3 + 7 * dimX] == 3073, "Axial slice voxel value matches formula");

    // Fetch and check coronal slice at y=7 (orientation 1)
    auto cor7 = store.getSlice(1, 7);
    TEST_ASSERT(cor7 != nullptr, "Coronal slice 7 retrieved");
    TEST_ASSERT(cor7->size() == static_cast<size_t>(dimX * dimZ), "Coronal slice 7 size correct");
    TEST_ASSERT((*cor7)[3 + 15 * dimX] == 3073, "Coronal slice voxel value matches formula");

    // Fetch and check sagittal slice at x=3 (orientation 0)
    auto sag3 = store.getSlice(0, 3);
    TEST_ASSERT(sag3 != nullptr, "Sagittal slice 3 retrieved");
    TEST_ASSERT(sag3->size() == static_cast<size_t>(dimY * dimZ), "Sagittal slice 3 size correct");
    TEST_ASSERT((*sag3)[7 + 15 * dimY] == 3073, "Sagittal slice voxel value matches formula");

    // Test getVoxelValue helper directly
    double v = store.getVoxelValue(3, 7, 15);
    TEST_ASSERT(static_cast<int>(v) == 3073, "getVoxelValue returned accurate cross-plane voxel value 3073");

    // Test out of bounds handling
    TEST_ASSERT(store.getSlice(2, -1) == nullptr, "Negative slice index rejected");
    TEST_ASSERT(store.getSlice(2, 999) == nullptr, "Excess slice index rejected");

    // Test loadFullVolume directly from disk-backed slices
    auto vol3D = store.loadFullVolume();
    TEST_ASSERT(vol3D != nullptr, "loadFullVolume created non-null ITK volume");
    if (vol3D) {
        auto volSize = vol3D->GetLargestPossibleRegion().GetSize();
        TEST_ASSERT(volSize[0] == dimX && volSize[1] == dimY && volSize[2] == dimZ, "3D volume dimensions match");
        VolumeType::IndexType idx = {3, 7, 15};
        TEST_ASSERT(vol3D->GetPixel(idx) == 3073, "3D volume voxel value matches at (3, 7, 15)");
    }

    // Cleanup tempDir
    fs::remove_all(tempDir);
}

int main(int argc, char** argv) {
    std::cout << "Running CT Lazy Loading & SliceStore Tests...\n";
    test_lru_cache_basic();
    test_lru_cache_concurrency();
    test_synthetic_slice_store();

    std::cout << "\n========================================\n";
    if (g_failures == 0) {
        std::cout << "All tests PASSED!\n";
        return 0;
    } else {
        std::cerr << g_failures << " test(s) FAILED!\n";
        return 1;
    }
}
