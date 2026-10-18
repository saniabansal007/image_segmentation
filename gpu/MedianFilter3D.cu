// 3D median filter on the GPU. This targets exactly the operation that
// was the likely bottleneck in the 900-slice CPU run: itk::MedianImageFilter
// at radius=1 does a 27-neighbor selection per voxel, ~900 million times.
// Each voxel's median is fully independent of every other voxel's, which
// is precisely the kind of embarrassingly-parallel workload a GPU is good
// at - unlike region growing or watershed, which are inherently sequential
// and are NOT ported here (see README's GPU section for why).
//
// NOTE: written and reasoned through carefully, but NOT built or run
// anywhere in this project's development, since no CUDA-capable GPU was
// available in the sandbox this project was developed in. Build and test
// this on your own machine before trusting its output - see gpu/README.md.

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace ctseg_gpu {

// One thread per output voxel. Each thread reads its 3x3x3 (radius=1)
// neighborhood from global memory, sorts the (up to) 27 values with a
// small fixed-size insertion sort (cheaper than std::sort for this size,
// and avoids any dynamic allocation inside the kernel), and writes the
// middle value as the result.
__global__ void medianFilter3DKernel(
        const unsigned short* input, unsigned short* output,
        int width, int height, int depth, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    unsigned short window[27]; // fixed size for radius=1; see note below for other radii
    int count = 0;

    for (int dz = -radius; dz <= radius; ++dz) {
        int zz = z + dz;
        if (zz < 0 || zz >= depth) continue;
        for (int dy = -radius; dy <= radius; ++dy) {
            int yy = y + dy;
            if (yy < 0 || yy >= height) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                int xx = x + dx;
                if (xx < 0 || xx >= width) continue;
                size_t idx = (static_cast<size_t>(zz) * height + yy) * width + xx;
                if (count < 27) window[count++] = input[idx];
            }
        }
    }

    // Small fixed-size insertion sort - faster than a general sort for
    // this few elements, and avoids any per-thread heap allocation.
    for (int i = 1; i < count; ++i) {
        unsigned short key = window[i];
        int j = i - 1;
        while (j >= 0 && window[j] > key) {
            window[j + 1] = window[j];
            --j;
        }
        window[j + 1] = key;
    }

    size_t outIdx = (static_cast<size_t>(z) * height + y) * width + x;
    output[outIdx] = window[count / 2];
}

// Host-side wrapper: allocates device buffers, copies the volume over,
// runs the kernel, copies the result back, and cleans up. Deliberately
// takes/returns plain contiguous buffers (not ITK types) so this file
// has zero ITK/VTK dependency and compiles standalone with nvcc - the
// ITK <-> raw buffer conversion happens in the caller (see
// gpu/GPUPreprocessing.hpp), keeping CUDA code and ITK code cleanly
// separated.
//
// IMPORTANT CAVEAT (see also gpu/README.md): this allocates the FULL
// input and output volumes on the GPU at once (2x the volume size in
// VRAM). A 1000x1000x900 16-bit volume is ~1.8GB, so this needs ~3.6GB+
// free VRAM. If your GPU has less, this will fail with a cudaErrorMemoryAllocation
// - the fix is to process the volume in Z-axis chunks (with radius-sized
// overlap between chunks), which is NOT implemented here; flagged as a
// known follow-up rather than silently guessed at.
void medianFilter3D(const std::vector<unsigned short>& input,
                     std::vector<unsigned short>& output,
                     int width, int height, int depth, int radius) {
    if (radius != 1) {
        throw std::runtime_error(
            "medianFilter3D: only radius=1 is implemented (fixed 27-element "
            "window) - extend the kernel's window[] size and loop bounds "
            "for other radii before using them.");
    }
    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    size_t bytes = numVoxels * sizeof(unsigned short);
    output.resize(numVoxels);

    unsigned short *d_input = nullptr, *d_output = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("cudaMalloc(d_input) failed: ") + cudaGetErrorString(err) +
        " - volume likely too large for available VRAM; see chunking note in this file.");

    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) { cudaFree(d_input); throw std::runtime_error(
        std::string("cudaMalloc(d_output) failed: ") + cudaGetErrorString(err)); }

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8); // 512 threads/block; reasonable default, not auto-tuned
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);

    medianFilter3DKernel<<<grid, block>>>(d_input, d_output, width, height, depth, radius);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input); cudaFree(d_output);
        throw std::runtime_error(std::string("medianFilter3DKernel launch failed: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

} // namespace ctseg_gpu
