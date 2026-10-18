// 3D binary erode/dilate on the GPU, matching the same structuring-element
// radius convention as itk::BinaryBallStructuringElement used in
// ROIProcessor's CPU morphology ops. Open/close are composed from these
// two on the host side (erode-then-dilate, dilate-then-erode) rather than
// duplicated as separate kernels.
//
// NOT built or run anywhere in this project's development (no GPU
// available in the sandbox) - see gpu/README.md before relying on this.

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace ctseg_gpu {

enum class MorphOp { Erode, Dilate };

// One thread per voxel. Erode: output is foreground only if EVERY voxel
// in the spherical neighborhood (radius r) is foreground. Dilate: output
// is foreground if ANY voxel in the neighborhood is foreground. Uses a
// spherical (not cubic) neighborhood test (dx^2+dy^2+dz^2 <= r^2) to
// match itk::BinaryBallStructuringElement's shape, not just its bounding box.
__global__ void morphologyKernel(
        const unsigned char* input, unsigned char* output,
        int width, int height, int depth, int radius, int opIsDilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int rSq = radius * radius;
    bool result = opIsDilate ? false : true; // dilate starts "off", erode starts "on"

    for (int dz = -radius; dz <= radius && (opIsDilate ? !result : result); ++dz) {
        int zz = z + dz;
        if (zz < 0 || zz >= depth) { if (!opIsDilate) result = false; continue; }
        for (int dy = -radius; dy <= radius; ++dy) {
            int yy = y + dy;
            bool yOOB = (yy < 0 || yy >= height);
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx*dx + dy*dy + dz*dz > rSq) continue; // outside the sphere
                int xx = x + dx;
                bool outOfBounds = yOOB || xx < 0 || xx >= width;
                unsigned char v = outOfBounds ? 0 : input[(static_cast<size_t>(zz) * height + yy) * width + xx];
                if (opIsDilate) {
                    if (v > 0) { result = true; }
                } else {
                    if (v == 0) { result = false; }
                }
            }
        }
    }

    output[(static_cast<size_t>(z) * height + y) * width + x] = result ? 1 : 0;
}

void morphology3D(const std::vector<unsigned char>& input,
                   std::vector<unsigned char>& output,
                   int width, int height, int depth, int radius, MorphOp op) {
    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    size_t bytes = numVoxels * sizeof(unsigned char);
    output.resize(numVoxels);

    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("cudaMalloc(d_input) failed: ") + cudaGetErrorString(err));
    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) { cudaFree(d_input); throw std::runtime_error(
        std::string("cudaMalloc(d_output) failed: ") + cudaGetErrorString(err)); }

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);

    morphologyKernel<<<grid, block>>>(d_input, d_output, width, height, depth, radius,
                                       op == MorphOp::Dilate ? 1 : 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input); cudaFree(d_output);
        throw std::runtime_error(std::string("morphologyKernel launch failed: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

// Open = erode then dilate; Close = dilate then erode. Composed here on
// the host from the two primitives above rather than as separate kernels.
inline void morphologyOpen3D(const std::vector<unsigned char>& input, std::vector<unsigned char>& output,
                              int width, int height, int depth, int radius) {
    std::vector<unsigned char> temp;
    morphology3D(input, temp, width, height, depth, radius, MorphOp::Erode);
    morphology3D(temp, output, width, height, depth, radius, MorphOp::Dilate);
}

inline void morphologyClose3D(const std::vector<unsigned char>& input, std::vector<unsigned char>& output,
                               int width, int height, int depth, int radius) {
    std::vector<unsigned char> temp;
    morphology3D(input, temp, width, height, depth, radius, MorphOp::Dilate);
    morphology3D(temp, output, width, height, depth, radius, MorphOp::Erode);
}

} // namespace ctseg_gpu
