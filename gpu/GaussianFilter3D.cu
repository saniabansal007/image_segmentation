#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ctseg_gpu {

#define MAX_KERNEL_RADIUS 16
#define MAX_KERNEL_SIZE (2 * MAX_KERNEL_RADIUS + 1)

__constant__ float d_KernelWeights[MAX_KERNEL_SIZE];

// Pass 1: Convolution along X axis
__global__ void gaussianConvXKernel(
        const unsigned short* input, float* output,
        int width, int height, int depth, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    size_t sliceOffset = (static_cast<size_t>(z) * height + y) * width;
    float sum = 0.0f;

    for (int k = -radius; k <= radius; ++k) {
        int nx = x + k;
        if (nx < 0) nx = 0;
        else if (nx >= width) nx = width - 1;

        float w = d_KernelWeights[k + radius];
        sum += w * static_cast<float>(input[sliceOffset + nx]);
    }

    output[sliceOffset + x] = sum;
}

// Pass 2: Convolution along Y axis
__global__ void gaussianConvYKernel(
        const float* input, float* output,
        int width, int height, int depth, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    size_t zOffset = static_cast<size_t>(z) * height * width;
    float sum = 0.0f;

    for (int k = -radius; k <= radius; ++k) {
        int ny = y + k;
        if (ny < 0) ny = 0;
        else if (ny >= height) ny = height - 1;

        float w = d_KernelWeights[k + radius];
        sum += w * input[zOffset + static_cast<size_t>(ny) * width + x];
    }

    output[zOffset + static_cast<size_t>(y) * width + x] = sum;
}

// Pass 3: Convolution along Z axis, writing back to unsigned short
__global__ void gaussianConvZKernel(
        const float* input, unsigned short* output,
        int width, int height, int depth, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    size_t xyOffset = static_cast<size_t>(y) * width + x;
    size_t sliceSize = static_cast<size_t>(height) * width;
    float sum = 0.0f;

    for (int k = -radius; k <= radius; ++k) {
        int nz = z + k;
        if (nz < 0) nz = 0;
        else if (nz >= depth) nz = depth - 1;

        float w = d_KernelWeights[k + radius];
        sum += w * input[static_cast<size_t>(nz) * sliceSize + xyOffset];
    }

    float clamped = fminf(65535.0f, fmaxf(0.0f, sum + 0.5f));
    output[static_cast<size_t>(z) * sliceSize + xyOffset] = static_cast<unsigned short>(clamped);
}

void gaussianFilter3D(const std::vector<unsigned short>& input,
                      std::vector<unsigned short>& output,
                      int width, int height, int depth, double sigma) {
    if (sigma <= 0.0) {
        output = input;
        return;
    }

    int radius = std::min(MAX_KERNEL_RADIUS, std::max(1, static_cast<int>(std::ceil(3.0 * sigma))));
    int kernelSize = 2 * radius + 1;
    std::vector<float> h_weights(kernelSize);
    float sumW = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        float val = std::exp(-0.5f * (static_cast<float>(k * k)) / static_cast<float>(sigma * sigma));
        h_weights[k + radius] = val;
        sumW += val;
    }
    for (int i = 0; i < kernelSize; ++i) {
        h_weights[i] /= sumW;
    }

    cudaMemcpyToSymbol(d_KernelWeights, h_weights.data(), kernelSize * sizeof(float));

    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    size_t ushortBytes = numVoxels * sizeof(unsigned short);
    size_t floatBytes = numVoxels * sizeof(float);
    output.resize(numVoxels);

    unsigned short* d_in = nullptr;
    unsigned short* d_out = nullptr;
    float* d_tmp1 = nullptr;
    float* d_tmp2 = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_in, ushortBytes);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&d_out, ushortBytes);
    if (err != cudaSuccess) { cudaFree(d_in); throw std::runtime_error("cudaMalloc d_out failed: " + std::string(cudaGetErrorString(err))); }
    err = cudaMalloc(&d_tmp1, floatBytes);
    if (err != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); throw std::runtime_error("cudaMalloc d_tmp1 failed: " + std::string(cudaGetErrorString(err))); }
    err = cudaMalloc(&d_tmp2, floatBytes);
    if (err != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp1); throw std::runtime_error("cudaMalloc d_tmp2 failed: " + std::string(cudaGetErrorString(err))); }

    cudaMemcpy(d_in, input.data(), ushortBytes, cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);

    gaussianConvXKernel<<<grid, block>>>(d_in, d_tmp1, width, height, depth, radius);
    cudaDeviceSynchronize();

    gaussianConvYKernel<<<grid, block>>>(d_tmp1, d_tmp2, width, height, depth, radius);
    cudaDeviceSynchronize();

    gaussianConvZKernel<<<grid, block>>>(d_tmp2, d_out, width, height, depth, radius);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_out, ushortBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
}

} // namespace ctseg_gpu
