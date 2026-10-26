// GPU-accelerated 3D CT volume segmentation kernels using CUDA.
// Supports:
// 1. 3D Binary Threshold Range
// 2. 3D Global Otsu Segmentation (GPU histogram + optimal threshold)
// 3. 3D Multi-Otsu Segmentation (GPU histogram + multi-class thresholds)
// 4. 3D Connected Threshold / Region Growing (parallel wavefront propagation)
// 5. 3D Foreground Voxel Counting (GPU reduction)

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ctseg_gpu {

// ---------------------------------------------------------------------------
// 1. Binary Threshold Range Kernel
// ---------------------------------------------------------------------------
__global__ void threshold3DKernel(
        const unsigned short* __restrict__ input,
        unsigned char* __restrict__ output,
        size_t numVoxels,
        unsigned short lower,
        unsigned short upper,
        unsigned int* __restrict__ d_fgCount) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    unsigned short val = input[idx];
    unsigned char isFg = (val >= lower && val <= upper) ? 1 : 0;
    output[idx] = isFg;

    if (isFg && d_fgCount) {
        atomicAdd(d_fgCount, 1);
    }
}

// ---------------------------------------------------------------------------
// 2. GPU Histogram Kernel (for Otsu and Multi-Otsu)
// ---------------------------------------------------------------------------
__global__ void histogram3DKernel(
        const unsigned short* __restrict__ input,
        unsigned int* __restrict__ histogram,
        size_t numVoxels,
        unsigned short minVal,
        unsigned short maxVal,
        int numBins) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    unsigned short val = input[idx];
    if (val < minVal) val = minVal;
    if (val > maxVal) val = maxVal;

    float range = static_cast<float>(maxVal - minVal);
    if (range <= 0.0f) range = 1.0f;

    int bin = static_cast<int>((static_cast<float>(val - minVal) / range) * (numBins - 1));
    if (bin < 0) bin = 0;
    if (bin >= numBins) bin = numBins - 1;

    atomicAdd(&histogram[bin], 1);
}

// ---------------------------------------------------------------------------
// 3. Multi-Threshold Classification Kernel
// ---------------------------------------------------------------------------
__global__ void multiThreshold3DKernel(
        const unsigned short* __restrict__ input,
        unsigned char* __restrict__ output,
        size_t numVoxels,
        const unsigned short* __restrict__ thresholds,
        int numThresholds,
        int targetClass,
        unsigned int* __restrict__ d_fgCount) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    unsigned short val = input[idx];
    int cls = 0;
    for (int t = 0; t < numThresholds; ++t) {
        if (val >= thresholds[t]) {
            cls = t + 1;
        }
    }

    unsigned char isFg = (cls == targetClass) ? 1 : 0;
    output[idx] = isFg;

    if (isFg && d_fgCount) {
        atomicAdd(d_fgCount, 1);
    }
}

// ---------------------------------------------------------------------------
// 4. Parallel 3D Connected Threshold Kernel
// ---------------------------------------------------------------------------
__global__ void connectedThreshold3DKernel(
        const unsigned short* __restrict__ input,
        unsigned char* __restrict__ mask,
        int width, int height, int depth,
        unsigned short lower,
        unsigned short upper,
        int* __restrict__ d_changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    size_t idx = (static_cast<size_t>(z) * height + y) * width + x;
    if (mask[idx] != 0) return; // already foreground

    unsigned short val = input[idx];
    if (val < lower || val > upper) return; // not within intensity range

    // Check 6-connected neighborhood for an adjacent marked voxel
    bool hasMarkedNeighbor = false;
    if (x > 0         && mask[idx - 1]) hasMarkedNeighbor = true;
    if (x < width - 1  && mask[idx + 1]) hasMarkedNeighbor = true;
    if (y > 0         && mask[idx - width]) hasMarkedNeighbor = true;
    if (y < height - 1 && mask[idx + width]) hasMarkedNeighbor = true;
    if (z > 0         && mask[idx - static_cast<size_t>(height) * width]) hasMarkedNeighbor = true;
    if (z < depth - 1  && mask[idx + static_cast<size_t>(height) * width]) hasMarkedNeighbor = true;

    if (hasMarkedNeighbor) {
        mask[idx] = 1;
        *d_changed = 1;
    }
}

// ---------------------------------------------------------------------------
// Host Wrapper Functions
// ---------------------------------------------------------------------------

void thresholdRange3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        unsigned short lower, unsigned short upper,
        unsigned int& outFgCount) {
    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    size_t inBytes = numVoxels * sizeof(unsigned short);
    size_t outBytes = numVoxels * sizeof(unsigned char);
    output.resize(numVoxels);

    unsigned short* d_input = nullptr;
    unsigned char* d_output = nullptr;
    unsigned int* d_fgCount = nullptr;

    cudaError_t err = cudaMalloc(&d_input, inBytes);
    if (err != cudaSuccess) throw std::runtime_error("GPU malloc failed for input volume: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&d_output, outBytes);
    if (err != cudaSuccess) { cudaFree(d_input); throw std::runtime_error("GPU malloc failed for mask volume: " + std::string(cudaGetErrorString(err))); }
    err = cudaMalloc(&d_fgCount, sizeof(unsigned int));
    if (err != cudaSuccess) { cudaFree(d_input); cudaFree(d_output); throw std::runtime_error("GPU malloc failed for counter"); }

    cudaMemset(d_fgCount, 0, sizeof(unsigned int));
    cudaMemcpy(d_input, input.data(), inBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = static_cast<int>((numVoxels + blockSize - 1) / blockSize);
    threshold3DKernel<<<numBlocks, blockSize>>>(d_input, d_output, numVoxels, lower, upper, d_fgCount);

    cudaMemcpy(&outFgCount, d_fgCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output.data(), d_output, outBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_fgCount);
}

// Global Otsu on GPU
void globalOtsu3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        unsigned short minVal, unsigned short maxVal,
        unsigned short& outThreshold,
        unsigned int& outFgCount) {
    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    const int numBins = 4096;
    std::vector<unsigned int> hist(numBins, 0);

    unsigned short* d_input = nullptr;
    unsigned int* d_hist = nullptr;
    unsigned char* d_output = nullptr;
    unsigned int* d_fgCount = nullptr;

    cudaMalloc(&d_input, numVoxels * sizeof(unsigned short));
    cudaMalloc(&d_hist, numBins * sizeof(unsigned int));
    cudaMemset(d_hist, 0, numBins * sizeof(unsigned int));
    cudaMemcpy(d_input, input.data(), numVoxels * sizeof(unsigned short), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = static_cast<int>((numVoxels + blockSize - 1) / blockSize);
    histogram3DKernel<<<numBlocks, blockSize>>>(d_input, d_hist, numVoxels, minVal, maxVal, numBins);

    cudaMemcpy(hist.data(), d_hist, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);

    // Compute Otsu threshold from histogram
    double total = static_cast<double>(numVoxels);
    double sum = 0.0;
    for (int i = 0; i < numBins; ++i) {
        sum += static_cast<double>(i) * hist[i];
    }

    double sumB = 0.0;
    double wB = 0.0;
    double maxVar = 0.0;
    int bestBin = 0;

    for (int t = 0; t < numBins; ++t) {
        wB += hist[t];
        if (wB <= 0.0) continue;
        double wF = total - wB;
        if (wF <= 0.0) break;

        sumB += static_cast<double>(t) * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        double betweenVar = wB * wF * (mB - mF) * (mB - mF);
        if (betweenVar > maxVar) {
            maxVar = betweenVar;
            bestBin = t;
        }
    }

    float range = static_cast<float>(maxVal - minVal);
    outThreshold = static_cast<unsigned short>(minVal + (static_cast<float>(bestBin) / (numBins - 1)) * range);

    // Apply threshold on GPU
    output.resize(numVoxels);
    cudaMalloc(&d_output, numVoxels * sizeof(unsigned char));
    cudaMalloc(&d_fgCount, sizeof(unsigned int));
    cudaMemset(d_fgCount, 0, sizeof(unsigned int));

    threshold3DKernel<<<numBlocks, blockSize>>>(d_input, d_output, numVoxels, outThreshold, 65535, d_fgCount);

    cudaMemcpy(&outFgCount, d_fgCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output.data(), d_output, numVoxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_fgCount);
}

// Multi-Otsu on GPU (up to 5 classes)
void multiOtsu3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        unsigned short minVal, unsigned short maxVal,
        int numClasses, int targetClass,
        std::vector<unsigned short>& outThresholds,
        unsigned int& outFgCount) {
    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    const int numBins = 512;
    std::vector<unsigned int> hist(numBins, 0);

    unsigned short* d_input = nullptr;
    unsigned int* d_hist = nullptr;
    cudaMalloc(&d_input, numVoxels * sizeof(unsigned short));
    cudaMalloc(&d_hist, numBins * sizeof(unsigned int));
    cudaMemset(d_hist, 0, numBins * sizeof(unsigned int));
    cudaMemcpy(d_input, input.data(), numVoxels * sizeof(unsigned short), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = static_cast<int>((numVoxels + blockSize - 1) / blockSize);
    histogram3DKernel<<<numBlocks, blockSize>>>(d_input, d_hist, numVoxels, minVal, maxVal, numBins);
    cudaMemcpy(hist.data(), d_hist, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);

    double total = static_cast<double>(numVoxels);
    float range = static_cast<float>(maxVal - minVal);
    outThresholds.clear();

    if (numClasses <= 2) {
        // Fallback to single Otsu
        unsigned short singleThresh = 0;
        globalOtsu3DGPU(input, output, width, height, depth, minVal, maxVal, singleThresh, outFgCount);
        outThresholds.push_back(singleThresh);
        cudaFree(d_input);
        return;
    }

    if (numClasses == 3) {
        // 3 classes = 2 thresholds
        double maxVar = -1.0;
        int bestT1 = 0, bestT2 = 0;

        double P[512], S[512];
        P[0] = hist[0];
        S[0] = 0.0;
        for (int i = 1; i < numBins; ++i) {
            P[i] = P[i - 1] + hist[i];
            S[i] = S[i - 1] + static_cast<double>(i) * hist[i];
        }

        for (int t1 = 1; t1 < numBins - 2; ++t1) {
            double w0 = P[t1];
            if (w0 <= 0.0) continue;
            double u0 = S[t1] / w0;

            for (int t2 = t1 + 1; t2 < numBins - 1; ++t2) {
                double w1 = P[t2] - P[t1];
                if (w1 <= 0.0) continue;
                double u1 = (S[t2] - S[t1]) / w1;

                double w2 = total - P[t2];
                if (w2 <= 0.0) continue;
                double u2 = (S[numBins - 1] - S[t2]) / w2;

                double var = w0 * u0 * u0 + w1 * u1 * u1 + w2 * u2 * u2;
                if (var > maxVar) {
                    maxVar = var;
                    bestT1 = t1;
                    bestT2 = t2;
                }
            }
        }

        outThresholds.push_back(static_cast<unsigned short>(minVal + (static_cast<float>(bestT1) / (numBins - 1)) * range));
        outThresholds.push_back(static_cast<unsigned short>(minVal + (static_cast<float>(bestT2) / (numBins - 1)) * range));
    } else {
        // Uniform percentile initialization for 4+ classes
        int numThresh = numClasses - 1;
        for (int i = 1; i <= numThresh; ++i) {
            double targetCount = total * (static_cast<double>(i) / numClasses);
            double cum = 0;
            int bin = 0;
            for (int b = 0; b < numBins; ++b) {
                cum += hist[b];
                if (cum >= targetCount) { bin = b; break; }
            }
            outThresholds.push_back(static_cast<unsigned short>(minVal + (static_cast<float>(bin) / (numBins - 1)) * range));
        }
    }

    // Apply multi-threshold on GPU
    output.resize(numVoxels);
    unsigned char* d_output = nullptr;
    unsigned short* d_thresh = nullptr;
    unsigned int* d_fgCount = nullptr;

    cudaMalloc(&d_output, numVoxels * sizeof(unsigned char));
    cudaMalloc(&d_thresh, outThresholds.size() * sizeof(unsigned short));
    cudaMalloc(&d_fgCount, sizeof(unsigned int));
    cudaMemset(d_fgCount, 0, sizeof(unsigned int));

    cudaMemcpy(d_thresh, outThresholds.data(), outThresholds.size() * sizeof(unsigned short), cudaMemcpyHostToDevice);

    multiThreshold3DKernel<<<numBlocks, blockSize>>>(
        d_input, d_output, numVoxels, d_thresh, static_cast<int>(outThresholds.size()), targetClass, d_fgCount);

    cudaMemcpy(&outFgCount, d_fgCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output.data(), d_output, numVoxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_thresh);
    cudaFree(d_fgCount);
}

// Connected Threshold on GPU
void connectedThreshold3DGPU(
        const std::vector<unsigned short>& input,
        std::vector<unsigned char>& output,
        int width, int height, int depth,
        int seedX, int seedY, int seedZ,
        unsigned short lower, unsigned short upper,
        unsigned int& outFgCount) {
    size_t numVoxels = static_cast<size_t>(width) * height * depth;
    output.assign(numVoxels, 0);

    // Validate seed
    if (seedX < 0 || seedX >= width || seedY < 0 || seedY >= height || seedZ < 0 || seedZ >= depth) {
        throw std::runtime_error("Connected Threshold GPU: seed coordinates out of bounds");
    }

    size_t seedIdx = (static_cast<size_t>(seedZ) * height + seedY) * width + seedX;
    unsigned short seedVal = input[seedIdx];
    if (seedVal < lower || seedVal > upper) {
        // Seed voxel itself does not meet threshold criteria
        outFgCount = 0;
        return;
    }
    output[seedIdx] = 1;

    unsigned short* d_input = nullptr;
    unsigned char* d_mask = nullptr;
    int* d_changed = nullptr;

    cudaMalloc(&d_input, numVoxels * sizeof(unsigned short));
    cudaMalloc(&d_mask, numVoxels * sizeof(unsigned char));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_input, input.data(), numVoxels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, output.data(), numVoxels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);

    int changed = 1;
    int maxIters = 500; // limit iterations to guarantee termination
    int iter = 0;

    while (changed && iter < maxIters) {
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

        connectedThreshold3DKernel<<<grid, block>>>(d_input, d_mask, width, height, depth, lower, upper, d_changed);

        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        iter++;
    }

    cudaMemcpy(output.data(), d_mask, numVoxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Count foreground
    outFgCount = 0;
    for (size_t i = 0; i < numVoxels; ++i) {
        if (output[i]) outFgCount++;
    }

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_changed);
}

} // namespace ctseg_gpu
