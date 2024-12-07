#pragma once
#include <cuda_runtime.h>

// Kernel declaration
__global__ void vectorAdd(const float* A, const float* B, float* C, int N);

// Wrapper class for kernel management and timing
class KernelManager {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock = 256);

    // Launch kernel with timing
    static float launchKernel(const float* A, const float* B, float* C, int N);
};
