#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel declaration
__global__ void prime_search(int* data, int size, int* output);

// Wrapper class for kernel management and timing
class KernelManagerPrimeSearch {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock);

    // Launch kernel with timing
    static float launchKernel(int* data, int* output);
};
