#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel declaration
__global__ void prime_search(long start_offset, int stride, int* primes_array, int* remainder_array, long* output, int* output_count);

// Wrapper class for kernel management and timing
class KernelManagerPrimeSearch {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock);

    // Launch kernel with timing
    static float launchKernel(long start_offset, int* primes_array, int* remainder_array, long primes_length, long* output, int* output_count);

    // Load in previously attained pruning arrays and return their length
    static int loadArraysFromCache(int* output_array, int* primes_array, const char* cache_folder);
};
