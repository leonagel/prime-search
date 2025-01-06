#include "prime_search.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <mma.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// #define DEBUG_PERMUTE
#define DEBUG_PRINT 
// #define DEBUG_SORT
// #define DEBUG_RANDOM

#define SIZE 1024

// #define TOTAL_PERMUTATIONS 1000000 
#define TOTAL_PERMUTATIONS 10000000000
// #define TOTAL_PERMUTATIONS 10
#define CHECK_DONE_PERMUTATIONS 1000000

using namespace nvcuda;
using namespace std;

__global__ void prime_search(int* data, int size, int* output) {
    output[threadIdx.x] = data[threadIdx.x] + 1;

    return;
}

// __device__ long fast_size_lower_bound(int* data, long lower_bound) {

// }


dim3 KernelManagerPrimeSearch::calculateGrid(int n, int threadsPerBlock) {
    // return dim3((INNER_DIM + threadsPerBlock - 1) / threadsPerBlock);
    return dim3(n);
}

float KernelManagerPrimeSearch::launchKernel(int* data, int* output) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int smCount = 0;
    for (int i = 0; i < deviceCount; i = i + 1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        smCount += prop.multiProcessorCount;
    }

    #ifdef DEBUG_PRINT
    printf("Number of SMs: %d\n", smCount);
    #endif

    int numBlocks = smCount * 2048 / SIZE;
    printf("Number of blocks: %d\n", numBlocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int threadsPerBlock = SIZE;
    dim3 grid = calculateGrid(numBlocks, threadsPerBlock);
    dim3 block(threadsPerBlock);
    #ifdef DEBUG_PRINT
    printf("Grid dimensions: %d x %d x %d\n", grid.x, grid.y, grid.z);
    #endif

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    prime_search<<<1, block>>>(data, SIZE, output);
    // prime_search<<<grid, block>>>(data, SIZE, output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    return milliseconds;
}