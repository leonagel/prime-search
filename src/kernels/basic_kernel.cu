#include "basic_kernel.cuh"

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

dim3 KernelManager::calculateGrid(int N, int threadsPerBlock) {
    return dim3((N + threadsPerBlock - 1) / threadsPerBlock);
}

float KernelManager::launchKernel(const float* A, const float* B, float* C, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int threadsPerBlock = 256;
    dim3 grid = calculateGrid(N, threadsPerBlock);
    dim3 block(threadsPerBlock);

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<grid, block>>>(A, B, C, N);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}
