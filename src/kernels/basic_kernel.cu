#include <cuda_runtime.h>

// Kernel declaration
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Wrapper class for kernel management and timing
class KernelManager {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock = 256) {
        return dim3((N + threadsPerBlock - 1) / threadsPerBlock);
    }

    // Launch kernel with timing
    static float launchKernel(const float* A, const float* B, float* C, int N) {
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
};
