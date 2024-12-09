#include <cuda_runtime.h>
#include <stdio.h>
#include "../src/kernels/basic_kernel.cuh"
#include "../src/kernels/bogo_sort.cuh"

// Simple test framework
#define RUN_TEST(test_func) do { \
    printf("Running %s...\n", #test_func); \
    if (test_func()) { \
        printf("PASSED\n"); \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

bool test_vector_add() {
    const int N = 1000;
    size_t size = N * sizeof(float);

    // Allocate and initialize host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerVectorAdd::launchKernel(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success;
}

bool test_bogo_sort() {
    const int N = 32; // Test with larger array
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[N];
    
    // Initialize input array
    for (int i = 0; i < N; i++) {
        h_input[i] = i; // Sequential values
        h_output[i] = 0; // Initialize output to 0
    }

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerBogoSort::launchKernel(d_input, N, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify all elements have been filled with random values
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] == 0) { // Check if still has initial value
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return success;

}
int main() {
    RUN_TEST(test_vector_add);
    RUN_TEST(test_bogo_sort);
    return 0;
}
