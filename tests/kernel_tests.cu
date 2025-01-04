#include <cuda_runtime.h>
#include <stdio.h>
// #include "../src/kernels/basic_kernel.cuh"
// #include "../src/kernels/bogo_sort.cuh"
#include "../src/kernels/prime_search.cuh"

#define COMPILE_ALL false

// Simple test framework
#define RUN_TEST(test_func) do { \
    printf("Running %s...\n", #test_func); \
    if (test_func()) { \
        printf("PASSED\n"); \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

bool test_2048_long_array_action() {
    int N = 1024;
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[N];
    
    // Initialize host arrays to 0
    for (int i = 0; i < N; i++) {
        h_input[i] = 0;
        h_output[i] = 0;
    }

    // Print input array
    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Expected output array
    int expected[N];
    for (int i = 0; i < N; i++) {
        // expected[i] = (i < num_zeroes) ? 0 : 1;
        expected[i] = 1;
    }

    // Print expected array
    printf("Expected array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", expected[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerPrimeSearch::launchKernel(d_input, d_output);

    cudaError_t err = cudaGetLastError();        // Get error code
    printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print output array
    printf("Output array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]); 
    }
    printf("\n");

    // Verify output matches expected array
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != expected[i]) {
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
    RUN_TEST(test_2048_long_array_action);
    return 0;
}
