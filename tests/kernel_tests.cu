#include <cuda_runtime.h>
#include <stdio.h>
#include "../src/kernels/prime_search.cuh"
#include "../src/kernels/prune.cuh"

#define COMPILE_ALL false

// #define TEST_MERSENNE_PRIME
// #define TEST_PRIME_NUMBERS
// # define TEST_PRIME_SEARCH

// Simple test framework
#define RUN_TEST(test_func) do { \
    printf("Running %s...\n", #test_func); \
    if (test_func()) { \
        printf("PASSED\n"); \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

#ifdef TEST_PRIME_SEARCH
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
#endif

bool test_2048_long_array_action_prune() {
    int N = 9;
    size_t size_N = N * sizeof(int);
    int primes_length = 1024;
    size_t size_primes_length = primes_length * sizeof(int);
    

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_primes = new int[primes_length];
    int *h_output = new int[primes_length];
    
    // generate good mersenne prime
    generate_mersenne_prime(257, h_input);
    // set output to 0
    for (int i = 0; i < primes_length; i++) {
        h_output[i] = 0;
    }
    // generate right prime numbers
    generate_prime_number_array(primes_length, h_primes);

    // Print input array
    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Print primes array
    printf("Primes array: ");
    for (int i = 0; i < primes_length; i++) {
        printf("%d ", h_primes[i]);
    }
    printf("\n");

    // Expected output array
    int expected[primes_length];
    for (int i = 0; i < primes_length; i++) {
        expected[i] = 1;  // Expect all values to be 1 since we're testing odd numbers
    }

    // Print expected array
    printf("Expected: ");
    printf("%d ", expected[0]);
    printf("\n");

    // Allocate device memory
    int *d_input, *d_primes, *d_output;
    cudaMalloc(&d_input, size_N);
    cudaMalloc(&d_primes, size_primes_length);
    cudaMalloc(&d_output, size_primes_length);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primes, h_primes, size_primes_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size_primes_length, cudaMemcpyHostToDevice);

    // Run kernel
    KernelManagerPrune::launchKernel(d_input, N, d_primes, primes_length, d_output);

    // cudaError_t err = cudaGetLastError();        // Get error code
    // printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size_primes_length, cudaMemcpyDeviceToHost);
    printf("\nDetailed Output:\n");
    printf("----------------\n");
    for (int i = 0; i < primes_length; i++) {
        if (i > 0 && i % 8 == 0) printf("\n");
        printf("%-4d ", h_output[i]);
    }
    printf("\n----------------\n");

    bool success;
    success = true;
    for (int i = 0; i < primes_length; i++) {
        if (h_output[i] != expected[i]) {
            success = false;
            break;
        }
    }

    // Cleanup
    delete[] h_input;
    delete[] h_primes;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_primes);
    cudaFree(d_output);

    return success;

}

#ifdef TEST_MERSENNE_PRIME
bool test_generate_mersenne_prime() {
    // Test for M31 (2^31 - 1)
    int expected_size = 1;  // 31 bits fits in one int
    int* output = new int[expected_size];
    
    // Generate M31
    generate_mersenne_prime(31, output);

    // Print the generated Mersenne prime
    printf("M31: 0x%08X\n", output[0]);
    
    // M31 should be 0x7FFFFFFF (2^31 - 1)
    bool success = (output[0] == 0x7FFFFFFF);
    
    // Test for M61 (needs 2 ints)
    int* output2 = new int[2];
    generate_mersenne_prime(61, output2);
    // Print M61 representation
    printf("M61: 0x%08X %08X\n", output2[1], output2[0]);
    
    // First int should be all 1s (0x7FFFFFFF)
    // Second int should be all 1s in first 30 bits (0x3FFFFFFF)
    success &= (output2[1] == 0x7FFFFFFF);
    success &= (output2[0] == 0x3FFFFFFF);

    // Test for M257
    int* output257 = new int[9];  // 257 bits needs 9 ints (8.29 ints exactly)
    generate_mersenne_prime(257, output257);
    
    // Print M257 representation
    printf("M257: ");
    for (int i = 8; i >= 0; i--) {
        printf("%08X ", output257[i]);
    }
    printf("\n");
    
    // First 8 ints should be all 1s (0x7FFFFFFF)
    for (int i = 8; i >= 1; i--) {
        success &= (output257[i] == 0x7FFFFFFF);
    }
    // Last int should have 8 bits set (0x000000FF)
    success &= (output257[0] == 0x000001FF);
    
    delete[] output257;
    
    delete[] output;
    delete[] output2;
    
    return success;
}
#endif

#ifdef TEST_PRIME_NUMBERS
bool test_generate_prime_numbers() {
    bool success = true;
    
    // Test first 10 primes
    int* first_10_primes = new int[10];
    generate_prime_number_array(10, first_10_primes);
    
    // Known first 10 primes
    int expected_10[10] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    
    printf("First 10 primes: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", first_10_primes[i]);
    }
    printf("\n");
    
    // Verify first 10 primes
    for (int i = 0; i < 10; i++) {
        success &= (first_10_primes[i] == expected_10[i]);
    }
    
    delete[] first_10_primes;

    // Test first 20 primes
    int* first_20_primes = new int[20];
    generate_prime_number_array(20, first_20_primes);
    
    // Known first 20 primes
    int expected_20[20] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 
                          31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
    
    printf("First 20 primes: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", first_20_primes[i]);
    }
    printf("\n");
    
    // Verify first 20 primes
    for (int i = 0; i < 20; i++) {
        success &= (first_20_primes[i] == expected_20[i]);
    }
    
    delete[] first_20_primes;

    // Test edge case - first prime
    int* single_prime = new int[1];
    generate_prime_number_array(1, single_prime);
    success &= (single_prime[0] == 2);
    
    delete[] single_prime;
    
    return success;
}
#endif



int main() {
    #ifdef TEST_PRIME_SEARCH
    RUN_TEST(test_2048_long_array_action);
    #endif

    RUN_TEST(test_2048_long_array_action_prune);

    #ifdef TEST_MERSENNE_PRIME
    RUN_TEST(test_generate_mersenne_prime);
    #endif

    #ifdef TEST_PRIME_NUMBERS
    RUN_TEST(test_generate_prime_numbers);
    #endif

    return 0;
}
