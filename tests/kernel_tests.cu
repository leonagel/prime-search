#include <cuda_runtime.h>
#include <stdio.h>
#include "../src/kernels/prime_search.cuh"
#include "../src/kernels/prune.cuh"

#define COMPILE_ALL false

#define PRIMES_LENGTH 262144

// #define TEST_MERSENNE_PRIME
// #define TEST_PRIME_NUMBERS
// #define TEST_PRUNE_ARRAY_GENERATION 
#define TEST_PRIME_SEARCH
// #define TEST_LOAD_ARRAYS_FROM_CACHE

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
    // Setup test parameters

    long start_offset = 2;
    int primes_length = PRIMES_LENGTH;
    const char* cache_folder = "../cache";  // Added cache folder parameter
    
    // Allocate host memory
    int *h_primes = new int[primes_length];
    int *h_remainder = new int[primes_length];
    long *h_output = new long[1024];  // Changed to array to match kernel output
    int *h_output_count = new int[1];
    
    // Initialize arrays
    for (int i = 0; i < primes_length; i++) {
        h_primes[i] = 0;
        if (i < 1024) {
            h_output[i] = -1;
        }
          // Initialize output array
        h_remainder[i] = 0;
    }
    *h_output_count = 0;
    printf("Here...\n");

    // Load arrays from cache
    int loaded_length = KernelManagerPrimeSearch::loadArraysFromCache(h_remainder, h_primes, cache_folder);
    if (loaded_length < 0) {
        printf("Failed to load arrays from cache\n");
        return false;
    }

    // Allocate device memory
    int *d_primes, *d_remainder;
    long *d_output;
    int *d_output_count;
    cudaMalloc(&d_primes, primes_length * sizeof(int));
    cudaMalloc(&d_remainder, primes_length * sizeof(int));
    cudaMalloc(&d_output, 1024 * sizeof(long));  // Changed size to match array
    cudaMalloc(&d_output_count, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_primes, h_primes, primes_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_remainder, h_remainder, primes_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, 1024 * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_count, h_output_count, sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel
    float time = KernelManagerPrimeSearch::launchKernel(start_offset, d_primes, d_remainder, 
        primes_length, d_output, d_output_count);

    cudaError_t err = cudaGetLastError();
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    printf("Kernel execution time: %f ms\n", time);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, 1024 * sizeof(long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Output array:\n");
    for (int i = 0; i < 1024; i++) {
        if (i > 0 && i % 8 == 0) printf("\n");  // Print 8 values per line
        printf("%-4ld ", h_output[i]);
    }
    printf("\nOutput count: %d\n", *h_output_count);

    bool success = true;
    for (int i = 0; i < *h_output_count; i++) {
        long number = h_output[i];
        for (int j = 0; j < primes_length; j++) {
            if ((number + h_remainder[j]) % h_primes[j] == 0) {
                success = false;
                break;
            }
        }
        if (!success) break;
    }

    // Cleanup
    delete[] h_primes;
    delete[] h_remainder;
    delete[] h_output;
    delete[] h_output_count;
    cudaFree(d_primes);
    cudaFree(d_remainder);
    cudaFree(d_output);
    cudaFree(d_output_count);

    return success;
}
#endif

#ifdef TEST_PRUNE_ARRAY_GENERATION
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
#endif

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

#ifdef TEST_LOAD_ARRAYS_FROM_CACHE
bool test_load_arrays_from_cache() {
    // Create test cache files
    FILE *output_file = fopen("../cache/output_array", "w");
    FILE *primes_file = fopen("../cache/primes_array", "w");
    
    if (output_file == NULL || primes_file == NULL) {
        printf("Failed to create test cache files\n");
        return false;
    }
    
    // Write test data
    const int expected_length = 5;
    int expected_output[] = {1, 2, 3, 4, 5};
    int expected_primes[] = {2, 3, 5, 7, 11};
    
    for (int i = 0; i < expected_length; i++) {
        fprintf(output_file, "%d\n", expected_output[i]);
        fprintf(primes_file, "%d\n", expected_primes[i]);
    }
    
    fclose(output_file);
    fclose(primes_file);
    
    // Test the load function
    int output_array[1024] = {0};  // Initialize with zeros
    int primes_array[1024] = {0};
    
    int result_length = KernelManagerPrimeSearch::loadArraysFromCache(output_array, primes_array);
    
    // Verify length
    bool success = (result_length == expected_length);
    
    // Verify contents
    for (int i = 0; i < expected_length && success; i++) {
        if (output_array[i] != expected_output[i] || primes_array[i] != expected_primes[i]) {
            success = false;
            printf("Mismatch at index %d:\n", i);
            printf("Output array: expected %d, got %d\n", expected_output[i], output_array[i]);
            printf("Primes array: expected %d, got %d\n", expected_primes[i], primes_array[i]);
        }
    }
    
    return success;
}
#endif

int main() {
    #ifdef TEST_PRIME_SEARCH
    RUN_TEST(test_2048_long_array_action);
    #endif

    #ifdef TEST_PRUNE_ARRAY_GENERATION
    RUN_TEST(test_2048_long_array_action_prune);
    #endif

    #ifdef TEST_MERSENNE_PRIME
    RUN_TEST(test_generate_mersenne_prime);
    #endif

    #ifdef TEST_PRIME_NUMBERS
    RUN_TEST(test_generate_prime_numbers);
    #endif

    #ifdef TEST_LOAD_ARRAYS_FROM_CACHE
    RUN_TEST(test_load_arrays_from_cache);
    #endif

    return 0;
}
