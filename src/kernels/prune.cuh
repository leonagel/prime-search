#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*******************************************************************************
 * We prune most of the search space of odd numbers to attain our prime        *
 * candidates. We do this by getting the remainder when dividing the current   *
 * largest prime by any of the first 100,000 primes. The 100,000 value was     *
 * empirically attained, and it should prune out around ~96% of odd prime      *
 * candidates. This is the code to do it. Will make a file noting these        *
 *******************************************************************************/


// Kernel declaration
__global__ void prune(int* number, int number_length, int* primes, int primes_length, int* output);

// Wrapper class for kernel management and timing
class KernelManagerPrune {
public:
    // Helper function to calculate grid dimensions
    static dim3 calculateGrid(int N, int threadsPerBlock);

    // Launch kernel with timing
    static float launchKernel(int* large_prime, long large_prime_length, int* primes_array, long primes_length, int* output);
};

void generate_mersenne_prime(long exponent, int* output);
void generate_prime_number_array(long length, int* output);