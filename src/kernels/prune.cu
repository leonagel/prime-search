#include "prune.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <mma.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// #define DEBUG_PRINT

#define SIZE 1024

using namespace nvcuda;
using namespace std;

__global__ void prune(int* number, long number_length, int* primes, int primes_length, int* output) {
    // #ifdef DEBUG_PRINT
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("\nInput number array: ");
    //     for (int i = 0; i < number_length; i++) {
    //         printf("%08X ", number[i]);
    //     }
    //     printf("\n");
        
    //     printf("Input primes array: ");
    //     for (int i = 0; i < min(32, primes_length); i++) {
    //         printf("%d ", primes[i]);
    //     }
    //     if (primes_length > 32) printf("...");
    //     printf("\n");
        
    //     printf("Number length: %ld\n", number_length);
    //     printf("Primes length: %d\n", primes_length);
    //     printf("\n");
    // }
    // #endif

    int prime = primes[blockIdx.x * SIZE + threadIdx.x];
    int int_max_mod_prime = 0x7FFFFFFF;
    int_max_mod_prime = (int_max_mod_prime % prime) + 1;
    long temp = 0;
    for (int i = 0; i < number_length; i++) {
        temp = (temp * int_max_mod_prime) % prime;
        temp = (temp + number[i] % prime) % prime;
    }

    output[blockIdx.x * SIZE + threadIdx.x] = (int) temp;

    // #ifdef DEBUG_PRINT
    // if (threadIdx.x == 0) {
    //     printf("\nOutput array: ");
    //     for (int i = 0; i < SIZE; i++) {
    //         if (i > 0 && i % 8 == 0) printf("\n");
    //         printf("%-4d ", output[i]);
    //     }
    //     printf("\n");
    // }
    // #endif

    return;
}


dim3 KernelManagerPrune::calculateGrid(int n, int threadsPerBlock) {
    return dim3(n);
}

float KernelManagerPrune::launchKernel(int* large_prime, long large_prime_length, int* primes_array, long primes_length, int* output) {
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

    int numBlocks = primes_length / 1024;
    
    #ifdef DEBUG_PRINT
    printf("Number of blocks: %d\n", numBlocks);
    #endif

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
    prune<<<grid, SIZE>>>(large_prime, large_prime_length, primes_array, primes_length, output);
    // prime_search<<<grid, block>>>(data, SIZE, output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    return milliseconds;
}

// generates M(exponent) efficiently
void generate_mersenne_prime(long exponent, int* output) {
    int remaining_bits = exponent % 31;
    int full_words = exponent / 31;
    
    // Initialize all words to 0x7FFFFFFF
    for (int i = 0; i <= full_words; i++) {
        output[i] = 0x7FFFFFFF;
    }
    
    // Handle remaining bits in least significant word (position 0)
    if (remaining_bits > 0) {
        output[0] = (1 << remaining_bits) - 1;
    }
}

// fills array with the (length)th first prime numbers (excluding 2)
void generate_prime_number_array(long length, int* output) {
    // Initialize array with 3 as first prime
    output[0] = 3;
    int count = 1;
    int num = 5;

    // Keep finding primes until we reach desired length
    while (count < length) {
        bool isPrime = true;
        
        // Check if num is divisible by any previously found prime
        for (int i = 0; i < count; i++) {
            if (num % output[i] == 0) {
                isPrime = false;
                break;
            }
            if (num < output[i] * output[i]) {
                break;
            }
        }

        // If prime found, add to array
        if (isPrime) {
            output[count] = num;
            count++;
        }

        num += 2; // Skip even numbers
    }
}