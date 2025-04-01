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

#define SIZE 256
#define PRIMES_LENGTH 262144
#define LARGEST_MERSENNE_EXPONENT 136279841

using namespace nvcuda;
using namespace std;

__global__ void prime_search(long start_offset, int stride, 
    int* primes_array, int* remainder_array, long primes_length, long* output, int* output_count) {
    extern __device__ int output_count_cached;
    // extern __device__ volatile bool done;
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     done = false;
    // }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        printf("start_offset: %ld\n", start_offset);
        printf("stride: %d\n", stride);
        printf("primes_length: %ld\n", primes_length);
        printf("primes_array (excerpt): ");
        for (int i = 0; i < 5; i++) {
            printf("%d ", primes_array[i]);
        }
        if (primes_length > 5) printf("... ");
        printf("\nremainder_array (excerpt): ");
        for (int i = 0; i < 5; i++) {
            printf("%d ", remainder_array[i]);
        }
        if (primes_length > 5) printf("... ");
        printf("\noutput (excerpt): ");
        for (int i = 0; i < 5; i++) {
            printf("%ld ", output[i]);
        }
        if (SIZE > 5) printf("... ");
        printf("\noutput_count: %d\n", *output_count);
    }

    extern __shared__ int is_prime;
    extern __shared__ int i;
    
    __syncthreads();
    extern __shared__ long offset;
    if (threadIdx.x == 0) {
        offset = start_offset + blockIdx.x * 2;
    }
    do {
        if (threadIdx.x == 0) {
            is_prime = 1;
            i = 0;
        }
        __syncthreads();
        do {
            if ((offset + remainder_array[i + threadIdx.x]) % primes_array[i + threadIdx.x] == 0) {
                atomicCAS(&is_prime, 1, 0);
                if (blockIdx.x == 0) {
                    printf("Offset: %ld, ", offset);
                    printf("primes_array[%d]: %d, ", i + threadIdx.x, primes_array[i + threadIdx.x]);
                    printf("is_prime: %d\n", is_prime);
                }
            }
            if (threadIdx.x == 0) {
                i = i + SIZE;
            }
            __syncthreads();
        } while((is_prime != 0) && (i < primes_length));
        if (threadIdx.x == 0) {
            offset = offset + stride;
        }
        __syncthreads();
    } while (is_prime == 0);
    // } while (false);

    // This is added in here because the code gets a seizure whenever I try to build this behavior into the
    // look itself. Genuinely unexplainable and cursed.
    offset = offset - 2048;

    if (threadIdx.x == 0) {
        output[atomicAdd(&output_count_cached, 1)] = offset - 2048;
    }

    *output_count = output_count_cached;

    return;
}

// Schönhage-Strassen algorithm to be implemented here.
// __device__ void multiply(int* factor_one, long factor_one_length, 
//     int* factor_two, long factor_two_length, int* output) {
//     return;
// }


dim3 KernelManagerPrimeSearch::calculateGrid(int n, int threadsPerBlock) {
    // return dim3((INNER_DIM + threadsPerBlock - 1) / threadsPerBlock);
    return dim3(n);
}

float KernelManagerPrimeSearch::launchKernel(long start_offset, 
    int* primes_array, int* remainder_array, long primes_length, long* output, int* output_count) {
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
    prime_search<<<1024, block>>>(start_offset, 2 * SIZE,
        primes_array, remainder_array, primes_length, output, output_count);
    // prime_search<<<grid, block>>>(data, SIZE, output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    return milliseconds;
}

int KernelManagerPrimeSearch::loadArraysFromCache(int* output_array, int* primes_array, const char* cache_folder) {
    FILE *output_file = fopen((std::string(cache_folder) + "/output_array.txt").c_str(), "r");
    FILE *primes_file = fopen((std::string(cache_folder) + "/primes_array.txt").c_str(), "r");

    if (output_file == NULL || primes_file == NULL) {
        printf("Error opening cache files\n");
        return -1;
    }

    // Count lines in output file
    int output_length = 0;
    char line[32];
    while (fgets(line, sizeof(line), output_file)) {
        output_length++;
    }
    rewind(output_file);

    // Count lines in primes file  
    int primes_length = 0;
    while (fgets(line, sizeof(line), primes_file)) {
        primes_length++;
    }
    rewind(primes_file);

    // Read output array
    int i = 0;
    while (fgets(line, sizeof(line), output_file)) {
        output_array[i++] = atoi(line);
    }

    // Read primes array
    i = 0;
    while (fgets(line, sizeof(line), primes_file)) {
        primes_array[i++] = atoi(line);
    }

    fclose(output_file);
    fclose(primes_file);
    
    if (output_length != primes_length) {
        printf("Error: output_length (%d) != primes_length (%d)\n", output_length, primes_length);
        return -1;
    }

    return output_length;
}
