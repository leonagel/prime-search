#include "bogo_sort.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>



__global__ void bogo_sort(int* data, int size, int* output) {
    extern __shared__ int shared_data[64];
    extern __shared__ curandStatePhilox4_32_10_t shared_random_state;
    extern __shared__ int shared_random;

    extern __shared__ bool parity;
    parity = false;
    __syncthreads();

    auto parity_shift = [](int p) { return p ? 32 : 0;};
    
    //curand_init(unsigned long long seed,
            // unsigned long long subsequence,
            // unsigned long long offset,
            // curandStatePhilox4_32_10_t *state)
    curand_init(0, 0, 0, &shared_random_state);
    shared_random = curand(&shared_random_state);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = shared_random;
    }

    if (threadIdx.x < 32 && threadIdx.x < size) {
        shared_data[threadIdx.x] = data[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("shared_data after loading: ");
        for (int i = 0; i < 64; i++) {
            printf("%d ", shared_data[i]);
        }
        printf("\n");
    }


    // bit shift right by threadIdx.x/2 bits, then 
    // check for parity to check if the threadIdx.x/2'th bit is 0 or 1
    if (threadIdx.x == 0) {
        printf("\nThread operations:\n");
    }

    int __shared__ offset;
    if (threadIdx.x == 0) {
        offset = 0;
    }
    if ((shared_random >> (threadIdx.x / 2 + offset)) % 2 == 1) {
        if (threadIdx.x % 2 < 2 / 2) {
            shared_data[threadIdx.x + 2 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        } else {
            shared_data[threadIdx.x - 2 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        }
    } else {
       shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
    }

    if (threadIdx.x == 0) {
        offset += 32 / 2;
        parity = !parity;
    }
    __syncthreads();

    if ((shared_random >> (threadIdx.x / 4 + offset)) % 2 == 1) {
        if (threadIdx.x % 4 < 4 / 2) {
            shared_data[threadIdx.x + 4 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        } else {
            shared_data[threadIdx.x - 4 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        }
    } else {
       shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
    }

    if (threadIdx.x == 0) {
        offset += 32 / 4;
        parity = !parity;
    }
    __syncthreads();

    if ((shared_random >> (threadIdx.x / 8 + offset)) % 2 == 1) {
        if (threadIdx.x % 8 < 8 / 2) {
            shared_data[threadIdx.x + 8 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        } else {
            shared_data[threadIdx.x - 8 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        }
    } else {
       shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
    }

    if (threadIdx.x == 0) {
        offset += 32 / 8;
        parity = !parity;
    }
    __syncthreads();

    if ((shared_random >> (threadIdx.x / 16 + offset)) % 2 == 1) {
        if (threadIdx.x % 16 < 16 / 2) {
            shared_data[threadIdx.x + 16 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        } else {
            shared_data[threadIdx.x - 16 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
        }
    } else {
       shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
    }

    if (threadIdx.x == 0) {
        offset += 32 / 16;
        parity = !parity;
    }
    __syncthreads();

    // if ((shared_random >> (threadIdx.x / 32 + offset)) % 2 == 1) {
    //     if (threadIdx.x % 32 < 32 / 2) {
    //         shared_data[threadIdx.x + 32 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
    //     } else {
    //         shared_data[threadIdx.x - 32 / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
    //     }
    // } else {
    //    shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
    // }

    // if (threadIdx.x == 0) {
    //     offset += 32 /32;
    //     parity = !parity;
    // }
    // __syncthreads();

    if (threadIdx.x == 0) {
        printf("shared_random binary: ");
        for (int i = 31; i >= 0; i--) {
            printf("%d:%d ", i, (shared_random >> i) & 1);
        }
        printf("\n");
        printf("shared_data: ");
        for (int i = 0; i < 64; i++) {
            printf("%d ", shared_data[i]);
        }
        printf("\n");
    }

    __syncthreads();
    
    // // Check first 16 bits of shared_random matches output pattern
    // if (threadIdx.x == 0) {
    //     int pattern = 0;
    //     // Build pattern from every second output value
    //     for (int i = 0; i < 16 && i*2 < size; i++) {
    //         pattern |= (output[i*2] & 1) << i;
    //     }
        
    //     // Compare with lowest 16 bits of shared_random
    //     if ((shared_random & 0xFFFF) != pattern) {
    //         asm("trap;"); // Throw hardware error if patterns don't match
    //     }
    // }
        
        
}

dim3 KernelManagerBogoSort::calculateGrid(int N, int threadsPerBlock) {
    return dim3((N + threadsPerBlock - 1) / threadsPerBlock);
}

float KernelManagerBogoSort::launchKernel(int* data, int size, int* output) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int threadsPerBlock = 32;
    dim3 grid = calculateGrid(size, threadsPerBlock);
    dim3 block(threadsPerBlock);

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    bogo_sort<<<grid, block>>>(data, size, output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}
