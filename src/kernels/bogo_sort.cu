#include "bogo_sort.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

__global__ void bogo_sort(int* data, int size, int* output) {
    extern __shared__ int shared_data[64];
    extern __shared__ curandStatePhilox4_32_10_t shared_random_state;
    extern __shared__ int shared_random;

    extern __device__ int done;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        done = 0;
    }
    extern __shared__ int local_done;
    if (threadIdx.x == 0) {
        local_done = 0;
    }

    extern __shared__ bool parity;
    parity = false;
    __shared__ bool is_sorted;
    if (threadIdx.x == 0) {
        is_sorted = false;
    }

     //curand_init(unsigned long long seed,
            // unsigned long long subsequence,
            // unsigned long long offset,
            // curandStatePhilox4_32_10_t *state)    
    curand_init(0, blockIdx.x, 0, &shared_random_state);
    __syncthreads();

    auto parity_shift = [](int p) { return p ? 32 : 0;};

    extern __shared__ long permutation;

    if (threadIdx.x == 0) {
        permutation = 0;
    }

    if (threadIdx.x < size) {
        shared_data[threadIdx.x] = data[threadIdx.x];
    }
    __syncthreads();

    do {
        shared_random = curand(&shared_random_state);
        // int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // if (threadIdx.x == 0) {
        //     printf("shared_data after loading: ");
        //     for (int i = 0; i < 64; i++) {
        //         printf("%d ", shared_data[i]);
        //     }
        //     printf("\n");
        // }

        // #define DEBUG_ITERATIONS 1
        // #if DEBUG_ITERATIONS
        // if (blockIdx.x == 0)  {
        //     if (threadIdx.x == 0) {
        //         permutation++;
        //     }
        //     if (threadIdx.x == 0 && permutation % 100000 == 0) {
        //         printf("Iteration %ld, shared_data: ", permutation);
        //         for (int i = 0; i < size; i++) {
        //             printf("%d ", shared_data[i + parity_shift(parity)]);
        //         }
        //         printf("\n");
        //     }
        // }
        // #endif

        // bit shift right by threadIdx.x/2 bits, then 
        // check for parity to check if the threadIdx.x/2'th bit is 0 or 1
        // if (threadIdx.x == 0) {
        //     printf("\nThread operations:\n");
        // }
        if (threadIdx.x == 0) {
            permutation++;
        }
        int __shared__ offset;
        if (threadIdx.x == 0) {
            offset = 0;
        }
        int __shared__ swap_length;
        for (swap_length = 2; swap_length <= 32; swap_length *= 2) {
            if ((shared_random >> (threadIdx.x / swap_length + offset)) % 2 == 1) {
                if (threadIdx.x % swap_length < swap_length / 2) {
                    shared_data[threadIdx.x + swap_length / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
                } else {
                    shared_data[threadIdx.x - swap_length / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
                }
            } else {
                shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
            }

            if (threadIdx.x == 0) {
                offset += 32 / swap_length;
                parity = !parity;
            }
            __syncthreads();
        }

        if (threadIdx.x != 0) {
            shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity) - 1];
        } else {
            shared_data[parity_shift(!parity)] = shared_data[size + parity_shift(parity) - 1]; 
            parity = !parity;
        }
        

        // if (threadIdx.x == 0) {
        //     printf("shared_random binary: ");
        //     for (int i = 31; i >= 0; i--) {
        //         printf("%d:%d ", i, (shared_random >> i) & 1);
        //     }
        //     printf("\n");
        //     printf("shared_data: ");
        //     for (int i = 0; i < 64; i++) {
        //         printf("%d ", shared_data[i]);
        //     }
        //     printf("\n");
        // }
        // __syncthreads();

        __syncthreads();
        verify_sort(shared_data + parity_shift(parity), 32, &is_sorted);
        __syncthreads();

        if (threadIdx.x == 0) {
            local_done = atomicAnd(&done, 1);
        }
        __syncthreads();
        if (local_done != 0) {
            if (threadIdx.x == 0 && blockIdx.x % 500 == 0) {
                printf("Block %d has tried %ld to end\n", blockIdx.x, permutation);
            }
            return;
        }
        __syncthreads();
    } while (!is_sorted);

    __syncthreads();
    output[threadIdx.x] = shared_data[threadIdx.x + parity_shift(parity)];
    if (threadIdx.x == 0) {
        printf("Before setting: %d\n", done);
        printf("Block %d found sorted array after %ld permutations\n", blockIdx.x, permutation);
        atomicCAS(&done, 0, 1);
        printf("After setting: %d\n", done);
    }
    __syncthreads();
    return;
}

__device__ void verify_sort(int* input, int size, bool* is_sorted) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *is_sorted = true;
    }
    __syncthreads();
    if (threadIdx.x < size - 1) {  // Don't check the last element since it has no right neighbor
        if (input[threadIdx.x] > input[threadIdx.x + 1]) {
            *is_sorted = false;
        }
    }
    __syncthreads();
}

// __global__ void bogo_sort_working(int* data, int size, int* output) {
//     extern __shared__ int shared_data[64];
//     extern __shared__ curandStatePhilox4_32_10_t shared_random_state;
//     extern __shared__ int shared_random;

//     extern __shared__ bool parity;
//     parity = false;
//     __syncthreads();

//     auto parity_shift = [](int p) { return p ? 32 : 0;};
    
//     //curand_init(unsigned long long seed,
//             // unsigned long long subsequence,
//             // unsigned long long offset,
//             // curandStatePhilox4_32_10_t *state)
//     curand_init(0, 0, 0, &shared_random_state);
//     shared_random = curand(&shared_random_state);
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         output[idx] = shared_random;
//     }

//     if (threadIdx.x < 32 && threadIdx.x < size) {
//         shared_data[threadIdx.x] = data[threadIdx.x];
//     }
//     __syncthreads();
//     // if (threadIdx.x == 0) {
//     //     printf("shared_data after loading: ");
//     //     for (int i = 0; i < 64; i++) {
//     //         printf("%d ", shared_data[i]);
//     //     }
//     //     printf("\n");
//     // }


//     // bit shift right by threadIdx.x/2 bits, then 
//     // check for parity to check if the threadIdx.x/2'th bit is 0 or 1
//     // if (threadIdx.x == 0) {
//     //     printf("\nThread operations:\n");
//     // }

//     int __shared__ offset;
//     if (threadIdx.x == 0) {
//         offset = 0;
//     }
//     int __shared__ swap_length;
//     for (swap_length = 2; swap_length <= 32; swap_length *= 2) {
//         if ((shared_random >> (threadIdx.x / swap_length + offset)) % 2 == 1) {
//             if (threadIdx.x % swap_length < swap_length / 2) {
//                 shared_data[threadIdx.x + swap_length / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
//             } else {
//                 shared_data[threadIdx.x - swap_length / 2 + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)];
//             }
//         } else {
//             shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity)]; 
//         }

//         if (threadIdx.x == 0) {
//             offset += 32 / swap_length;
//             parity = !parity;
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x != 0) {
//         shared_data[threadIdx.x + parity_shift(!parity)] = shared_data[threadIdx.x + parity_shift(parity) - 1];
//     } else {
//         shared_data[parity_shift(!parity)] = shared_data[size + parity_shift(parity) - 1]; 
//         parity = !parity;
//     }
//     __syncthreads();

//     // if (threadIdx.x == 0) {
//     //     printf("shared_random binary: ");
//     //     for (int i = 31; i >= 0; i--) {
//     //         printf("%d:%d ", i, (shared_random >> i) & 1);
//     //     }
//     //     printf("\n");
//     //     printf("shared_data: ");
//     //     for (int i = 0; i < 64; i++) {
//     //         printf("%d ", shared_data[i]);
//     //     }
//     //     printf("\n");
//     // }
//     // __syncthreads();

//     output[threadIdx.x] = shared_data[threadIdx.x + parity_shift(parity)];        
//     __shared__ bool is_sorted;
//     if (threadIdx.x == 0) {
//         is_sorted = true;
//     }
//     __syncthreads();

//     verify_sort(shared_data + parity_shift(parity), 32, &is_sorted);
//     __syncthreads();
        
// }

dim3 KernelManagerBogoSort::calculateGrid(int N, int threadsPerBlock) {
    // return dim3((N + threadsPerBlock - 1) / threadsPerBlock);
    return dim3(N);
}

float KernelManagerBogoSort::launchKernel(int* data, int* output) {
    int size = 32;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int smCount = 0;
    for (int i = 0; i < deviceCount; i = i + 1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        smCount += prop.multiProcessorCount;
    }

    printf("Number of SMs: %d\n", smCount);

    int numBlocks = smCount * 32;
    printf("Number of blocks: %d\n", numBlocks);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int threadsPerBlock = 32;
    dim3 grid = calculateGrid(numBlocks, threadsPerBlock);
    dim3 block(threadsPerBlock);
    printf("Grid dimensions: %d x %d x %d\n", grid.x, grid.y, grid.z);

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    bogo_sort<<<grid, block>>>(data, size, output);
    // bogo_sort<<<100, block>>>(data, size, output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}
