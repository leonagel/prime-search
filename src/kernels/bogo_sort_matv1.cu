#include "bogo_sort_matv1.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <mma.h>
#include <cuda.h>
#include <cuda_fp16.h>

// #define DEBUG_PERMUTE
// #define DEBUG_PRINT 
// #define DEBUG_SORT
#define PERMUTE_MATRIX_WIDTH 16
#define PERMUTATION_LENGTH 32
#define PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024 1024
#define PERMUTATION_VECTORS_32x16_FLAT_LENGTH_512 512
#define LOWER_ROW 512
#define NEXT_BLOCK 16

#define OUTER_WIDTH 16
#define INNER_DIM 16
#define OUTER_HEIGHT 16

using namespace nvcuda;
using namespace std;

__global__ void bogo_sort_matv1(int* data, int size, int* output) {
    extern __device__ int done;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        done = false;
    }
    __syncthreads();

    extern __shared__ __half permutation_matrix[PERMUTATION_MATRIX_32x32_FLAT_LENGTH_1024]; // 32x32 array
    extern __shared__ __half permutation_vectors[PERMUTATION_VECTORS_32x16_FLAT_LENGTH_512];
    extern __shared__ int temp_permutation[PERMUTATION_LENGTH];

    #ifdef DEBUG_PERMUTE
    for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
        data[i * PERMUTATION_LENGTH + threadIdx.x] = __float2half(threadIdx.x);
    }
    __syncthreads();
    #endif
    
    // Initialize random states and generate random ints
    extern __shared__ curandStatePhilox4_32_10_t random_states[PERMUTATION_LENGTH];
    extern __shared__ int random_ints[PERMUTATION_LENGTH];
    curand_init(0, threadIdx.x, 0, &random_states[threadIdx.x]);
    __syncthreads();
    
    random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    bogo_sort_permutation_gen(temp_permutation, size, random_ints);

    for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
        random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
        __syncthreads();
        bogo_sort_permutation_gen(temp_permutation, size, random_ints);
        permutation_vectors[i * PERMUTATION_LENGTH + threadIdx.x] = __float2half(data[temp_permutation[threadIdx.x]]);
        __syncthreads();
    }

    random_ints[threadIdx.x] = curand(&random_states[threadIdx.x]);
    __syncthreads();
    bogo_sort_basis_gen(permutation_matrix, size, random_ints);
    __syncthreads();

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("Before Matmul Permutation vectors:\n");
        for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
            printf("  Row %2d: ", i);
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%.1f ", __half2float(permutation_vectors[i * PERMUTATION_LENGTH + j]));
            }
            printf("\n");
        }
        printf("\n");
    }
    __syncthreads();
    #endif

    extern __shared__ long permutations_tried;
    if (threadIdx.x == 0) {
        permutations_tried = 0;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_ne_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_nw_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_se_frag;
    wmma::fragment<wmma::matrix_a, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::row_major> mat_sw_frag;
    wmma::fragment<wmma::matrix_b, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::col_major> vec_up_frag;
    wmma::fragment<wmma::matrix_b, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half, wmma::col_major> vec_down_frag;
    wmma::fragment<wmma::accumulator, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half> prod_up_frag;
    wmma::fragment<wmma::accumulator, OUTER_WIDTH, INNER_DIM, OUTER_HEIGHT, half> prod_down_frag;

    wmma::load_matrix_sync(vec_up_frag, permutation_vectors, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(vec_down_frag, permutation_vectors + NEXT_BLOCK, PERMUTATION_LENGTH);

    wmma::load_matrix_sync(mat_nw_frag, permutation_matrix, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_ne_frag, permutation_matrix + NEXT_BLOCK, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_sw_frag, permutation_matrix + LOWER_ROW, PERMUTATION_LENGTH);
    wmma::load_matrix_sync(mat_se_frag, permutation_matrix + LOWER_ROW + NEXT_BLOCK, PERMUTATION_LENGTH);

    while (permutations_tried < 1000000) {

        wmma::fill_fragment(prod_up_frag, 0.0f);
        wmma::mma_sync(prod_up_frag, mat_nw_frag, vec_up_frag, prod_up_frag);
        wmma::mma_sync(prod_up_frag, mat_ne_frag, vec_down_frag, prod_up_frag);
        wmma::store_matrix_sync(permutation_vectors, prod_up_frag, PERMUTATION_LENGTH, wmma::mem_col_major);

        wmma::fill_fragment(prod_down_frag, 0.0f);
        wmma::mma_sync(prod_down_frag, mat_sw_frag, vec_up_frag, prod_down_frag);
        wmma::mma_sync(prod_down_frag, mat_se_frag, vec_down_frag, prod_down_frag);
        wmma::store_matrix_sync(permutation_vectors + NEXT_BLOCK, prod_down_frag, PERMUTATION_LENGTH, wmma::mem_col_major);

        if (threadIdx.x == 0) {
            permutations_tried++;
        }

        __syncthreads();

        bool is_sorted;
        for (int i = 0; i < 16; i++) {
            verify_sort_matv1(permutation_vectors + i * 32, 32, &is_sorted);
            // if (is_sorted) {
            //     output[threadIdx.x] = permutation_vectors[i * 32 + threadIdx.x];
            //     atomicCAS(&done, 0, 1);
            //     return;
            // }
        }

    }

    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("After Matmul Permutation vectors:\n");
        for (int i = 0; i < PERMUTE_MATRIX_WIDTH; i++) {
            printf("  Row %2d: ", i);
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%5.1f ", __half2float(permutation_vectors[i * PERMUTATION_LENGTH + j]));
            }
            printf("\n");
        }
        printf("\n");
    }
    __syncthreads();
    #endif
    
    #ifdef DEBUG_PRINT
    if (threadIdx.x == 0) {
        printf("Output data: ");
        for (int i = 0; i < size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }
    #endif



    return;
}

__device__ void verify_sort_matv1(__half* input, int size, bool* is_sorted) {
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

__device__ void bogo_sort_basis_gen(__half* data, int size, int* random_ints) {
    extern __shared__ int sorted_ints[PERMUTATION_LENGTH * 2];
    auto parity_shift = [](int p) { return p ? PERMUTATION_LENGTH : 0;};
    
    __syncthreads();
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", random_ints[i]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();

    // Copy random ints to sorted_ints initial parity section
    sorted_ints[threadIdx.x] = random_ints[threadIdx.x];
    __syncthreads();

    extern __shared__ int step_size;
    extern __shared__ bool parity;
    extern __shared__ int merge_indices[PERMUTATION_LENGTH];
    if (threadIdx.x == 0) {
        step_size = 2;
        parity = false;
    }
    __syncthreads();

    while (step_size <= PERMUTATION_LENGTH) {
        if (threadIdx.x % step_size == 0) {
            int left_merge_counter = threadIdx.x;
            int right_merge_counter = threadIdx.x + 1;
            #ifdef DEBUG_SORT
            printf("Thread %d: left_merge_counter=%d, right_merge_counter=%d\n", 
                   threadIdx.x, left_merge_counter, right_merge_counter);
            #endif

            merge_indices[left_merge_counter] = 0;
            merge_indices[right_merge_counter] = 0;
            int print_thread_idx = 0;
            for (int i=0; i < step_size; i++) {
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("Thread %d, Step %d, Iteration %d:\n", threadIdx.x, step_size, i);
                }
                #endif
                
                int left_idx = threadIdx.x + merge_indices[left_merge_counter] + parity_shift(parity);
                int right_idx = threadIdx.x + merge_indices[right_merge_counter] + step_size/2 + parity_shift(parity);
                int dest_idx = threadIdx.x + i + parity_shift(!parity);
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  Left index: %d (value: %d)\n", left_idx, sorted_ints[left_idx]);
                    printf("  Right index: %d (value: %d)\n", right_idx, sorted_ints[right_idx]);
                    printf("  Destination index: %d\n", dest_idx);
                }
                #endif
                
                bool take_from_left = merge_indices[right_merge_counter] == step_size/2 ||
                    (sorted_ints[left_idx] < sorted_ints[right_idx] && 
                     merge_indices[left_merge_counter] < step_size/2);
                
                if (take_from_left) {
                    sorted_ints[dest_idx] = sorted_ints[left_idx];
                    merge_indices[left_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from left array\n");
                    }
                    #endif
                } else {
                    sorted_ints[dest_idx] = sorted_ints[right_idx]; 
                    merge_indices[right_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from right array\n");
                    }
                    #endif
                }
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  New value at destination: %d\n", sorted_ints[dest_idx]);
                    printf("  Left merge index: %d, Right merge index: %d\n\n", 
                           merge_indices[left_merge_counter], merge_indices[right_merge_counter]);
                    printf("  Current array state: ");
                    for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                        printf("%d ", sorted_ints[j + parity_shift(!parity)]);
                    }
                    printf("\n");
                }
                #endif
            }
        }
        if (threadIdx.x == 0) {
            step_size *= 2;
            parity = !parity;
        }
        __syncthreads();
        #ifdef DEBUG_SORT
        if (threadIdx.x == 0) {
            printf("Random ints after step %d: ", step_size);
            for (int i = 0; i < PERMUTATION_LENGTH; i++) {
                printf("%d ", sorted_ints[i + parity_shift(parity)]);
            }
            printf("\n");
        }
        #endif
        __syncthreads();
    }
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Sorted random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", sorted_ints[i + parity_shift(parity)]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
    int my_value = random_ints[threadIdx.x];
    int final_index = -1;
    for (int i = 0; i < PERMUTATION_LENGTH; i++) {
        if (sorted_ints[i + parity_shift(parity)] == my_value) {
            final_index = i;
            break;
        }
    }
    for (int i = 0; i < PERMUTATION_LENGTH; i++) {
        data[threadIdx.x * PERMUTATION_LENGTH + i] = __float2half(i == final_index ? 1.0f : 0.0f);
    }
    __syncthreads();

    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Final indices matrix:\n");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("  ");
            for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                printf("%d ", data[i * PERMUTATION_LENGTH + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    #endif

    // data[threadIdx.x] = sorted_ints[threadIdx.x + parity_shift(!parity)];
    __syncthreads();
}

__device__ void bogo_sort_permutation_gen(int* data, int size, int* random_ints) {
    extern __shared__ int sorted_ints[64];
    auto parity_shift = [](int p) { return p ? PERMUTATION_LENGTH : 0;};
    
    __syncthreads();
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", random_ints[i]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
    // Copy random ints to sorted_ints initial parity section
    sorted_ints[threadIdx.x] = random_ints[threadIdx.x];
    __syncthreads();
    extern __shared__ int step_size;
    extern __shared__ bool parity;
    extern __shared__ int merge_indices[PERMUTATION_LENGTH];
    if (threadIdx.x == 0) {
        step_size = 2;
        parity = false;
    }
    __syncthreads();
    while (step_size <= PERMUTATION_LENGTH) {
        if (threadIdx.x % step_size == 0) {
            int left_merge_counter = threadIdx.x;
            int right_merge_counter = threadIdx.x + 1;
            #ifdef DEBUG_SORT
            printf("Thread %d: left_merge_counter=%d, right_merge_counter=%d\n", 
                   threadIdx.x, left_merge_counter, right_merge_counter);
            #endif
            merge_indices[left_merge_counter] = 0;
            merge_indices[right_merge_counter] = 0;
            int print_thread_idx = 0;
            for (int i=0; i < step_size; i++) {
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("Thread %d, Step %d, Iteration %d:\n", threadIdx.x, step_size, i);
                }
                #endif
                
                int left_idx = threadIdx.x + merge_indices[left_merge_counter] + parity_shift(parity);
                int right_idx = threadIdx.x + merge_indices[right_merge_counter] + step_size/2 + parity_shift(parity);
                int dest_idx = threadIdx.x + i + parity_shift(!parity);
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  Left index: %d (value: %d)\n", left_idx, sorted_ints[left_idx]);
                    printf("  Right index: %d (value: %d)\n", right_idx, sorted_ints[right_idx]);
                    printf("  Destination index: %d\n", dest_idx);
                }
                #endif
                
                bool take_from_left = merge_indices[right_merge_counter] == step_size/2 ||
                    (sorted_ints[left_idx] < sorted_ints[right_idx] && 
                     merge_indices[left_merge_counter] < step_size/2);
                
                if (take_from_left) {
                    sorted_ints[dest_idx] = sorted_ints[left_idx];
                    merge_indices[left_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from left array\n");
                    }
                    #endif
                } else {
                    sorted_ints[dest_idx] = sorted_ints[right_idx]; 
                    merge_indices[right_merge_counter]++;
                    #ifdef DEBUG_SORT
                    if (threadIdx.x == print_thread_idx) {
                        printf("  Taking from right array\n");
                    }
                    #endif
                }
                
                #ifdef DEBUG_SORT
                if (threadIdx.x == print_thread_idx) {
                    printf("  New value at destination: %d\n", sorted_ints[dest_idx]);
                    printf("  Left merge index: %d, Right merge index: %d\n\n", 
                           merge_indices[left_merge_counter], merge_indices[right_merge_counter]);
                    printf("  Current array state: ");
                    for (int j = 0; j < PERMUTATION_LENGTH; j++) {
                        printf("%d ", sorted_ints[j + parity_shift(!parity)]);
                    }
                    printf("\n");
                }
                #endif
            }
        }
        if (threadIdx.x == 0) {
            step_size *= 2;
            parity = !parity;
        }
        __syncthreads();
        #ifdef DEBUG_SORT
        if (threadIdx.x == 0) {
            printf("Random ints after step %d: ", step_size);
            for (int i = 0; i < PERMUTATION_LENGTH; i++) {
                printf("%d ", sorted_ints[i + parity_shift(parity)]);
            }
            printf("\n");
        }
        #endif
        __syncthreads();
    }
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Sorted random ints: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", sorted_ints[i + parity_shift(parity)]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
    int my_value = random_ints[threadIdx.x];
    int my_index = -1;
    for (int i = 0; i < PERMUTATION_LENGTH; i++) {
        if (sorted_ints[i + parity_shift(parity)] == my_value) {
            my_index = i;
            break;
        }
    }
    data[threadIdx.x] = my_index;
    #ifdef DEBUG_SORT
    if (threadIdx.x == 0) {
        printf("Final sorted array: ");
        for (int i = 0; i < PERMUTATION_LENGTH; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }
    #endif
    __syncthreads();
}


dim3 KernelManagerBogoSortMatV1::calculateGrid(int n, int threadsPerBlock) {
    // return dim3((INNER_DIM + threadsPerBlock - 1) / threadsPerBlock);
    return dim3(n);
}

float KernelManagerBogoSortMatV1::launchKernel(int* data, int* output) {
    int size = 32;
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

    int numBlocks = smCount * 64;
    #ifdef DEBUG_PRINT
    printf("Number of blocks: %d\n", numBlocks);
    #endif

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int threadsPerBlock = 32;
    dim3 grid = calculateGrid(numBlocks, threadsPerBlock);
    dim3 block(threadsPerBlock);
    #ifdef DEBUG_PRINT
    printf("Grid dimensions: %d x %d x %d\n", grid.x, grid.y, grid.z);
    #endif

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    bogo_sort_matv1<<<grid, block>>>(data, size, output);
    // bogo_sort_matv1<<<1, block>>>(data, size, output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}
