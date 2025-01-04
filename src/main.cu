#include <cuda_runtime.h>
#include <stdio.h>
#include "./kernels/prune.cuh"

// #define DEBUG_PRINT

#define MERSENNE_EXPONENT 136279841

int main() {
    int N = (MERSENNE_EXPONENT - 1) / 31 + 1;
    size_t size_N = N * sizeof(int);
    int primes_length = 262144;
    size_t size_primes_length = primes_length * sizeof(int);
    

    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_primes = new int[primes_length];
    int *h_output = new int[primes_length];
    
    // generate good mersenne prime
    generate_mersenne_prime(MERSENNE_EXPONENT, h_input);
    // set output to 0
    for (int i = 0; i < primes_length; i++) {
        h_output[i] = 0;
    }
    // generate right prime numbers
    generate_prime_number_array(primes_length, h_primes);

    // #ifdef DEBUG_PRINT
    // // Print input array
    // printf("Input array: ");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", h_input[i]);
    // }
    // printf("\n");

    // // Print primes array
    // printf("Primes array: ");
    // for (int i = 0; i < primes_length; i++) {
    //     printf("%d ", h_primes[i]);
    // }
    // printf("\n");
    // #endif

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
    printf("Done!\n");

    // Print output array
    printf("Output array (first 10): ");
    for (int i = 0; i < 10 && i < primes_length; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    printf("Output array (last 10): ");
    for (int i = primes_length - 10; i < primes_length; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Write output array to file BEFORE cleanup
    FILE *output_file = fopen("../cache/output_array.txt", "w");
    if (output_file == NULL) {
        printf("Error opening output file!\n");
        return 1;
    }
    for (int i = 0; i < primes_length; i++) {
        fprintf(output_file, "%d\n", h_output[i]);
    }
    fclose(output_file);

    // Write primes array to file 
    FILE *primes_file = fopen("../cache/primes_array.txt", "w");
    if (primes_file == NULL) {
        printf("Error opening primes file!\n");
        return 1;
    }
    for (int i = 0; i < primes_length; i++) {
        fprintf(primes_file, "%d\n", h_primes[i]);
    }
    fclose(primes_file);

    // Cleanup
    delete[] h_input;
    delete[] h_primes;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_primes);
    cudaFree(d_output);

    return 0;
}
