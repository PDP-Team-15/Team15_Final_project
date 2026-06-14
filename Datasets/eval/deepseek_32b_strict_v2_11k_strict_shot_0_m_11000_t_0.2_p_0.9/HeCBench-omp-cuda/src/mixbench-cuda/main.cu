#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

__global__ void benchmark_func(float *cd, int grid_dim, int block_dim, int compute_iterations) {
    const unsigned int blockSize = block_dim;
    const int stride = blockSize;
    int idx = blockIdx.x * blockSize * granularity + threadIdx.x;
    const int big_stride = grid_dim * blockSize * granularity;
    
    float tmps[granularity];
    for(int k=0; k<fusion_degree; k++) {
        for(int j=0; j<granularity; j++) {
            tmps[j] = cd[idx + j*stride + k*big_stride];
            
            for(int i=0; i<compute_iterations; i++)
                tmps[j] = tmps[j]*tmps[j] + seed;
        }
        
        float sum = 0;
        for(int j=0; j<granularity; j+=2)
            sum += tmps[j] * tmps[j+1];
        
        for(int j=0; j<granularity; j++)
            cd[idx + k*big_stride] = sum;
    }
}

void mixbenchGPU(long size, int repeat) {
    const char *benchtype = "compute with global memory (block strided)";
    printf("Trade-off type:%s\n", benchtype);
    
    float *cd_host = (float*) malloc(size * sizeof(float));
    for(int i=0; i<size; i++) cd_host[i] = 0;
    
    float *cd_device;
    cudaMalloc((void**)&cd_device, size * sizeof(float));
    cudaMemcpy(cd_device, cd_host, size * sizeof(float), cudaMemcpyHostToDevice);
    
    const long reduced_grid_size = size / granularity / 128;
    const int block_dim = 256;
    const int grid_dim = reduced_grid_size / block_dim;
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    for(int i=0; i<repeat; i++) {
        benchmark_func<<<grid_dim, block_dim>>>(cd_device, grid_dim, block_dim, i);
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(start);
    for(int i=0; i<repeat; i++) {
        benchmark_func<<<grid_dim, block_dim>>>(cd_device, grid_dim, block_dim, i);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Total kernel execution time: %f (s)\n", time / 1000.0f);
    
    cudaMemcpy(cd_host, cd_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool ok = true;
    for(int i=0; i<size; i++) {
        if(cd_host[i] != 0) {
            if(fabsf(cd_host[i] - 0.050807f) > 1e-6f) {
                ok = false;
                printf("Verification failed at index %d: %f\n", i, cd_host[i]);
                break;
            }
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    
    free(cd_host);
    cudaFree(cd_device);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);
    
    unsigned int datasize = VECTOR_SIZE * sizeof(float);
    printf("Buffer size: %dMB\n", datasize / (1024*1024));
    
    mixbenchGPU(VECTOR_SIZE, repeat);
    
    return 0;
}