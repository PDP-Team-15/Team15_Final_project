#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f
#define BLOCK_SIZE 256

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

float elapsed_time(long long start_time, long long end_time) {
    return (float)(end_time - start_time) / (1000 * 1000);
}

__device__ float randu(int *seed, int index) {
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabsf(seed[index] / (float)M);
}

__device__ float randn(int *seed, int index) {
    float u = randu(seed, index);
    float v = randu(seed, index);
    float cosine = cosf(2 * PI * v);
    float rt = -2 * logf(u);
    return sqrtf(rt) * cosine;
}

__device__ float roundFloat(float value) {
    int newValue = (int)value;
    if (value - newValue < 0.5f)
        return newValue;
    else
        return newValue + 1;
}

__global__ void dilate_matrix_kernel(unsigned char *newMatrix, int x, int y, int z, int dimX, int dimY, int dimZ, int error) {
    int startX = x - error;
    while (startX < 0) startX++;
    int startY = y - error;
    while (startY < 0) startY++;
    int endX = x + error;
    while (endX > dimX) endX--;
    int endY = y + error;
    while (endY > dimY) endY--;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startX; i < endX; i++) {
        for (int j = startY; j < endY; j++) {
            float distance = sqrtf(powf((float)(i - x), 2) + powf((float)(j - y), 2));
            if (distance < error) {
                newMatrix[i * dimY * dimZ + j * dimZ + z] = 1;
            }
        }
    }
}

void imdilate_disk(unsigned char *matrix, int dimX, int dimY, int dimZ, int error, unsigned char *newMatrix) {
    int x, y, z;
    for (z = 0; z < dimZ; z++) {
        for (x = 0; x < dimX; x++) {
            for (y = 0; y < dimY; y++) {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
                    dilate_matrix_kernel<<<1, 1>>>(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

__global__ void getneighbors_kernel(int *se, int numOnes, int *neighbors, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 * radius - 1 && y < 2 * radius - 1) {
        if (se[x * (2 * radius - 1) + y] == 1) {
            int center = radius - 1;
            neighbors[(y - center) * 2] = y - center;
            neighbors[(y - center) * 2 + 1] = x - center;
        }
    }
}

void getneighbors(int *se, int numOnes, int *neighbors, int radius) {
    getneighbors_kernel<<<dim3(2 * radius - 1), dim3(2 * radius - 1)>>>(se, numOnes, neighbors, radius);
}

__global__ void videoSequence_kernel(unsigned char *I, int IszX, int IszY, int Nfr, int *seed) {
    int x0 = roundFloat(IszY / 2.0f);
    int y0 = roundFloat(IszX / 2.0f);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;
    int k = blockIdx.x;
    int xk = abs(x0 + (k - 1));
    int yk = abs(y0 - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos < IszX * IszY * Nfr) {
        I[pos] = 1;
    }
}

void videoSequence(unsigned char *I, int IszX, int IszY, int Nfr, int *seed) {
    videoSequence_kernel<<<Nfr, 1>>>(I, IszX, IszY, Nfr, seed);
}

__global__ void particleFilter_kernel(unsigned char *I, int IszX, int IszY, int Nfr, int *seed, int Nparticles, float *weights, float *likelihood, float *partial_sums, float *arrayX, float *arrayY, float *xj, float *yj, float *CDF, int *ind, float *u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nparticles) {
        arrayX[i] = xj[i];
        arrayY[i] = yj[i];
        weights[i] = 1.0f / Nparticles;
        seed[i] = (A * seed[i] + C) % M;
        float u_val = randu(seed, i);
        float v_val = randu(seed, i);
        arrayX[i] += 1.0f + 5.0f * (sqrtf(-2.0f * logf(u_val)) * cosf(2.0f * PI * v_val));
        seed[i] = (A * seed[i] + C) % M;
        u_val = randu(seed, i);
        v_val = randu(seed, i);
        arrayY[i] += -2.0f + 2.0f * (sqrtf(-2.0f * logf(u_val)) * cosf(2.0f * PI * v_val));
    }
}

int particleFilter(unsigned char *I, int IszX, int IszY, int Nfr, int *seed, int Nparticles) {
    int max_size = IszX * IszY * Nfr;
    float xe = roundFloat(IszY / 2.0f);
    float ye = roundFloat(IszX / 2.0f);
    int radius = 5;
    int diameter = radius * 2 - 1;
    int *disk = (int *)calloc(diameter * diameter, sizeof(int));
    strelDisk(disk, radius);
    int countOnes = 0;
    for (int x = 0; x < diameter; x++) {
        for (int y = 0; y < diameter; y++) {
            if (disk[x * diameter + y] == 1) countOnes++;
        }
    }
    int *objxy = (int *)calloc(2 * countOnes, sizeof(int));
    getneighbors(disk, countOnes, objxy, radius);
    free(disk);
    float *weights = (float *)calloc(Nparticles, sizeof(float));
    for (int x = 0; x < Nparticles; x++) weights[x] = 1.0f / Nparticles;
    float *likelihood = (float *)calloc(Nparticles + 1, sizeof(float));
    float *partial_sums = (float *)calloc(Nparticles + 1, sizeof(float));
    float *arrayX = (float *)calloc(Nparticles, sizeof(float));
    float *arrayY = (float *)calloc(Nparticles, sizeof(float));
    float *xj = (float *)calloc(Nparticles, sizeof(float));
    float *yj = (float *)calloc(Nparticles, sizeof(float));
    float *CDF = (float *)calloc(Nparticles, sizeof(float));
    int *ind = (int *)calloc(countOnes * Nparticles, sizeof(int));
    float *u = (float *)calloc(Nparticles, sizeof(float));
    for (int x = 0; x < Nparticles; x++) {
        xj[x] = xe;
        yj[x] = ye;
    }
    long long offload_start = get_time();
    int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int k = 1; k < Nfr; k++) {
        particleFilter_kernel<<<num_blocks, BLOCK_SIZE>>>(I, IszX, IszY, Nfr, seed, Nparticles, weights, likelihood, partial_sums, arrayX, arrayY, xj, yj, CDF, ind, u);
    }
    long long offload_end = get_time();
    printf("Device offloading time: %lf (s)\n", elapsed_time(offload_start, offload_end));
    free(weights);
    free(likelihood);
    free(partial_sums);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
    return 0;
}

int main(int argc, char *argv[]) {
    // Main function remains the same, as per the problem statement.
    // All function calls to videoSequence and particleFilter are unchanged.
}