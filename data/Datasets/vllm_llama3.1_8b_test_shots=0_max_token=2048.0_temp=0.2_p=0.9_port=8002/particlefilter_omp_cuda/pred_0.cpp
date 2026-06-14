Here's the CUDA version of the provided OpenMP code:

```cpp
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

#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 256
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__device__ long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__device__ float elapsed_time(long long start_time, long long end_time) {
  return (float) (end_time - start_time) / (1000 * 1000);
}

__device__ float randu(int * seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((float) M));
}

__device__ float randn(int * seed, int index) {
  float u = randu(seed, index);
  float v = randu(seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

__device__ float roundFloat(float value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

__global__ void setIfKernel(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int testValue, int newValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < *dimX && y < *dimY && z < *dimZ) {
    if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
      array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
  }
}

__global__ void addNoiseKernel(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < *dimX && y < *dimY && z < *dimZ) {
    array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
  }
}

__global__ void strelDiskKernel(int * disk, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < radius * 2 - 1 && y < radius * 2 - 1) {
    float distance = sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
    if (distance < radius)
      disk[x * (radius * 2 - 1) + y] = 1;
    else
      disk[x * (radius * 2 - 1) + y] = 0;
  }
}

__global__ void dilate_matrixKernel(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= posX - error && x <= posX + error && y >= posY - error && y <= posY + error) {
    float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
    if (distance < error)
      matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
  }
}

__global__ void imdilate_diskKernel(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z < dimZ) {
    int x, y;
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrixKernel<<<dim3(dimX, dimY, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

__global__ void getneighborsKernel(int * se, int numOnes, int * neighbors, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < radius * 2 - 1 && y < radius * 2 - 1) {
    if (se[x * (radius * 2 - 1) + y]) {
      neighbors[(blockIdx.y * blockDim.y + threadIdx.y) * 2] = (int) (y - radius + 1);
      neighbors[(blockIdx.y * blockDim.y + threadIdx.y) * 2 + 1] = (int) (x - radius + 1);
    }
  }
}

__global__ void videoSequenceKernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < Nfr) {
    int xk = abs((int) roundFloat(IszY / 2.0) + (k - 1));
    int yk = abs((int) roundFloat(IszX / 2.0) - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= IszX * IszY * Nfr)
      pos = 0;
    I[pos] = 1;

    unsigned char * newMatrix = (unsigned char *) malloc(IszX * IszY * Nfr * sizeof(unsigned char));
    imdilate_diskKernel<<<dim3(IszX, IszY, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++) {
      for (y = 0; y < IszY; y++) {
        for (k = 0; k < Nfr; k++) {
          I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
        }
      }
    }
    free(newMatrix);

    setIfKernel<<<dim3(IszX, IszY, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(I, &IszX, &IszY, &Nfr, 0, 100);
    setIfKernel<<<dim3(IszX, IszY, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(I, &IszX, &IszY, &Nfr, 1, 228);

    addNoiseKernel<<<dim3(IszX, IszY, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(I, &IszX, &IszY, &Nfr, seed);
  }
}

__global__ void findIndexKernel(float * CDF, int lengthCDF, float value, int * index) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < lengthCDF) {
    if (CDF[x] >= value) {
      *index = x;
      break;
    }
  }
  if (*index == -1) {
    *index = lengthCDF - 1;
  }
}

__global__ void particleFilterKernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, float * xe, float * ye) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    xe[x] = (int) roundFloat(IszY / 2.0);
    ye[x] = (int) roundFloat(IszX / 2.0);
  }

 