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

__global__ void get_time_kernel(long long *time) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  *time = (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void elapsed_time_kernel(float *time, long long start_time, long long end_time) {
  *time = (float) (end_time - start_time) / (1000 * 1000);
}

__global__ void randu_kernel(float *randu, int *seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  randu[index] = fabs(seed[index] / ((float) M));
}

__global__ void randn_kernel(float *randn, int *seed, int index) {
  float u = randu_kernel<<<1, 1>>>(randu, seed, index);
  float v = randu_kernel<<<1, 1>>>(randu, seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  randn[index] = sqrt(rt) * cosine;
}

__global__ void roundFloat_kernel(float *value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    value = newValue;
  else
    value = newValue++;
}

__global__ void setIf_kernel(unsigned char *array3D, int *dimX, int *dimY, int *dimZ, int testValue, int newValue) {
  int x, y, z;
  for (x = blockIdx.x * blockDim.x + threadIdx.x; x < *dimX; x += blockDim.x * gridDim.x) {
    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < *dimY; y += blockDim.y * gridDim.y) {
      for (z = blockIdx.z * blockDim.z + threadIdx.z; z < *dimZ; z += blockDim.z * gridDim.z) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}

__global__ void addNoise_kernel(unsigned char *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
  int x, y, z;
  for (x = blockIdx.x * blockDim.x + threadIdx.x; x < *dimX; x += blockDim.x * gridDim.x) {
    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < *dimY; y += blockDim.y * gridDim.y) {
      for (z = blockIdx.z * blockDim.z + threadIdx.z; z < *dimZ; z += blockDim.z * gridDim.z) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn_kernel<<<1, 1>>>(randn, seed, 0));
      }
    }
  }
}

__global__ void strelDisk_kernel(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = blockIdx.x * blockDim.x + threadIdx.x; x < diameter; x += blockDim.x * gridDim.x) {
    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < diameter; y += blockDim.y * gridDim.y) {
      float distance = sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}

__global__ void dilate_matrix_kernel(unsigned char *matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = blockIdx.x * blockDim.x + threadIdx.x; x < endX; x += blockDim.x * gridDim.x) {
    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < endY; y += blockDim.y * gridDim.y) {
      float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

__global__ void imdilate_disk_kernel(unsigned char *matrix, int dimX, int dimY, int dimZ, int error, unsigned char *newMatrix) {
  int z, x, y;
  for (z = blockIdx.z * blockDim.z + threadIdx.z; z < dimZ; z += blockDim.z * gridDim.z) {
    for (x = blockIdx.x * blockDim.x + threadIdx.x; x < dimX; x += blockDim.x * gridDim.x) {
      for (y = blockIdx.y * blockDim.y + threadIdx.y; y < dimY; y += blockDim.y * gridDim.y) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix_kernel<<<dim3(dimX, dimY, 1), dim3(16, 16, 1)>>>(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

__global__ void getneighbors_kernel(int *se, int numOnes, int *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = blockIdx.x * blockDim.x + threadIdx.x; x < diameter; x += blockDim.x * gridDim.x) {
    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < diameter; y += blockDim.y * gridDim.y) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int) (y - center);
        neighbors[neighY * 2 + 1] = (int) (x - center);
        neighY++;
      }
    }
  }
}

__global__ void videoSequence_kernel(unsigned char *I, int IszX, int IszY, int Nfr, int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  int x0 = (int) roundFloat_kernel<<<1, 1>>>(roundFloat, IszY / 2.0);
  int y0 = (int) roundFloat_kernel<<<1, 1>>>(roundFloat, IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;
  for (k = 1; k < Nfr; k++) {
    int xk = abs(x0 + (k - 1));
    int yk = abs(y0 - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }
  unsigned char *newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk_kernel<<<dim3(IszX, IszY, 1), dim3(16, 16, 1)>>>(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = blockIdx.x * blockDim.x + threadIdx.x; x < IszX; x += blockDim.x * gridDim.x) {
    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < IszY; y += blockDim.y * gridDim.y) {
      for (k = blockIdx.z * blockDim.z + threadIdx.z; k < Nfr; k += blockDim.z * gridDim.z) {
        I