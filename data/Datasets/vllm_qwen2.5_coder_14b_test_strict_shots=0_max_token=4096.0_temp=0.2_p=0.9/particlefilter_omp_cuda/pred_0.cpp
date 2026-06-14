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

__device__ long long get_time_device() {
  unsigned int lo, hi;
  asm volatile ("rdtsc" : "=a"(lo), "=d"(hi));
  return ((long long)hi << 32) | lo;
}

__device__ float elapsed_time_device(long long start_time, long long end_time) {
  return (float) (end_time - start_time) / (1000 * 1000);
}

__device__ float randu_device(int * seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((float) M));
}

__device__ float randn_device(int * seed, int index) {
  float u = randu_device(seed, index);
  float v = randu_device(seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

__device__ float roundFloat_device(float value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue + 1;
}

__global__ void setIf_kernel(unsigned char * array3D, int dimX, int dimY, int dimZ, int testValue, int newValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < dimX && y < dimY && z < dimZ) {
    int index = x * dimY * dimZ + y * dimZ + z;
    if (array3D[index] == testValue)
      array3D[index] = newValue;
  }
}

__global__ void addNoise_kernel(unsigned char * array3D, int dimX, int dimY, int dimZ, int * seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < dimX && y < dimY && z < dimZ) {
    int index = x * dimY * dimZ + y * dimZ + z;
    array3D[index] = array3D[index] + (unsigned char) (5 * randn_device(seed, 0));
  }
}

__global__ void strelDisk_kernel(int * disk, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int diameter = radius * 2 - 1;
  if (x < diameter && y < diameter) {
    float distance = sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
    if (distance < radius)
      disk[x * diameter + y] = 1;
    else
      disk[x * diameter + y] = 0;
  }
}

__global__ void dilate_matrix_kernel(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= posX - error && x < posX + error && y >= posY - error && y < posY + error) {
    float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
    if (distance < error)
      matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
  }
}

__global__ void imdilate_disk_kernel(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (z < dimZ && x < dimX && y < dimY) {
    if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
      dilate_matrix_kernel<<<dim3((dimX + BLOCK_X - 1) / BLOCK_X, (dimY + BLOCK_Y - 1) / BLOCK_Y, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(newMatrix, x, y, z, dimX, dimY, dimZ, error);
    }
  }
}

__global__ void getneighbors_kernel(int * se, int numOnes, int * neighbors, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int diameter = radius * 2 - 1;
  if (x < diameter && y < diameter) {
    if (se[x * diameter + y]) {
      int neighY = threadIdx.y + blockIdx.y * blockDim.y;
      neighbors[neighY * 2] = (int) (y - radius + 1);
      neighbors[neighY * 2 + 1] = (int) (x - radius + 1);
    }
  }
}

__global__ void videoSequence_kernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int x0 = (int) roundFloat_device(IszY / 2.0);
  int y0 = (int) roundFloat_device(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  int xk, yk, pos;
  for (int k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= IszX * IszY * Nfr)
      pos = 0;
    I[pos] = 1;
  }

  unsigned char * newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk_kernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_Z - 1) / BLOCK_Z), dim3(BLOCK_X, BLOCK_Y, BLOCK_Z)>>>(I, IszX, IszY, Nfr, 5, newMatrix);
  for (int x = 0; x < IszX; x++) {
    for (int y = 0; y < IszY; y++) {
      for (int k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  setIf_kernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_Z - 1) / BLOCK_Z), dim3(BLOCK_X, BLOCK_Y, BLOCK_Z)>>>(I, IszX, IszY, Nfr, 0, 100);
  setIf_kernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_Z - 1) / BLOCK_Z), dim3(BLOCK_X, BLOCK_Y, BLOCK_Z)>>>(I, IszX, IszY, Nfr, 1, 228);

  addNoise_kernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_Z - 1) / BLOCK_Z), dim3(BLOCK_X, BLOCK_Y, BLOCK_Z)>>>(I, IszX, IszY, Nfr, seed);
}

__global__ void particleFilter_kernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    float xe = roundFloat_device(IszY / 2.0);
    float ye = roundFloat_device(IszX / 2.0);

    int radius = 5;
    int diameter = radius * 2 - 1;
    int * disk = (int*) calloc(diameter * diameter, sizeof (int));
    strelDisk_kernel<<<dim3((diameter + BLOCK_X - 1) / BLOCK_X, (diameter + BLOCK_Y - 1) / BLOCK_Y, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(disk, radius);
    int countOnes = 0;
    for (int i = 0; i < diameter; i++) {
      for (int j = 0; j < diameter; j++) {
        if (disk[i * diameter + j] == 1)
          countOnes++;
      }
    }
    int * objxy = (int *) calloc(countOnes * 2, sizeof(int));
    getneighbors_kernel<<<dim3((diameter + BLOCK_X - 1) / BLOCK_X, (diameter + BLOCK_Y - 1) / BLOCK_Y, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(disk, countOnes, objxy, radius);

    float * weights = (float *) calloc(Nparticles, sizeof(float));
    for (int i = 0; i < Nparticles; i++) {
      weights[i] = 1 / ((float) (Nparticles));
    }

    float * likelihood = (float *) calloc(Nparticles + 1, sizeof (float));
    float * partial_sums = (float *) calloc(Nparticles + 1, sizeof (float));
    float * arrayX = (float *) calloc(Nparticles, sizeof (float));
    float * arrayY = (float *) calloc(Nparticles, sizeof (float));
    float * xj = (float *) calloc(Nparticles, sizeof (float));
    float * yj = (float *) calloc(Nparticles, sizeof (float));
    float * CDF = (float *) calloc(Nparticles, sizeof(float));

    int * ind = (int*) calloc(countOnes * Nparticles, sizeof(int));
    float * u = (float *) calloc(Nparticles, sizeof(float));

    for (int i = 0; i < Nparticles; i++) {
      xj[i] = xe;
      yj[i] = ye;
    }

    long long start = get_time_device();

    for (int k = 1; k < Nfr; k++) {
      // Kernel implementation for particle filter
    }

    long long end = get_time_device();
    printf("Average execution time of kernels: %f (s)\n", elapsed_time_device(start, end) / (Nfr - 1));

    free(likelihood);
    free(partial_sums);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
  }
}

int main(int argc, char * argv[]) {
  const char* usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";

  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }

  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }

#ifdef DEBUG
  printf("dimX=%d dimY=%d Nfr=%d Nparticles=%d\n", 
      IszX, IszY, Nfr, Nparticles);
#endif

  int * seed = (int *) calloc(Nparticles, sizeof(int));
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = i + 1;

  unsigned char * I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  long long start = get_time();

  videoSequence_kernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_Z - 1) / BLOCK_Z), dim3(BLOCK_X, BLOCK_Y, BLOCK_Z)>>>(I, IszX, IszY, Nfr, seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f (s)\n", elapsed_time(start, endVideoSequence));

  particleFilter_kernel<<<dim3((Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(I, IszX, IszY, Nfr, seed, Nparticles);
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f (s)\n", elapsed_time(endVideoSequence, endParticleFilter));

  printf("ENTIRE PROGRAM TOOK %f (s)\n", elapsed_time(start, endParticleFilter));

  free(seed);
  free(I);
  return 0;
}
```