```cuda
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
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  return (long long) (elapsed_time * 1000);
}

__host__ float elapsed_time(long long start_time, long long end_time) {
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

__host__ float roundFloat(float value) {
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
          dilate_matrixKernel<<<dim3(dimX, dimY, 1), dim3(16, 16, 1)>>>(newMatrix, x, y, z, dimX, dimY, dimZ, error);
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
      neighbors[(blockIdx.z * blockDim.z + threadIdx.z) * 2] = (int) (y - radius + 1);
      neighbors[(blockIdx.z * blockDim.z + threadIdx.z) * 2 + 1] = (int) (x - radius + 1);
    }
  }
}

__global__ void videoSequenceKernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (k < Nfr) {
    int xk = abs((int) roundFloat(IszY / 2.0) + (k - 1));
    int yk = abs((int) roundFloat(IszX / 2.0) - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= IszX * IszY * Nfr)
      pos = 0;
    I[pos] = 1;
  }
}

__global__ void imdilate_diskKernel2(unsigned char * I, int IszX, int IszY, int Nfr, int error, unsigned char * newMatrix) {
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z < Nfr) {
    int x, y;
    for (x = 0; x < IszX; x++) {
      for (y = 0; y < IszY; y++) {
        newMatrix[x * IszY * Nfr + y * Nfr + z] = I[x * IszY * Nfr + y * Nfr + z];
      }
    }
  }
}

__global__ void setIfKernel2(unsigned char * I, int IszX, int IszY, int Nfr, int testValue, int newValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < IszX && y < IszY && k < Nfr) {
    if (I[x * IszY * Nfr + y * Nfr + k] == testValue)
      I[x * IszY * Nfr + y * Nfr + k] = newValue;
  }
}

__global__ void addNoiseKernel2(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < IszX && y < IszY && k < Nfr) {
    I[x * IszY * Nfr + y * Nfr + k] = I[x * IszY * Nfr + y * Nfr + k] + (unsigned char) (5 * randn(seed, 0));
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

__global__ void particleFilterKernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, float * weights, float * arrayX, float * arrayY, float * xj, float * yj, float * CDF, float * u, int * ind) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    arrayX[x] = xj[x];
    arrayY[x] = yj[x];
    weights[x] = 1.0f / ((float) (Nparticles));
    seed[x] = (A * seed[x] + C) % M;
    float u1 = fabsf(seed[x] / ((float) M));
    seed[x] = (A * seed[x] + C) % M;
    float v = fabsf(seed[x] / ((float) M));
    arrayX[x] += 1.0f + 5.0f * (sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * v));
    seed[x] = (A * seed[x] + C) % M;
    u1 = fabsf(seed[x] / ((float) M));
    seed[x] = (A * seed[x] + C) % M;
    v = fabsf(seed[x] / ((float) M));
    arrayY[x] += -2.0f + 2.0f * (sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * v));
    int y;
    for (y = 0; y < 8; y++) {
      int iX = arrayX[x];
      int iY = arrayY[x];
      int rnd_iX = (arrayX[x] - iX) < .5f ? iX : iX++;
      int rnd_iY = (arrayY[x] - iY) < .5f ? iY : iY++;
      int indX = rnd_iX + (int) (y % 2 == 0 ? 0 : 4);
      int indY = rnd_iY + (int) (y % 2 == 0 ? 4 : 0);
      ind[x * 8 + y] = abs(indX * IszY * Nfr + indY * Nfr + 0);
      if (ind[x * 8 + y] >= IszX * IszY * Nfr)
        ind[x * 8 + y] = 0;
    }
    float likelihoodSum = 0.0f;
    for (int y = 0; y < 8; y++) {
      likelihoodSum += ((I[ind[x * 8 + y]] - 100) * (I[ind[x * 8 + y]] - 100) -
          (I[ind[x * 8 + y]] - 228) * (I[ind[x * 8 + y]] - 228)) / 50.0f;
    }
    weights[x] = weights[x] * expf(likelihoodSum / 8 - SCALE_FACTOR);
  }
}

__global__ void reduceKernel(float * weights, int Nparticles, float * partial_sums) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    partial_sums[blockIdx.x] += weights[x];
  }
}

__global__ void normalizeKernel(float * weights, int Nparticles, float * partial_sums) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    weights[x] /= partial_sums[0];
  }
}

__global__ void cumulativeSumKernel(float * weights, int Nparticles, float * CDF) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    CDF[x] = weights[x] + (x > 0 ? CDF[x - 1] : 0);
  }
}

__global__ void particleFilterKernel2(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, float * weights, float * arrayX, float * arrayY, float * xj, float * yj, float * CDF, float * u, int * ind) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < Nparticles) {
    int index = -1;
    for (int y = 0; y < Nparticles; y++) {
      if (CDF[y] >= u[x]) {
        index = y;
        break;
      }
    }
    if (index == -1) {
      index = Nparticles - 1;
    }
    xj[x] = arrayX[index];
    yj[x] = arrayY[index];
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
  printf("dimX=%d dimY=%d Nfr=%d Nparticles=%d\n", IszX, IszY, Nfr, Nparticles);
#endif
  int * seed = (int *) calloc(Nparticles, sizeof(int));
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = i + 1;
  unsigned char * I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  long long start = get_time();
  videoSequenceKernel<<<dim3(1, 1, Nfr), dim3(1, 1, 1)>>>(I, IszX, IszY, Nfr, seed);
  cudaDeviceSynchronize();
  imdilate_diskKernel2<<<dim3(1, 1, Nfr), dim3(1, 1, 1)>>>(I, IszX, IszY, Nfr, 5, (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char)));
  cudaDeviceSynchronize();
  setIfKernel2<<<dim3(1, 1, IszX), dim3(16, 16, 1)>>>(I, IszX, IszY, Nfr, 0, 100);
  cudaDeviceSynchronize();
  setIfKernel2<<<dim3(1, 1, IszX), dim3(16, 16, 1)>>>(I, IszX, IszY, Nfr, 1, 228);
  cudaDeviceSynchronize();
  addNoiseKernel2<<<dim3(1, 1, Nfr), dim3(16, 16, 1)>>>(I, IszX, IszY, Nfr, seed);
  cudaDeviceSynchronize();
  particleFilterKernel<<<dim3(1, 1, Nparticles), dim3(16, 16, 1)>>>(I, IszX, IszY, Nfr, seed, Nparticles, (float *) calloc(Nparticles, sizeof(float)), (float *) calloc(Nparticles, sizeof(float)), (float *) calloc(Nparticles, sizeof(float)), (float *) calloc(Nparticles, sizeof(float)), (float *) calloc(Nparticles, sizeof(float)), (float *) calloc(Nparticles, sizeof(float)), (int *) calloc(Nparticles * 8, sizeof(int)));
  cudaDeviceSynchronize();
  float * weights = (float *) calloc(Nparticles, sizeof(float));
  float * arrayX = (float *) calloc(Nparticles, sizeof(float));
  float * arrayY = (float *) calloc(Nparticles, sizeof(float));
  float * xj = (float *) calloc(Nparticles, sizeof(float));
  float * yj = (float *) calloc(Nparticles, sizeof(float));
  float * CDF = (float *) calloc(Nparticles, sizeof(float));
  float * u = (float *) calloc(Nparticles, sizeof(float));
  int * ind = (int *) calloc(Nparticles * 8, sizeof(int));
  for (int i = 0; i < Nparticles; i++) {
    weights[i] = 1.0f / ((float) (Nparticles));
    seed[i] = (A * seed[i] + C) % M;
    float u1 = fabsf(seed[i] / ((float) M));
    seed[i] = (A * seed[i] + C) % M;
    float v = fabsf(seed[i] / ((float) M));
    arrayX[i] = 1.0f + 5.0f * (sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * v));
    seed[i] = (A * seed[i] + C) % M;
    u1 = fabsf(seed[i] / ((float) M));
    seed[i] = (A * seed[i] + C) % M;
    v = fabsf(seed[i] / ((float) M));
    arrayY[i] = -2.0f + 2.0f * (sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * v));
    int y;
    for (y = 0; y < 8; y++) {
      int iX = arrayX[i];
      int iY = arrayY[i];
      int rnd_iX = (arrayX[i] - iX) < .5f ? iX : iX++;
      int rnd_iY = (arrayY[i] - iY) < .5f ? iY : iY++;
      int indX = rnd_iX + (int) (y % 2 == 0 ? 0 : 4);
      int indY = rnd_iY + (int) (y % 2 == 0 ? 4 : 0);
      ind[i * 8 + y] = abs(indX * IszY * Nfr + indY * Nfr + 0);
      if (ind[i * 8 + y] >= IszX * IszY * Nfr)
        ind[i * 8 + y] = 0;
    }
    float likelihoodSum = 0.0f;
    for (int y = 0; y < 8; y++) {
      likelihoodSum += ((I[ind[i * 8 + y]] - 100) * (I[ind[i * 8 + y]] - 100) -
          (I[ind[i * 8 + y]] - 228) * (I[ind[i * 8 + y]] - 228)) / 50.0f;
    }
    weights[i] = weights[i] * expf(likelihoodSum / 8 - SCALE_FACTOR);
  }
  reduceKernel<<<(Nparticles + 255) / 256, 256>>>(weights, Nparticles, (float *) calloc((Nparticles + 255) / 256, sizeof(float)));
  cudaDeviceSynchronize();
  normalizeKernel<<<(Nparticles + 255) / 256, 256>>>(weights, Nparticles, (float *) calloc((Nparticles + 255) / 256, sizeof(float)));
  cudaDeviceSynchronize();
  cumulativeSumKernel<<<(Nparticles + 255) / 256, 256>>>(weights, Nparticles, CDF);
  cudaDeviceSynchronize();
  particleFilterKernel2<<<dim3(1, 1, Nparticles), dim3(16, 16, 1)>>>(I, IszX, IszY, Nfr, seed, Nparticles, weights, arrayX, arrayY, xj, yj, CDF, u, ind);
  cudaDeviceSynchronize();
  FILE *fid;
  fid = fopen("output.txt", "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return -1;
  }
  float xe = 0;
  float ye = 0;
  for (int i = 0; i < Nparticles; i++) {
    xe += xj[i] * weights[i];
    ye += yj[i] * weights[i];
  }
  float distance = sqrt(pow((float) (xe - (int) roundFloat(IszY / 2.0)), 2) + pow((float) (ye - (int) roundFloat(IszX / 2.0)), 2));
  fprintf(fid, "XE: %lf\n", xe);
  fprintf(fid, "YE: %lf\n", ye);
  fprintf(fid, "distance: %lf\n", distance);
  fclose(fid);
  free(seed);
  free(I);
  return 0;
}
```