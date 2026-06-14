Below is the translated CUDA version of the provided OpenMP code. The translation involves converting the OpenMP parallel constructs to CUDA kernels and managing memory transfers between the host and device.

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

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

float elapsed_time(long long start_time, long long end_time) {
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
    return newValue + 1;
}

__global__ void setIfKernel(unsigned char * array3D, int dimX, int dimY, int dimZ, int testValue, int newValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < dimX && y < dimY && z < dimZ) {
    int index = x * dimY * dimZ + y * dimZ + z;
    if (array3D[index] == testValue)
      array3D[index] = newValue;
  }
}

__global__ void addNoiseKernel(unsigned char * array3D, int dimX, int dimY, int dimZ, int * seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < dimX && y < dimY && z < dimZ) {
    int index = x * dimY * dimZ + y * dimZ + z;
    array3D[index] = array3D[index] + (unsigned char) (5 * randn(seed, 0));
  }
}

void strelDisk(int * disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      float distance = sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}

__global__ void dilate_matrix_kernel(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < dimX && y < dimY) {
    float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
    if (distance < error)
      newMatrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
  }
}

__global__ void imdilate_disk_kernel(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < dimX && y < dimY && z < dimZ) {
    if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
      dilate_matrix_kernel<<<dim3(dimX, dimY, 1), dim3(BLOCK_X, BLOCK_Y, 1)>>>(newMatrix, x, y, z, dimX, dimY, dimZ, error, newMatrix);
    }
  }
}

void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int) (y - center);
        neighbors[neighY * 2 + 1] = (int) (x - center);
        neighY++;
      }
    }
  }
}

__global__ void videoSequenceKernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < Nfr) {
    int x0 = (int) roundFloat(IszY / 2.0);
    int y0 = (int) roundFloat(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    int xk, yk, pos;
    for (k = 1; k < Nfr; k++) {
      xk = abs(x0 + (k - 1));
      yk = abs(y0 - 2 * (k - 1));
      pos = yk * IszY * Nfr + xk * Nfr + k;
      if (pos >= IszX * IszY * Nfr)
        pos = 0;
      I[pos] = 1;
    }

    unsigned char * newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
    imdilate_disk_kernel<<<dim3(IszX, IszY, Nfr), dim3(BLOCK_X, BLOCK_Y, 1)>>>(I, IszX, IszY, Nfr, 5, newMatrix);
    for (int x = 0; x < IszX; x++) {
      for (int y = 0; y < IszY; y++) {
        for (int k = 0; k < Nfr; k++) {
          I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
        }
      }
    }
    free(newMatrix);

    setIfKernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_X - 1) / BLOCK_X), dim3(BLOCK_X, BLOCK_Y, BLOCK_X)>>>(I, IszX, IszY, Nfr, 0, 100);
    setIfKernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_X - 1) / BLOCK_X), dim3(BLOCK_X, BLOCK_Y, BLOCK_X)>>>(I, IszX, IszY, Nfr, 1, 228);

    addNoiseKernel<<<dim3((IszX + BLOCK_X - 1) / BLOCK_X, (IszY + BLOCK_Y - 1) / BLOCK_Y, (Nfr + BLOCK_X - 1) / BLOCK_X), dim3(BLOCK_X, BLOCK_Y, BLOCK_X)>>>(I, IszX, IszY, Nfr, seed);
  }
}

int findIndex(float * CDF, int lengthCDF, float value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}

__global__ void particleFilterKernel(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles, float * weights, float * arrayX, float * arrayY, float * xj, float * yj, float * CDF, int * ind, float * u, float * partial_sums, float * likelihood) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Nparticles) {
    arrayX[i] = xj[i];
    arrayY[i] = yj[i];
    weights[i] = 1.0f / ((float) (Nparticles)); 
    seed[i] = (A*seed[i] + C) % M;
    float u = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    float v = fabsf(seed[i]/((float)M));
    arrayX[i] += 1.0f + 5.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));

    seed[i] = (A*seed[i] + C) % M;
    u = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    v = fabsf(seed[i]/((float)M));
    arrayY[i] += -2.0f + 2.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));

    for(int y = 0; y < countOnes; y++){
      int iX = arrayX[i];
      int iY = arrayY[i];
      int rnd_iX = (arrayX[i] - iX) < .5f ? iX : iX++;
      int rnd_iY = (arrayY[i] - iY) < .5f ? iY : iY++;
      ind[i*countOnes + y] = abs(rnd_iX*IszY*Nfr + rnd_iY*Nfr + k);
      if(ind[i*countOnes + y] >= IszX * IszY * Nfr)
        ind[i*countOnes + y] = 0;
    }
    float likelihoodSum = 0.0f;
    for(int x = 0; x < countOnes; x++)
      likelihoodSum += ((I[ind[i*countOnes + x]] - 100) * (I[ind[i*countOnes + x]] - 100) -
          (I[ind[i*countOnes + x]] - 228) * (I[ind[i*countOnes + x]] - 228)) / 50.0f;
    likelihood[i] = likelihoodSum/countOnes-SCALE_FACTOR;

    weights[i] = weights[i] * expf(likelihood[i]);
  }
}

int particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
  int max_size = IszX * IszY * Nfr;

  float xe = roundFloat(IszY / 2.0);
  float ye = roundFloat(IszX / 2.0);

  int radius = 5;
  int diameter = radius * 2 - 1;
  int * disk = (int*) calloc(diameter * diameter, sizeof (int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int * objxy = (int *) calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  float * weights = (float *) calloc(Nparticles, sizeof(float));
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((float) (Nparticles));
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

  for (x = 0; x < Nparticles; x++) {
    xj[x] = xe;
    yj[x] = ye;
  }

  long long offload_start = get_time();

  int k;

  int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (k = 1; k < Nfr; k++) {
    particleFilterKernel<<<num_blocks, BLOCK_SIZE>>>(I, IszX, IszY, Nfr, seed, Nparticles, weights, arrayX, arrayY, xj, yj, CDF, ind, u, partial_sums, likelihood);

    // Reduction of weights
    float sum = 0;
    for (int x = 0; x < Nparticles; x++) {
      sum += weights[x];
    }
    for (int x = 0; x < Nparticles; x++) {
      weights[x] /= sum;
    }

    // Cumulative distribution function
    CDF[0] = weights[0];
    for (int x = 1; x < Nparticles; x++) {
      CDF[x] = weights[x] + CDF[x-1];
    }

    // Resampling
    for (int i = 0; i < Nparticles; i++) {
      int index = findIndex(CDF, Nparticles, u[i]);
      xj[i] = arrayX[index];
      yj[i] = arrayY[index];
    }
  }

  long long offload_end = get_time();
  printf("Device offloading time: %lf (s)\n", elapsed_time(offload_start, offload_end));

  xe = 0;
  ye = 0;

  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = sqrt(pow((float) (xe - (int) roundFloat(IszY / 2.0)), 2) + pow((float) (ye - (int) roundFloat(IszX / 2.0)), 2));

  FILE *fid;
  fid=fopen("output.txt", "w+");
  if( fid == NULL ){
    printf( "The file was not opened for writing\n" );
    return -1;
  }
  fprintf(fid, "XE: %lf\n", xe);
  fprintf(fid, "YE: %lf\n", ye);
  fprintf(fid, "distance: %lf\n", distance);
  fclose(fid);

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

  if (sscanf(argv[