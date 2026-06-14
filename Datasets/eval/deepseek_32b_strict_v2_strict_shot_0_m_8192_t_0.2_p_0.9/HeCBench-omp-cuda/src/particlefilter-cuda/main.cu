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
#include "helper_cuda.h"

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

__global__ void updateParticles(float * arrayX, float * arrayY, float * xj, float * yj, int * seed, int Nparticles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Nparticles) {
    arrayX[i] = xj[i];
    arrayY[i] = yj[i];
    seed[i] = (A * seed[i] + C) % M;
    float u = randu(seed, i);
    float v = randu(seed, i);
    arrayX[i] += 1.0f + 5.0f * (sqrtf(-2.0f * logf(u)) * cosf(2.0f * PI * v));
    seed[i] = (A * seed[i] + C) % M;
    u = randu(seed, i);
    v = randu(seed, i);
    arrayY[i] += -2.0f + 2.0f * (sqrtf(-2.0f * logf(u)) * cosf(2.0f * PI * v));
  }
}

__global__ void computeLikelihood(unsigned char * I, float * likelihood, int * ind, float * arrayX, float * arrayY, int * objxy, int countOnes, int max_size, int k, int IszX, int IszY, int Nfr, int Nparticles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Nparticles) {
    int rnd_iX = (arrayX[i] - (int)arrayX[i]) < 0.5f ? (int)arrayX[i] : (int)arrayX[i] + 1;
    int rnd_iY = (arrayY[i] - (int)arrayY[i]) < 0.5f ? (int)arrayY[i] : (int)arrayY[i] + 1;
    for (int y = 0; y < countOnes; y++) {
      int indX = rnd_iX + objxy[y * 2 + 1];
      int indY = rnd_iY + objxy[y * 2];
      ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
      if (ind[i * countOnes + y] >= max_size)
        ind[i * countOnes + y] = 0;
    }
    float likelihoodSum = 0.0f;
    for (int x = 0; x < countOnes; x++) {
      likelihoodSum += ((I[ind[i * countOnes + x]] - 100) * (I[ind[i * countOnes + x]] - 100) -
                       (I[ind[i * countOnes + x]] - 228) * (I[ind[i * countOnes + x]] - 228)) / 50.0f;
    }
    likelihood[i] = likelihoodSum / countOnes - SCALE_FACTOR;
  }
}

__global__ void updateWeights(float * weights, float * likelihood, float * partial_sums, int Nparticles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Nparticles) {
    weights[i] = weights[i] * expf(likelihood[i]);
  }
}

__global__ void computeCDF(float * CDF, float * weights, int Nparticles) {
  if (threadIdx.x == 0) {
    CDF[0] = weights[0];
    for (int x = 1; x < Nparticles; x++) {
      CDF[x] = weights[x] + CDF[x - 1];
    }
  }
}

__global__ void resampleParticles(float * u, float * CDF, float * arrayX, float * arrayY, float * xj, float * yj, int Nparticles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < Nparticles) {
    int index = -1;
    for (int x = 0; x < Nparticles; x++) {
      if (CDF[x] >= u[i]) {
        index = x;
        break;
      }
    }
    if (index == -1)
      index = Nparticles - 1;
    xj[i] = arrayX[index];
    yj[i] = arrayY[index];
  }
}

int particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
  int max_size = IszX * IszY * Nfr;

  float xe = roundFloat(IszY / 2.0f);
  float ye = roundFloat(IszX / 2.0f);

  int radius = 5;
  int diameter = radius * 2 - 1;
  int * disk = (int*)calloc(diameter * diameter, sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  for (int x = 0; x < diameter; x++) {
    for (int y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int * objxy = (int *)calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  float * weights = (float *)calloc(Nparticles, sizeof(float));
  for (int x = 0; x < Nparticles; x++) {
    weights[x] = 1.0f / ((float)Nparticles);
  }

  float * likelihood = (float *)calloc(Nparticles + 1, sizeof(float));
  float * partial_sums = (float *)calloc(Nparticles + 1, sizeof(float));
  float * arrayX = (float *)calloc(Nparticles, sizeof(float));
  float * arrayY = (float *)calloc(Nparticles, sizeof(float));
  float * xj = (float *)calloc(Nparticles, sizeof(float));
  float * yj = (float *)calloc(Nparticles, sizeof(float));
  float * CDF = (float *)calloc(Nparticles, sizeof(float));
  int * ind = (int*)calloc(countOnes * Nparticles, sizeof(int));
  float * u = (float *)calloc(Nparticles, sizeof(float));

  unsigned char * d_I;
  float * d_likelihood, *d_partial_sums, *d_arrayX, *d_arrayY, *d_xj, *d_yj, *d_CDF;
  int * d_ind, *d_objxy;
  float * d_weights;
  int * d_seed;

  checkCudaErrors(cudaMalloc((void**)&d_I, IszX * IszY * Nfr * sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void**)&d_likelihood, (Nparticles + 1) * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_partial_sums, (Nparticles + 1) * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_arrayX, Nparticles * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_arrayY, Nparticles * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_xj, Nparticles * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_yj, Nparticles * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_CDF, Nparticles * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_ind, countOnes * Nparticles * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_weights, Nparticles * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_seed, Nparticles * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_objxy, 2 * countOnes * sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_I, I, IszX * IszY * Nfr * sizeof(unsigned char), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_objxy, objxy, 2 * countOnes * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_seed, seed, Nparticles * sizeof(int), cudaMemcpyHostToDevice));

  for (int k = 1; k < Nfr; k++) {
    int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    updateParticles<<<num_blocks, BLOCK_SIZE>>>(d_arrayX, d_arrayY, d_xj, d_yj, d_seed, Nparticles);
    checkCudaErrors(cudaDeviceSynchronize());

    computeLikelihood<<<num_blocks, BLOCK_SIZE>>>(d_I, d_likelihood, d_ind, d_arrayX, d_arrayY, d_objxy, countOnes, max_size, k, IszX, IszY, Nfr, Nparticles);
    checkCudaErrors(cudaDeviceSynchronize());

    updateWeights<<<num_blocks, BLOCK_SIZE>>>(d_weights, d_likelihood, d_partial_sums, Nparticles);
    checkCudaErrors(cudaDeviceSynchronize());

    computeCDF<<<1, 1>>>(d_CDF, d_weights, Nparticles);
    checkCudaErrors(cudaDeviceSynchronize());

    resampleParticles<<<num_blocks, BLOCK_SIZE>>>(d_u, d_CDF, d_arrayX, d_arrayY, d_xj, d_yj, Nparticles);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  checkCudaErrors(cudaMemcpy(I, d_I, IszX * IszY * Nfr * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(seed, d_seed, Nparticles * sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_I));
  checkCudaErrors(cudaFree(d_likelihood));
  checkCudaErrors(cudaFree(d_partial_sums));
  checkCudaErrors(cudaFree(d_arrayX));
  checkCudaErrors(cudaFree(d_arrayY));
  checkCudaErrors(cudaFree(d_xj));
  checkCudaErrors(cudaFree(d_yj));
  checkCudaErrors(cudaFree(d_CDF));
  checkCudaErrors(cudaFree(d_ind));
  checkCudaErrors(cudaFree(d_weights));
  checkCudaErrors(cudaFree(d_seed));
  checkCudaErrors(cudaFree(d_objxy));

  free(disk);
  free(objxy);
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
  // Main function remains the same, calling particleFilter with CUDA implementation
}