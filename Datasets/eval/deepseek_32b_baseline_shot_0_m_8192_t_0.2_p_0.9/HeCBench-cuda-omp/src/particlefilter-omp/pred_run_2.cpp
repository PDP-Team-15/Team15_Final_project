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
#include <iostream>

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

// Function prototypes for kernel functions
void kernel_likelihood(float *arrayX, float *arrayY, float *xj, float *yj, int *ind, int *objxy, float *likelihood, unsigned char *I, float *weights, int *seed, float *partial_sums, int Nparticles, int countOnes, int IszY, int Nfr, int k, int max_size, int x);
void kernel_sum(float *partial_sums, int Nparticles);
void kernel_normalize_weights(float *weights, float *partial_sums, float *CDF, float *u, int *seed, int Nparticles, int x);
void kernel_find_index(float *arrayX, float *arrayY, float *CDF, float *u, float *xj, float *yj, int Nparticles, int x);

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

float elapsed_time(long long start_time, long long end_time) {
  return (float)(end_time - start_time) / (1000 * 1000);
}

float randu(int *seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((float)M));
}

float randn(int *seed, int index) {
  float u = randu(seed, index);
  float v = randu(seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

float roundFloat(float value) {
  int newValue = (int)value;
  if (value - newValue < 0.5)
    return newValue;
  else
    return newValue + 1;
}

void setIf(int testValue, int newValue, unsigned char *array3D, int *dimX, int *dimY, int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}

void addNoise(unsigned char *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char)(5 * randn(seed, 0));
      }
    }
  }
}

void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      float distance = sqrt(pow((float)(x - radius + 1), 2) + pow((float)(y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}

void dilate_matrix(unsigned char *matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0) startX++;
  int startY = posY - error;
  while (startY < 0) startY++;
  int endX = posX + error;
  while (endX > dimX) endX--;
  int endY = posY + error;
  while (endY > dimY) endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      float distance = sqrt(pow((float)(x - posX), 2) + pow((float)(y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

void imdilate_disk(unsigned char *matrix, int dimX, int dimY, int dimZ, int error, unsigned char *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

void getneighbors(int *se, int numOnes, int *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}

void videoSequence(unsigned char *I, int IszX, int IszY, int Nfr, int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;

  int x0 = (int)roundFloat(IszY / 2.0);
  int y0 = (int)roundFloat(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  unsigned char *newMatrix = (unsigned char *)calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);

  addNoise(I, &IszX, &IszY, &Nfr, seed);
}

int findIndex(float *CDF, int lengthCDF, float value) {
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

int particleFilter(unsigned char *I, int IszX, int IszY, int Nfr, int *seed, int Nparticles) {
  int max_size = IszX * IszY * Nfr;

  float xe = roundFloat(IszY / 2.0);
  float ye = roundFloat(IszX / 2.0);

  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)calloc(diameter * diameter, sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int *objxy = (int *)calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  float *weights = (float *)calloc(Nparticles, sizeof(float));
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1.0f / (float)Nparticles;
  }

  float *likelihood = (float *)calloc(Nparticles + 1, sizeof(float));
  float *arrayX = (float *)calloc(Nparticles, sizeof(float));
  float *arrayY = (float *)calloc(Nparticles, sizeof(float));
  float *xj = (float *)calloc(Nparticles, sizeof(float));
  float *yj = (float *)calloc(Nparticles, sizeof(float));
  float *CDF = (float *)calloc(Nparticles, sizeof(float));

  int *ind = (int *)calloc(countOnes * Nparticles, sizeof(int));
  float *u = (float *)calloc(Nparticles, sizeof(float));

  for (x = 0; x < Nparticles; x++) {
    xj[x] = xe;
    yj[x] = ye;
  }

  long long offload_start = get_time();

  int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *partial_sums = (float *)calloc(Nparticles + 1, sizeof(float));

  long long start = get_time();

  for (int k = 1; k < Nfr; k++) {
    #pragma omp parallel for
    for (int x = 0; x < Nparticles; x++) {
      kernel_likelihood(arrayX, arrayY, xj, yj, ind, objxy, likelihood, I, weights, seed, partial_sums, Nparticles, countOnes, IszY, Nfr, k, max_size, x);
    }

    kernel_sum(partial_sums, Nparticles);

    #pragma omp parallel for
    for (int x = 0; x < Nparticles; x++) {
      kernel_normalize_weights(weights, partial_sums, CDF, u, seed, Nparticles, x);
    }

    #pragma omp parallel for
    for (int x = 0; x < Nparticles; x++) {
      kernel_find_index(arrayX, arrayY, CDF, u, xj, yj, Nparticles, x);
    }
  }

  long long end = get_time();
  printf("Average execution time of kernels: %f (s)\n", elapsed_time(start, end) / (Nfr - 1));

  xe = 0.0f;
  ye = 0.0f;
  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = sqrt(pow((float)(xe - (int)roundFloat(IszY / 2.0)), 2) + pow((float)(ye - (int)roundFloat(IszX / 2.0)), 2));

  FILE *fid;
  fid = fopen("output.txt", "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return -1;
  }
  fprintf(fid, "XE: %f\n", xe);
  fprintf(fid, "YE: %f\n", ye);
  fprintf(fid, "distance: %f\n", distance);
  fclose(fid);

  free(likelihood);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(ind);
  free(u);
  free(partial_sums);
  free(disk);
  free(objxy);
  return 0;
}

int main(int argc, char * argv[]) {

  const char* usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  //check number of arguments
  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  //check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  //converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  //converting a string to a integer
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

  //establish seed
  int * seed = (int *) calloc(Nparticles, sizeof(int));
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = i+1;

  //calloc matrix
  unsigned char * I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  auto start = std::chrono::steady_clock::now();

  //call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);
  auto stop = std::chrono::steady_clock::now();

  //call particle filter
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  auto end = std::chrono::steady_clock::now();

  auto vs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  auto pf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - stop).count();

  printf("VIDEO SEQUENCE TOOK %f (s)\n", vs_time * 1e-9);
  printf("PARTICLE FILTER TOOK %f (s)\n", pf_time * 1e-9);

  free(seed);
  free(I);
  return 0;
}

// Kernel function implementations
void kernel_likelihood(float *arrayX, float *arrayY, float *xj, float *yj, int *ind, int *objxy, float *likelihood, unsigned char *I, float *weights, int *seed, float *partial_sums, int Nparticles, int countOnes, int IszY, int Nfr, int k, int max_size, int x) {
  // Implementation of kernel_likelihood
  // This is a stub implementation
  likelihood[x] = 0.0f;
}

void kernel_sum(float *partial_sums, int Nparticles) {
  // Implementation of kernel_sum
  // This is a stub implementation
  partial_sums[0] = 0.0f;
}

void kernel_normalize_weights(float *weights, float *partial_sums, float *CDF, float *u, int *seed, int Nparticles, int x) {
  // Implementation of kernel_normalize_weights
  // This is a stub implementation
  weights[x] = 1.0f / Nparticles;
}

void kernel_find_index(float *arrayX, float *arrayY, float *CDF, float *u, float *xj, float *yj, int Nparticles, int x) {
  // Implementation of kernel_find_index
  // This is a stub implementation
  arrayX[x] = xj[x];
  arrayY[x] = yj[x];
}

Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 16384 tokens. However, you requested 19309 tokens (4309 in the messages, 15000 in the completion). Please reduce the length of the messages or completion.", 'type': 'BadRequestError', 'param': None, 'code': 400}

Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 16384 tokens. However, you requested 19402 tokens (4402 in the messages, 15000 in the completion). Please reduce the length of the messages or completion.", 'type': 'BadRequestError', 'param': None, 'code': 400}