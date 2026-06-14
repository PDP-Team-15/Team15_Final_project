```c
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

float randu(int * seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((float) M));
}

float randn(int * seed, int index) {
  float u = randu(seed, index);
  float v = randu(seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

float roundFloat(float value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) {
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

void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
      }
    }
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

void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
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
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
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

void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int k;
  int max_size = IszX * IszY * Nfr;

  int x0 = (int) roundFloat(IszY / 2.0);
  int y0 = (int) roundFloat(IszX / 2.0);
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

  unsigned char * newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
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

  int num_threads = omp_get_max_threads();
  float* likelihood_GPU;
  float* arrayX_GPU;
  float* arrayY_GPU;
  float* xj_GPU;
  float* yj_GPU;
  float* CDF_GPU;
  float* partial_sums_GPU;
  float* u_GPU;
  int* objxy_GPU;
  int* ind_GPU;
  int* seed_GPU;
  float* weights_GPU;
  unsigned char* I_GPU;

  likelihood_GPU = (float*) malloc((Nparticles + 1)*sizeof(float));
  arrayX_GPU = (float*) malloc(Nparticles*sizeof(float));
  arrayY_GPU = (float*) malloc(Nparticles*sizeof(float));
  xj_GPU = (float*) malloc(Nparticles*sizeof(float));
  yj_GPU = (float*) malloc(Nparticles*sizeof(float));
  CDF_GPU = (float*) malloc(Nparticles*sizeof(float));
  u_GPU = (float*) malloc(Nparticles*sizeof(float));
  objxy_GPU = (int*) malloc(2*countOnes*sizeof(int));
  ind_GPU = (int*) malloc(countOnes*Nparticles*sizeof(int));
  seed_GPU = (int*) malloc(Nparticles*sizeof(int));
  weights_GPU = (float*) malloc(Nparticles*sizeof(float));
  I_GPU = (unsigned char*) malloc(IszX * IszY * Nfr * sizeof(unsigned char));

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    xj_GPU[i] = xj[i];
    yj_GPU[i] = yj[i];
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    weights_GPU[i] = weights[i];
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles + 1; i++) {
    likelihood_GPU[i] = 0.0f;
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    arrayX_GPU[i] = 0.0f;
    arrayY_GPU[i] = 0.0f;
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    CDF_GPU[i] = 0.0f;
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    u_GPU[i] = 0.0f;
  }

  #pragma omp parallel for
  for (int i = 0; i < countOnes*Nparticles; i++) {
    ind_GPU[i] = 0;
  }

  #pragma omp parallel for
  for (int i = 0; i < 2*countOnes; i++) {
    objxy_GPU[i] = 0;
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    seed_GPU[i] = seed[i];
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    I_GPU[i * IszY * Nfr] = I[i * IszY * Nfr];
  }

  long long start = get_time();
  for (int k = 1; k < Nfr; k++) {
    #pragma omp parallel for
    for (int i = 0; i < Nparticles + 1; i++) {
      likelihood_GPU[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < Nparticles; i++) {
      arrayX_GPU[i] = 0.0f;
      arrayY_GPU[i] = 0.0f;
    }

    kernel_likelihood(arrayX_GPU, arrayY_GPU, xj_GPU, yj_GPU, ind_GPU, objxy_GPU, likelihood_GPU, I_GPU, weights_GPU, seed_GPU, partial_sums_GPU, Nparticles, countOnes, IszY, Nfr, k, max_size);

    #pragma omp parallel for
    for (int i = 0; i < Nparticles + 1; i++) {
      partial_sums_GPU[i] = 0.0f;
    }

    kernel_sum(partial_sums_GPU, Nparticles);

    #pragma omp parallel for
    for (int i = 0; i < Nparticles; i++) {
      weights_GPU[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < Nparticles; i++) {
      CDF_GPU[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < Nparticles; i++) {
      u_GPU[i] = 0.0f;
    }

    kernel_normalize_weights(weights_GPU, partial_sums_GPU, CDF_GPU, u_GPU, seed_GPU, Nparticles);

    kernel_find_index(arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, Nparticles);
  }

  long long end = get_time();
  printf("Average execution time of kernels: %f (s)\n", elapsed_time(start, end) / (Nfr-1));

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    arrayX[i] = arrayX_GPU[i];
    arrayY[i] = arrayY_GPU[i];
  }

  #pragma omp parallel for
  for (int i = 0; i < Nparticles; i++) {
    weights[i] = weights_GPU[i];
  }

  free(likelihood_GPU);
  free(arrayX_GPU);
  free(arrayY_GPU);
  free(xj_GPU);
  free(yj_GPU);
  free(CDF_GPU);
  free(partial_sums_GPU);
  free(objxy_GPU);
  free(u_GPU);
  free(ind_GPU);
  free(seed_GPU);
  free(weights_GPU);
  free(I_GPU);

  long long offload_end = get_time();
  printf("Device offloading time: %f (s)\n", elapsed_time(offload_start, offload_end));

  xe = 0;
  ye = 0;

  for (int x = 0; x < Nparticles; x++) {
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
   