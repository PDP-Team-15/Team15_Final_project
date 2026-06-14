```cpp
#include <sys/types.h>
#include <chrono>
#include <cuda_runtime.h>
#include "3D_helper.h"

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

#define FACTOR_CHIP  0.5

float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;
  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  float* tIn      = (float*) calloc(size,sizeof(float));
  float* pIn      = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*)malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));
  float* sel; 

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  long long start = get_time();

  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);

  dim3 blocks(numCols, numRows, 1);
  dim3 threads(256, 1, 1);

  float *d_tIn, *d_pIn, *d_tOut;
  cudaMalloc((void**)&d_tIn, size * sizeof(float));
  cudaMalloc((void**)&d_pIn, size * sizeof(float));
  cudaMalloc((void**)&d_tOut, size * sizeof(float));
  cudaMemcpy(d_tIn, tIn, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pIn, pIn, size * sizeof(float), cudaMemcpyHostToDevice);

  for(int j = 0; j < iterations; j++)
  {
    kernel<<<blocks, threads>>>(d_tIn, d_pIn, d_tOut, numCols, numRows, layers, ce, cw, cn, cs, ct, cb, cc, stepDivCap, amb_temp);
    cudaDeviceSynchronize();
    float *temp = tIn;
    tIn = tOut;
    tOut = temp;
  }

  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start_event, stop_event);
  printf("Average kernel execution time %f (us)\n", kernel_time);

  if (iterations & 01) {
    cudaMemcpy(tIn, d_tIn, size * sizeof(float), cudaMemcpyDeviceToHost);
  }
  else {
    cudaMemcpy(tIn, d_tOut, size * sizeof(float), cudaMemcpyDeviceToHost);
  }

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  cudaFree(d_tIn);
  cudaFree(d_pIn);
  cudaFree(d_tOut);
  return 0;
}

__global__ void kernel(float *tIn, float *pIn, float *tOut, int numCols, int numRows, int layers, float ce, float cw, float cn, float cs, float ct, float cb, float cc, float stepDivCap, float amb_temp)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int c = i + j * numCols;
  int xy = numCols * numRows;

  int W = (i == 0)        ? c : c - 1;
  int E = (i == numCols-1)     ? c : c + 1;
  int N = (j == 0)        ? c : c - numCols;
  int S = (j == numRows-1)     ? c : c + numCols;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  for (int k = 1; k < layers-1; ++k) {
    temp1 = temp2;
    temp2 = temp3;
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
              + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
}

```